"""Push-driven map traits for B01 Q10 devices.

Map-related state arrives on two independent streams:

* map and trace packets are decoded by the protocol layer;
* restricted zones and virtual walls arrive as ordinary DPS values.

``MapDpsTrait`` owns the low-level DPS read model. ``MapContentTrait`` depends
on it and combines that state with the latest map/trace packets through the pure
functions in :mod:`roborock.map.b01_q10_render`. The high-level trait keeps only
one grouped source snapshot and one replace-whole rendered result; calibration,
path placement and overlay placement are not independently mutable trait state.
"""

import logging
from dataclasses import dataclass, field, replace

from roborock.data import RoborockBase
from roborock.data.b01_q10.b01_q10_code_mappings import B01_Q10_DP
from roborock.devices.traits.common import DpsDataConverter, TraitUpdateListener
from roborock.exceptions import RoborockException
from roborock.map.b01_grid_layers import GridCalibration
from roborock.map.b01_q10_map_parser import (
    B01Q10MapParserConfig,
    Q10MapPacket,
    Q10Point,
    Q10Room,
    Q10TracePacket,
)
from roborock.map.b01_q10_overlays import Q10Zone, parse_virtual_wall_blob, parse_zone_blob
from roborock.map.b01_q10_render import Q10MapOverlays, Q10MapRender, draw_path_on_map, render_q10_map

from .common import UpdatableTrait

_LOGGER = logging.getLogger(__name__)


@dataclass
class MapDps(RoborockBase):
    """Low-level map values delivered in the Q10 DPS stream."""

    restricted_zone_up: str | None = field(default=None, metadata={"dps": B01_Q10_DP.RESTRICTED_ZONE_UP})
    virtual_wall_up: str | None = field(default=None, metadata={"dps": B01_Q10_DP.VIRTUAL_WALL_UP})


class MapDpsTrait(MapDps, UpdatableTrait):
    """Converter-backed read model for map-related DPS values."""

    _CONVERTER = DpsDataConverter.from_dataclass(MapDps)

    def __init__(self) -> None:
        MapDps.__init__(self)
        UpdatableTrait.__init__(self, command=None, logger=_LOGGER)


@dataclass(frozen=True)
class Q10MapSource:
    """The latest map-protocol inputs, replaced atomically on every push."""

    map_packet: Q10MapPacket | None = None
    trace_packet: Q10TracePacket | None = None


class MapContentTrait(TraitUpdateListener):
    """High-level composed Q10 map view.

    Map and trace packet updates replace :attr:`_source`; DPS updates are owned
    by the injected :class:`MapDpsTrait`. Rendering always produces one new
    :class:`Q10MapRender`, keeping the externally visible fields consistent.
    """

    def __init__(
        self,
        map_dps: MapDpsTrait | None = None,
        *,
        map_parser_config: B01Q10MapParserConfig | None = None,
    ) -> None:
        TraitUpdateListener.__init__(self, logger=_LOGGER)
        self._config = map_parser_config or B01Q10MapParserConfig()
        self._map_dps = map_dps or MapDpsTrait()
        self._source = Q10MapSource()
        self._render: Q10MapRender | None = None
        self._map_dps.add_update_listener(self._map_dps_updated)

    @property
    def image_content(self) -> bytes | None:
        """The rendered base map PNG, if a map has been pushed."""
        return self._render.image_content if self._render else None

    @property
    def rooms(self) -> list[Q10Room]:
        """Rooms reported by the device."""
        return self._render.rooms if self._render else []

    @property
    def calibration(self) -> GridCalibration | None:
        """The calibration used by the current rendered result."""
        return self._render.calibration if self._render else None

    @property
    def path(self) -> list[Q10Point]:
        """Full path from the latest trace packet."""
        trace = self._source.trace_packet
        return trace.points if trace else []

    @property
    def robot_position(self) -> Q10Point | None:
        """Current robot position from the latest trace packet."""
        trace = self._source.trace_packet
        return trace.robot_position if trace else None

    @property
    def robot_heading(self) -> int | None:
        """Current robot heading from the latest trace packet."""
        trace = self._source.trace_packet
        return trace.heading if trace else None

    @property
    def zones(self) -> list[Q10Zone]:
        """Restricted zones decoded from the low-level DPS trait."""
        return parse_zone_blob(self._map_dps.restricted_zone_up)

    @property
    def virtual_walls(self) -> list[Q10Zone]:
        """Virtual walls decoded from the low-level DPS trait."""
        return parse_virtual_wall_blob(self._map_dps.virtual_wall_up)

    def update_from_map_packet(self, packet: Q10MapPacket) -> None:
        """Replace the current map packet and compose a consistent result."""
        source = replace(self._source, map_packet=packet)
        render = self._compose(source)
        if render is None:
            return
        self._source = source
        self._render = render
        self._notify_update()

    def update_from_trace_packet(self, packet: Q10TracePacket) -> None:
        """Replace the complete current-session trace packet."""
        self._source = replace(self._source, trace_packet=packet)
        if self._source.map_packet is not None:
            self._rebuild()
        self._notify_update()

    def render_path_on_map(
        self,
        *,
        line_color: tuple[int, int, int, int] = (235, 64, 52, 255),
        position_color: tuple[int, int, int, int] = (255, 211, 0, 255),
    ) -> bytes:
        """Return a PNG with the path, robot, zones and walls drawn."""
        if self._render is None:
            raise RoborockException("No map available; no map has been pushed yet")
        if self._render.calibration is None:
            raise RoborockException(
                "No calibration available; a cleaning path must be captured (pushed) during a clean"
            )
        return draw_path_on_map(
            self._render,
            config=self._config,
            line_color=line_color,
            position_color=position_color,
        )

    def _map_dps_updated(self) -> None:
        """Recompose placed overlays after the low-level DPS state changes."""
        if self._source.map_packet is not None:
            self._rebuild()
        self._notify_update()

    def _compose(self, source: Q10MapSource) -> Q10MapRender | None:
        """Compose a source snapshot, preserving the previous result on error."""
        if source.map_packet is None:
            return None
        try:
            return render_q10_map(
                source.map_packet,
                source.trace_packet,
                Q10MapOverlays(zones=tuple(self.zones), virtual_walls=tuple(self.virtual_walls)),
                config=self._config,
            )
        except RoborockException as ex:
            _LOGGER.debug("Failed to render Q10 map packet: %s", ex)
            return None

    def _rebuild(self) -> None:
        """Replace the derived render from the current source snapshot."""
        render = self._compose(self._source)
        if render is not None:
            self._render = render

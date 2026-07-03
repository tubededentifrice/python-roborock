"""Map content trait for B01 Q10 devices.

Unlike the v1 / Q7 maps, the Q10 has no synchronous "get map" command, so this
trait is purely push-driven and mirrors the Q10 ``StatusTrait`` contract:

- The device pushes its current map/path as protocol-301 ``MAP_RESPONSE``
  messages (a ``dpRequestDps`` nudges it to do so). The protocol layer decodes
  each push into a typed :class:`~roborock.map.b01_q10_map_parser.Q10MapPacket`
  or :class:`~roborock.map.b01_q10_map_parser.Q10TracePacket`, and the
  ``Q10PropertiesApi`` subscribe loop routes those to
  :meth:`MapContentTrait.update_from_map_packet` /
  :meth:`MapContentTrait.update_from_trace_packet`.
- The no-go / no-mop zones and virtual walls arrive separately as status data
  points, fanned in via :meth:`update_from_dps`.

The trait is deliberately just state management: it accumulates those pushed
inputs (the map packet, the path, the overlays, a solved calibration) and, on
every change, asks :func:`~roborock.map.b01_q10_render.render_q10_map` to compose
them into one :class:`~roborock.map.b01_q10_render.Q10MapRender`. The
low-level pixel work (layer decomposition, erase blanking, world->pixel overlay
placement, path drawing) lives in that map module, not here. Consumers read the
cached fields (``image_content``, ``map_data``, ``rooms``, ``layers``, ``path``,
``robot_position``, ``zones``, ``virtual_walls``) or register a callback with
:meth:`add_update_listener` to learn when new content arrives.

Unlike the Q7, the Q10 map payload is unencrypted, so no map key is required.
"""

import logging
from typing import Any

from vacuum_map_parser_base.map_data import MapData

from roborock.data.b01_q10.b01_q10_code_mappings import B01_Q10_DP
from roborock.devices.traits.common import TraitUpdateListener
from roborock.exceptions import RoborockException
from roborock.map.b01_grid_layers import GridCalibration, GridLayers
from roborock.map.b01_q10_map_parser import (
    B01Q10MapParserConfig,
    Q10MapPacket,
    Q10Point,
    Q10Room,
    Q10TracePacket,
)
from roborock.map.b01_q10_overlays import (
    Q10Zone,
    parse_virtual_wall_blob,
    parse_zone_blob,
)
from roborock.map.b01_q10_render import Q10MapRender, draw_path_on_map, render_q10_map, solve_q10_calibration

_LOGGER = logging.getLogger(__name__)


class MapContentTrait(TraitUpdateListener):
    """Trait holding the most recently pushed parsed map content for Q10 devices.

    Holds the pushed inputs plus the single composed
    :class:`~roborock.map.b01_q10_render.Q10MapRender` derived from them (see the
    module docstring). Every mutator updates one input, rebuilds the render in one
    shot, and notifies listeners, so the exposed state is always consistent with
    one set of inputs rather than a pile of independently-mutated fields.
    """

    def __init__(
        self,
        *,
        map_parser_config: B01Q10MapParserConfig | None = None,
    ) -> None:
        TraitUpdateListener.__init__(self, logger=_LOGGER)
        self._config = map_parser_config or B01Q10MapParserConfig()
        # Pushed inputs, accumulated from the device's map / trace / DPS streams.
        self._packet: Q10MapPacket | None = None
        self._path: list[Q10Point] = []
        self._robot_position: Q10Point | None = None
        self._robot_heading: int | None = None
        self._zones: list[Q10Zone] = []
        self._virtual_walls: list[Q10Zone] = []
        self._calibration: GridCalibration | None = None
        # The single composed result, rebuilt from the inputs on every change.
        self._render: Q10MapRender | None = None

    # -- Read-only view of the composed map --------------------------------------

    @property
    def image_content(self) -> bytes | None:
        """The rendered base map (PNG), or ``None`` if no map has been pushed."""
        return self._render.image_content if self._render else None

    @property
    def map_data(self) -> MapData | None:
        """Parsed map data (image metadata, room names, placed overlays)."""
        return self._render.map_data if self._render else None

    @property
    def rooms(self) -> list[Q10Room]:
        """Rooms (segments) reported by the device, with ids and names."""
        return self._render.rooms if self._render else []

    @property
    def layers(self) -> GridLayers | None:
        """Separable map layers (background / wall / floor / per-room) in
        grid-pixel space, each renderable to a transparent PNG for compositing."""
        return self._render.layers if self._render else None

    @property
    def calibration(self) -> GridCalibration | None:
        """World<->pixel transform, solved from a cleaning path via
        :meth:`solve_calibration` (``None`` until one has been fitted)."""
        return self._calibration

    @property
    def path(self) -> list[Q10Point]:
        """Full path of the current cleaning session (oldest point first).

        The robot accumulates this server-side and serves the whole trajectory so
        far in one packet, so it is complete even if we connect mid-session. Only
        populated while a cleaning session is active."""
        return self._path

    @property
    def robot_position(self) -> Q10Point | None:
        """Current robot position (the most recent path point), if known."""
        return self._robot_position

    @property
    def robot_heading(self) -> int | None:
        """Current robot heading in degrees from the trace packet (``0`` = +x,
        ``+90`` = +y, ``±180`` = −x, ``−90`` = −y), if a trace has been pushed."""
        return self._robot_heading

    @property
    def zones(self) -> list[Q10Zone]:
        """Restricted zones (no-go / no-mop) in world coordinates, from the
        device's ``dpRestrictedZoneUp``. See :meth:`load_overlays`."""
        return self._zones

    @property
    def virtual_walls(self) -> list[Q10Zone]:
        """Virtual walls (line segments) in world coordinates."""
        return self._virtual_walls

    # -- Push handlers -----------------------------------------------------------

    def update_from_map_packet(self, packet: Q10MapPacket) -> None:
        """Render a pushed full-map packet into the cached map view.

        Rendering failures are logged and skipped (the previous map and listeners
        are left untouched) so a single bad push cannot tear down the subscribe
        loop.
        """
        render = self._compose(packet)
        if render is None:
            return
        self._packet = packet
        self._render = render
        self._notify_update()

    def update_from_trace_packet(self, packet: Q10TracePacket) -> None:
        """Cache the path / robot position / heading from a pushed trace packet."""
        self._path = packet.points
        self._robot_position = packet.robot_position
        self._robot_heading = packet.heading
        # The path/position only reach the rendered map_data once a calibration
        # places them in pixel space, so there is nothing to recompose until then
        # (skipping the rebuild keeps the frequent pre-calibration trace pushes
        # cheap -- the raster is unchanged by the path).
        if self._calibration is not None:
            self._rebuild()
        self._notify_update()

    def update_from_dps(self, decoded_dps: dict[B01_Q10_DP, Any]) -> None:
        """Decode any vector-overlay data points present in a DPS push.

        The Q10 pushes no-go / no-mop zones (``dpRestrictedZoneUp``) and virtual
        walls (``dpVirtualWallUp``) as status data points rather than inside the
        map packet, so the map trait joins the ``Q10PropertiesApi`` DPS fan-out
        like the other read-model traits instead of being special-cased by the
        orchestrator. Data points absent from this push leave the existing
        overlays untouched (a partial status push must not wipe them); a push
        carrying neither is a no-op.
        """
        if B01_Q10_DP.RESTRICTED_ZONE_UP not in decoded_dps and B01_Q10_DP.VIRTUAL_WALL_UP not in decoded_dps:
            return
        self.load_overlays(
            restricted_zone_up=decoded_dps.get(B01_Q10_DP.RESTRICTED_ZONE_UP),
            virtual_wall_up=decoded_dps.get(B01_Q10_DP.VIRTUAL_WALL_UP),
        )
        self._notify_update()

    def load_overlays(
        self,
        *,
        restricted_zone_up: bytes | str | None = None,
        virtual_wall_up: bytes | str | None = None,
    ) -> None:
        """Decode the device's vector-overlay blobs (from the status DPs).

        Pass the raw ``dpRestrictedZoneUp`` / ``dpVirtualWallUp`` values
        (``Q10Status.restricted_zone_up`` / ``virtual_wall_up``). Stores them as
        world-coordinate :attr:`zones` / :attr:`virtual_walls`, and -- once a
        calibration is available -- the rebuild places them onto ``map_data`` as
        ``no_go_areas`` / ``no_mopping_areas`` / ``walls`` in pixel space.

        ``None`` means "data point absent from this update" and leaves the
        existing value untouched (a partial status push must not wipe overlays).
        An explicit empty blob does clear them. Unlike the push handlers this does
        not notify listeners; :meth:`update_from_dps` does after calling it.
        """
        if restricted_zone_up is not None:
            self._zones = parse_zone_blob(restricted_zone_up)
        if virtual_wall_up is not None:
            self._virtual_walls = parse_virtual_wall_blob(virtual_wall_up)
        # As with the path, the zones/walls are only placed onto the map once a
        # calibration exists, so there is nothing to recompose until then.
        if self._calibration is not None:
            self._rebuild()

    # -- Calibration + rendering -------------------------------------------------

    def solve_calibration(self) -> GridCalibration | None:
        """Fit and cache the world<->pixel calibration from the current path.

        When the map packet's grid-frame header carries a calibration origin
        (ss07), only the resolution is fit -- around that fixed origin -- so a
        short path suffices; otherwise the full origin + resolution fit is used,
        which needs a reasonably dense cleaning path. Both inputs arrive as device
        pushes (the path is only populated during an active clean). Caches and
        returns the calibration (also stored on :attr:`calibration`), rebuilding
        the map so the erase zones and overlays are applied, or ``None`` if there
        is no map or the path is too short/featureless to fit.
        """
        if self._render is None or self._packet is None:
            return None
        calibration = solve_q10_calibration(self._render.layers, self._packet.header_calibration, self._path)
        if calibration is not None:
            self._calibration = calibration
            self._rebuild()
        return calibration

    def render_path_on_map(
        self,
        *,
        line_color: tuple[int, int, int, int] = (235, 64, 52, 255),
        position_color: tuple[int, int, int, int] = (255, 211, 0, 255),
    ) -> bytes:
        """Return the map image (PNG) with the session path + robot position drawn.

        Solves the calibration on demand if not already cached. Raises
        :class:`RoborockException` if there is no map, or no calibration can be
        fitted (e.g. no cleaning path captured yet).
        """
        if self._render is None:
            raise RoborockException("No map available; no map has been pushed yet")
        if self._calibration is None:
            self.solve_calibration()
        if self._render.calibration is None:
            raise RoborockException(
                "No calibration available; a cleaning path must be captured (pushed) during a clean"
            )
        return draw_path_on_map(
            self._render,
            config=self._config,
            path=self._path,
            robot_position=self._robot_position,
            robot_heading=self._robot_heading,
            zones=self._zones,
            virtual_walls=self._virtual_walls,
            line_color=line_color,
            position_color=position_color,
        )

    # -- Internals ---------------------------------------------------------------

    def _compose(self, packet: Q10MapPacket) -> Q10MapRender | None:
        """Compose ``packet`` with the current inputs, or ``None`` on failure."""
        try:
            return render_q10_map(
                packet,
                calibration=self._calibration,
                path=self._path,
                robot_position=self._robot_position,
                zones=self._zones,
                virtual_walls=self._virtual_walls,
                config=self._config,
            )
        except RoborockException as ex:
            _LOGGER.debug("Failed to render Q10 map packet: %s", ex)
            return None

    def _rebuild(self) -> None:
        """Recompose the cached map packet with the current inputs (no-op if no
        map has been pushed yet, e.g. overlays/trace arriving before the map)."""
        if self._packet is None:
            return
        render = self._compose(self._packet)
        if render is not None:
            self._render = render

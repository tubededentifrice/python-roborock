"""Tests for the Q10 B01 map content trait.

The Q10 map API is push-driven: the device publishes ``MAP_RESPONSE`` messages
which the protocol layer decodes into typed map/trace packets; the trait updates
its cached state from them via ``update_from_map_packet`` /
``update_from_trace_packet`` (there is no synchronous get-map request). These
tests cover that state management; the pixel/geometry work it drives is tested in
``tests/map/test_b01_q10_render.py``.
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from roborock.data.b01_q10.b01_q10_code_mappings import B01_Q10_DP
from roborock.devices.traits.b01.q10 import Q10PropertiesApi, create
from roborock.devices.traits.b01.q10.map import MapContentTrait
from roborock.exceptions import RoborockException
from roborock.map.b01_grid_layers import GridCalibration
from roborock.map.b01_q10_map_parser import (
    Q10HeaderCalibration,
    Q10Point,
    Q10TracePacket,
    parse_map_packet,
    parse_trace_packet,
)
from roborock.roborock_message import RoborockMessage, RoborockMessageProtocol

FIXTURE = Path("tests/map/testdata/b01_q10_map.bin")
TRACE_SESSION_FIXTURE = Path("tests/map/testdata/b01_q10_trace_session.bin")

# A header calibration whose pixel origin (0, 5) is usable (not a keepalive
# frame), so a short path can calibrate the fixture map.
_USABLE_HEADER = Q10HeaderCalibration(origin_x=0, origin_y=50, resolution=5, charger_x=0, charger_y=0, charger_phi=0)


def _map_message(
    payload: bytes, protocol: RoborockMessageProtocol = RoborockMessageProtocol.MAP_RESPONSE
) -> RoborockMessage:
    return RoborockMessage(protocol=protocol, payload=payload, version=b"B01")


def _trait_with_map() -> MapContentTrait:
    """A trait with the fixture map already pushed into it."""
    trait = MapContentTrait()
    trait.update_from_map_packet(parse_map_packet(FIXTURE.read_bytes()))
    return trait


def _floor_world_points(trait: MapContentTrait, cal: GridCalibration, count: int) -> list[Q10Point]:
    """``count`` world points lying on the map's floor under ``cal``."""
    layers = trait.layers
    assert layers is not None
    floor = [
        (px, py)
        for py in range(layers.height)
        for px in range(layers.width)
        if layers.cell_class(layers.grid[py * layers.width + px]) == "floor"
    ]
    return [Q10Point(*(int(v) for v in cal.pixel_to_world(px, py))) for px, py in floor[:count]]


def test_update_from_map_packet_populates_image_and_rooms() -> None:
    """A pushed 01 01 map packet populates the image, rooms and map data."""
    packet = parse_map_packet(FIXTURE.read_bytes())
    trait = MapContentTrait()
    updates: list[None] = []
    trait.add_update_listener(lambda: updates.append(None))

    trait.update_from_map_packet(packet)

    assert trait.image_content is not None
    assert trait.image_content[:8] == b"\x89PNG\r\n\x1a\n"
    assert {room.id: room.name for room in trait.rooms} == {2: "Living Room", 3: "Bedroom"}
    assert trait.map_data is not None
    assert trait.layers is not None
    assert len(updates) == 1


def test_update_from_trace_packet_populates_path_and_position() -> None:
    """A pushed 02 01 trace packet populates the path, position and heading."""
    trace = parse_trace_packet(TRACE_SESSION_FIXTURE.read_bytes())
    trait = MapContentTrait()
    updates: list[None] = []
    trait.add_update_listener(lambda: updates.append(None))

    trait.update_from_trace_packet(trace)

    assert len(trait.path) == 14
    assert (trait.path[0].x, trait.path[0].y) == (41, 64)
    assert trait.robot_position is not None
    assert (trait.robot_position.x, trait.robot_position.y) == (276, -1)
    assert trait.robot_heading == -34
    assert len(updates) == 1


def test_map_push_populates_layers() -> None:
    """A pushed map is also decomposed into separable layers."""
    trait = _trait_with_map()
    assert trait.layers is not None
    assert trait.layers.class_counts.get("floor") == 26
    assert {room.id for room in trait.layers.rooms} == {2, 3}


# --- Integration through the Q10PropertiesApi subscribe loop -----------------


@pytest.fixture
def message_queue() -> asyncio.Queue[RoborockMessage]:
    return asyncio.Queue()


@pytest.fixture
def mock_channel(message_queue: asyncio.Queue[RoborockMessage]) -> AsyncMock:
    async def mock_stream() -> AsyncGenerator[RoborockMessage, None]:
        while True:
            yield await message_queue.get()

    channel = AsyncMock()
    channel.subscribe_stream = Mock(return_value=mock_stream())
    return channel


@pytest.fixture
async def q10_api(mock_channel: AsyncMock) -> AsyncGenerator[Q10PropertiesApi, None]:
    api = create(mock_channel)
    await api.start()
    yield api
    await api.close()


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def test_subscribe_loop_routes_map_push(
    q10_api: Q10PropertiesApi,
    message_queue: asyncio.Queue[RoborockMessage],
) -> None:
    """A map pushed onto the stream is routed to the map trait by the loop."""
    assert q10_api.map.image_content is None

    message_queue.put_nowait(_map_message(FIXTURE.read_bytes()))

    await _wait_for(lambda: q10_api.map.image_content is not None)
    assert {room.id: room.name for room in q10_api.map.rooms} == {2: "Living Room", 3: "Bedroom"}


async def test_subscribe_loop_routes_trace_push(
    q10_api: Q10PropertiesApi,
    message_queue: asyncio.Queue[RoborockMessage],
) -> None:
    """A trace pushed onto the stream is routed to the map trait by the loop."""
    assert not q10_api.map.path

    message_queue.put_nowait(_map_message(TRACE_SESSION_FIXTURE.read_bytes()))

    await _wait_for(lambda: bool(q10_api.map.path))
    assert q10_api.map.robot_position is not None


# --- Calibration + rendering -------------------------------------------------


def test_solve_calibration_needs_map_and_dense_path() -> None:
    """No map -> no calibration, even with a path pushed."""
    trait = MapContentTrait()
    trait.update_from_trace_packet(Q10TracePacket(points=[Q10Point(i, 0) for i in range(30)]))
    assert trait.solve_calibration() is None  # no map pushed yet


def test_solve_calibration_uses_header_origin_with_short_path() -> None:
    """A grid-frame header origin lets a short path calibrate and caches it."""
    trait = MapContentTrait()
    trait.update_from_map_packet(replace(parse_map_packet(FIXTURE.read_bytes()), header_calibration=_USABLE_HEADER))
    true = GridCalibration(resolution=20.0, origin_x=0.0, origin_y=5.0, y_sign=1)
    trait.update_from_trace_packet(Q10TracePacket(points=_floor_world_points(trait, true, 6)))
    assert len(trait.path) < 20  # far too short for the full origin+resolution fit

    cal = trait.solve_calibration()

    assert cal is not None
    assert (cal.origin_x, cal.origin_y) == (0.0, 5.0)  # straight from the header
    assert trait.calibration is cal
    # The solved calibration is applied to the map so overlays are placed.
    assert trait.map_data is not None
    assert trait.map_data.path is not None


def test_solve_calibration_short_path_without_header_returns_none() -> None:
    """Without a header origin a short path is still too sparse for the full fit."""
    trait = _trait_with_map()  # the fixture header is a keepalive frame
    true = GridCalibration(resolution=10.0, origin_x=0.0, origin_y=5.0, y_sign=1)
    trait.update_from_trace_packet(Q10TracePacket(points=_floor_world_points(trait, true, 6)))
    assert trait.solve_calibration() is None
    assert trait.calibration is None


def test_render_path_on_map_requires_map() -> None:
    trait = MapContentTrait()
    with pytest.raises(RoborockException, match="No map available"):
        trait.render_path_on_map()


def test_render_path_on_map_solves_and_renders() -> None:
    """render_path_on_map solves the calibration on demand and returns a PNG."""
    trait = MapContentTrait()
    trait.update_from_map_packet(replace(parse_map_packet(FIXTURE.read_bytes()), header_calibration=_USABLE_HEADER))
    true = GridCalibration(resolution=20.0, origin_x=0.0, origin_y=5.0, y_sign=1)
    trait.update_from_trace_packet(Q10TracePacket(points=_floor_world_points(trait, true, 6)))

    png = trait.render_path_on_map()

    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert trait.calibration is not None  # solved and cached on demand


def test_render_path_on_map_without_path_cannot_calibrate() -> None:
    """A map but no cleaning path -> no calibration -> a clear error."""
    trait = _trait_with_map()
    with pytest.raises(RoborockException, match="No calibration available"):
        trait.render_path_on_map()


# --- Overlays ----------------------------------------------------------------


def test_load_overlays_places_zones_after_calibration() -> None:
    """Decoded no-go / no-mop zones are placed on MapData once calibrated."""
    trait = MapContentTrait()
    trait.update_from_map_packet(replace(parse_map_packet(FIXTURE.read_bytes()), header_calibration=_USABLE_HEADER))
    true = GridCalibration(resolution=20.0, origin_x=0.0, origin_y=5.0, y_sign=1)
    trait.update_from_trace_packet(Q10TracePacket(points=_floor_world_points(trait, true, 6)))
    assert trait.solve_calibration() is not None

    def rect(zone_type: int, corners: list[tuple[int, int]]) -> bytes:
        out = bytes([zone_type, len(corners)])
        for x, y in corners:
            out += int.to_bytes(x & 0xFFFF, 2, "big") + int.to_bytes(y & 0xFFFF, 2, "big")
        return out.ljust(18, b"\x00")

    blob = bytes([1, 1]) + rect(0, [(0, 0), (40, 0), (40, 40), (0, 40)])
    trait.load_overlays(restricted_zone_up=blob)

    assert len(trait.zones) == 1
    assert trait.map_data is not None
    assert len(trait.map_data.no_go_areas or []) == 1


def test_load_overlays_partial_update_keeps_existing_zones() -> None:
    """A status push without the zone DP (None) must not wipe loaded zones."""
    trait = MapContentTrait()
    blob = (
        bytes([1, 1])
        + bytes([0, 4])
        + b"".join(int.to_bytes(v & 0xFFFF, 2, "big") for xy in [(0, 0), (4, 0), (4, 4), (0, 4)] for v in xy)
    )
    trait.load_overlays(restricted_zone_up=blob)
    assert len(trait.zones) == 1
    # A later partial update carrying only the (empty) virtual-wall DP.
    trait.load_overlays(restricted_zone_up=None, virtual_wall_up=b"\x00")
    assert len(trait.zones) == 1  # zones preserved
    assert trait.virtual_walls == []


def test_update_from_dps_decodes_overlay_data_points() -> None:
    """The map trait picks the overlay DPs out of a DPS push and decodes them."""
    trait = MapContentTrait()
    blob = (
        bytes([1, 1])
        + bytes([0, 4])
        + b"".join(int.to_bytes(v & 0xFFFF, 2, "big") for xy in [(0, 0), (4, 0), (4, 4), (0, 4)] for v in xy)
    )
    notified = []
    trait.add_update_listener(lambda: notified.append(True))

    trait.update_from_dps({B01_Q10_DP.RESTRICTED_ZONE_UP: blob})

    assert len(trait.zones) == 1
    assert notified  # listeners learn the overlays changed


def test_update_from_dps_without_overlay_data_points_is_noop() -> None:
    """A DPS push carrying neither overlay DP leaves the trait untouched."""
    trait = MapContentTrait()
    notified = []
    trait.add_update_listener(lambda: notified.append(True))

    trait.update_from_dps({B01_Q10_DP.BATTERY: 50})

    assert trait.zones == []
    assert trait.virtual_walls == []
    assert not notified

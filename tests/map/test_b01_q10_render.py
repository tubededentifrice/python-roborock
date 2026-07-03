"""Tests for composing a Q10 map packet + overlays into a rendered result.

The pixel-level machinery (erase blanking, world->pixel overlay placement, path
drawing, calibration fitting) lives in ``b01_q10_render``; these exercise it with
explicit calibrations so the geometry is deterministic. The map trait's own tests
cover the state management that drives this module.
"""

import io
from dataclasses import replace
from pathlib import Path

from PIL import Image

from roborock.map.b01_grid_layers import GridCalibration
from roborock.map.b01_q10_map_parser import (
    B01Q10MapParserConfig,
    Q10EraseZone,
    Q10HeaderCalibration,
    Q10Point,
    parse_map_packet,
)
from roborock.map.b01_q10_overlays import ZONE_TYPE_NO_GO, ZONE_TYPE_NO_MOP, Q10Zone
from roborock.map.b01_q10_render import (
    _Q10_RESOLUTIONS,
    draw_path_on_map,
    render_q10_map,
    solve_q10_calibration,
)

FIXTURE = Path("tests/map/testdata/b01_q10_map.bin")
CONFIG = B01Q10MapParserConfig()

# identity-ish calibration used across the geometry tests: world (x, y) -> grid
# pixel (x, 5 - y) over the fixture's 8x6 grid (top-down, no flip).
IDENTITY = GridCalibration(resolution=1.0, origin_x=0.0, origin_y=5.0, y_sign=1)


def _packet():
    return parse_map_packet(FIXTURE.read_bytes())


def _render(packet=None, **kwargs):
    packet = packet if packet is not None else _packet()
    args = dict(calibration=None, path=[], robot_position=None, zones=[], virtual_walls=[], config=CONFIG)
    args.update(kwargs)
    return render_q10_map(packet, **args)


def _floor_world_points(layers, cal: GridCalibration, count: int) -> list[Q10Point]:
    """``count`` world points lying on the map's floor under ``cal``."""
    floor = [
        (px, py)
        for py in range(layers.height)
        for px in range(layers.width)
        if layers.cell_class(layers.grid[py * layers.width + px]) == "floor"
    ]
    return [Q10Point(*(int(v) for v in cal.pixel_to_world(px, py))) for px, py in floor[:count]]


def test_render_base_map_without_calibration() -> None:
    """Without a calibration only the base raster/layers/rooms are produced."""
    render = _render()
    assert render.image_content[:8] == b"\x89PNG\r\n\x1a\n"
    assert render.map_data is not None
    assert render.calibration is None
    assert {room.id: room.name for room in render.rooms} == {2: "Living Room", 3: "Bedroom"}
    assert render.layers.class_counts.get("floor") == 26
    # Overlays are world-coordinate only, so nothing is placed yet.
    assert render.map_data.path is None


def test_render_places_path_and_position() -> None:
    """A calibration places the path + robot position onto MapData in pixels."""
    path = [Q10Point(1, 2), Q10Point(3, 2)]
    render = _render(calibration=IDENTITY, path=path, robot_position=Q10Point(3, 2))
    assert render.map_data.path is not None
    assert render.map_data.vacuum_position is not None
    # world (3, 2) -> grid pixel (3, 5 - 2) = (3, 3)
    assert (render.map_data.vacuum_position.x, render.map_data.vacuum_position.y) == (3.0, 3.0)


def test_render_places_zones_and_charger() -> None:
    """Decoded no-go / no-mop zones become pixel-space MapData areas + charger."""
    zones = [
        Q10Zone(type=ZONE_TYPE_NO_GO, vertices=[(0, 0), (4, 0), (4, 4), (0, 4)]),
        Q10Zone(type=ZONE_TYPE_NO_MOP, vertices=[(1, 1), (2, 1), (2, 2), (1, 2)]),
    ]
    render = _render(calibration=IDENTITY, path=[Q10Point(1, 1)], zones=zones)
    assert len(render.map_data.no_go_areas or []) == 1
    assert len(render.map_data.no_mopping_areas or []) == 1
    # charger = path origin in pixels: (1, 5 - 1) = (1, 4)
    assert render.map_data.charger is not None
    assert (render.map_data.charger.x, render.map_data.charger.y) == (1.0, 4.0)


def test_render_applies_erase_zones() -> None:
    """With a calibration, erase-zone cells are blanked from layers + image."""
    base = _render()
    before_floor = base.layers.class_counts.get("floor")
    assert before_floor and before_floor > 0

    # A rectangle covering the whole grid in world coords erases every cell.
    packet = replace(_packet(), erase_zones=[Q10EraseZone(vertices=[(0, 0), (7, 0), (7, 5), (0, 5)])])
    render = _render(packet, calibration=IDENTITY)

    assert render.layers.class_counts.get("floor", 0) == 0  # all floor erased
    assert render.image_content != base.image_content  # re-rendered


def test_render_partial_erase() -> None:
    """An erase rectangle only blanks the cells it covers, leaving the rest."""
    before_floor = _render().layers.class_counts.get("floor", 0)

    # Cover only the top two grid rows (pixel y 0..1 -> world y 4..5).
    packet = replace(_packet(), erase_zones=[Q10EraseZone(vertices=[(0, 4), (7, 4), (7, 5), (0, 5)])])
    render = _render(packet, calibration=IDENTITY)

    after_floor = render.layers.class_counts.get("floor", 0)
    assert 0 < after_floor < before_floor  # some, not all, floor removed


def test_draw_path_on_map_draws_position() -> None:
    """The robot position is drawn at the mapped pixel."""
    path = [Q10Point(1, 2), Q10Point(3, 2)]
    render = _render(calibration=IDENTITY, path=path, robot_position=Q10Point(3, 2))
    png = draw_path_on_map(
        render,
        config=CONFIG,
        path=path,
        robot_position=Q10Point(3, 2),
        robot_heading=None,
        zones=[],
        virtual_walls=[],
        position_color=(255, 211, 0, 255),
    )
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    # world (3, 2) -> grid pixel (3, 3) -> image (12, 12) at scale 4 (no flip).
    assert img.size == (8 * 4, 6 * 4)
    assert img.getpixel((12, 12)) == (255, 211, 0, 255)


def test_draw_path_on_map_draws_heading_indicator() -> None:
    """A known heading draws a facing tick from the robot marker.

    With heading 0 (= +x world) and the identity-ish calibration, the tick
    extends to the right of the robot pixel; with the marker at image (12, 12)
    the tick covers pixels at x > 12 along y == 12.
    """
    path = [Q10Point(1, 2), Q10Point(3, 2)]
    render = _render(calibration=IDENTITY, path=path, robot_position=Q10Point(3, 2))
    png = draw_path_on_map(
        render,
        config=CONFIG,
        path=path,
        robot_position=Q10Point(3, 2),
        robot_heading=0,  # facing +x
        zones=[],
        virtual_walls=[],
        position_color=(255, 211, 0, 255),
    )
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    # tick runs +x from the marker (4 * radius = 16 px at scale 4)
    assert img.getpixel((20, 12)) == (255, 211, 0, 255)
    # ...and not behind it (the marker is a small disc; sample well to the left)
    assert img.getpixel((4, 12)) != (255, 211, 0, 255)


def test_solve_q10_calibration_uses_header_origin_with_short_path() -> None:
    """A grid-frame header origin lets a short path calibrate (origin is exact)."""
    layers = _render().layers
    # Header origin in 5 mm units -> pixel origin (0, 5); not a keepalive frame.
    header = Q10HeaderCalibration(origin_x=0, origin_y=50, resolution=5, charger_x=0, charger_y=0, charger_phi=0)
    true = GridCalibration(resolution=20.0, origin_x=0.0, origin_y=5.0, y_sign=1)
    path = _floor_world_points(layers, true, 6)
    assert len(path) < 20  # far too short for the full origin+resolution fit

    cal = solve_q10_calibration(layers, header, path)
    assert cal is not None
    # Origin comes straight from the header (exact); only the resolution is fit,
    # so it lands on one of the candidates (the exact pick is grid-quantized).
    assert (cal.origin_x, cal.origin_y) == (0.0, 5.0)
    assert cal.resolution in _Q10_RESOLUTIONS


def test_solve_q10_calibration_short_path_without_header_returns_none() -> None:
    """Without a header origin a short path is too sparse for the full fit."""
    layers = _render().layers
    true = GridCalibration(resolution=10.0, origin_x=0.0, origin_y=5.0, y_sign=1)
    path = _floor_world_points(layers, true, 6)
    assert solve_q10_calibration(layers, None, path) is None

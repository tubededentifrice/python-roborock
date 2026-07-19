"""Compose a Q10 (B01/ss07) map into a single rendered result.

The :class:`~roborock.map.b01_q10_map_parser.B01Q10MapParser` turns wire bytes
into a :class:`~roborock.map.b01_q10_map_parser.Q10MapPacket`; this module takes
that packet plus the *other* inputs the device streams separately -- the cleaning
path, the vector overlays (no-go / no-mop zones, virtual walls) and a solved
world<->pixel calibration -- and composes them into one :class:`Q10MapRender` result object
(image + ``MapData`` + layers).

It exists so the map trait stays about state management: the trait accumulates
the pushed inputs and calls :func:`render_q10_map` once per change, holding the
returned object rather than mutating a pile of derived fields itself. All the
low-level pixel work (erase-zone blanking, world->pixel overlay placement, path
drawing) and the calibration policy live here, next to the rest of the map code.
"""

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image, ImageDraw
from vacuum_map_parser_base.map_data import Area, MapData, Path, Point, Wall

from roborock.exceptions import RoborockException

from .b01_grid_layers import (
    GridCalibration,
    GridLayers,
    solve_calibration,
    solve_calibration_with_origin,
)
from .b01_q10_map_parser import (
    B01Q10MapParser,
    B01Q10MapParserConfig,
    Q10HeaderCalibration,
    Q10MapPacket,
    Q10Point,
    Q10Room,
    erased_packet,
)
from .b01_q10_overlays import ZONE_TYPE_NO_GO, ZONE_TYPE_NO_MOP, Q10Zone

# Path-units-per-pixel candidates for calibration. A dense ss07 path lands a
# best fit of 20.0 around the header origin -- ground-truthed June 2026 on the
# R1: a corridor drive registered at 20 (matching the format author's
# independent "20 path-units/px"), and the dock->corridor span lined up with the
# ruler-measured 8.81 m corridor. With the header resolution=5 (50 mm/px grid)
# that makes one path-unit exactly 50/20 = 2.5 mm -- so a path-unit is NOT a
# millimetre (the open scale question). An earlier [10.0..18.0] range couldn't
# reach 20 (it railed at the bound), biasing the fit. A dense cleaning path
# selects the best fit within this bracket.
_Q10_RESOLUTIONS = [step * 0.5 for step in range(24, 53)]  # 12.0 .. 26.0
# A path needs enough shape to constrain a full (origin + resolution) fit; a few
# points cannot.
_MIN_CALIBRATION_POINTS = 20
# When the grid-frame header supplies the origin, only the resolution is fit, so
# a much shorter path suffices to confirm it (early in a clean, not just a dense
# one). See :func:`solve_calibration_with_origin`.
_MIN_HEADER_CALIBRATION_POINTS = 4


@dataclass
class Q10MapRender:
    """The fully composed result of rendering a Q10 map packet.

    Built by :func:`render_q10_map` from the packet plus the current path,
    overlays and calibration, so every derived field is consistent with one set
    of inputs. Analogous to :class:`~roborock.map.map_parser.ParsedMapData`, but
    also carrying the separable :attr:`layers` and the :attr:`calibration` used
    to place the vector overlays.
    """

    image_content: bytes
    """The rendered base map (PNG) with erase zones blanked, path not drawn."""

    map_data: MapData
    """Parsed map data: image metadata, room names, and -- once a calibration is
    known -- the path / robot position / zones / walls placed in pixel space."""

    layers: GridLayers
    """Separable map layers (background / wall / floor / per-room) in grid-pixel
    space, each renderable to a transparent PNG for frontend compositing."""

    rooms: list[Q10Room]
    """Rooms (segments) reported by the device, with ids and names."""

    calibration: GridCalibration | None
    """World<->pixel transform used to place the overlays, or ``None`` if no
    calibration was available (the overlays are then absent from ``map_data``)."""


def render_q10_map(
    packet: Q10MapPacket,
    *,
    calibration: GridCalibration | None,
    path: Sequence[Q10Point],
    robot_position: Q10Point | None,
    zones: Sequence[Q10Zone],
    virtual_walls: Sequence[Q10Zone],
    config: B01Q10MapParserConfig,
) -> Q10MapRender:
    """Compose a Q10 map packet and its overlays into a :class:`Q10MapRender`.

    With a ``calibration`` the erase zones are blanked out of the raster and the
    path / robot position / restricted zones / virtual walls are placed onto
    ``map_data`` in pixel space; without one only the base raster is rendered
    (the overlays are world-coordinate only and can't be placed yet). Raises
    :class:`RoborockException` if the packet fails to render.
    """
    parser = B01Q10MapParser(config)
    layers = packet.layers

    render_packet = packet
    if calibration is not None:
        cells = _erased_cells(layers, packet.erase_zones, calibration)
        if cells:
            # Blank the erase-zone cells and re-derive the raster/layers from the
            # modified packet so the phantom areas disappear (as the app shows).
            render_packet = erased_packet(packet, cells)
            layers = render_packet.layers

    parsed = parser.parsed_from_packet(render_packet)
    if parsed.image_content is None or parsed.map_data is None:
        raise RoborockException("Failed to render Q10 map image")
    map_data = parsed.map_data

    if calibration is not None:
        _place_path(map_data, calibration, path, robot_position)
        _place_zones(map_data, calibration, path, zones, virtual_walls)

    return Q10MapRender(
        image_content=parsed.image_content,
        map_data=map_data,
        layers=layers,
        rooms=packet.rooms,
        calibration=calibration,
    )


def solve_q10_calibration(
    layers: GridLayers,
    header_calibration: Q10HeaderCalibration | None,
    path: Sequence[Q10Point],
) -> GridCalibration | None:
    """Fit the world<->pixel calibration from the current cleaning path.

    When the map packet's grid-frame header carries a calibration origin (ss07),
    only the resolution is fit -- around that fixed origin -- so a short path
    suffices and the origin is exact rather than recovered by a slide. Otherwise
    the full origin + resolution fit is used, which needs a reasonably dense
    cleaning path. Returns ``None`` if the path is too short/featureless to fit.
    """
    points: list[tuple[float, float]] = [(point.x, point.y) for point in path]
    return _calibration_from_header(layers, header_calibration, points) or _calibration_from_fit(layers, points)


def _calibration_from_header(
    layers: GridLayers,
    header_calibration: Q10HeaderCalibration | None,
    points: list[tuple[float, float]],
) -> GridCalibration | None:
    """Calibrate around the header-supplied origin (resolution fit to a path)."""
    if header_calibration is None or len(points) < _MIN_HEADER_CALIBRATION_POINTS:
        return None
    origin = header_calibration.origin_pixels()
    if origin is None:  # keepalive frame -- no usable origin
        return None
    return solve_calibration_with_origin(layers, points, origin, resolutions=_Q10_RESOLUTIONS)


def _calibration_from_fit(layers: GridLayers, points: list[tuple[float, float]]) -> GridCalibration | None:
    """Full origin + resolution fit; needs a reasonably dense path."""
    if len(points) < _MIN_CALIBRATION_POINTS:
        return None
    return solve_calibration(layers, points, resolutions=_Q10_RESOLUTIONS)


def _erased_cells(layers: GridLayers, erase_zones: Sequence, calibration: GridCalibration) -> set[int]:
    """Grid-cell indices covered by the erase zones (axis-aligned bbox fill)."""
    if not erase_zones:
        return set()
    width, height = layers.width, layers.height
    cells: set[int] = set()
    for zone in erase_zones:
        pixels = [calibration.world_to_pixel(x, y) for x, y in zone.vertices]
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        x0, x1 = int(min(xs)), int(max(xs))
        y0, y1 = int(min(ys)), int(max(ys))
        for py in range(max(0, y0), min(height, y1 + 1)):
            for px in range(max(0, x0), min(width, x1 + 1)):
                cells.add(py * width + px)
    return cells


def _place_path(
    map_data: MapData,
    calibration: GridCalibration,
    path: Sequence[Q10Point],
    robot_position: Q10Point | None,
) -> None:
    """Fill ``MapData.path`` / ``vacuum_position`` in grid-pixel coords.

    Points are stored in grid-pixel space (origin top-left), matching the Q10's
    top-down, un-flipped raster so they line up with the rendered image.
    """
    pixels = [Point(*calibration.world_to_pixel(point.x, point.y)) for point in path]
    map_data.path = Path(len(pixels), 1, 0, [pixels])
    if robot_position is not None:
        px, py = calibration.world_to_pixel(robot_position.x, robot_position.y)
        map_data.vacuum_position = Point(px, py)


def _place_zones(
    map_data: MapData,
    calibration: GridCalibration,
    path: Sequence[Q10Point],
    zones: Sequence[Q10Zone],
    virtual_walls: Sequence[Q10Zone],
) -> None:
    """Convert world-coordinate zones/walls into pixel-space ``MapData`` layers."""

    def to_area(zone: Q10Zone) -> Area | None:
        if len(zone.vertices) != 4:
            return None  # MapData.Area is a quad
        pts = [calibration.world_to_pixel(x, y) for x, y in zone.vertices]
        return Area(pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1], pts[3][0], pts[3][1])

    no_go = [area for zone in zones if zone.type == ZONE_TYPE_NO_GO and (area := to_area(zone))]
    no_mop = [area for zone in zones if zone.type == ZONE_TYPE_NO_MOP and (area := to_area(zone))]
    map_data.no_go_areas = no_go or None
    map_data.no_mopping_areas = no_mop or None

    walls: list[Wall] = []
    for zone in virtual_walls:
        if len(zone.vertices) >= 2:
            (x0, y0), (x1, y1) = zone.vertices[0], zone.vertices[1]
            p0 = calibration.world_to_pixel(x0, y0)
            p1 = calibration.world_to_pixel(x1, y1)
            walls.append(Wall(p0[0], p0[1], p1[0], p1[1]))
    map_data.walls = walls or None

    # The robot starts a session at its dock, so the path origin is the charger.
    if path:
        cx, cy = calibration.world_to_pixel(path[0].x, path[0].y)
        map_data.charger = Point(cx, cy)


def draw_path_on_map(
    render: Q10MapRender,
    *,
    config: B01Q10MapParserConfig,
    path: Sequence[Q10Point],
    robot_position: Q10Point | None,
    robot_heading: int | None,
    zones: Sequence[Q10Zone],
    virtual_walls: Sequence[Q10Zone],
    line_color: tuple[int, int, int, int] = (235, 64, 52, 255),
    position_color: tuple[int, int, int, int] = (255, 211, 0, 255),
) -> bytes:
    """Draw the session path + robot position + overlays onto the base map (PNG).

    ``render`` must carry a calibration (its :attr:`Q10MapRender.calibration`) --
    the caller is responsible for solving one first. Returns a fresh PNG; the
    ``render.image_content`` base raster is left untouched.
    """
    calibration = render.calibration
    if calibration is None:
        raise RoborockException("No calibration available; a cleaning path must be captured during a clean")

    scale = config.map_scale
    base = Image.open(io.BytesIO(render.image_content)).convert("RGBA")

    def world_to_image(x: float, y: float) -> tuple[float, float]:
        px, py = calibration.world_to_pixel(x, y)
        # The ss07 grid renders top-down (no flip), so grid-pixel (px, py) maps
        # straight to image space, only upscaled by ``scale``.
        return (px * scale, py * scale)

    def to_image(point: Q10Point) -> tuple[float, float]:
        return world_to_image(point.x, point.y)

    draw = ImageDraw.Draw(base, "RGBA")

    # Erase zones are applied to the raster itself (cells blanked), so they are
    # not drawn here -- the base image already reflects them.

    # No-go (blue) and no-mop (magenta) zones beneath the path.
    for zone in zones:
        if len(zone.vertices) < 3:
            continue
        polygon = [world_to_image(x, y) for x, y in zone.vertices]
        fill = (0, 120, 255, 70) if zone.type == ZONE_TYPE_NO_GO else (255, 0, 200, 70)
        outline = (0, 80, 200, 255) if zone.type == ZONE_TYPE_NO_GO else (200, 0, 160, 255)
        draw.polygon(polygon, fill=fill, outline=outline)

    # Virtual walls (line segments, not polygons) drawn over the zones.
    for wall in virtual_walls:
        if len(wall.vertices) < 2:
            continue
        draw.line(
            [world_to_image(x, y) for x, y in wall.vertices[:2]],
            fill=(255, 64, 64, 255),
            width=max(2, scale),
        )

    if len(path) >= 2:
        draw.line([to_image(point) for point in path], fill=line_color, width=max(1, scale // 2))
    if path:  # path origin == dock / charger
        dx, dy = to_image(path[0])
        draw.ellipse([dx - scale, dy - scale, dx + scale, dy + scale], outline=(40, 200, 40, 255), width=2)
    if robot_position is not None:
        cx, cy = to_image(robot_position)
        radius = scale
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=position_color)
        if robot_heading is not None:
            # Heading is world-space degrees (0 = +x, +90 = +y). Map a unit
            # world-space facing vector through the same transform (so the
            # Y-flip/scale match the marker), then normalize to a fixed
            # pixel-length tick so it reads at any calibration resolution.
            angle = math.radians(robot_heading)
            hx, hy = world_to_image(
                robot_position.x + math.cos(angle),
                robot_position.y + math.sin(angle),
            )
            norm = math.hypot(hx - cx, hy - cy)
            if norm > 0:
                tick = 4 * radius
                draw.line(
                    [cx, cy, cx + (hx - cx) / norm * tick, cy + (hy - cy) / norm * tick],
                    fill=position_color,
                    width=max(1, scale // 2),
                )
    buffer = io.BytesIO()
    base.save(buffer, format="PNG")
    return buffer.getvalue()

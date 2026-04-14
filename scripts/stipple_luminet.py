#!/usr/bin/env python3
"""Render a Luminet-style black hole plate from a deterministic observer-plane mesh.

This script follows the paper's geometry more closely than the earlier photon
sampler: it evaluates direct and first-order images on a dense mesh of
isoradials, interpolates the observed flux onto the observer's plate, and can
optionally convert the resulting continuous field into a stippled image.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from luminet.black_hole import BlackHole
from luminet import black_hole_math as bhmath


@dataclass
class FluxField:
    image: np.ndarray
    x: np.ndarray
    y: np.ndarray
    x_extent: tuple[float, float]
    y_extent: tuple[float, float]


def polar_to_plate(alpha: np.ndarray, impact_parameter: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = impact_parameter * np.sin(alpha)
    y = -impact_parameter * np.cos(alpha)
    return x, y


def build_structured_triangles(n_radii: int, n_angles: int) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for i in range(n_radii - 1):
        row0 = i * n_angles
        row1 = (i + 1) * n_angles
        for j in range(n_angles):
            jn = (j + 1) % n_angles
            a = row0 + j
            b = row1 + j
            c = row0 + jn
            d = row1 + jn
            triangles.append((a, b, c))
            triangles.append((b, d, c))
    return np.asarray(triangles, dtype=int)


def triangle_edge_lengths(x: np.ndarray, y: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    pts = np.stack([x, y], axis=1)
    tri_pts = pts[triangles]
    d01 = np.linalg.norm(tri_pts[:, 0] - tri_pts[:, 1], axis=1)
    d12 = np.linalg.norm(tri_pts[:, 1] - tri_pts[:, 2], axis=1)
    d20 = np.linalg.norm(tri_pts[:, 2] - tri_pts[:, 0], axis=1)
    return np.stack([d01, d12, d20], axis=1)


def collect_branch_mesh(
    bh: BlackHole,
    radii: np.ndarray,
    order: int,
    angular_resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if order == 0:
        bh.calc_isoradials(direct_r=radii, ghost_r=[])
    elif order == 1:
        bh.calc_isoradials(direct_r=[], ghost_r=radii)
    else:
        raise ValueError("Only direct (0) and first-order ghost (1) images are supported.")

    isoradials = sorted(
        [ir for ir in bh.isoradials if ir.order == order and ir.radius in set(radii)],
        key=lambda ir: ir.radius,
    )
    if len(isoradials) != len(radii):
        raise ValueError(f"Expected {len(radii)} isoradials for order {order}, got {len(isoradials)}.")

    angles = np.asarray([ir.angles[:angular_resolution] for ir in isoradials], dtype=float)
    impact_parameters = np.asarray(
        [ir.impact_parameters[:angular_resolution] for ir in isoradials], dtype=float
    )
    redshift_factors = np.asarray(
        [ir.redshift_factors[:angular_resolution] for ir in isoradials], dtype=float
    )
    radius_grid = np.repeat(radii[:, None], angular_resolution, axis=1)
    flux = bhmath.calc_flux_observed(radius_grid, bh.acc, bh.mass, redshift_factors)

    x, y = polar_to_plate(angles, impact_parameters)
    return x.ravel(), y.ravel(), flux.ravel(), impact_parameters.ravel()


def interpolate_branch(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    n_radii: int,
    n_angles: int,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    edge_clip_factor: float,
) -> np.ndarray:
    triangles = build_structured_triangles(n_radii, n_angles)
    finite_mask = np.all(np.isfinite(values[triangles]), axis=1)
    edge_lengths = triangle_edge_lengths(x, y, triangles)
    finite_edges = edge_lengths[np.all(np.isfinite(edge_lengths), axis=1)]
    if finite_edges.size == 0:
        return np.full_like(x_grid, np.nan, dtype=float)
    edge_threshold = np.median(finite_edges) * edge_clip_factor
    long_edge_mask = np.max(edge_lengths, axis=1) > edge_threshold
    bad_vertex_mask = ~np.all(
        np.isfinite(np.stack([x[triangles], y[triangles]], axis=-1)),
        axis=(1, 2),
    )
    mask = (~finite_mask) | long_edge_mask | bad_vertex_mask

    triangulation = mtri.Triangulation(x, y, triangles=triangles, mask=mask)
    interpolator = mtri.LinearTriInterpolator(triangulation, values)
    image = np.asarray(interpolator(x_grid, y_grid), dtype=float)
    image[image <= 0] = np.nan
    return image


def estimate_flux_field_mesh(
    bh: BlackHole,
    width: int,
    height: int,
    radial_resolution: int,
    angular_resolution: int,
    edge_clip_factor: float,
    blur_sigma: float,
    gamma: float,
    clip_quantile: float,
) -> FluxField:
    radii = np.linspace(bh.disk_inner_edge, bh.disk_outer_edge, radial_resolution)
    direct_x, direct_y, direct_flux, direct_b = collect_branch_mesh(
        bh, radii, order=0, angular_resolution=angular_resolution
    )
    ghost_x, ghost_y, ghost_flux, ghost_b = collect_branch_mesh(
        bh, radii, order=1, angular_resolution=angular_resolution
    )

    all_x = np.concatenate([direct_x, ghost_x])
    all_y = np.concatenate([direct_y, ghost_y])
    all_b = np.concatenate([direct_b, ghost_b])

    xmax = float(np.nanmax(np.abs(all_x)))
    ymax = float(np.nanmax(np.abs(all_y)))
    bmax = float(np.nanmax(all_b))
    x_extent = (-xmax * 1.03, xmax * 1.03)
    y_extent = (-ymax * 1.03, ymax * 1.03)

    x_lin = np.linspace(x_extent[0], x_extent[1], width)
    y_lin = np.linspace(y_extent[0], y_extent[1], height)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)

    direct_img = interpolate_branch(
        direct_x,
        direct_y,
        direct_flux,
        radial_resolution,
        angular_resolution,
        x_grid,
        y_grid,
        edge_clip_factor,
    )
    ghost_img = interpolate_branch(
        ghost_x,
        ghost_y,
        ghost_flux,
        radial_resolution,
        angular_resolution,
        x_grid,
        y_grid,
        edge_clip_factor,
    )

    field = np.nan_to_num(direct_img, nan=0.0) + np.nan_to_num(ghost_img, nan=0.0)

    shadow_radius = bh.critical_b
    shadow_mask = np.hypot(x_grid, y_grid) <= shadow_radius
    field[shadow_mask] = 0.0

    if blur_sigma > 0:
        field = gaussian_filter(field, sigma=blur_sigma)
        field[shadow_mask] = 0.0

    positive = field[field > 0]
    if positive.size:
        clip_value = float(np.quantile(positive, clip_quantile))
        clip_value = max(clip_value, 1e-12)
        field = np.clip(field, 0, clip_value) / clip_value
        field = np.power(field, gamma)

    return FluxField(
        image=field,
        x=x_grid,
        y=y_grid,
        x_extent=x_extent,
        y_extent=y_extent,
    )


def sample_stipples(
    field: np.ndarray,
    count: int,
    seed: int,
    min_separation_px: float,
    background_floor: float,
    density_power: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.power(np.clip(field, 0, 1), density_power)
    weights += float(background_floor)
    flat = weights.ravel()
    flat /= flat.sum()

    height, width = field.shape
    points: list[tuple[float, float]] = []
    candidates = rng.choice(flat.size, size=max(count * 4, count), replace=True, p=flat)

    cell_size = max(min_separation_px, 1.0)
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def accepted(px: float, py: float) -> bool:
        if min_separation_px <= 0:
            return True
        gx = int(px // cell_size)
        gy = int(py // cell_size)
        for ix in range(gx - 1, gx + 2):
            for iy in range(gy - 1, gy + 2):
                for ox, oy in grid.get((ix, iy), []):
                    if (px - ox) ** 2 + (py - oy) ** 2 < min_separation_px**2:
                        return False
        grid.setdefault((gx, gy), []).append((px, py))
        return True

    for idx in candidates:
        row, col = divmod(int(idx), width)
        px = col + rng.random()
        py = row + rng.random()
        if accepted(px, py):
            points.append((px, py))
        if len(points) >= count:
            break

    while len(points) < count:
        idx = int(rng.choice(flat.size, p=flat))
        row, col = divmod(idx, width)
        px = col + rng.random()
        py = row + rng.random()
        if accepted(px, py):
            points.append((px, py))

    return np.asarray(points, dtype=float)


def render_stipple_image(
    field: FluxField,
    stipples: np.ndarray,
    output: Path,
    dot_size: float,
    background: str,
    foreground: str,
) -> None:
    height, width = field.image.shape
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    ax.scatter(
        stipples[:, 0],
        height - stipples[:, 1],
        s=dot_size,
        c=foreground,
        marker="o",
        linewidths=0,
    )
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=background, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_flux_preview(field: FluxField, output: Path, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(field.image.shape[1] / 100, field.image.shape[0] / 100), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(field.image, cmap=cmap, origin="lower")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/luminet_stipple_mesh.png"))
    parser.add_argument("--field-output", type=Path, default=Path("assets/luminet_flux_field_mesh.png"))
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--radial-resolution", type=int, default=180)
    parser.add_argument("--angular-resolution", type=int, default=240)
    parser.add_argument("--incl", type=float, default=np.deg2rad(80.0))
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--acc", type=float, default=1.0)
    parser.add_argument("--outer-edge", type=float, default=40.0)
    parser.add_argument("--edge-clip-factor", type=float, default=3.5)
    parser.add_argument("--blur-sigma", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--clip-quantile", type=float, default=0.995)
    parser.add_argument("--dots", type=int, default=45000)
    parser.add_argument("--dot-size", type=float, default=1.4)
    parser.add_argument("--min-separation-px", type=float, default=1.35)
    parser.add_argument("--background-floor", type=float, default=0.0015)
    parser.add_argument("--density-power", type=float, default=1.35)
    parser.add_argument("--seed", type=int, default=1979)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bh = BlackHole(
        mass=args.mass,
        incl=args.incl,
        acc=args.acc,
        outer_edge=args.outer_edge,
        angular_resolution=args.angular_resolution,
        radial_resolution=args.radial_resolution,
    )
    field = estimate_flux_field_mesh(
        bh=bh,
        width=args.width,
        height=args.height,
        radial_resolution=args.radial_resolution,
        angular_resolution=args.angular_resolution,
        edge_clip_factor=args.edge_clip_factor,
        blur_sigma=args.blur_sigma,
        gamma=args.gamma,
        clip_quantile=args.clip_quantile,
    )
    stipples = sample_stipples(
        field=field.image,
        count=args.dots,
        seed=args.seed,
        min_separation_px=args.min_separation_px,
        background_floor=args.background_floor,
        density_power=args.density_power,
    )
    render_stipple_image(
        field=field,
        stipples=stipples,
        output=args.output,
        dot_size=args.dot_size,
        background="black",
        foreground="white",
    )
    render_flux_preview(field=field, output=args.field_output, cmap="gray")
    print(f"saved {args.output}")
    print(f"saved {args.field_output}")


if __name__ == "__main__":
    main()

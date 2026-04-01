#!/usr/bin/env python3
"""Render a Luminet-style stippled black hole image.

This script uses the repo's photon solver to estimate the observed flux on the
observer's plate, then turns that continuous brightness field into a stippled
image by placing more dots where the field is brighter.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from luminet.black_hole import BlackHole


@dataclass
class FluxField:
    image: np.ndarray
    x_extent: tuple[float, float]
    y_extent: tuple[float, float]


def _photons_to_plate_coordinates(photons):
    b = np.array([ph.impact_parameter for ph in photons], dtype=float)
    a = np.array([ph.alpha for ph in photons], dtype=float)
    flux = np.array([ph.flux_o for ph in photons], dtype=float)
    x = b * np.sin(a)
    y = -b * np.cos(a)
    return x, y, flux


def estimate_flux_field(
    bh: BlackHole,
    photons_per_branch: int,
    width: int,
    height: int,
    blur_sigma: float,
    gamma: float,
    clip_quantile: float,
) -> FluxField:
    direct, ghost = bh.sample_photons(photons_per_branch)

    x0, y0, f0 = _photons_to_plate_coordinates(direct)
    x1, y1, f1 = _photons_to_plate_coordinates(ghost)

    x = np.concatenate([x0, x1])
    y = np.concatenate([y0, y1])
    flux = np.concatenate([f0, f1])

    xmax = float(np.max(np.abs(x)))
    ymax = float(np.max(np.abs(y)))
    pad = 0.05
    x_extent = (-xmax * (1 + pad), xmax * (1 + pad))
    y_extent = (-ymax * (1 + pad), ymax * (1 + pad))

    field, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=(width, height),
        range=[x_extent, y_extent],
        weights=flux,
    )
    field = field.T

    if blur_sigma > 0:
        field = gaussian_filter(field, sigma=blur_sigma)

    field = np.log1p(field)
    if np.max(field) > 0:
        clip_value = np.quantile(field[field > 0], clip_quantile) if np.any(field > 0) else np.max(field)
        clip_value = max(float(clip_value), 1e-12)
        field = np.clip(field, 0, clip_value) / clip_value
    field = np.power(field, gamma)
    return FluxField(
        image=field,
        x_extent=(float(xedges[0]), float(xedges[-1])),
        y_extent=(float(yedges[0]), float(yedges[-1])),
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
    weights = field.astype(float).copy()
    weights = np.power(np.clip(weights, 0, 1), density_power)
    weights += float(background_floor)
    flat = weights.ravel()
    total = flat.sum()
    if total <= 0:
        raise ValueError("Flux field is empty; cannot place stipples.")
    flat /= total

    height, width = field.shape
    candidates = rng.choice(flat.size, size=max(count * 4, count), replace=True, p=flat)

    points: list[tuple[float, float]] = []
    if min_separation_px > 0:
        cell_size = max(min_separation_px, 1.0)
        grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    else:
        cell_size = 1.0
        grid = {}

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

    return np.array(points, dtype=float)


def render_stipple_image(
    field: FluxField,
    stipples: np.ndarray,
    output: Path,
    dot_size: float,
    background: str,
    foreground: str,
) -> None:
    height, width = field.image.shape
    fig_w = width / 100
    fig_h = height / 100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
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
    ax.imshow(field.image, cmap=cmap, origin="lower")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("assets/luminet_stipple.png"))
    parser.add_argument("--field-output", type=Path, default=Path("assets/luminet_flux_field.png"))
    parser.add_argument("--photons", type=int, default=150000, help="Photons per branch: direct and ghost.")
    parser.add_argument("--dots", type=int, default=65000, help="Number of stipple dots to draw.")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--incl", type=float, default=np.deg2rad(80.0))
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--acc", type=float, default=1.0)
    parser.add_argument("--outer-edge", type=float, default=40.0)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--clip-quantile", type=float, default=0.995)
    parser.add_argument("--dot-size", type=float, default=1.8)
    parser.add_argument("--min-separation-px", type=float, default=1.6)
    parser.add_argument("--background-floor", type=float, default=0.01)
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
    )
    field = estimate_flux_field(
        bh=bh,
        photons_per_branch=args.photons,
        width=args.width,
        height=args.height,
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

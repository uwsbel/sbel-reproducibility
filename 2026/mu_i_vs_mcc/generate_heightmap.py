#!/usr/bin/env python3
"""
Generate a cratered heightmap for the lander demo.

Dependencies: numpy, Pillow, noise
"""

import argparse
import os
import time

import numpy as np
from PIL import Image
import math


def normalize(arr):
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-9:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def normalize_signed(arr):
    return normalize(arr) * 2.0 - 1.0


def fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def hash2d(ix, iy, seed):
    ix = ix.astype(np.uint64, copy=False)
    iy = iy.astype(np.uint64, copy=False)
    h = ix * np.uint64(374761393) + iy * np.uint64(668265263) + np.uint64(seed) * np.uint64(1442695041)
    h &= np.uint64(0xFFFFFFFF)
    h = (h ^ (h >> np.uint64(13))) * np.uint64(1274126177)
    h &= np.uint64(0xFFFFFFFF)
    h = h ^ (h >> np.uint64(16))
    return h.astype(np.uint32, copy=False)


def gradients(ix, iy, seed):
    h = hash2d(ix, iy, seed)
    angles = (h & np.uint32(0xFFFF)).astype(np.float32) / 65535.0 * (2.0 * math.pi)
    return np.cos(angles), np.sin(angles)


def perlin2d(x, y, seed):
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)
    xf = x - xi
    yf = y - yi

    u = fade(xf)
    v = fade(yf)

    g00x, g00y = gradients(xi, yi, seed)
    g10x, g10y = gradients(xi + 1, yi, seed)
    g01x, g01y = gradients(xi, yi + 1, seed)
    g11x, g11y = gradients(xi + 1, yi + 1, seed)

    dot00 = g00x * xf + g00y * yf
    dot10 = g10x * (xf - 1.0) + g10y * yf
    dot01 = g01x * xf + g01y * (yf - 1.0)
    dot11 = g11x * (xf - 1.0) + g11y * (yf - 1.0)

    x1 = dot00 + u * (dot10 - dot00)
    x2 = dot01 + u * (dot11 - dot01)
    return x1 + v * (x2 - x1)


def perlin_grid_builtin(xs, ys, scale_m, octaves, base):
    xv, yv = np.meshgrid(xs, ys)
    x = xv / scale_m
    y = yv / scale_m

    total = np.zeros_like(x, dtype=np.float32)
    amp = 1.0
    freq = 1.0
    amp_sum = 0.0
    for i in range(octaves):
        total += amp * perlin2d(x * freq, y * freq, base + i * 101)
        amp_sum += amp
        amp *= 0.5
        freq *= 2.0

    return total / amp_sum


def perlin_grid(xs, ys, scale_m, octaves, base, use_noise_lib):
    if use_noise_lib:
        import noise

        grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                grid[j, i] = noise.pnoise2(x / scale_m, y / scale_m, octaves=octaves, base=base)
        return grid

    return perlin_grid_builtin(xs, ys, scale_m, octaves, base)


def clamp_noise_base(value):
    return int(value) & 0x7FFFFFFF


def choose_crater_centers(field, xs, ys, count, min_sep_m, margin_m):
    flat = field.ravel()
    order = np.argsort(flat)[::-1]
    centers = []
    for idx in order:
        y_idx, x_idx = np.unravel_index(idx, field.shape)
        x = float(xs[x_idx])
        y = float(ys[y_idx])
        if x < margin_m or x > xs[-1] - margin_m:
            continue
        if y < margin_m or y > ys[-1] - margin_m:
            continue
        too_close = False
        for cx, cy in centers:
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) < min_sep_m * min_sep_m:
                too_close = True
                break
        if too_close:
            continue
        centers.append((x, y))
        if len(centers) >= count:
            break
    return centers


def main():
    parser = argparse.ArgumentParser(description="Generate cratered terrain heightmap.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--length", type=float, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--height-min", type=float, required=True)
    parser.add_argument("--height-max", type=float, required=True)
    parser.add_argument("--max-craters", type=int, default=4)
    parser.add_argument("--max-crater-diameter", type=float, default=2.0)
    parser.add_argument("--max-crater-depth", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-noise-lib", action="store_true",
                        help="Use the C-extension noise library (faster, but may be less stable).")
    args = parser.parse_args()

    if args.resolution < 2:
        raise ValueError("resolution must be >= 2")
    if args.length <= 0.0 or args.width <= 0.0:
        raise ValueError("length and width must be > 0")
    if args.height_max <= args.height_min:
        raise ValueError("height-max must be greater than height-min")
    if args.max_craters < 0:
        raise ValueError("max-craters must be >= 0")

    seed = args.seed if args.seed is not None else int(time.time_ns() & 0xFFFFFFFF)
    base_seed = clamp_noise_base(seed)
    rng = np.random.default_rng(seed)

    xs = np.linspace(0.0, args.length, args.resolution, dtype=np.float32)
    ys = np.linspace(0.0, args.width, args.resolution, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    height_range = args.height_max - args.height_min
    base_height = args.height_min + 0.85 * height_range

    base_noise = normalize_signed(
        perlin_grid(xs, ys, scale_m=max(args.length, args.width) / 2.5, octaves=3, base=base_seed,
                    use_noise_lib=args.use_noise_lib)
    )
    bump_noise = normalize_signed(
        perlin_grid(xs, ys, scale_m=max(args.length, args.width) / 8.0, octaves=4,
                    base=clamp_noise_base(base_seed + 19), use_noise_lib=args.use_noise_lib)
    )

    base_amp = 0.06 * height_range
    bump_amp = 0.03 * height_range
    height = base_height + base_amp * base_noise + bump_amp * bump_noise

    crater_noise = normalize(
        perlin_grid(xs, ys, scale_m=max(args.length, args.width) / 1.5, octaves=2,
                    base=clamp_noise_base(base_seed + 101), use_noise_lib=args.use_noise_lib)
    )
    if args.max_craters > 0:
        num_features = int(rng.integers(1, args.max_craters + 1))
    else:
        num_features = 0

    max_radius = 0.5 * args.max_crater_diameter
    min_sep = 0.6 * args.max_crater_diameter
    centers = choose_crater_centers(crater_noise, xs, ys, num_features, min_sep, max_radius)

    while len(centers) < num_features:
        cx = float(rng.uniform(max_radius, args.length - max_radius))
        cy = float(rng.uniform(max_radius, args.width - max_radius))
        centers.append((cx, cy))

    if num_features > 0:
        crater_count = int(rng.integers(0, num_features + 1))
    else:
        crater_count = 0
    crater_indices = set(rng.choice(num_features, size=crater_count, replace=False)) if crater_count > 0 else set()
    bump_count = num_features - crater_count

    min_diameter = 0.3 * args.max_crater_diameter
    for idx, (cx, cy) in enumerate(centers):
        diameter = float(rng.uniform(min_diameter, args.max_crater_diameter))
        magnitude = float(rng.uniform(0.3 * args.max_crater_depth, args.max_crater_depth))
        radius = 0.5 * diameter
        dist = np.sqrt((xv - cx) ** 2 + (yv - cy) ** 2)
        mask = dist <= radius
        profile = 1.0 - (dist / radius) ** 2
        if idx in crater_indices:
            height[mask] -= magnitude * profile[mask]

            rim_mask = (dist > 0.8 * radius) & (dist <= radius)
            rim_height = 0.15 * magnitude
            rim_t = (dist[rim_mask] - 0.8 * radius) / (0.2 * radius)
            height[rim_mask] += rim_height * (1.0 - rim_t)
        else:
            height[mask] += magnitude * profile[mask]

    height = np.clip(height, args.height_min, args.height_max)
    normalized = (height - args.height_min) / height_range
    img = np.clip(normalized * 65535.0 + 0.5, 0, 65535).astype(np.uint16)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    Image.fromarray(img, mode="I;16").save(args.output)
    print(f"Wrote heightmap to {args.output} (seed={seed}, craters={crater_count}, bumps={bump_count})")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Generate batched TransformerPayne spectra with fixed mu=1.

This script is intended for training-data generation and simple throughput
benchmarking. It samples stellar parameters in the compact
    (teff, logg, mh, alpha_m)
space, expands them to the full TransformerPayne parameter vector, runs
batched inference with mu fixed to 1.0, and saves the result to .npz.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transformer_payne import TransformerPayne  # noqa: E402


ALPHA_ELEMENTS = ("O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Ti")


def build_full_parameters(
    emulator: TransformerPayne,
    teff: float,
    logg: float,
    mh: float,
    alpha_m: float,
    vmic: float = 1.0,
) -> np.ndarray:
    """Map (teff, logg, mh, alpha_m) to the full TransformerPayne parameter vector."""
    params = np.array(
        emulator.to_parameters(
            {
                "logteff": np.log10(teff),
                "logg": logg,
                "vmic": vmic,
            },
            relative=False,
        ),
        dtype=np.float32,
    )
    params = np.array(
        emulator.set_group_of_abundances_relative_to_solar(
            params,
            mh,
            emulator.metals,
        ),
        dtype=np.float32,
    )
    params = np.array(
        emulator.set_abundances_relative_to_arbitrary_element(
            params,
            alpha_m,
            list(ALPHA_ELEMENTS),
            reference_element="Fe",
        ),
        dtype=np.float32,
    )
    return params


def sample_compact_parameters(n_samples: int, seed: int) -> np.ndarray:
    """Sample compact stellar parameters uniformly from the documented model range."""
    rng = np.random.default_rng(seed)
    teff = rng.uniform(4000.0, 8000.0, size=n_samples)
    logg = rng.uniform(2.0, 5.0, size=n_samples)
    mh = rng.uniform(-2.5, 1.0, size=n_samples)
    alpha_m = rng.uniform(-1.0, 1.0, size=n_samples)
    return np.column_stack([teff, logg, mh, alpha_m]).astype(np.float32)


def expand_parameter_batch(
    emulator: TransformerPayne,
    compact_params: np.ndarray,
    vmic: float = 1.0,
) -> np.ndarray:
    """Expand a batch of compact parameters to the full emulator parameter space."""
    expanded = [
        build_full_parameters(
            emulator,
            teff=float(row[0]),
            logg=float(row[1]),
            mh=float(row[2]),
            alpha_m=float(row[3]),
            vmic=vmic,
        )
        for row in compact_params
    ]
    return np.stack(expanded, axis=0).astype(np.float32)


def make_batch_infer_fn(emulator: TransformerPayne, log_wavelengths: np.ndarray):
    """Create a compiled batched inference function with fixed mu=1."""
    log_wavelengths_jax = jnp.asarray(log_wavelengths, dtype=jnp.float32)

    def infer_single(full_params: jnp.ndarray) -> jnp.ndarray:
        return emulator(log_wavelengths_jax, 1.0, full_params)

    return jax.jit(jax.vmap(infer_single))


def generate_dataset(
    emulator: TransformerPayne,
    compact_params: np.ndarray,
    wavelengths: np.ndarray,
    batch_size: int,
    vmic: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Run batched mu=1 inference and return spectra plus timing stats."""
    full_params = expand_parameter_batch(emulator, compact_params, vmic=vmic)
    batch_fn = make_batch_infer_fn(emulator, np.log10(wavelengths))

    spectra_batches = []
    compile_seconds = 0.0
    steady_seconds = 0.0
    n_batches = 0

    for start in range(0, len(full_params), batch_size):
        stop = min(start + batch_size, len(full_params))
        batch = jnp.asarray(full_params[start:stop], dtype=jnp.float32)

        t0 = time.perf_counter()
        out = batch_fn(batch)
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - t0

        if n_batches == 0:
            compile_seconds = elapsed
        else:
            steady_seconds += elapsed

        spectra_batches.append(np.asarray(out, dtype=np.float32))
        n_batches += 1

    spectra = np.concatenate(spectra_batches, axis=0)
    steady_batches = max(n_batches - 1, 1)
    steady_samples = max(len(full_params) - batch_size, 1)

    stats = {
        "compile_first_batch_sec": compile_seconds,
        "steady_total_sec": steady_seconds,
        "steady_batch_mean_sec": steady_seconds / steady_batches,
        "steady_sample_mean_sec": steady_seconds / steady_samples,
        "steady_throughput_samples_per_sec": steady_samples / steady_seconds if steady_seconds > 0 else 0.0,
    }
    return full_params, spectra, stats


def write_dataset_npz(
    output_path: Path,
    wavelengths: np.ndarray,
    compact_params: np.ndarray,
    full_params: np.ndarray,
    spectra: np.ndarray,
) -> None:
    """Write one dataset shard to a compressed npz file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        wavelengths_angstrom=wavelengths,
        compact_params=compact_params,
        model_params=full_params,
        spectra=spectra,
        mu=np.array([1.0], dtype=np.float32),
    )


def generate_sharded_dataset(
    emulator: TransformerPayne,
    compact_params: np.ndarray,
    wavelengths: np.ndarray,
    batch_size: int,
    samples_per_shard: int,
    output_path: Path,
    vmic: float = 1.0,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Generate a large dataset and write it shard-by-shard to disk."""
    batch_fn = make_batch_infer_fn(emulator, np.log10(wavelengths))
    shard_records: list[dict[str, object]] = []
    compile_seconds = 0.0
    steady_seconds = 0.0
    n_batches = 0

    output_path.mkdir(parents=True, exist_ok=True)

    n_samples = len(compact_params)
    shard_index = 0
    for shard_start in range(0, n_samples, samples_per_shard):
        shard_stop = min(shard_start + samples_per_shard, n_samples)
        shard_compact = compact_params[shard_start:shard_stop]
        shard_full = expand_parameter_batch(emulator, shard_compact, vmic=vmic)

        spectra_batches = []
        for batch_start in range(0, len(shard_full), batch_size):
            batch_stop = min(batch_start + batch_size, len(shard_full))
            batch = jnp.asarray(shard_full[batch_start:batch_stop], dtype=jnp.float32)

            t0 = time.perf_counter()
            out = batch_fn(batch)
            jax.block_until_ready(out)
            elapsed = time.perf_counter() - t0

            if n_batches == 0:
                compile_seconds = elapsed
            else:
                steady_seconds += elapsed

            spectra_batches.append(np.asarray(out, dtype=np.float32))
            n_batches += 1

        shard_spectra = np.concatenate(spectra_batches, axis=0)
        shard_name = f"mu1_batch_spectra_shard_{shard_index:05d}.npz"
        shard_path = output_path / shard_name
        write_dataset_npz(
            shard_path,
            wavelengths=wavelengths,
            compact_params=shard_compact,
            full_params=shard_full,
            spectra=shard_spectra,
        )
        shard_records.append(
            {
                "path": shard_name,
                "start": shard_start,
                "stop": shard_stop,
                "n_samples": int(shard_stop - shard_start),
            }
        )
        shard_index += 1

    steady_batches = max(n_batches - 1, 1)
    steady_samples = max(n_samples - batch_size, 1)
    stats = {
        "compile_first_batch_sec": compile_seconds,
        "steady_total_sec": steady_seconds,
        "steady_batch_mean_sec": steady_seconds / steady_batches,
        "steady_sample_mean_sec": steady_seconds / steady_samples,
        "steady_throughput_samples_per_sec": steady_samples / steady_seconds if steady_seconds > 0 else 0.0,
    }
    return shard_records, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate batched TransformerPayne spectra with fixed mu=1.")
    parser.add_argument("--n-samples", type=int, default=256, help="Number of spectra to generate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for vmap inference.")
    parser.add_argument("--n-wave", type=int, default=10000, help="Number of wavelength points.")
    parser.add_argument("--wave-min", type=float, default=3360.0, help="Minimum wavelength in Angstrom.")
    parser.add_argument("--wave-max", type=float, default=10200.0, help="Maximum wavelength in Angstrom.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for compact parameter sampling.")
    parser.add_argument("--vmic", type=float, default=1.0, help="Microturbulence value used for expansion.")
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=0,
        help="If > 0, write multiple shard files with at most this many samples each.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mu1_batch_spectra.npz"),
        help="Output .npz file path for single-file mode, or output directory for sharded mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be > 0")
    if args.n_wave <= 1:
        raise ValueError("--n-wave must be > 1")
    if args.samples_per_shard < 0:
        raise ValueError("--samples-per-shard must be >= 0")

    emulator = TransformerPayne.download()
    wavelengths = np.geomspace(args.wave_min, args.wave_max, args.n_wave).astype(np.float32)
    compact_params = sample_compact_parameters(args.n_samples, args.seed)

    if args.samples_per_shard > 0:
        shard_records, stats = generate_sharded_dataset(
            emulator=emulator,
            compact_params=compact_params,
            wavelengths=wavelengths,
            batch_size=args.batch_size,
            samples_per_shard=args.samples_per_shard,
            output_path=args.output,
            vmic=args.vmic,
        )
        manifest = {
            "mu": 1.0,
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "n_wave": args.n_wave,
            "wave_min": args.wave_min,
            "wave_max": args.wave_max,
            "vmic": args.vmic,
            "samples_per_shard": args.samples_per_shard,
            "shards": shard_records,
        }
        manifest_path = args.output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved sharded dataset to: {args.output}")
        print(f"n shards: {len(shard_records)}")
        print(f"last shard samples: {shard_records[-1]['n_samples']}")
    else:
        full_params, spectra, stats = generate_dataset(
            emulator=emulator,
            compact_params=compact_params,
            wavelengths=wavelengths,
            batch_size=args.batch_size,
            vmic=args.vmic,
        )
        write_dataset_npz(
            args.output,
            wavelengths=wavelengths,
            compact_params=compact_params,
            full_params=full_params,
            spectra=spectra,
        )
        print(f"Saved dataset to: {args.output}")
        print(f"spectra shape: {spectra.shape}")

    print(f"compile first batch: {stats['compile_first_batch_sec']:.4f} s")
    print(f"steady batch mean:   {stats['steady_batch_mean_sec']:.4f} s")
    print(f"steady sample mean:  {stats['steady_sample_mean_sec']:.6f} s")
    print(f"steady throughput:   {stats['steady_throughput_samples_per_sec']:.3f} samples/s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Combine compatible tutorial NPZ datasets and renumber configuration IDs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED = {"signals", "baselines", "P", "configuration_id", "frequency_mhz"}
SCALAR_KEYS = ("simulator", "voltage_unit", "feature_mode", "label_source", "polarization_method")


def scalar_value(data, key):
    if key not in data:
        return None
    value = data[key]
    return str(value.item()) if value.shape == () else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("Supply at least two input datasets")
    if args.output.suffix != ".npz" or args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Choose a new .npz output path")

    loaded = [np.load(path, allow_pickle=False) for path in args.inputs]
    try:
        missing = [(str(path), sorted(REQUIRED - set(data.files))) for path, data in zip(args.inputs, loaded)
                   if REQUIRED - set(data.files)]
        if missing:
            parser.error(f"Missing required arrays: {missing}")
        frequency = loaded[0]["frequency_mhz"]
        for path, data in zip(args.inputs[1:], loaded[1:]):
            if not np.array_equal(data["frequency_mhz"], frequency):
                parser.error(f"{path} uses a different frequency grid")
        scalars = {key: scalar_value(loaded[0], key) for key in SCALAR_KEYS if key in loaded[0]}
        for path, data in zip(args.inputs[1:], loaded[1:]):
            for key, value in scalars.items():
                if scalar_value(data, key) != value:
                    parser.error(f"{path} disagrees on {key}")

        arrays = {}
        row_count = [len(data["P"]) for data in loaded]
        concat_keys = set.intersection(*(set(data.files) for data in loaded))
        for key in sorted(concat_keys):
            first = loaded[0][key]
            if key == "frequency_mhz" or first.shape == ():
                continue
            if all(data[key].shape[:1] == (count,) for data, count in zip(loaded, row_count)):
                arrays[key] = np.concatenate([data[key] for data in loaded], axis=0)

        offset, config_ids = 0, []
        for data in loaded:
            ids = data["configuration_id"].astype(np.int64, copy=True)
            config_ids.append(ids + offset)
            offset += int(ids.max()) + 1
        arrays["configuration_id"] = np.concatenate(config_ids)
        arrays["frequency_mhz"] = frequency
        for key, value in scalars.items():
            arrays[key] = np.array(value)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
        report = {
            "inputs": [{"path": str(path), "rows": int(count),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                       for path, count in zip(args.inputs, row_count)],
            "rows": int(sum(row_count)),
            "configuration_id_policy": "Input configuration_id values were offset to avoid collisions.",
            "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        }
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
        print(f"Saved {report['rows']} combined rows to {args.output}")
    finally:
        for data in loaded:
            data.close()


if __name__ == "__main__":
    main()

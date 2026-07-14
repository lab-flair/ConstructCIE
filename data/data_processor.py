"""Rebuild per-split dataset files from a single JSONL dataset + id maps.

Default layout (all under this script's dir, with <name>-scoped subfolders):

  raw_data/<name>.json                 — hierarchical JSONL, one record per line
  templates/<name>/split{i}/index.json — {"train": [...], "valid": [...], "test": [...]}
  processed_data/<name>/split{i}/...   — output tree (train.json, val.json, test.json, index.json)

Override any of these with --input / --template-dir / --output.
Use `--name my_ds` to reuse for other datasets — expects `my_ds.json` under
--input and `my_ds/split*/index.json` under --template-dir.
"""

import argparse
import json
import os
import re
from os.path import abspath, dirname, isdir, isfile, join


SCRIPT_DIR = dirname(abspath(__file__))
DEFAULT_INPUT    = join(SCRIPT_DIR, "raw_data")
DEFAULT_TEMPLATE = join(SCRIPT_DIR, "templates")
DEFAULT_OUTPUT   = join(SCRIPT_DIR, "processed_data")


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def index_by_id(records, id_key="id"):
    by_id = {}
    for r in records:
        rid = r.get(id_key)
        if rid is None or rid in by_id:
            continue
        by_id[rid] = r
    return by_id


def rebuild_splits(data_path, template_dir, output_dir, id_key="id"):
    if not isfile(data_path):
        raise FileNotFoundError(f"data file not found: {data_path}")
    if not isdir(template_dir):
        raise FileNotFoundError(f"template dir not found: {template_dir}")

    records = load_jsonl(data_path)
    by_id = index_by_id(records, id_key)
    print(f"Loaded {len(records)} records ({len(by_id)} unique) from {data_path}")
    print(f"Reading maps from {template_dir}")
    print(f"Writing splits to  {output_dir}")

    split_names = [
        d for d in os.listdir(template_dir)
        if isdir(join(template_dir, d)) and re.match(r'^split\d+$', d)
    ]
    split_names.sort(key=natural_key)

    label_to_file = {
        "train": "train.json",
        "valid": "val.json",
        "val":   "val.json",
        "test":  "test.json",
    }

    for name in split_names:
        src_dir = join(template_dir, name)
        index_path = join(src_dir, "index.json")
        if not isfile(index_path):
            print(f"  [skip] {name}: no index.json")
            continue

        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)

        out_split = join(output_dir, name)
        os.makedirs(out_split, exist_ok=True)

        counts = {}
        missing = []
        for label, ids in idx.items():
            out_name = label_to_file.get(label)
            if out_name is None:
                print(f"  [warn] {name}: unknown label '{label}', skipping")
                continue
            subset = []
            for rid in ids:
                rec = by_id.get(rid)
                if rec is None:
                    missing.append((label, rid))
                    continue
                subset.append(rec)
            write_jsonl(subset, join(out_split, out_name))
            counts[label] = len(subset)

        # Copy the id map alongside so the output folder is self-contained.
        with open(join(out_split, "index.json"), "w", encoding="utf-8") as f:
            json.dump(idx, f, indent=4, ensure_ascii=False)

        summary = ", ".join(f"{k}={v}" for k, v in counts.items())
        miss_note = f"  [{len(missing)} missing ids]" if missing else ""
        print(f"  [OK]   {name}: {summary}{miss_note}")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild per-split train/val/test JSONL files from a full dataset + id maps."
    )
    parser.add_argument("-n", "--name", default="data",
                        help="Dataset name (default: data). Expects <input>/<name>.json.")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT,
                        help=f"Directory holding <name>.json (default: {DEFAULT_INPUT})")
    parser.add_argument("-t", "--template-dir", default=DEFAULT_TEMPLATE, dest="template_dir",
                        help=f"Directory holding <name>/split{{i}}/index.json (default: {DEFAULT_TEMPLATE})")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help=f"Output directory; splits go under <output>/<name>/ (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    data_path    = join(args.input,        f"{args.name}.json")
    template_dir = join(args.template_dir, args.name)
    output_dir   = join(args.output,       args.name)
    os.makedirs(output_dir, exist_ok=True)

    rebuild_splits(data_path, template_dir, output_dir)


if __name__ == "__main__":
    main()

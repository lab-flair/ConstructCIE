# Data

Put your data file in `raw_data/` (as `<name>.json`), then run `data_processor.py`
with the matching name. It picks up the id maps under `templates/<name>/` and
rebuilds the per-split `train.json` / `val.json` / `test.json` files under
`processed_data/<name>/`.

## Usage

```bash
python data_processor.py --name <dataset_name>
```

`<dataset_name>` must match both the file in `raw_data/` and an existing
template folder in `templates/` (e.g. `raw_data/<name>.json` +
`templates/<name>/`). New datasets can be added over time, each under its own
name — drop both in place and run the command with that name.

Options (all optional): `-n/--name` dataset name (default `data`),
`-i/--input` raw data dir (default `raw_data/`), `-t/--template-dir` id-map dir
(default `templates/`), `-o/--output` output dir (default `processed_data/`).

```
data/
├── raw_data/<name>.json                       # full dataset, JSONL
├── templates/<name>/split{i}/index.json       # id maps per split
└── processed_data/<name>/split{i}/            # output: train.json, val.json, test.json, index.json
```

## Data format

Note: this is **not** the standard TextEE format. Each line of
`raw_data/<name>.json` is one JSON record:

```JSON5
{
  "id": 162549.015,  // unique id, matched against the ids in templates/<name>/split{i}/index.json
  "accident_report": "{'text': [...], 'children': {...}}",  // hierarchical annotation (stringified dict); each node is an extraction ({text, keywords, children}) or classification ({value}) node
  "accident_type": {"type": "classification", "value": ["struck-by"]}
}
```
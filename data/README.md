# Data

Place the raw datasets under `data/raw/` as described in the top-level README. Running `cpu/00_prepare_data.py` (or `cpu/00b_prepare_westermo.py`) writes the cleaned splits to `data/<dataset>/`.

The cleaning statistics of the runs reported in the paper are included for reference:

* `data/<dataset>/prep_stats.json` — raw rows, rows after cleaning, de-duplication and conflict removal, train/test sizes;
* `data/<dataset>/class_table.csv` — per-class counts before and after cleaning and in the train/test split.

Note: in `class_table.csv` and `prep_stats.json`, the per-class counts labelled "dedup" are counted after both de-duplication and removal of conflicting rows.

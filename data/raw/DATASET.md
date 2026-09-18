# Raw data

`train.csv` (45,593 food-delivery orders) is **not committed** — fetch it with:

```powershell
python scripts/fetch_data.py
```

The script downloads from the public mirror recorded in `configs/config.yaml`
(`ingestion.source_url`) and is idempotent (skips when a sane file already
exists). See the repo README's Data card for dataset provenance.

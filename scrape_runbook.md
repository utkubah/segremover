# segremover scrape upgrade runbook

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

No new Python dependency is required for the scraper beyond the existing repo stack.

## 2. Copy files into the repo

```bash
cp src/scrape.py /path/to/segremover/src/scrape.py
cp data/source_registry.csv /path/to/segremover/data/source_registry.csv
cp scripts/slurm/*.sbatch /path/to/segremover/scripts/slurm/
cp scripts/merge_candidate_logs.sh /path/to/segremover/scripts/
cp notebooks/eda_scrape_additions.py /path/to/segremover/notebooks/
```

## 3. Local smoke test

```bash
python src/scrape.py \
  --sources data/source_registry.csv \
  --target-total 20 \
  --max-per-channel 5 \
  --dry-run \
  --candidate-log data/candidates_smoke.csv
```

## 4. Local pilot scrape

```bash
python src/scrape.py \
  --sources data/source_registry.csv \
  --target-total 300 \
  --max-per-channel 40 \
  --min-words 500 \
  --cookies-from-browser chrome \
  --candidate-log data/candidates_pilot.csv
```

If browser-cookie extraction fails on your machine, export a Netscape-format cookies file and use:

```bash
python src/scrape.py \
  --sources data/source_registry.csv \
  --target-total 300 \
  --cookies cookies.txt \
  --candidate-log data/candidates_pilot.csv
```

## 5. Full local run

```bash
python src/scrape.py \
  --sources data/source_registry.csv \
  --target-total 5000 \
  --max-per-channel 120 \
  --min-words 500 \
  --sleep-sec 3.0 \
  --cookies-from-browser chrome \
  --candidate-log data/candidates_full.csv
```

## 6. HPC run

```bash
mkdir -p logs data/raw data/hpc scripts/slurm
sbatch scripts/slurm/scrape_pilot.sbatch
```

After the pilot finishes:

```bash
bash scripts/merge_candidate_logs.sh data/candidates_pilot.csv data/candidates_pilot_shard_*.csv
```

Then inspect `notebooks/eda.ipynb` with the cells from `notebooks/eda_scrape_additions.py`. If the pilot looks healthy, run:

```bash
sbatch scripts/slurm/scrape_full.sbatch
bash scripts/merge_candidate_logs.sh data/candidates_full.csv data/candidates_full_shard_*.csv
```

## 7. Expected outputs

```text
data/raw/<genre>/<video_id>.json
data/candidates_pilot.csv
data/candidates_full.csv
data/scrape_checkpoint*.jsonl
data/processed/corpus_metadata.csv
data/processed/corpus_summary.csv
```

## 8. Final genre set

The source registry uses seven top-level genres:

1. `lecture_cs_math`
2. `coding_tutorials`
3. `science_explainers_news`
4. `commentary_essays`
5. `podcasts_interviews`
6. `entertainment_food_lifestyle`
7. `subtitled_sewries`

`subtitled_series` allows Turkish series candidates, but they are kept only if an English transcript/subtitle file is actually downloaded.

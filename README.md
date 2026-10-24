# sweet.tv TV program recommender (LLM4Rec midterm)

Predicts each user's Top-5 TV shows for weeks 31-42 of the Kaggle competition
"sweet.tv - TV program recommender" (metric MAP@5).

## Requirements

Python 3.14 with pandas 3.0.5, pyarrow 25.0.1, numpy 2.5.3, scipy 1.18.1,
scikit-learn 1.9.1 and LightGBM 4.7.0 (the versions used). No GPU is needed.
The report builder also needs python-docx and pymupdf.

## Data

Download the competition files from Kaggle (Data tab, "Download All") into `data/`:
`dataset11-30.csv`, `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`,
`export_arh_31-42-final.csv`, `submission.csv`. The raw files are never edited.

## Run

    cd src
    python run_all.py                 # every step, about 45 minutes on a laptop CPU
    python run_all.py rerank.py       # one step only

Each step writes its output to `results/logs/<step>.log`. Reruns give byte-identical files.

| Step | What it does | Output |
|---|---|---|
| build_interactions.py | joins views with the schedule, share of each airing seen in one view, drops tv_show_id 0 | data/processed/*.parquet |
| baselines.py | popularity and own-history reference points | log |
| validate.py | every method on fold A, blends (round 1, watch-time truth) | results/validation.json |
| experiments2.py | sources of true items, schedule-aware rate, small PPR alpha | results/validation2.json |
| experiments3.py | title family, channel affinity, combinations | results/validation3.json |
| tune.py | round 1 grid on folds A and B | results/tuning.json |
| predict.py | round 1 Kaggle files | submissions/sub_rate_mf.csv and others, results/test_lists.json |
| two_users.py | the two users analysed in the report | results/two_users.json |
| discovery.py | familiar vs new hits per method (round 1) | results/discovery.json |
| table_methods.py | round 1 report table | results/methods_table.json |
| round2.py | organisers' rule: count airings seen over 80% in one view; blend grid | results/round2.json |
| predict2.py | first layer: schedule-aware count hybrid | submissions/sub_counts.csv, results/test_lists_round2.json |
| table_round2.py | every method under the organisers' rule, familiar vs new | results/round2_report.json |
| rerank_features.py | second layer: candidates and features, incl. expected exposure | data/processed/view_profile.parquet, data/processed/rerank/*.parquet |
| rerank.py | second layer: LightGBM checks against the first layer, final file | results/rerank.json, submissions/sub_rerank.csv |

## Validation

A user's true Top-5 is the five shows with the most airings seen over 80% in one
view without switching channel (ties: watch seconds, then id). This is the
organisers' rule as reported by the second-place author (N. Melentev, Towards
Data Science, 2021); his code was not used.

- Fold A: history weeks 11-18, future weeks 19-30.
- Fold B: history weeks 11-22, future weeks 23-30.

Only shows on air in the predicted weeks are candidates (their schedule is provided
by the competition). Viewing from the predicted weeks is never used.

The re-ranker is checked twice against the first layer:
- trained on users outside submission.csv, tested on the 1,260 submission users (fold A);
- trained on weeks 11-16 -> 17-22, tested on fold B (later weeks).

## Submission files

| File | Method | Fold A | Fold B | Kaggle public | Kaggle private |
|---|---|---|---|---|---|
| sub_rate_mf.csv | watch-time model (round 1) | 0.2988 | 0.2632 | 0.31182 | 0.28032 |
| sub_counts.csv | first layer: count hybrid | 0.4010 | 0.3547 | 0.44058 | 0.42458 |
| sub_rerank.csv | two layers: hybrid + LightGBM re-ranker (final) | 0.4222 | 0.3698 | 0.45820 | 0.45124 |

Fold scores use the organisers' rule. `sub_rate.csv` and `sub_history.csv` are
round 1 reference files. Write your full name in Russian in the Kaggle submission
description.

## Report

    cd report
    python build_report_final.py kazanfarov_timur_midterm_report.docx --s1 0.31182 0.28032 --s2 0.44058 0.42458 --s3 0.45820 0.45124 --best-private 0.42912 --screenshot kaggle_submissions.png

Then export the .docx to PDF from Word.

## Note on user ids

User ids have up to 20 digits, above the int64 range. pandas' pyarrow engine parses
them as floating point even with `dtype=str`, which rounds them; the code reads the
column as text with `pyarrow.csv` instead.

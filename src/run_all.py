# -*- coding: utf-8 -*-
"""Run the whole pipeline in order and keep each step's output in results/logs/.
Usage: python run_all.py [step ...]   (no arguments = every step)
Total time on a laptop CPU is about 45 minutes; validate.py and tune.py take longest.
"""
import os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "results" / "logs"
STEPS = [
    ("build_interactions.py", [str(HERE.parent / "data")]),  # data report + view/schedule join
    ("baselines.py", []),       # popularity and history reference points
    ("validate.py", []),        # every method on fold A, blends
    ("experiments2.py", []),    # truth sources, schedule-aware rate, small alpha
    ("experiments3.py", []),    # title family, channel, combinations
    ("tune.py", []),            # final grid on folds A and B
    ("predict.py", []),         # Kaggle files + test lists
    ("two_users.py", []),       # the two users of the report
    ("discovery.py", []),       # familiar vs new hits per method
    ("table_methods.py", []),   # report table: every method on both folds
    ("round2.py", []),          # organisers' rule: airings seen over 80% in one view
    ("predict2.py", []),        # round 2 Kaggle file (count-based blend)
    ("table_round2.py", []),    # report numbers for the final method
    ("rerank_features.py", []), # second layer: candidates, exposure and other features
    ("rerank.py", []),          # second layer: LightGBM checks and the third Kaggle file
]

LOGS.mkdir(parents=True, exist_ok=True)
wanted = set(sys.argv[1:])
for script, args in STEPS:
    if wanted and script not in wanted:
        continue
    t0 = time.time()
    with open(LOGS / (script.replace(".py", ".log")), "w", encoding="utf-8") as log:
        code = subprocess.call([sys.executable, str(HERE / script)] + args, cwd=HERE, stdout=log,
                               stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print("%-24s exit %d  %.0f s" % (script, code, time.time() - t0), flush=True)
    if code != 0:
        sys.exit("stopped: %s failed, see %s" % (script, LOGS))

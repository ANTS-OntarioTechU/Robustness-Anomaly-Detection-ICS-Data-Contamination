# Data availability and terms

## The SWaT dataset is not included in this repository

All experiments in this work use the **Secure Water Treatment (SWaT)** dataset
from iTrust, Centre for Research in Cyber Security, Singapore University of
Technology and Design (SUTD).

**No SWaT data — raw, sampled, or preprocessed — is distributed here.** The
iTrust terms of use explicitly prohibit sharing the dataset, in either a private
or a public setting. Every request must go through iTrust directly.

Request access here:
<https://itrust.sutd.edu.sg/itrust-labs_datasets/>

iTrust states that requests may take up to three working days to process.

## What this repository does contain

Only derived artifacts, which are the outputs of the analysis rather than the
data itself:

- **Code** — detector implementations, contamination routines, Slurm orchestration.
- **Result tables** (`tables/`) — aggregate evaluation metrics per run:
  accuracy, precision, recall, F1, TP/TN/FP/FN counts, FNR, FPR, ROC-AUC,
  PR-AUC, plus the run identifiers (model, seed, attack, rate) and timings.
  These are summary statistics over the test set. They contain no process
  measurements, no sensor or actuator tags, and no per-timestamp records, so
  the dataset cannot be reconstructed from them.
- **Figures** (`figures/`) — the four plots that appear in the paper.

## Obligations if you use this repository

The iTrust terms bind anyone who obtains the dataset. If you run this code and
publish the outcome, you must:

1. **Credit iTrust explicitly** — "iTrust, Centre for Research in Cyber
   Security, Singapore University of Technology and Design" — in any published
   work, in any medium.
2. **Not redistribute the dataset.** Direct others to the iTrust request form
   rather than sharing your copy.
3. **Notify iTrust** once work using the dataset is published.

iTrust provides the datasets in good faith and on an "as is" basis.

## Running the code

Place the SWaT CSVs where the code expects them, then point the environment
variables at that location:

```bash
export SWAT_DATA_DIR=/path/to/swat_data     # must contain normal.csv and attack.csv
export SWAT_OUTPUT_DIR=/path/to/outputs
```

`.gitignore` is configured to keep `normal.csv`, `attack.csv`, and any
`swat_data/` or `data/` directory out of version control. Do not override this.

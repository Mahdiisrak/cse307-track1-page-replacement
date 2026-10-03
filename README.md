# CSE-307 Term Paper — Track 1

## Learning-Augmented Page Replacement

This project implements **FIFO**, **LRU**, and **Optimal (Belady's)** page replacement, then compares them with a small learned eviction policy based on a decision tree.

### Repository contents

- `track1_page_replacement.py` — reproducible implementation and experiment.
- `results/policy_results.csv` — hit ratio and fault results by phase.
- `results/trace.csv` — generated page-reference trace.
- `results/hit_ratio_by_phase.png` — hit-ratio chart.
- `results/faults_by_phase.png` — page-fault chart.
- `report/term_paper_report.pdf` — 2–3 page written report.
- `DEMO_GUIDE.md` — 3–5 minute walkthrough script.

## Setup

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python track1_page_replacement.py
```

The command regenerates the CSV files and charts in `results/`. The random generator uses a fixed seed (`307`) so the results are reproducible.

## Experiment summary

The first 400 references use locality-heavy and partly sequential access. The next 400 references switch to a random/bursty workload. The learned model is trained on a separate synthetic trace. At each miss, it scores resident pages using only information available so far: age, access frequency, recency rank, and page identifier.

## AI assistance disclosure

An AI coding assistant was used to help structure the implementation, explain standard library APIs, and draft documentation. The workload design, experiment execution, generated measurements, and interpretation should be reviewed and edited by the student before submission.

## Student details to edit

Before submitting, replace the student-name placeholder in the PDF report and README metadata if your instructor requires it. Also create the GitHub repository and push this folder.

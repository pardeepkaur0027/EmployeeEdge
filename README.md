# EmployeeEdge — Employee Attrition Risk & Retention Planner

An end-to-end HR analytics project that predicts **which employees are about to
leave**, explains **why** (SHAP), and hands HR a **ready-to-run retention action
plan** — with what-if levers they can pull without retraining.

> **Status:** complete. All notebooks executed; model, SHAP explainers and the
> Flask web app are verified end-to-end (`smoke_test.py` + `app_smoke_test.py`).

---

## 1. The problem

- Dataset: IBM HR Analytics / Kaggle `WA_Fn-UseC_-HR-Employee-Attrition.csv`
  (1,470 employees x 35 fields, no missing values, 16.1% attrition).
- Business question: *given everything HR knows about an employee, how
  likely are they to leave in the next 12 months — and what can we actually do?*
- This is a classic **imbalanced classification** problem: only ~1 in 6
  leaves, so accuracy is a trap; recall/precision on the minority class and
  ROC-AUC are the metrics that matter.

## 2. Pipeline (no leakage, end to end)

```
01_EDA.ipynb          distributions, attrition-by-category, correlations, insights
02_Preprocessing.ipynb  drop junk cols (EmployeeCount, Over18, StandardHours,
                       EmployeeNumber) -> stratified 80/20 split
03_Modeling.ipynb      LR vs HistGradientBoosting vs XGBoost
                       (class_weight balanced + scale_pos_weight),
                       grid-search by ROC-AUC, threshold scan,
                       SHAP global + waterfall, cost-benefit analysis
src/preprocess.py      reusable load/split/preprocessor (single source of truth)
src/explain.py         SHAP explanation + retention action plan (used by app + notebooks)
src/train.py           training/tuning/eval/cost-benefit helpers
app/                   Flask web app (form -> risk % + why + action plan + what-if)
smoke_test.py          end-to-end check of model + SHAP + action plan
app_smoke_test.py      end-to-end check of the Flask routes
```

Every model is a `Pipeline(preprocessor, model)` — scaling + one-hot encoding
are **fit inside cross-validation**, then baked into the saved artifact, so
serving uses the exact same feature engineering as training (including
`handle_unknown='ignore'` for categories the app may see that training didn't).

## 3. Results

| Metric | Score |
|---|---|
| Test ROC-AUC | **0.81** (best: tuned LogisticRegression) |
| Recall (leavers) | 0.68 with `class_weight='balanced'` |
| Precision | 0.38 (conservative — many flagged need interview, that's the point) |
| Best F1 threshold | 0.55 (~F1 0.52) |

All three models land in the same performance band — a good sign the signal is
real and the pipeline is honest. LogisticRegression won on AUC *and* gives us
free interpretability, so it is the deployed model.

### Business value (conservative)

Intervening on the **top 30 highest-risk** test employees:

- Replacement cost 1.5x annual salary (industry rule of thumb)
- Intervention spend $2,500 / person (stay interviews, small raises)
- → **~$1.65M net savings / ~22x ROI** if even a third of the caught leavers
  are retained.

## 4. The app (Flask)

`python app/app.py` → http://127.0.0.1:5000

1. Enter or paste an employee profile (all 30 model fields, grouped).
2. See their **0–100% attrition risk** and a **SHAP waterfall** explaining why.
3. Read the **top pushing factors** and a **retention action plan**.
4. Test **what-if levers** — remove overtime, +20% income, satisfaction=4 —
   each re-runs the real pipeline, no retraining.

### Render deployment

On Render (free web service):

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn -w 1 --timeout 120 -b 0.0.0.0:$PORT app.app:app`
- The model artifact lives in `outputs/models/model.joblib` and loads at first
  request (lazy, ~1 s on Render's free tier).

## 5. Project layout

```
EmployeeEdge/
├── data/                 # raw CSV (untracked, restore from Kaggle if needed)
├── notebooks/            # 01 EDA, 02 Preprocessing, 03 Modeling (executed)
├── outputs/
│   ├── models/           # model.joblib (Pipeline), preprocessor.joblib
│   └── plots/            # EDA + SHAP images
├── src/                  # preprocess.py, explain.py, train.py (+ __init__)
├── app/                  # app.py, templates/, static/, requirements.txt
├── smoke_test.py         # model + SHAP + action plan verification
├── app_smoke_test.py     # Flask routes verification
└── requirements.txt      # full local dev requirements
```

> `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` is bundled in the repo so
> everything re-runs from a fresh clone (228 KB).

## 6. Run everything from scratch

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows PowerShell
pip install -r requirements.txt
jupyter notebook notebooks/      # re-run 01 -> 03 to retrain
python smoke_test.py             # verify model + explanations
python app_smoke_test.py         # verify Flask routes
python app/app.py                # launch the web app
```

---

Built as an HR-analytics capstone: the three things that separate this from a
plain notebook demo are **no data leakage**, **business-value framing** (AUC,
recall-with-threshold, ROI), and **actionable output** (SHAP + action plan +
what-if) instead of just a number.
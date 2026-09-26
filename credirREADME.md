# Credit Scoring Model for Loan Approval Decisions

A cost-aware credit scoring model built on the German Credit (Statlog) dataset. The model predicts an applicant's probability of default and converts that score into an approve/decline decision using a threshold justified by the asymmetric cost of lending mistakes, rather than a default 0.5 cutoff.

## Project Structure

```
├── germancredit.ipynb   # Main notebook: data prep, models, threshold selection, SHAP
├── note.md              # Documentation: imbalance/cost handling, threshold justification, SHAP findings
└── README.md            # This file
```

## Dataset

- **Source:** German Credit (Statlog) Dataset — UCI Machine Learning Repository (id=144)
- **Size:** 1,000 applicants, labeled good (700) or bad (300) credit risk
- **Features:** 20 application-time attributes (checking account status, credit history, purpose, amount, employment, age, housing, etc.)
- **Cost matrix:** classifying a bad applicant as good is 5x more costly than the reverse

## Setup

```bash
pip install pandas numpy scikit-learn xgboost shap
```

Data can be loaded either from the local `german.data` file or directly via:
```python
from ucimlrepo import fetch_ucirepo
german_credit = fetch_ucirepo(id=144)
```

## Approach

1. **Data preparation** — loaded the raw space-separated file, mapped coded categorical values (e.g. `A11`, `A34`) to human-readable labels based on the dataset documentation, and one-hot encoded them for modeling.
2. **Imbalance handling** — used `class_weight="balanced"` (Logistic Regression) and `scale_pos_weight` (XGBoost) so the models pay adequate attention to the minority (bad credit) class.
3. **Modeling** — trained a Logistic Regression baseline and an XGBoost model, then compared them.
4. **Threshold selection** — instead of the default 0.5 cutoff, searched a range of thresholds and selected the one minimizing total cost (5×False Negatives + 1×False Positives).
5. **Explainability** — used SHAP to identify which features drive each prediction, so decisions can be defended to a customer or regulator.

## Results

| Model | Optimal Threshold | Cost (5×FN + 1×FP) |
|---|---|---|
| **Logistic Regression (final model)** | **0.46** | **90** |
| XGBoost | 0.07 | 103 |

Logistic Regression was selected as the final model — it achieved the lowest cost, and its optimal threshold stayed close to the default 0.5, suggesting well-calibrated probabilities.

## Key SHAP Findings

- **Credit amount** and **duration** are the strongest risk drivers — higher amounts and longer terms increase predicted risk, consistent with real-world credit practice.
- Two counter-intuitive findings were flagged for further scrutiny: applicants with an "unknown/no savings account" status and those with a negative checking account balance both showed *lower* predicted risk than expected — likely reflecting a distinct demographic segment in the dataset rather than a data error.

See `note.md` for the full step-by-step reasoning behind each decision.
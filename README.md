# AI Solution Evaluation and Future Enhancement Proposal

**Week 4 project | Predictive Analytics Using Historical Data**

This repository contains a small, reproducible demonstration of a predictive-analytics workflow. It uses a generated sample dataset to demonstrate data preparation, model training, evaluation, and future enhancement recommendations.

> **Important:** The included dataset is synthetic and for educational demonstration only. The model results are not evidence of real-world business performance. Replace the sample data with an authorised, suitable real dataset before making real decisions.

## Repository contents

- `data/historical_data.csv` — synthetic monthly observations for demonstration.
- `src/train_evaluate.py` — trains a regression baseline and random-forest model, then reports MAE and RMSE.
- `reports/Week4_AI_Solution_Evaluation_Future_Enhancement_Proposal.docx` — detailed Week 4 evaluation report.
- `requirements.txt` — Python dependencies.
- `README.md` — project setup, method, metrics, limitations, and future plan.

## Problem statement

Historical records can be used to estimate a future numeric outcome. This demonstration predicts `sales` using time-related and business activity features. It is intentionally simple so the evaluation process is easy to understand.

## How to run

1. Install Python 3.10 or later.
2. Open a terminal in the repository folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the evaluation:

   ```bash
   python src/train_evaluate.py
   ```

The script prints MAE and RMSE for a baseline and a Random Forest model. It uses a chronological train/test split to reduce the risk of training on future observations.

## Methodology

1. Load the historical CSV and check required columns.
2. Sort records by month.
3. Use earlier observations for training and later observations for testing.
4. Fit a simple baseline and a Random Forest regression model.
5. Compare test-set performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Evaluation metrics

- **MAE:** average absolute prediction error; lower is better.
- **RMSE:** penalises larger errors more strongly; lower is better.
- **Efficiency:** record runtime and resource use when testing on a real workload.
- **Scalability:** assess performance as data volume and request load increase.
- **User experience:** test whether users understand forecasts, uncertainty, and limitations.

For a classification task, suitable alternatives include precision, recall, F1-score, and a confusion matrix. Metrics must match the actual task and data.

## Limitations and risks

- The dataset is synthetic and does not represent a real business.
- Results depend on the generated sample and should not be generalised.
- A small sample cannot establish production readiness or performance across seasons.
- Data drift, missing values, bias, leakage, and unrepresentative training data can degrade predictions.
- Predictions are estimates, not guarantees.

## Future enhancement plan

1. Replace synthetic data with a properly authorised and documented dataset.
2. Add automated checks for missing values, duplicates, invalid ranges, and leakage.
3. Compare several models using a time-aware validation strategy.
4. Analyse residuals and performance across relevant groups or periods.
5. Add explainability, uncertainty intervals where appropriate, and a user-friendly dashboard.
6. Track dataset/model versions and monitor errors and data drift after deployment.
7. Use access controls, data minimisation, backups, and a rollback plan.
8. Pilot with human oversight before any high-impact use.

## References

- scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- scikit-learn common pitfalls: https://scikit-learn.org/stable/common_pitfalls.html
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
- Google Rules of Machine Learning: https://developers.google.com/machine-learning/guides/rules-of-ml

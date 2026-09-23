# AutoDS Final Model Card

**Problem Type:** classification

**Target:** Churn

**Primary Metric:** f1_weighted

## Verified Final Metrics

- accuracy: 0.8055
- precision_macro: 0.753
- recall_macro: 0.7268
- f1_macro: 0.7376
- f1_weighted: 0.8002
- roc_auc: 0.8419

## Data Quality

Quality score: 98.0

## Known Limitations

- Performance is dependent on the training dataset.
- The model should be validated on future unseen data.
- Feature importance and SHAP values indicate model influence, not causal relationships.
- Data drift may reduce future model performance.
## Leakage experiment A (REAL data): scaler + SelectKBest k=10 + logistic regression

| Metric | Wrong mean | Wrong std | Right mean | Right std | Gap (wrong - right) |
|---|---|---|---|---|---|
| roc_auc | 0.9878 | 0.0087 | 0.9877 | 0.0087 | +0.0000 |
| recall | 0.9278 | 0.0439 | 0.9283 | 0.0443 | -0.0005 |
| precision | 0.9467 | 0.0306 | 0.9467 | 0.0313 | -0.0000 |
| f1 | 0.9364 | 0.0276 | 0.9366 | 0.0284 | -0.0003 |

## Leakage experiment B (SYNTHETIC data): 569 x 500 pure noise features with the real labels, SelectKBest k=10 + logistic regression

| Metric | Wrong mean | Wrong std | Right mean | Right std | Gap (wrong - right) |
|---|---|---|---|---|---|
| roc_auc | 0.6654 | 0.0516 | 0.4737 | 0.0386 | +0.1917 |
| recall | 0.3664 | 0.0580 | 0.2160 | 0.0565 | +0.1505 |
| precision | 0.6129 | 0.0940 | 0.3473 | 0.0705 | +0.2656 |
| f1 | 0.4544 | 0.0578 | 0.2643 | 0.0601 | +0.1901 |

# EEG Model Evaluation Report

**Protocol:** sample-level stratified cross-validation

## Aggregate Metrics

| Metric | Mean | Std |
|--------|------|-----|
| accuracy | 0.3530 | 0.1020 |
| balanced_accuracy | 0.5144 | 0.0378 |
| macro_f1 | 0.3631 | 0.0783 |

## Primary Metrics (calibrated when enabled)

- macro-F1: **0.5072** ± 0.0175
- accuracy: **0.6549** ± 0.0492

## Per-fold

- Fold 1: acc=0.2750, bal_acc=0.4657, macro_f1=0.2860
  - calibrated macro_f1=0.5106
- Fold 2: acc=0.3912, bal_acc=0.5301, macro_f1=0.3925
  - calibrated macro_f1=0.5210
- Fold 3: acc=0.2582, bal_acc=0.4770, macro_f1=0.2922
  - calibrated macro_f1=0.4848
- Fold 4: acc=0.3052, bal_acc=0.5314, macro_f1=0.3460
  - calibrated macro_f1=0.4896
- Fold 5: acc=0.5352, bal_acc=0.5677, macro_f1=0.4990
  - calibrated macro_f1=0.5299

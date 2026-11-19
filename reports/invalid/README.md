# Invalid runs: do not use

First run of `src/evaluate_test.py` (2026-11-19). `feature_columns()` was called after the
Dixon-Coles probabilities were merged in, so `dc_p_home/draw/away` were LR inputs. DC values
existed only for the evaluated seasons, so:

- 2023/24 and 2025/26 (training data had no DC values): DC columns were all-missing in
  training, got zero coefficients, and LR predictions are identical to the frozen model
  (max diff 8e-14).
- 2024/25 and 2026/27 (the previous evaluated season was in the training data, with DC
  values): LR was partly stacked on DC, so the model differs from the one frozen in
  src/config.py (LR probabilities differ by up to 0.09).

The bug was found from an sklearn warning in the run output, not from the scores. The code
was fixed (LR features are taken before the DC merge, with an assertion), and both splits were
rerun with no other change. Valid results: reports/test/ and reports/forward_*/.

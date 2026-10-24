# Engineering log — defects and surprises found while building the solution

Each entry: detection → root cause → fix → confirmation. Numbers come from the files named.

1. **Schedule dates are day-first.** Detection: the schedule's first row reads `09.03.2020 06:00:00`, the view log `2020-03-09 00:00:02`.
   Root cause: two date formats; a generic parser may read `09.03` as 3 September. Fix: explicit formats only, no inference
   (`tools/sweettv_lib.py::to_unix`). Confirmation: `09.03.2020 06:00:00 → 2020-03-09 06:00`, `01.04.2020 → 2020-04-01`.
2. **Clock alignment (trap check, no defect).** Shifting the view log by −4…+4 h: the share of view starts within 2 min of an airing
   start peaks at 0 h (0.0395 vs ≤ 0.0336) and so does the number of watched pairs (`results/eda/clock_alignment.csv`).
3. **TF.js tensor leak.** Detection: self-check “leaves no tensors behind” failed — PPR left 7 tensors per run, MF 2.
   Root cause: intermediates (`tf.equal`, `tf.sum`, `tf.oneHot` input, the pre-transpose division) and the random initialisers passed
   to `tf.variable` (which copies them) were never disposed. Fix: `tf.tidy` around them. Confirmation: 0 leaked tensors, same MAP@5.
4. **`tf.gather` gradient unusable in Node.** Detection: one gradient step did not finish in 100 s on the CPU backend; the WASM backend
   raised “Kernel 'UnsortedSegmentSum' not registered”. Root cause: the gather gradient is a segment sum that WASM lacks and the CPU
   backend computes very slowly. Fix: optimise the implicit-MF loss over the full user × programme matrix with `tf.matMul`
   (no gather). Confirmation: same objective as ALS — Adam reaches loss 231,103 vs ALS 232,854; MAP@5 0.3151 vs 0.3144 (k = 32, λ = 50).
5. **Checkpoint resume is not bit-identical.** `tests/test_checkpoint.js`: 6 straight Adam steps vs. the same steps over 4 processes
   differ by ≤ 3e-6 per factor, because on restore TF.js recomputes Adam's bias-correction powers as βᵗ instead of the running product
   (float32 rounding). Tolerance set to 1e-4; disclosed for the MF test refit.
6. **“Popularity over 2 weeks = over 4 weeks” (not a defect).** Identical MAP@5 looked suspicious; the top-5 lists are identical,
   differences start at rank 6. Popularity gives every user the same five, so the metric cannot differ.
7. **Python vs JS “own history” on all 4,030 users (0.31136 vs 0.31739 in results/summary.csv).** Different tie-break and padding list (viewers vs. views); the plain popularity
   reference matches exactly in both implementations (0.21386).
8. **Selection at the edge of the grid.** MF's best λ = 50 was the largest value tried; the grid was extended to λ ≤ 500 with α, which
   confirmed an interior optimum (λ = 100 → 0.289, 500 → 0.266).
9. **Duplicate settings in the cache.** Settings with a different key order were evaluated twice (identical MAP@5 — determinism);
   fixed with canonical (sorted-key) ids.
10. **Long jobs vs. a 3-minute command limit.** Background processes are killed with the command; the runner became a resumable,
    shardable job list (`--budget`, `--shard`).

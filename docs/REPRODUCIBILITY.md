# Extraction and reproducibility

## Source scope

Source: `Test_20epoh_Kmeans.ipynb`. Cell numbers below are zero-based.

| Notebook cells | Treatment |
| --- | --- |
| 0 | Extracted cleaning, feature alignment, and binning |
| 1 | Main 2014 → 2015 predictor, K=13; converted threshold loop to one configurable run |
| 4 | Main 2013 → 2014 predictor, K=12; settings and seeds extracted |
| 7 | Main 2012 → 2013 predictor, K=8; settings and seeds extracted |
| 2, 5, 9 | Excluded explicit ablations |
| 3, 6, 10 | Excluded zero-gate variants |
| 8, 11–15 | Excluded hard-coded summaries, sensitivity plots, and empty content |

The spectral-clustering branch is unused by the main K-means runs and is excluded. Sampling helpers, duplicate imports, global notebook variables, unused `entmax15`, and the always-zero auxiliary loss are excluded. The main cells' LR reference result is retained by reusing the same fitted risk scorer.

## Behavior retained

- Labels and four features; log1p(DTI); 32 uniform bins, no scaling of employment length.
- In historical mode, reject rows are deduplicated before percentage strings are parsed, preserving the notebook's treatment of distinct strings such as `10%` and `10.0%`. `train_only` deduplicates after numeric conversion.
- Stratified 80/20 accepted-loan split with seed 42.
- LR C grid `[0.001, 0.01, 0.1, 1, 10, 100, 1000]`, `lbfgs`, `max_iter=10000`, best validation AUC with first tie retained.
- Strict `default_probability > threshold` filtering and descending candidate order.
- K-means `n_init=10`, Euclidean distance to each cluster's mean, nearest real samples rather than synthetic centroids; clusters smaller than top-k contribute fewer samples.
- Original model parameter names and initialization order, including an unused `gate` layer. This layer is retained solely because removing its initialization shifts the subsequent classifier's random initial weights; no trainable gate is used in forward.
- Sparsemax attention, one head, hidden dimension 64, fixed fusion weight 0.3, classifier `Linear(64,64) → ReLU → Linear(64,1)`.
- BCE training, original Adam settings, strict validation-loss improvement, patience 3, restoration of the best state.
- The original **unweighted mean of per-batch validation losses** is retained, even when the final batch is smaller. Training loss in the new log is sample-weighted and does not affect optimization or model selection.
- Threshold 0.5 for binary metrics; KS as `max(TPR-FPR)`; population standard deviation across seeds.

## Explicit adjustments

1. Device selection uses CUDA index 0 or CPU instead of hard-coded `cuda:1`.
2. The auxiliary prototype BCE has coefficient zero in the source, so it is removed. This also avoids its `squeeze()` shape error on a singleton batch. The supervised objective is unchanged for finite inputs.
3. Training cleaning now rejects invalid dates, nonnumeric and infinite features predictably; unlike the source it does not suppress all warnings. Well-formed numeric input follows the same operations, but malformed input handling need not match.
4. `notebook_yearwise` retains the original case-sensitive employment mapping and per-year bin fitting. `train_only` is an explicitly different protocol: split first, fit bins on accepted training rows only, and normalize employment strings. New-data prediction does not refit bins.
5. Saved binning state stores interior cut points instead of a pickled sklearn estimator. Ordinal transforms use the same right-sided boundary rule and clipping into first/last bins, including constant features.
6. Configs use a single example threshold 0.2. The notebook scans several values and does not provide one unambiguous final setting for each main run. No validation-based threshold search has been added. Match the intended experimental setting before comparing with existing results.
7. Checkpoints, structured metrics, dependency records, data-count diagnostics, and command-line error handling are new. Existing output directories are protected against overwrite.

## Limits

`notebook_yearwise` fits on all accepted/rejected covariates within each year, including held-out validation and test covariates. Its test results characterize that historical/transductive preprocessing protocol, not training-only future-data evaluation. The saved test-year binner is provided only to make that protocol explicit and reproducible. Neither preprocessing mode reveals actual outcomes for rejected applicants; test metrics are computed on accepted applicants with observed labels.

Five seeds characterize initialization, K-means, and batch shuffling while the accepted train/validation split remains fixed. CPU repeatability is tested. Different PyTorch/sklearn versions, hardware, numerical libraries, or CUDA kernels can still change results. The notebook metadata supplies Python 3.12.4 but no full dependency lock; exact original dependency versions are unknown. `environment.json` records each new run's versions. The package dependency ranges are installation bounds, not a claim that every combination has been tested.

The source notebook, private datasets, stored cell outputs, and trained artifacts are not bundled. Synthetic checks cannot establish equivalence of full-dataset metrics to a paper. No paper publication status, DOI, authorship metadata, or reuse license has been inferred.


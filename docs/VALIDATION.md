# Local validation record

Validated on 2026-10-13 using Python 3.12.14, macOS 15.6 ARM64, and CPU execution.

| Dependency | Observed version |
| --- | --- |
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| scikit-learn | 1.9.1 |
| PyTorch | 2.14.1 |
| entmax | 1.3 |
| pytest | 9.1.1 |
| build | 1.6.1 |

## Checks completed

- **15 tests passed**, including both preprocessing modes, checkpoint reload, CSV inference, repeatability, feature boundaries, singleton batches, and historical reject deduplication order.
- The documented `make-demo` and `train` commands completed on synthetic data; the `predict` command was also exercised by integration tests.
- All four JSON example configs loaded successfully.
- Source distribution and wheel built successfully.
- A separate comparison extracted the original preprocessing function, prototype builder, model class, dataset class, metrics, and training function from the supplied notebook. On synthetic standard-format records, the historical preprocessing arrays and splits, selected prototype rows, initialized parameters, forward outputs, and every model parameter after two training epochs matched exactly on CPU (`rtol=0`, `atol=0`). This comparison used one seed and the 2014 → 2015 setting; it is not a full-data reproduction claim.

Source notebook SHA-256:

```text
3f11291cded8060803d60d9affe3fbe206f19d8c2d151b42552f8d74ae858045
```

The notebook itself and its stored outputs are not included. Full Lending Club experiments and published table values have **not** been rerun. GPU execution and the GitHub-hosted CI matrix have not been exercised locally. The local sandbox prevented physical CPU-core detection; joblib emitted a warning and fell back to the logical core count. All checks completed despite that environment warning.


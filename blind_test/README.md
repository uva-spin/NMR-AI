# Blind two-site d-butanol test

`butanol-multisite-20runs-blind.npz` is a synthetic, ordered NMR challenge dataset based on the repository's fitted two-site d-butanol example. It is not a new experimental acquisition.

The compressed NumPy file contains only:

| Array | Shape | Meaning |
| --- | --- | --- |
| `frequency_mhz` | `(500,)` | Shared frequency grid from 32.3000000 to 33.0628213 MHz |
| `signals` | `(20000, 500)` | Combined recorded-unit sweeps, one spectrum per row |

Both arrays are `float64`. Rows are in acquisition order. The sequence spans 20 runs of 1,000 spectra. Each run begins near zero polarization and progresses toward a positive or negative polarization. Baseline shapes and levels change between runs and remain nearly stable within each run. The signal position drifts across the sequence; discrete phase changes and intermittent Gaussian and sinusoidal noise make extraction harder.

The file deliberately contains no polarization labels, run IDs, baseline references, or separate signal and noise components. A local truth key is retained by the dataset owner and is not part of this repository.

```python
import numpy as np

with np.load('blind_test/butanol-multisite-20runs-blind.npz') as data:
    frequency_mhz = data['frequency_mhz']
    signals = data['signals']
```

Blind file SHA-256: `eaf80e94fc91e5c38ba82561b1082a92c6fcbb18f2938a08d4043e73f99f4dc6`.

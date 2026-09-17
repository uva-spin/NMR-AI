# Python tools

Run commands from the repository root with the environment activated.
The [guided tutorial](https://zetanaut.github.io/NMR-AI/walkthrough.html) supplies
the arguments and expected outputs for each step. Use `--help` to see a
command's options.

| Task | Command |
| --- | --- |
| Preview the baseline measurement | `python tools/preview_baseline.py` |
| Fit the baseline | `python tools/fit_baseline.py` |
| Fit the single-site butanol example | `python tools/match_experimental_signals.py` |
| Compare the two-site butanol model | `python tools/match_butanol.py` |
| Fit raw UVA-ND3 sweeps | `python tools/match_uva_nd3.py` |
| Plot your saved signal fits | `python tools/plot_fit.py` |
| Generate examples from the single-site fits | `python tools/generate_matched_data.py` |
| Combine compatible generated datasets | `python tools/combine_datasets.py` |
| Train a network | `python tools/train_model.py` |
| Predict with a saved model | `python tools/predict.py` |
| Analyze prediction errors | `python tools/analyze_predictions.py` |

## Read the implementation

| Module | Contains |
| --- | --- |
| [circuit.py](circuit.py) | Coil, cable, tuning circuit, and detector |
| [lineshape.py](lineshape.py), [material_lineshapes.py](material_lineshapes.py) | Complex spin-1 susceptibility and the two-site response |
| [baseline_data.py](baseline_data.py), [experimental_data.py](experimental_data.py), [uva_nd3_data.py](uva_nd3_data.py) | Input readers and checks |
| [fit_tuned_baseline.py](fit_tuned_baseline.py), [baseline_parameters.py](baseline_parameters.py) | Baseline fitting and parameter definitions |
| [learning_data.py](learning_data.py) | Input grid, channels, units, and preprocessing |
| [nmr_lab.py](nmr_lab.py) | Simulation, calibration, networks, and grouped data splits |

Architectures include `mlp`, `dnn`, `dnn_extra`, `compact`, and
`physics_multiscale`. The extra-layer DNN is for the advanced targeted-accuracy
challenge; compare it against the best data-only `dnn` result.

## Further exercises and maintenance

`generate_data.py` and `run_model_comparison.py` run the optional
[TE-area benchmarks](../docs/reference/benchmarks.md). `benchmark_network.py`
measures forward-pass timing.

The `export_*.py` scripts rebuild the website's reference figures and reports.
They are described in the [contributor guide](../CONTRIBUTING.md).
`prepare_uva_nd3.py` rebuilds the included ND3 working file; the reader uses
`_data/` internally to verify its interpolation. For lessons, load the files
in `examples/` and view your own results with `plot_fit.py`.

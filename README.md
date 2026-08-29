# PipeLens

PipeLens is a framework for debugging and optimizing data-science pipelines through **causally guided interventions**. It repairs failing pipelines by prioritizing minimal changes to component parameters or pipeline structure using historical executions and data profiles.

Published in PVLDB 19(11), 2026: [PipeLens: Identifying Interventions for Resolving Malfunctioning Data Science Pipelines](https://doi.org/10.14778/3836663.3836682). A local copy is available as [`p3188-hasan.pdf`](p3188-hasan.pdf).

## Key features

- Supports machine-learning, entity-matching, and regression pipelines.
- Supports module insertion, deletion, swapping, and parameter interventions.
- Provides glass-box and opaque-box optimization modes.
- Uses historical executions to learn a proxy utility model.
- Prioritizes minimal and causally effective interventions.

The two main implementations are:

- `glassbox_optimizer.py`: glass-box optimization using profiles and intermediate information.
- `opaque_optimizer.py`: opaque-box optimization using final utility observations.

## Reproduce the example

The reference environment uses Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python reproduce_example.py
```

The final command executes `example.ipynb` from the repository root and writes the executed notebook to `artifacts/example.executed.ipynb`. To explore it interactively, run `jupyter lab example.ipynb`.

## Reproducibility inputs

The tutorial configuration is in `config_example.json`. The example uses:

- Main HMDA data: `data/hmda/hmda_Orleans_X_train_1.csv` and `data/hmda/hmda_Orleans_X_test_1.csv`.
- Tutorial profiles: `historical_data/tutorial/`.
- Pipeline components: `modules/` and their adapters in `pipeline_component/`.

The smoke test verifies imports, configuration, and SHA-256 checksums for all tutorial inputs. Random seeds are set to `42` by the notebook and implementation where stochastic behavior is used.

## Repository structure

```text
.
├── glassbox_optimizer.py
├── opaque_optimizer.py
├── pipeline_execution.py
├── modules/
├── pipeline_component/
├── data/
├── historical_data/tutorial/
├── config_example.json
├── example.ipynb
├── reproduce_example.py
└── tests/
```

## Extending PipeLens

Implementations of pipeline operations live in `modules/`. The corresponding handlers in `pipeline_component/` expose those operations to `PipelineExecutor`. Dataset loading and preprocessing paths are defined in `LoadDataset.py`.

## Citation

Please cite the PVLDB paper using the metadata in `CITATION.cff`.

## License

The published paper is distributed under the Creative Commons BY-NC-ND 4.0 license stated in the paper. A software license has not yet been specified for this source code.

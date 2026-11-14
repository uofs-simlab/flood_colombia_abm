# Flood Colombia ABM

This repository contains the Santa Lucia, Colombia adaptation of Flood-ABM and the analysis code for the associated study, *Community flood defence and rescue logistics: an agent-based scenario analysis for Santa Lucia, Colombia* (Daniel Alejandro Pineda Sanchez, Obeng A. Addai, and Raymond J. Spiteri).

The model represents a synthetic population responding to a flood scenario. It explores community-organization assumptions, access to shelters, rescue logistics, and related behavioural and clinical alternatives. It is an exploratory scenario model, not a flood forecast or a calibrated estimate of real-world intervention effects.

The adaptation builds on [Flood-ABM](https://github.com/uofs-simlab/flood_abm), described by Addai et al. (2025), [doi:10.1016/j.ijdrr.2025.105698](https://doi.org/10.1016/j.ijdrr.2025.105698). The associated Santa Lucia manuscript is maintained separately.

## Quick start

Use Python 3.12. Create and activate a virtual environment, then install the pinned direct dependencies:

```bash
git clone https://github.com/uofs-simlab/flood_colombia_abm.git
cd flood_colombia_abm
python -m venv .venv
```

Activate the environment (`source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\Activate.ps1` in PowerShell), then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run/run_study.py --dry-run
python run/structural/production.py --dry-run
```

The dry run prints the experiment design and does not start simulations. To run one smoke-test job after installing the spatial inputs described in [docs/data.md](docs/data.md):

```bash
python run/run_study.py --family organization --max-jobs 1 --workers 1 --skip-analysis
```

This creates an incomplete study cohort for a software smoke test; it is not a research result and must not be reported as a completed experiment. Full experiments require substantial compute. The primary design contains 480 runs, the one-at-a-time sensitivity study contains 960 additional runs, and the manuscript also reports 1,260 runs testing alternative behavioural and clinical rules.

## Data

Full runs use twelve spatial archives listed in [docs/data.md](docs/data.md). The study release packages these inputs separately from the source-code repository. The layer sources, QGIS processing, route calculations, and source-specific attributions are summarized in [docs/spatial-data-sources.md](docs/spatial-data-sources.md).

## Tests

Run the test suite with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Tests that construct the full model require the spatial inputs. Tests for individual analysis and utility functions may run without them.

## Repository layout

- `agents/`, `model/`, `space/`: agent, model, and spatial-simulation code.
- `run/`: study designs, experiment runners, and verification tools.
- `analysis/`: statistical analysis and generated manuscript assets.
- `tests/`: model, runner, and analysis tests.
- `santa_lucia_map_data/`: expected location for the separately obtained spatial inputs.
- `docs/`: data requirements and project documentation.

Simulation outputs are written under `data_collection/` and are not tracked in Git. Generated publication assets are kept under `analysis/paper_generated/`.

## Assumptions and limitations

The population is synthetic. Flood depths, protective decisions, rescue throughput, clinical progression, and institutional capacities include model-defined assumptions; they are not locally calibrated forecasts. Replication intervals describe stochastic variation under the specified model settings, not uncertainty in real-world outcomes. See the associated manuscript for the full methods and limitations.

## Licensing

The study code is licensed under the MIT License in [LICENSE](LICENSE). The upstream Flood-ABM notice is retained in [UPSTREAM-LICENSE.txt](UPSTREAM-LICENSE.txt). Spatial inputs retain their source-specific terms and attributions, as listed in [docs/spatial-data-sources.md](docs/spatial-data-sources.md).
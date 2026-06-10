# CaDRe

Reference implementation of **CaDRe** (**Ca**usal **D**iscovery and **Re**presentation learning), ICML 2026.

CaDRe jointly recovers the latent dynamic process and the observed causal graph from a multivariate time series, with nonparametric identifiability guarantees.

- **Paper:** [OpenReview](https://openreview.net/forum?id=MZxhwFT7yF) · [arXiv](https://arxiv.org/abs/2501.12500)
- **Authors:** Minghao Fu, Biwei Huang, Zijian Li, Yujia Zheng, Ignavier Ng, Guangyi Chen,
  Yingyao Hu&dagger;, Kun Zhang&dagger;

<sub>&dagger; Equal advising. Affiliations: MBZUAI, CMU, UC San Diego, Johns Hopkins University.</sub>

## Install

```bash
git clone https://github.com/MinghaoFu/CaDRe.git
cd CaDRe
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

| Path | What lives here |
|---|---|
| `SSM/` | Main CaDRe trainers and evaluators (CESM2, WeatherBench). |
| `LiLY/` | VAE encoders / decoders, flow priors, latent dynamics modules. |
| `Caulimate/` | Lightweight utilities for data simulation, graph operations, metrics. |
| `forecasting/` | Forecasting evaluation notebooks. |
| `analyze/` | Climate visualisation and downstream analysis. |
| `dataset/` | Data pre-processing scripts. |
| `scripts/` | Training and inference launchers. |
| `scripts/repro/` | One-line reproducer for each paper experiment. |
| `tests/` | Smoke tests. |

## Reproduce the paper

Set the environment variables (each script reads them):

```bash
export PROJECT_ROOT="$(pwd)"
export DATA_DIR=/path/to/data        # CESM2 / WeatherBench / ERSST
export CKPT_DIR=/path/to/checkpoints
export LOG_DIR=/path/to/logs
```

Run any of the following:

```bash
./scripts/repro/run_synthetic.sh         # §5.1
./scripts/repro/run_neural_baselines.sh  # App. D, Fig. 8
./scripts/repro/run_causalrivers.sh      # App. D, Fig. 9(b)
./scripts/repro/run_higher_order.sh      # App. D, Table 16
./scripts/repro/run_cesm2.sh             # §5.2, Tables 5, 6
./scripts/repro/run_weatherbench.sh      # §5.2, Tables 5, 6
./scripts/repro/run_ersst.sh             # §5.2, Table 6
./scripts/repro/run_forecasting.sh       # §5.2, Tables 5, 18-20
./scripts/repro/run_visualize.sh         # §5.2, Figs. 5, 10, 11
./scripts/repro/run_all.sh               # everything, sequentially
```

## Data

We use four public datasets. We do not redistribute the raw data; download from the official sources and place under `$DATA_DIR/`:

- **CESM2 Pacific SST** — NCAR CESM2 control run.
- **ERSST v5** — [NOAA NCEI](https://www.ncei.noaa.gov/products/extended-reconstructed-sst).
- **WeatherBench** — [Rasp et al. 2020](https://github.com/pangeo-data/WeatherBench).
- **Weather (Jena)** — [BGC Jena](https://www.bgc-jena.mpg.de/wetter/).

## Citation

```bibtex
@inproceedings{fu2026learning,
  title     = {Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis},
  author    = {Fu, Minghao and Huang, Biwei and Li, Zijian and Zheng, Yujia and
               Ng, Ignavier and Chen, Guangyi and Hu, Yingyao and Zhang, Kun},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Contact

Minghao Fu — `isminghaofu[at]gmail.com`

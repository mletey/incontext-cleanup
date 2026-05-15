# Asymptotic Theory of In-Context Learning by Linear Attention

This repository is our code used to create the figures for our paper "Asymptotic theory of in-context learning by linear attention"
by _Yue M. Lu, Mary I. Letey, Jacob A. Zavatone-Veth, Anindita Maiti, and Cengiz
Pehlevan_

Paper links:

- arXiv: <https://arxiv.org/abs/2405.11751>
- DOI: <https://doi.org/10.1073/pnas.2502599122>

## Paper Overview

This paper studies in-context learning in a controlled regression setting. The main analytical object is a simplified linear attention model trained on linear regression tasks. This model is exactly solvable, allowing us to derive sharp asymptotic predictions, while still retaining the key quadratic interaction between tokens that makes attention useful for ICL. Phenomenologically, we identify

- key data variables: token dimension, context length, pretraining sample size, and task diversity
- asympototic scaling regime of these variables
- double descent in sample size
- non-monotonicity of ICL performance in context length
- phase transition in task diversity.

This repository is holds the code release for figure generation and reproduction for the paper. It will collect the source code needed to regenerate the main experiments, aggregate raw outputs into processed data, and produce the final figure panels.

## Figure Roadmap

The cleaned code will be organized around regenerating the paper figures.

#### Figure 1: Sample-Wise Double Descent

Compares asymptotic theory and finite-dimensional simulations for ICL andin-distribution generalization (IDG) error as the number of pretraining examples varies. _The code here will involve simulating the reduced linear attention model._

#### Figure 2: Dependence on Context Length

Studies how performance changes with context length. The figure compares theory and simulations for both ICL and IDG errors. _The code here will involve simulating the reduced linear attention model._

#### Figure 3: Memorization-to-ICL Transition

Shows the transition controlled by pretraining task diversity. _The code here will involve simulating the reduced linear attention model and dMMSE estimator._

#### Figure 4: Double Descent in Nonlinear Transformers

Tests whether the sample-complexity scaling and double descent predicted by the linear theory also appear in full nonlinear Transformer models. _The code here involves training various transformer architectures, plotting their final test losses, as well as numerically determining the location of interpolation threshhold._ 

#### Figure 5: Task-Diversity Transition in Nonlinear Transformers

Tests whether full nonlinear Transformers maintain both the task diversity scaling and phase transition predicted by the theory. _The code here involves training various transformer architectures and plotting their final test losses, as well as numerically evaluating the dMMSE estimator._ 

## Repository Organisation

- `theory_runs/`: Reduced linear-attention theory and simulation code. This computes $\Gamma^*$ from sampled data numerically and evaluates the reduced-linear attention MSE loss on these parameters, for both the ICL and IDG distribution. This folder also includes the finite-task Bayes estimator (dMMSE) simulation code used for the task-diversity comparisons.
- `transformer_codebase/`: Transformer model, data, and training code.
- `test_runs/`: small CPU tests runs. These are intentionally low-dimensional
  (`d=5`) checks over task diversity and are meant to provide an easy proof-of-concept run for the reduced model and the transformers. 
- `easy_runs/`: existing plotting/aggregation scripts plus prepopulated
  processed data from our previous runs.
- `run_from_scratch/`: reserved for full reproduction scripts. 

To run the quick-and-easy tests, use

```bash
bash test_runs/run_theory_task_diversity_cpu.sh
bash test_runs/run_transformer_task_diversity_cpu.sh
bash test_runs/run_all_cpu_smoke_tests.sh
```

The test scripts write CSV outputs under `test_runs/outputs/`, which is ignored
by git.

## Environment

We provide both a conda environment file and pip requirements files. If `conda
env create` is killed while collecting package metadata, use the pip/virtualenv
path below. That usually means the conda solver ran out of memory on a
constrained login node, not that the experiment code is broken.

### Recommended Cluster Setup

This path avoids the conda solver and is usually the most reliable on clusters:

```bash
cd incontext-cleanup

# If your cluster uses environment modules, load Python first, e.g.
# module load python/3.10.12-fasrc01

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda12.txt
```

For CPU-only local development or quick smoke tests, use:

```bash
python -m pip install -r requirements-cpu.txt
```

For CUDA 13 systems with a sufficiently recent NVIDIA driver, use:

```bash
python -m pip install -r requirements-cuda13.txt
```

### Conda Setup

If conda works on your machine, you can instead create and activate the
environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate incontext-asymptotics
```

If `mamba` is available, it is usually faster and uses less memory:

```bash
mamba env create -f environment.yml
conda activate incontext-asymptotics
```

After activating the environment, double check everything's actually been imported 

```bash
python -c "import numpy, scipy, matplotlib, seaborn, tqdm, jax, flax, optax; print('imports ok')"
```

For GPU runs, which we will use for the nonlinear Transformer figures, request
a GPU node to check JAX.

```bash
python -c "import jax; print(jax.devices())"
```

The output should include a CUDA device. If it only
shows `CpuDevice`, then JAX installed but is not using the GPU. 

The default GPU setup uses `jax[cuda12]`, which requires an NVIDIA driver
version of at least 525. The CUDA 13 setup uses `jax[cuda13]`, which requires
driver version at least 580. On clusters, create the environment where internet
access is available, then run the JAX device check on an allocated GPU node.
Avoid loading old CUDA or cuDNN modules unless necessary, since the JAX CUDA
wheels include their own CUDA/cuDNN libraries and older cluster modules can
conflict through `LD_LIBRARY_PATH`.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{lu2025asymptotic,
  title={Asymptotic theory of in-context learning by linear attention},
  author={Lu, Yue M. and Letey, Mary I. and Zavatone-Veth, Jacob A. and Maiti, Anindita and Pehlevan, Cengiz},
  journal={Proceedings of the National Academy of Sciences},
  volume={122},
  number={28},
  pages={e2502599122},
  year={2025},
  doi={10.1073/pnas.2502599122}
}
```

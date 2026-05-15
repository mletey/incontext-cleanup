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

theory runs code base
transformer code base
test runs (gives some cpu-runnable example code)
easy runs (using prepopulated data)
run-from-scratch scripts

## Environment

We provide an `environment.yml` file to install the dependencies needed for our code. From the repository root, create and activate the environment with

```bash
conda env create -f environment.yml
conda activate incontext-asymptotics
```

If you want you can also use

```bash
mamba env create -f environment.yml
conda activate incontext-asymptotics
```

After activating the environment, double check everything's been imported 

```bash
python -c "import numpy, scipy, matplotlib, seaborn, tqdm, jax, flax, optax; print('imports ok')"
```

For GPU runs, which we will use for the the nonlinear Transformer figures, request a GPU node before checking JAX:

```bash
python -c "import jax; print(jax.devices())"
```
The output should include a CUDA device.

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

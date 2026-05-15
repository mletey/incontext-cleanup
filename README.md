# Asymptotic Theory of In-Context Learning by Linear Attention

This repository is our code used to create the figures for our paper
> "Asymptotic theory of in-context learning by linear attention"
> 
> _Yue M. Lu, Mary I. Letey, Jacob A. Zavatone-Veth, Anindita Maiti, and Cengiz
Pehlevan_

Paper links:

- arXiv: <https://arxiv.org/abs/2405.11751>
- DOI: <https://doi.org/10.1073/pnas.2502599122>

## Project Overview

The paper studies in-context learning (ICL) in a controlled regression setting.
The main analytical object is a simplified linear attention model trained on
linear regression tasks. This model is exactly tractable enough to derive sharp
asymptotic predictions, while still retaining the key quadratic interaction
between tokens that makes attention useful for ICL.

The central questions are:

- How many pretraining examples are needed for ICL to emerge?
- How should context length scale with input dimension?
- How diverse must the pretraining tasks be for the model to generalize to new
  tasks rather than memorize the training tasks?
- Do the same scaling phenomena appear in full nonlinear Transformer
  architectures?

The theory identifies a joint asymptotic regime in which input dimension grows,
context length and task diversity scale linearly with dimension, and the number
of pretraining examples scales quadratically with dimension. In this regime, the
model exhibits sample-wise double descent and a transition from memorization to
genuine in-context learning as task diversity increases.

## Purpose of This Repository

This cleanup repository is intended to become the reproducible code release for
the paper. It will collect the source code needed to regenerate the main
experiments, aggregate raw outputs into processed data, and produce the final
figure panels.

The current goal is to make the repository suitable as a grant deliverable and
as a usable artifact for readers who want to inspect or reproduce the results.
The previous experiment archive contains many exploratory runs, logs, cached
pickles, and intermediate files. This repository should instead contain only the
curated code, documented data products, and clear reproduction instructions.

## Figure Roadmap

The cleaned code will be organized around regenerating the paper figures.

### Figure 1: Sample-Wise Double Descent in Linear Attention

Compares asymptotic theory and finite-dimensional simulations for ICL and
in-distribution generalization (IDG) error as the number of pretraining examples
varies. This figure demonstrates the predicted double-descent behavior and the
interpolation threshold in the linear attention model.

### Figure 2: Dependence on Context Length

Studies how performance changes as context length scales with input dimension.
The figure compares theory and simulations for both ICL and IDG errors and
shows that ICL error can be non-monotonic in context length, while IDG error is
monotonic in the plotted regimes.

### Figure 3: Memorization-to-ICL Transition

Shows the transition controlled by pretraining task diversity. At low task
diversity, the model behaves like it has memorized the finite set of training
tasks and is close to the dMMSE estimator. At high task diversity, it approaches
the ridge estimator and generalizes to new task vectors, indicating genuine
in-context learning.

### Figure 4: Double Descent in Nonlinear Transformers

Tests whether the sample-complexity scaling predicted by the linear theory also
appears in full nonlinear Transformer models. The figure shows double descent
in nonlinear Transformer ICL error and checks that the interpolation threshold
scales quadratically with input dimension.

### Figure 5: Task-Diversity Transition in Nonlinear Transformers

Tests whether full nonlinear Transformers also transition from memorization-like
behavior to ICL-like behavior as task diversity increases. The figure compares
the nonlinear model to the same ridge and dMMSE reference estimators used in the
linear theory.

### Supplementary and Appendix Figures

Additional figures will document robustness checks, alternate parameter
settings, finite-sample estimators, and supporting comparisons used in the
supplementary information. These will be added to the roadmap as the cleanup
process identifies the final scripts and data products needed for each panel.

## Repository Status

This repository is currently a cleanup scaffold. The environment, directory
layout, command-line entry points, and data manifest are still being defined.

Planned next steps:

- Add a reproducible Python environment specification.
- Define the final repository topology for source code, experiment configs,
  processed data, and figures.
- Add small smoke-test runs that can be executed without a cluster.
- Add figure-specific reproduction commands.
- Decide which processed data files should be tracked in git and which raw or
  large artifacts should live in an external archive.

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

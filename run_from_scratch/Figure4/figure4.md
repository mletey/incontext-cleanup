# Figure 4 data sweep parameters and instructions

This figure requires both finite-$\Gamma*$ reduced linear attention simulations, and also dMMSE simulations.

### Finite-$\Gamma^*$ Simulations
To generate the data used for panel A of figure 1 (logscale task generalisation), run `run_gtask_from_data.py` for the follow sweeps of values

> `kappa=[0.2,0.5,1,10,50]`
> `alpha=[6,10,20]`
> `d=100`
> `tau=0.2*alpha`
> `rho=0.01`

To generate the data used for panel B, run `run_gtask_from_data.py` for the follow sweeps of values

> `kappa=[0.2,0.5,1,10,50]`
> `alpha=[10,20,40]`
> `d=100`
> `tau=0.2*alpha`
> `rho=0.01`


`run_gtask_from_data.py` can be called by 

```
python run_gtask_from_data.py --d=100 --kappa=1 --alpha=1 --tau=5 --rho=0.01 --lam=0.00001 --numavg=10 
```

It returns 2 values, mean `g_task = e_ICL - e_IDG`, and standard.dev. of `g_task` over the `numavg` simulations. These values are saved to a csv file. The data is added to the csv in a threading-safe way so that the calls to this python file over the parameter values listed above can be parallelised over, say, a job array. 

### dMMSE Simulations

`run_gtask_dmmse.py` can be called by 

```
python run_gtask_dmmse.py --d=20 --kappa=1 --alpha=1 --rho=0.01 --numavg=1000 
```

Note that $\tau$ and $\lambda$ are no longer necessary parameters as the estimator does not use these. Similar to `run_gtask_from_data.py` this will save `g_task` mean and stdv over `numavg` simulations; we recommend using far more averaging for this estimator as it concentrates far slower. These values are saved, in a threading safe way, to a (different) csv for later use. 
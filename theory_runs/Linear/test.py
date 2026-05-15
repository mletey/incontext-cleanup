from common import *

d=16; alpha=1; kappa=1; tau=1.1; rho=0.01; lam=0.0001; numavg=20
a,b,c,d = errors_from_DATA(d, alpha, kappa, tau, rho, lam, numavg)
print('data = ', a)
print('theory = ', ridge_icl(alpha, kappa, tau, rho, lam))
print('theory = ', ridgeless_icl(alpha, kappa, tau, rho))
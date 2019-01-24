# note: need to have Rcpp and RcppEigen packages installed
library(Rcpp)

sourceCpp("../src/spd.cpp",showOutput = TRUE)
sourceCpp("../src/linzengspd.cpp",showOutput = TRUE)

source("dataGen.R")

# generate some data
# parameters
gamma = rnorm(5); beta = rnorm(4)
# disease prevalence
ksi = 0.05
ds = caseControlData(80,80,gamma,beta,ksi,"genX","normalY","sigma2" = 1.33)

# initialize parameters
th0 = rand_params(ds,ymodel = "LZ")

# fit models:
res1 = fit_lzspd(ds$X,ds$Y,ds$dd,ds$ksi,th0)
res2 = fit2pd(ds$X,ds$Y,ds$dd,ds$ksi,verb = 2,TOL=1e-6)

beta
res2$beta
c(res2$beta)

library(numDeriv)

# checking derivative/Hessian of models
grad_n = numDeriv::grad(spdLL,x = th0,dd = ds$dd,X = ds$X,Y = ds$Y,ksi = ds$ksi)
grad_a = spdGrad(ds$dd,ds$Y,ds$X,ds$ksi,th0)
crossprod(grad_n - grad_a) # should be ~ 0

# LZ model Hessian
hess_n = numDeriv::hessian(lzLL,x = th0,d = ds$dd,X = ds$X,y = ds$Y,rate = ds$ksi)
hess_a = c_hess(th0,d = ds$dd,X = ds$X,y = ds$Y,rate = ds$ksi)
hess_n - hess_a


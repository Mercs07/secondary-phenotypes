library(Rcpp)

sourceCpp("../src/spd.cpp",showOutput = TRUE)
sourceCpp("../src/linzengspd.cpp",showOutput = TRUE)

source("dataGen.R")

# generate some data
gamma = rnorm(5); beta = rnorm(4)
ds = caseControlData(80,80,gamma,beta,0.05,"genX","normalY")

# initialize parameters
th0 = rand_params(ds)

# fit models:
res1 = fit_lzspd(ds$X,ds$Y,ds$dd,ds$ksi,th0)
res2 = fit2pd(ds$X,ds$Y,ds$dd,ds$ksi)

beta
res1$beta
c(res2$beta)

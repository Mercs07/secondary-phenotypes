# data generation for secondary phenotypes data
library(MASS)

# these functions specify the form of X, then Y|X
# and are looked up by name via match.fun() later on.
# change them (or create more versions, etc.) as desired

genX = function(N,P){
	C1 = floor(3*runif(N))-1  # an alelle with 3 equally likely types
	X = cbind(matrix(runif(N*(P-1),-1,1),nrow=N),C1) # non-negative for gammas
	round(X,1) # testing UX stuff
	# exp(cbind(C1-1,matrix(runif(N*(P-1),-1,1),nrow=N)))  # for gammas
}

# for multivariate outcomes with different margins
# helper generating correlated uniforms. S can be covariance or correlation
corrU = function(S,N){
  S2 = 2*sin((pi/6)*cov2cor(S))
  pnorm(mvrnorm(N,rep(0,nrow(S)),S2))
}

# ... needs arguments "Sigma', 'sd', and 'shape'
normal_gamma = function(xb,...){
  dots = list(...)
  stopifnot(all(c("Sigma","sd","shape") %in% names(dots)))
  stopifnot(ncol(xb)>=2,nrow(dots$Sigma)==2)
  N = nrow(xb)
  U = corrU(dots$Sigma,N)
  cbind(qnorm(U[,1],xb[,1],dots$sd),
        qgamma(U[,2],shape = dots$shape, rate = exp(xb[,2])))
}

# Y is a function of xb and possibly dispersion (specify baseline dist'n here)
# normal deviates

normalY = function(mu,...){
	N = NROW(mu)
	Q = NCOL(mu)
	exArg = list(...)
	sigL = pmatch(tolower(names(exArg)),c("sig","sig2","sigma","sigma2"))
	if(sum(!is.na(sigL))==0){
		sig2 = 1 # default (co)variance
	} else {
		sig2 = exArg[[ which(!is.na(sigL))[1L] ]]
	}
	if(Q==1){   # univariate
		sig2 = abs(sig2[1])
		res = rnorm(length(mu),mu,sqrt(sig2))
	} else {	# multivariate
		if(!is.matrix(sig2)) sig2 = diag(abs(sig2[1]),Q)
		require(MASS)
		res = t(apply(mu,1,mvrnorm,n=1,Sigma=sig2))
	}
	round(res,1)
}

smat_y = function(xb,...){
  ea = list(...)
  ysig = match.arg(names(ea),choices=c("sigma","Sigma","Sigma2","sigma2","ySig","ySigma"))
  S2 = ea[[ysig]] # covariance matrix
  stopifnot(nrow(S2)==ncol(xb),isSymmetric(S2))
  t(apply(sweep(xb,2,sqrt(diag(S2)),`*`),1,"mvrnorm",n=1,Sigma=S2))
}

# gamma deviates: we may include a multivariate version for correlated gammas
# expect two parameter (vectors), mu and sd but check these!
pyGamma = function(mu,...){
	exArg = list(...)
	sdL = pmatch(names(exArg),c("sd","s.d.","stddev","std.dev."))
	if(sum(!is.na(sdL))==0){
		sd = 1 # default
	} else {
		sd = exArg[[ which(!is.na(sdL))[1L] ]]
	}
	# ensure positivity of shape, rate
	EPS = 1e-1
	if(min(mu) < 0){mu = mu - min(mu) + EPS}
	if(min(sd) < 0){sd = sd - min(sd) + EPS}
	# shp = mu*mu/(sd*sd) # do we need constant shape for this to work?
	# rat = mu/(sd*sd)
	res = rgamma(length(mu),shape=mu,rate = sd)
	#print(range(res))
	res
}


# simple case when y is logistic given x
pyLogis = function(mu){
  pi = 1/(1+exp(-mu))
  rbinom(length(mu),1,pi)
}

# the intercept-adjustment function (nice function but of questionable need)

# Rcpp::sourceCpp("C:/Users/skm/Dropbox/2pd/source/gamma0.cpp") # findG0 is the exported function

find_g0 = function(XY,gamma,ksi){
  xyg = XY%*%gamma
  f0 = function(g){mean(1/(1+exp(-g-xyg))) - ksi}
  lb = -1; ub = 1
  while(f0(lb) > 0) lb = lb - 1
  while(f0(ub) < 0) ub = ub + 1
  uniroot(f0,interval = c(lb,ub))$root
}

# calculate, roughly, a large enough population size to sample from (but not too large!)
# for disease prevalence xi and desired case sample size N, find a population size P which generates
# enough cases with at least 1-eps probability
calcP = function(N,xi,eps){
	P0 = min(N/xi,5000); MAXIT = 20   # initial guess
	ep = pnorm(N/P0,xi,sqrt(xi*(1-xi)/P0))
	i = 0
	while(ep > eps & i < MAXIT){
		i = i+1
		P0 = P0*1.25
		ep = pnorm(N/P0,xi,sqrt(xi*(1-xi)/P0))
	}
	if(i==MAXIT) stop("Expected population is too large!")
	round(P0)
}

# this function assumes that the first element in beta is the intercept
# and doesn't use this except adding it to Y.  Returned X doesn't have intercept col. either
# for compatibility with Lin-Zeng model

caseControlData = function(nCase,nControl,gamma,beta,ksi,Xname,Yname,printInfo=FALSE,y_trans = "identity",...){
  beta = as.matrix(beta)
	P = nrow(beta); Q = ncol(beta); N = nCase + nControl
	stopifnot(ksi > 0 && ksi < 1)
	# create a 'large' population to sample from (since we should only calculate gamma_0 once)
	popSize = calcP(nCase,min(ksi,1-ksi),1.0e-8)  # at least 20x the input size should pretty much guarantee enough cases + controls
	if(printInfo) cat(paste0("Population size: ",popSize,".\n"))
	if(length(gamma) != P+Q){  # beta0 doesn't factor into D, but gamma has its own intercept
		stop(paste0("gamma/beta dim. mismatch: X has ",P," columns, Y has ",Q,", but input gamma is length ",length(gamma),"."))
	}
	Xf = match.fun(Xname)
	Xpop = Xf(popSize,P-1)  # NOTE: subtract one so as to not include intercept!
	xb = cbind(1,Xpop)%*%beta # intercept(s) added here
	Yf = match.fun(Yname)
	Ypop = match.fun(y_trans)(as.matrix(Yf(xb,...)))
	#print(paste0("col(X) = ",ncol(Xpop),"; col(y) = ",ncol(Ypop),", len(gam) = ",length(gamma)))
	XY = cbind(Xpop,Ypop)
	gamma[1] = find_g0(XY,gamma[-1],ksi)
	if(printInfo) cat(paste0("Gamma intercept = ",sprintf("%.3f",gamma[1]),"\n"))
	pis = 1/(1+exp(-gamma[1] - XY%*%gamma[-1]))
	Dpop = rbinom(popSize,1,pis)
	if(printInfo) cat(paste0("Disease prevalence in population: ",round(mean(Dpop),4),".\n"))
	popCase = sum(Dpop)
	if(nCase > popCase || nControl > popSize - popCase){
		# if not enough, call recursively with double the population size. calcP should avert this almost always
		return(caseControlData(2*nCase,2*nControl,gamma,beta,ksi,Xname,Yname,printInfo=FALSE,...))
	}
	caseInx = which(Dpop == 1); controlInx = which(Dpop == 0)
	Xcase = as.matrix(Xpop[caseInx[1:nCase],])
	Xcontrol = as.matrix(Xpop[controlInx[1:nControl],])
	Ycase = as.matrix(Ypop[caseInx[1:nCase],])
	Ycontrol = as.matrix(Ypop[controlInx[1:nControl],])
	w0 = N*ksi/nCase
	w1 = N*(1-ksi)/(nControl)
	list('X' = rbind(Xcase,Xcontrol),'Y' = rbind(Ycase,Ycontrol),
		'dd' = c(rep(1,nCase),rep(0,nControl)),'int' = gamma[1],
		'ksi' = mean(Dpop),'w' = c(rep(w0,nCase),rep(w1,nControl)))
}

# generate random (hopefully decent) initial parameter values
# from the output of caseControlData
rand_params = function(ds,ymodel = "drm"){
  # shared parameters: gamma (since intercept is different for beta)
  g0 = unname(coef(glm(ds$dd ~ cbind(ds$X,ds$Y),family = binomial())))
  g0[1] = ds$int
  if(tolower(ymodel) == "drm"){
    b0 = unname(c(0.33*as.matrix(coef(lm(ds$Y~ds$X)))[-1,]))
    mY = nrow(unique(ds$Y))
    return(c(runif(mY-1,-0.1,0.1),b0,g0))
  } else if(tolower(ymodel) == "lz") {
    m0 = lm(ds$Y~ds$X)
    b0 = unname(c(0.33*coef(m0)))
    sig0 = sigma(m0)^2
    return(c(b0,sig0,g0))
  }
  stop("ymodel value ",ymodel," not supported!")
}


# should get moved elsewhere - not logically part of this module
# format a numeric matrix for pretty-printing in a LaTeX array
texmatrix = function(x,dig=2){
  cfmt = paste0("%.",dig,"f")
  endl = function(){cat("\\\\\n")}
  hp = "\\hphantom{-}"
  hasNeg = colSums(x<0) > 0
  cat("\\begin{pmatrix}\n")
  for(i in 1:nrow(x)){
    for(j in 1:(ncol(x)-1)){
      if(hasNeg[j] & x[i,j] >= 0){
        cat(hp)
      }
      cat(sprintf(cfmt,x[i,j]));cat(" & ")
    }
    if(hasNeg[j+1] & x[i,j+1] >= 0){
      cat(hp)
    }
    cat(sprintf(cfmt,x[i,j+1]));endl()
  }
  cat("\\end{pmatrix}\n")
}



#

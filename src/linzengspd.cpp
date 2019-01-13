// Model fitting for secondary phenotypes regression

// [[Rcpp::depends(RcppEigen)]]
#include "gauher.h"  // gaussian quadrature subroutine
#include "util.h"  // uniq() and other helpers
#include "logistic.h" // initializing gamma
#include "linemins.h" // objective function maximization via BFGS
#include <cmath>
#include <algorithm> // std::copy
#include <array>
#include <iostream>
#include <type_traits>

using MapVxd = Eigen::Map<Eigen::VectorXd>;
using MapMxd = Eigen::Map<Eigen::MatrixXd>;
using MapVxi = Eigen::Map<Eigen::VectorXi>;
using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using ArrRef = const Eigen::Ref<const Eigen::ArrayXd>&;
using Arr2Ref = const Eigen::Ref<const Eigen::ArrayXXd>&;

using namespace Eigen;

void printRng(MatRef v,const char* name){
	Rcpp::Rcout << name << " (" << v.rows() << " x " << v.cols() << ") has range: (" << v.minCoeff() << ", " << v.maxCoeff() << ")" << std::endl;
}

inline double flam(double lam,ArrRef z,ArrRef nz,double n){
	return (nz/(lam*z+n)).sum() - 1.;
}

// z are differences E_k[pi(y)] - xi; for a feasible solution we need @ least one > 0 and one < 0
double calcLambda(ArrRef z,ArrRef nz){
	const double N = nz.sum();
    constexpr double eps = 1.0e-5, conv = 1.0e-7;
	constexpr unsigned maxit = 150;
	double zMin = z.minCoeff(), zMax = z.maxCoeff(), mid, fL = 1.0; // bounding asymptotes
	if(zMin >= 0. || zMax <= 0.) return 0; 							// trivial solution
	double aLeft = -N/zMax, aRight = -N/zMin, fp0 = -(z*nz).sum();  // fp0 is slope @ lambda = 0
	if(fp0 > 0.){
		aRight = -eps;
		aLeft += eps;
	} else {
		aLeft = eps;
		aRight -= eps;
	}
	for(unsigned i=0;i<maxit;++i){
		if(std::abs(fL) < conv) break;
		mid = 0.5*(aLeft + aRight);
		fL = flam(mid,z,nz,N);
		if(fL*flam(aLeft,z,nz,N) > 0) aLeft = mid;
		else aRight = mid; 
	}
	return mid;
}

// functor class for use with dfpmin - implementing operator() and method df() for dfpmin routines
// NOTE for consistency with other routines, do not include an intercept column in X - it gets added in the constructor.
class lzspd {
	public:
	const MatrixXd X,XY;
	const VectorXd y,d;
    const double xi;
    const int n,p,ngq,nParam;
    VectorXi nix; // cannot be const with current uniq() implementation, but this is not used directly anyways
    const MatrixXd UX; // XY is for multiplying with gamma: [X,Y]. UX gives the unique X matrix. NOTE: X should come with an intercept column!!
    const VectorXd nx;
    const MatrixXd nxX;
	const int m;  // # of uniquely observed X values
    VectorXd w,x;  // Gaussian quadrature weights and abscissae

	// constructor with Map inputs
	lzspd(const MapMxd xx,const MapVxd yy,const MapVxd dd,double inxi,int numWeights = 20) :
	    X(cbind1(xx)), XY(cbind2(xx,yy,true)), y(yy), d(dd), xi(inxi), n(X.rows()), p(X.cols()),
        ngq(numWeights), nParam(2*p+2), nix(n), UX(uniq(X,nix)), nx(nix.cast<double>()),
        nxX(UX.array().colwise()*nx.array()), m(UX.rows()) { init(); }
	// constructor with matrix inputs
	lzspd(MatRef xx,VecRef yy,VecRef dd,double inxi,int numWeights = 20) :
	    X(cbind1(xx)), XY(cbind2(xx,yy,true)), y(yy), d(dd), xi(inxi), n(X.rows()), p(X.cols()),
        ngq(numWeights), nParam(2*p+2), nix(n), UX(uniq(X,nix)), nx(nix.cast<double>()),
        nxX(UX.array().colwise()*nx.array()), m(UX.rows()) { init(); }
	
	// gaussian quadrature subroutine. gamma2 is the gamma coefficient for y. The vector 'x' really has y values in this context.
    // how the hell is it possible to declare an alias to this function with specialized template parameters????????????????
    // like 'using calcW0 = calcW<0,0>;'
    template<unsigned py,unsigned piLVL>
	ArrayXXd calcW(VecRef xb,VecRef xg,const double sigma2,const double gamma2) const;
    
	double operator()(VecRef) const;  // log-likelihood
	void df(VecRef,Ref<VectorXd>) const;   // gradient
    VectorXd grad(VecRef) const;
	void Hess(VecRef,Ref<MatrixXd>) const;  // Hessian
    MatrixXd opg(VecRef theta) const; // outer-prod. of gradients estimate of covariance (only valid @ MLE)
    double lambda(VecRef theta) const {
        const VectorXd gamma(theta.tail(p+1));
        const VectorXd uxb(UX*theta.head(p)), uxg(UX*gamma.head(p));
        ArrayXXd ww = calcW<0,0>(uxb,uxg,theta(p),gamma(p));
        return calcLambda(ww.col(0) - xi,nx);
    }
    // point masses describing the empirical/profiled distribution of X
    VectorXd pX(VecRef theta) const {
        const VectorXd gamma(theta.tail(p+1));
        const VectorXd uxb(UX*theta.head(p)), uxg(UX*gamma.head(p));
        ArrayXXd ww = calcW<0,0>(uxb,uxg,theta(p),gamma(p));
        const ArrayXd zi(ww.col(0) - xi);
        double lam = calcLambda(zi,nx);
        VectorXd pks = nx.array()/(n + lam*zi);
        return pks/pks.sum(); // upweight based on multiplicities and re-norm to make an estimated X-distribution
    }
	private:
	static constexpr double sqrtPiInv{1./std::sqrt(3.141592654)};
	void init(){ // initial data processing before model fit
		w = VectorXd(ngq);
		x = VectorXd(ngq);
        try {
            auto gq = gauher<double>(static_cast<size_t>(ngq)); // calculate quadrature points & weights, store in x and w respectively
            // stupid copying time...apparently no easy way to initialize an Eigen vector without using the map class...which doesn't own its own memory
            std::copy(gq.first.cbegin(),gq.first.cend(),w.data());
            std::copy(gq.second.cbegin(),gq.second.cend(),x.data());
        } catch(...){
            Rcpp::stop("quadrature failed...oy vey!");
        }
	}
};

// calculating weights of profiled X distribution, and related quantities for gradient/Hessian
// py: power of y; for likelihood, it's zero, gradient and Hessian involve higher-order terms up to y^3
// piLVL: which 'moment' of logistic probabilities. '0' - just 'pi' (see eArg), '1': variance pi(1-pi); '2': third moment pi(1-pi)(1-2pi)
template<unsigned py,unsigned piLVL>
Eigen::ArrayXXd lzspd::calcW(VecRef xb,VecRef xg,const double sigma2,const double gamma2) const {
    ArrayXd eArg(ngq), yp(ngq);
    const ArrayXd xadj = std::sqrt(2.*sigma2)*x;  // add xb(k) to get mean for kth X
    const int M = xb.size(); // sometimes we need UX*beta, sometimes (scores) X*beta
    if(xg.size() != M) Rcpp::stop("bad call to calcW!");
    ArrayXXd W(M,py+1);
    for(int k=0;k<M;++k){ // @ each point mass of profiled X distribution
        eArg = 1./(1.0 + (-xg(k) - gamma2*(xadj + xb(k))).exp());
        if(piLVL == 1) eArg *= (1. - eArg); // any d_gamma term
        else if(piLVL == 2) eArg *= (1. - eArg)*(1. - 2.*eArg); // d_gamma - d_gamma terms
        yp = w;
        for(unsigned p=0;p <= py; ++p){
            W(k,p) = (yp*eArg).sum();
            yp *= (xadj + xb(k)); // increment when differentating according to analytical form
            //yp *= x.array(); // power to increment when differentiating according to quadrature approximation
        }
    }
    return sqrtPiInv*W;
}

double lzspd::operator()(VecRef theta) const {
	const VectorXd beta(theta.head(p)), gamma(theta.tail(p+1));
	const double sigma2 = theta(p);
	const VectorXd uxb(UX*beta), uxg(UX*gamma.head(p)), gxy(XY*gamma);
	double LL = - 0.5*(n*std::log(sigma2) + (y-X*beta).squaredNorm()/sigma2) + d.dot(gxy) - (1. + gxy.array().exp()).log().sum();
    const ArrayXd zi = calcW<0,0>(uxb,uxg,sigma2,gamma(p)).col(0) - xi;
	const double lam = calcLambda(zi,nx);
	LL -= nx.dot((n + lam*zi).log().matrix()); // interaction term
	return -LL/n;
}

// [[Rcpp::export]]
double lzLL(Eigen::VectorXd theta,Eigen::MatrixXd X,Eigen::VectorXd y,Eigen::VectorXd d,
    double rate,int ngq = 20){
    const lzspd D(X,y,d,rate,ngq);
    return D(theta);
}

// this one differentiates according to how the model's actually calculated (weighted sum)
// as opposed to calculing derivative of the theoretical model which we numerically approximate
VectorXd lzspd::grad(VecRef theta) const {
    VectorXd g(theta.size());
    const VectorXd beta(theta.head(p)), gamma(theta.tail(p+1));
	const double sigma2 = theta(p), s4 = sigma2*sigma2; // gammaY = gamma(p);
	const VectorXd xb(X*beta), gxy(XY*gamma), uxg(UX*gamma.head(p));
	const ArrayXd uxb(UX*beta);
	const ArrayXXd WW = calcW<2,0>(uxb,uxg,sigma2,gamma(p));
    const ArrayXXd wpi = calcW<1,1>(uxb,uxg,sigma2,gamma(p));
    const ArrayXd zi(WW.col(0) - xi);
	const double lam = calcLambda(zi,nx);
	const ArrayXd denoms(lam*zi + n);
	const VectorXd pis = 1./(1. + ((-gxy.array()).exp()));
	g.head(p) = (1./sigma2)*X.transpose()*(y-xb);  // normal gradient
	g(p) = -n/(2.*sigma2) + (0.5/s4)*(y-xb).squaredNorm();  // normal sigma2
	g.tail(p+1) = XY.transpose()*(d-pis);   // logistic gradient
    g.head(p) -= (lam/sigma2)*(nxX.transpose()*((WW.col(1) - uxb*WW.col(0))/denoms).matrix()); //V1
    // VectorXd W0 = WW.col(0);
    // g.head(p) -= gammaY*lam*(nxX.transpose()*(WW.col(0)/denoms).matrix()); //V2
    g.segment(p+1,p) -= lam*(nxX.transpose()*(wpi.col(0)/denoms).matrix());//V1
    // g.segment(p+1,p) -= lam*(nxX.transpose()*(WW.col(0)/denoms).matrix());//V2
	g(p) -= (0.5*lam/s4)*(nx.array()*(WW.col(2)-2.*uxb*WW.col(1) + (uxb*uxb-sigma2)*WW.col(0))/denoms).sum(); // interaction sigma2
    // g(p) -= (gammaY*lam/(std::sqrt(2.*sigma2))) * ((nx.array()*WW.col(1))/denoms).sum();//V1
	g(2*p+1) -= lam*(nx.array()*wpi.col(1)/denoms).sum();  // gamma(Y) gradient
    // g(2*p+1) -= lam*((std::sqrt(2.*sigma2)*WW.col(1) + uxb*WW.col(0))/denoms).sum();//V2
	return -(1./n)*g;
}

MatrixXd lzspd::opg(VecRef theta) const {
    const VectorXd beta(theta.head(p)), gamma(theta.tail(p+1));
	const double sigma2 = theta(p), s4 = sigma2*sigma2;
	const ArrayXd xb(X*beta), gxy(XY*gamma);
	const ArrayXXd WW = calcW<2,0>(xb,gxy,sigma2,gamma(p));
    const ArrayXXd wpi = calcW<1,1>(xb,gxy,sigma2,gamma(p));
    const ArrayXd zi(WW.col(0) - xi);
	const double lam = calcLambda(zi,VectorXd::Constant(n,1.));
	const ArrayXd denoms(lam*zi + n);
	const VectorXd pis = 1./(1. + ((-gxy.array()).exp()));
    MatrixXd scores(n,nParam);
    ArrayXd beps(y - xb.matrix()), deps(d - pis);
    // normal/logistic model scores
    scores.leftCols(p) = (1./sigma2)*(X.array().colwise()*beps); // normal scores
    scores.col(p) = -1./(2*sigma2) + (0.5/s4)*(beps*beps);
	scores.rightCols(p+1) = XY.array().colwise()*deps;
    // Rcpp::Rcout << "score sums before interaction:\n" << scores.colwise().sum() << std::endl;
    // interaction term scores
    scores.leftCols(p) -= (lam/sigma2)*(X.array().colwise()*((WW.col(1) - xb*WW.col(0))/denoms)).matrix();
    scores.col(p) -= (0.5*lam/s4)*((WW.col(2)-2.*xb*WW.col(1) + (xb*xb-sigma2)*WW.col(0))/denoms).matrix();
    scores.block(0,p+1,n,p) -= lam*(X.array().colwise()*(wpi.col(0)/denoms)).matrix();
    scores.col(nParam-1) -= lam*(wpi.col(1)/denoms).matrix();

    ArrayXd scoreSums = scores.colwise().sum();
    // Rcpp::Rcout << "final score sums: " << scoreSums.transpose() << std::endl;
    // double maxC = scoreSums.abs().maxCoeff();
    // if(maxC > n*1e-4) Rcpp::Rcout << "Warning: scores not numerically zero: sums = " << scoreSums.transpose() << std::endl;
    return scores.transpose()*scores;
}

// [[Rcpp::export]]
Rcpp::NumericVector grad2(Eigen::VectorXd theta,Eigen::MatrixXd X,Eigen::VectorXd y,Eigen::VectorXd d,
    double rate,int ngq = 20){
    const lzspd D(X,y,d,rate,ngq);
    VectorXd gd = D.grad(theta);
    return Rcpp::wrap(gd);
}

// outer-product-of-gradients estimator of parameter covariance
// [[Rcpp::export]]
Rcpp::NumericMatrix spd_opg(Eigen::VectorXd theta,Eigen::MatrixXd X,Eigen::VectorXd y,Eigen::VectorXd d,
    double rate,int ngq = 20){

    const lzspd D(X,y,d,rate,ngq);
    return Rcpp::wrap(D.opg(theta));
}

// [[Rcpp::export]]
Rcpp::List GQ(int n){
    auto res = gauher<double>(static_cast<size_t>(n));
    Rcpp::NumericVector W(res.first.cbegin(),res.first.cend());
    Rcpp::NumericVector X(res.second.cbegin(),res.second.cend());
    return Rcpp::List::create(
        Rcpp::Named("weights") = W,
        Rcpp::Named("abscissae") = X
    );
}

void lzspd::df(VecRef theta,Eigen::Ref<Eigen::VectorXd> g) const {
	const VectorXd beta(theta.head(p)), gamma(theta.tail(p+1));
	const double sigma2 = theta(p), s4 = sigma2*sigma2;
	const VectorXd xb(X*beta), gxy(XY*gamma), uxg(UX*gamma.head(p));
	const ArrayXd uxb(UX*beta);
	const ArrayXXd WW = calcW<2,0>(uxb,uxg,sigma2,gamma(p));
    const ArrayXXd wpi = calcW<1,1>(uxb,uxg,sigma2,gamma(p));
    const ArrayXd zi(WW.col(0) - xi);
	const double lam = calcLambda(zi,nx);
	const ArrayXd denoms(lam*zi + n);
	const VectorXd pis = 1./(1. + ((-gxy.array()).exp()));
	g.head(p) = (1./sigma2)*X.transpose()*(y-xb);  // normal gradient
	g(p) = -n/(2.*sigma2) + (0.5/s4)*(y-xb).squaredNorm();  // normal sigma2
	g.tail(p+1) = XY.transpose()*(d-pis);   // logistic gradient
    g.head(p) -= (lam/sigma2)*(nxX.transpose()*((WW.col(1) - uxb*WW.col(0))/denoms).matrix());
    g.segment(p+1,p) -= lam*(nxX.transpose()*(wpi.col(0)/denoms).matrix());
	g(p) -= (0.5*lam/s4)*(nx.array()*(WW.col(2)-2.*uxb*WW.col(1) + (uxb*uxb-sigma2)*WW.col(0))/denoms).sum(); // interaction sigma2
	g(2*p+1) -= lam*(nx.array()*wpi.col(1)/denoms).sum();  // gamma(Y) gradient
	g *= -1./n; // need negative gradient for dfpmin
}

// [[Rcpp::export]]
Rcpp::NumericVector lzGrad(Eigen::VectorXd theta,Eigen::MatrixXd X,Eigen::VectorXd y,Eigen::VectorXd d,
    double rate,int ngq = 20){
    const lzspd D(X,y,d,rate,ngq);
    VectorXd grad(D.nParam);
    D.df(theta,grad);
    return Rcpp::wrap(grad);
}

void lzspd::Hess(VecRef theta,Ref<MatrixXd> H) const {
	H.setZero();
	// ingredients
	const VectorXd beta(theta.head(p)), gamma(theta.tail(p+1));
	const double sigma2 = theta(p);
    const double s4 = sigma2*sigma2;
    const double s6 = s4*sigma2, s8 = s4*s4;
	const VectorXd xb(X*beta), gxy(XY*gamma), uxg(UX*gamma.head(p));
	const ArrayXd uxb(UX*beta);
    // the various 'w' matrices
	const ArrayXXd WW = calcW<4,0>(uxb,uxg,sigma2,gamma(p)); // for beta and sigma2
    const ArrayXXd wpi = calcW<3,1>(uxb,uxg,sigma2,gamma(p)); // up to y^3 (1-pi)
	const ArrayXXd wpi2 = calcW<2,2>(uxb,uxg,sigma2,gamma(p)); // piLVL2 - for gamma-gamma terms
    const ArrayXd zi(WW.col(0)-xi);
	const double lam = calcLambda(zi,nx);
    const double lam2 = lam*lam;
    
	const ArrayXd denoms(lam*zi + n), uxb2(uxb*uxb);
	const VectorXd pis = 1./(1. + (-gxy.array()).exp());
	const ArrayXd d2 = denoms*denoms, pi2 = pis.array()*(1.-pis.array()), msd(uxb2-sigma2);
	// basic parts of Hessian (normal and logistic)
	H.topLeftCorner(p,p) = -(1./sigma2)*(X.transpose()*X); // normal Hessian
	H.block(p,0,1,p) = -(1./s4)*(X.transpose()*(y-xb)).transpose(); // normal beta-sigma2
	H(p,p) = (1./s4) * (0.5*n - (y-xb).squaredNorm()/sigma2); // normal sigma2-sigma2
	H.block(p+1,p+1,p+1,p+1) = -XY.transpose()*(XY.array().colwise()*pi2).matrix(); // logistic gamma-gamma
	// interaction terms
	const ArrayXXd aX(UX.array());
	const ArrayXd uk(lam*nx.array()/denoms);  // a 'universal constant' appearing in all Hessian terms
    const ArrayXd ndk(nx.array()/denoms);
	const ArrayXd gqB(WW.col(1) - uxb*WW.col(0));  // used in beta parts
	const ArrayXd gqSig(WW.col(2) - 2.*uxb*WW.col(1) + msd*WW.col(0)); // used in sigma2 parts
	const VectorXd bb = (1./s4) * (lam*gqB*gqB/denoms - (WW.col(2)-2.*uxb*WW.col(1) + uxb2*WW.col(0))); // for beta-beta
	const VectorXd bg1 = (-1./sigma2) * (wpi.col(1) - uxb*wpi.col(0));
    const VectorXd bg2 = (lam/sigma2) * wpi.col(0)*gqB/denoms;  // for beta-gamma
	const ArrayXd gs1 = wpi.col(2) - 2*uxb*wpi.col(1) + msd*wpi.col(0), gs2 = gqSig*wpi.col(0)/denoms;  // for gamma-sigma
	const ArrayXd bs1 = WW.col(3) - 3*uxb*WW.col(2) + (3*uxb-sigma2)*WW.col(1) + uxb*msd*WW.col(0); // for beta-sigma
	VectorXd tc(m);
	for(int a=0;a<p;a++){
		// beta - beta and gammaX - gammaX
		for(int b=0;b<p;b++){
			tc = uk*aX.col(a)*aX.col(b);
			if(b<=a){
				H(a,b) += tc.dot(bb);   // Jensen's inequality
				H(p+1+a,p+1+b) += -tc.dot(wpi2.col(0).matrix()) + lam*(tc.array()*wpi.col(0)*wpi.col(0)/denoms).sum();
			}
			// beta - gammaX
			H(p+1+a,b) = tc.dot(bg1) - tc.dot(bg2);
		}
		// beta - gamma2
		H(2*p+1,a) = -(1./sigma2)*(uk*aX.col(a)*(wpi.col(2) - uxb*wpi.col(1))).sum() + (lam/sigma2)*(uk*aX.col(a)*(wpi.col(1)*gqB)/denoms).sum();
		// beta - sigma
		H(p,a) += -(0.5/s6) * (uk*aX.col(a)*bs1).sum() + (0.5*lam/s6) * (uk*aX.col(a)*gqSig*gqB/denoms).sum();
		// gammaX - sigma
		H(p+1+a,p) += (0.5*lam/s4) * (uk*gs2/denoms).sum() - (0.5/s4) * (uk*gs1).sum();
		// gammaX - gamma2
		H(2*p+1,p+a+1) += lam * (uk*aX.col(a)*wpi.col(1)*wpi.col(1)/denoms).sum() - (uk*aX.col(a)*wpi2.col(1)).sum();
	}
	// sigma - sigma
	H(p,p) += (1./s6) * (uk*gqSig).sum() + (0.5*lam2/s8) * (gqSig*gqSig/d2).sum();
	H(p,p) -= (0.25*lam/s8) * (ndk * (WW.col(4)-4*uxb*WW.col(3) + (6*uxb-2*sigma2)*WW.col(2) - 4*uxb*msd*WW.col(1) + msd*msd*WW.col(0)) ).sum();
	// sigma - gamma2
	H(2*p+1,p) = -(0.5/s4) * (uk*(wpi.col(3)-2*uxb*wpi.col(2)+msd*wpi.col(1))).sum() + (0.5*lam/s4) * (uk*gqSig*wpi.col(1)/denoms).sum();
	// gamma2 - gamma2
	H(2*p+1,2*p+1) += lam2 * (nx.array()*wpi.col(2)*wpi.col(2)/d2).sum() - (lam*ndk*wpi2.col(2)).sum();
	// symmetry
	for(int i=1;i<nParam;i++){
		for(int j=0;j<i;j++){
			H(j,i) = H(i,j);
		}
	}
}

// [[Rcpp::export]]
Rcpp::NumericMatrix lzHess(Eigen::VectorXd theta,Eigen::MatrixXd X,Eigen::VectorXd y,
                           Eigen::VectorXd d,double rate,int ngq = 20) {
    const lzspd D(X,y,d,rate,ngq);
    MatrixXd H(D.nParam,D.nParam);
    D.Hess(theta,H);
    return Rcpp::wrap(H);
}

constexpr static size_t NR_OFFSET = 1;

template<typename T,size_t offset>
T** Tmatrix(const size_t nrow,const size_t ncol){
    static_assert(std::is_fundamental<T>::value,"cannot allocate type of matrix!");
    T **m = new T*[nrow + offset]{}; // default value initialization (to zero for numeric types)
    m[offset] = new T[nrow*ncol + offset];
    for(size_t rp = offset+1;rp <= nrow; ++rp) m[rp] = m[rp-1] + ncol;
    return m;
}

constexpr auto &dmatrix = Tmatrix<double,NR_OFFSET>;

// copy constructor (the result does not share the same memory as the input!)
// include an offset for compatibility with NR
template<size_t offset>
double** Matrix_copy(MatRef M){ // we can't pass as a const ref because of operator&
	const size_t R = M.rows(), C = M.cols();
	double** M2 = new double*[R + offset]; // pointers to each row
	M2[offset] = new double[R*C + offset]; // the memory is all contiguous; each row pointer just points to start of each C-width segment
	for(size_t i=offset;i<R + offset; ++i){
		if(i > offset) M2[i] = M2[i-1] + C; // locate the next pointer @ beginning of the next row
		for(size_t j=offset;j<C+offset;j++){ // cannot use std::copy/memcpy since we're converting from col-major to row-major
			M2[i][j] = M(i,j); // no need to treat the MatrixXd as a raw array - causes more issues than it's worth
		}
	}
	return M2;
}

template<typename T,size_t offset>
void freeMat(T** M){
	delete[] M[offset]; // beginning of the contiguous memory used to store rows
	delete[] M;
}

// copy constructor for vectors. It's the caller's responsibility to free the memory
// according to http://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html#TopicUsingRefClass
template<size_t offset>
double* Vector_copy(VecRef V){
	double *res = new double[V.size() + offset];
	if(offset > 0) std::fill(res,res + offset,0.); // this memory should not be accessed, but try to prevent any mischief/explosions
	std::copy(V.data(),V.data() + V.size(),res + offset); // Eigen class members don't have begin() / end() members but data() good enough
	return res;
}

/*
// pure C implementation of Hessian matrix
void spregn_d2logL(int n, int pX, double **X, double *Y, double *D,
    double *beta, double sigma, double *gamma, double rate, double *betaX,
    double *gammaX, double *gammaZ, double **tmpgammaZ, double **Wi, int ngq,
    double *w, double *x, double *pXi, double *lambda, double **d2theta) {

    static constexpr double sqrtPi = std::sqrt(3.141592654); // defining 'PI' is bad idea since Rmath.h wants to have this as macro name!
    int i, j, k;
    const int nParam{2*pX + 4};
    double tmp1, tmp2, tmp3, sum1{0}, sum2{0}, sum3{0}, sum4{0}, sum5{0};
    const double sqrt2s = std::sqrt(2.0*sigma);

    double* tmp4 = new double[nParam + NR_OFFSET]{}; // default zero-initialization
    double** Zstar = dmatrix(n,pX+2); // Zstar = (1, X, Y)
    double** tmp5 = dmatrix(nParam,nParam);

    for (i=1; i<=n; i++) {
        Zstar[i][1] = 1;
        for (j=1; j<=pX; j++)
            Zstar[i][j+1] = X[i][j];
        Zstar[i][pX+2] = Y[i];
    }

    for (j=1; j<=nParam; j++)
        for (k=1; k<=nParam; k++)
            d2theta[j][k] = 0;

    for(i=1; i<=n; i++){
        for (j=1; j<=pX+1; j++)
            for (k=1; k<=j; k++)
                d2theta[j][k] += -Zstar[i][j]*Zstar[i][k]/sigma;
        for (k=1; k<=pX+1; k++)
            d2theta[pX+2][k] += -(Y[i]-betaX[i])*Zstar[i][k]/pow(sigma,2);
        d2theta[pX+2][pX+2] += 0.5/pow(sigma,2) - pow(Y[i]-betaX[i],2.0)/pow(sigma,3.0);

        tmp1 = 1.0/(1+exp(gammaZ[i]));
        tmp2 = tmp1*(1-tmp1);
        for (j=1; j<=pX+2; j++)
            for (k=1; k<=j; k++)
                d2theta[pX+2+j][pX+2+k] += -tmp2*Zstar[i][j]*Zstar[i][k];

        // for (j=1; j<=nParam; j++) {
        //     tmp4[j] = 0;
        //     for (k=1; k<=nParam; k++)		
        //         tmp5[j][k] = 0;
        // }
        for (k=1; k<=ngq; k++) {
            tmp1 = 1.0/(1+exp(tmpgammaZ[i][k]));
            tmp2 = tmp1*(1-tmp1)*w[k]*lambda[1]/sqrtPi;
            tmp3 = tmp2*(2*tmp1-1);
            sum1 += tmp2;
            sum2 += tmp2*x[k];
            sum3 += tmp3;
            sum4 += tmp3*x[k];
            sum5 += tmp3*x[k]*x[k];
        }
    for (j=1; j<=pX+1; j++)
        tmp4[j] = sum1*gamma[pX+2]*Zstar[i][j];
    tmp4[pX+2] = sum2*gamma[pX+2]/sqrt2s;

    for (j=1; j<=pX+1; j++)
        tmp4[pX+2+j] = sum1*Zstar[i][j];
    tmp4[nParam] = sum1*betaX[i] + sum2*sqrt2s;
        
    for (j=1; j<=pX+1; j++)
        for (k=1; k<=j; k++)
            tmp5[j][k] = sum3*pow(gamma[pX+2],2)*Zstar[i][j]*Zstar[i][k];

    for (k=1; k<=pX+1; k++)
        tmp5[pX+2][k] = sum4*pow(gamma[pX+2],2)*Zstar[i][k]/sqrt2s;
    
    for (j=1; j<=pX+1; j++)
        for (k=1; k<=pX+1; k++)
            tmp5[pX+2+j][k] = sum3*gamma[pX+2]*Zstar[i][j]*Zstar[i][k];
    
    for (k=1; k<=pX+1; k++)
        tmp5[nParam][k] = (sum3*gamma[pX+2]*betaX[i] + sum4*gamma[pX+2]*sqrt2s)*Zstar[i][k];
    tmp5[pX+2][pX+2] = sum5*pow(gamma[pX+2],2)/sqrt2s;
    
    for (k=1; k<=pX+1; k++)
        tmp5[pX+2+k][pX+2] = sum4*gamma[pX+2]*Zstar[i][k]/sqrt2s;
    tmp5[nParam][pX+2] = (gamma[pX+2]/sqrt2s) * (sum4*betaX[i] + sum5*sqrt2s);
    
    for (j=1; j<=pX+1; j++)
        for (k=1; k<=j; k++)
            tmp5[pX+2+j][pX+2+k] = sum3*Zstar[i][j]*Zstar[i][k];
    
    for (k=1; k<=pX+1; k++)
        tmp5[nParam][pX+2+k] = (sum3*betaX[i]+sum4*sqrt2s)*Zstar[i][k];
    tmp5[nParam][nParam] = sum3*pow(betaX[i],2) + sum4*2*betaX[i]*sqrt2s + sum5*2.*sigma;

    for (k=1; k<=pX+1; k++)
        tmp5[nParam][k] += sum1*Zstar[i][k];
    tmp5[pX+2][pX+2] += -sum2*gamma[pX+2]*pow(2*sigma,-1.5);
    tmp5[nParam][pX+2] += sum2/sqrt2s;

    for (j=1; j <= nParam; j++)
        for (k=1; k<=j; k++)
            d2theta[j][k] += pXi[i]*pXi[i]*tmp4[j]*tmp4[k] - pXi[i]*tmp5[j][k];
    }
    for (j=1; j <= nParam; j++){
        for (k=1; k<=j; k++){
            d2theta[j][k] /= static_cast<double>(n);
            d2theta[k][j] = d2theta[j][k];
        }
    }
    
    delete[] tmp4;
    freeMat<double,NR_OFFSET>(Zstar);
    freeMat<double,NR_OFFSET>(tmp5);
}

*/

	// Given an input spd object D, we can set up all the pointers and call spreg functions:
	/*
	  To call the functions in spreg_d2logL, we need the following elements:
	  int n               sample size
	  int pX              X.cols() (p-1 relative to spd)
	  double **X          X, but WITHOUT an intercept column!
	  double *Y           y
	  double *D           d
      double *beta        beta
	  double sigma        sigma2
	  double *gamma       gamma
	  double rate         xi
	  double *betaX       X*beta
      double *gammaX      X*gammaX
	  double *gammaZ      cbind(1,X,Y)*gamma
	  double **tmpgammaZ  tmpgammaZ[i][j] = gammaX[i] + gamma2*(betaX[i]+sqrt(2*sigma2)*x[j])   // same as eArg in calcW above
	  double **Wi	      # this is not really a double ** - only column 1 is used
	  int ngq			  # of Gaussian quadrature points (length of w and x below)
      double *w			  Gaussian quadrature weights
	  double *x			  Gaussian quadrature locations
	  double *pXi		  1/denoms (X_i point masses) NOTE: these are not unique!
	  double *lambda      a pointer to length-2 array, but only lambda[1] is used
	  double **d2theta    just for spregn_d2logL, the returned Hessian matrix
	*/

/*
MatrixXd spd_Hess(const lzspd& D,VecRef theta){
	VectorXd xb(D.X*theta.head(D.p)); // xBeta
	VectorXd xg(D.X*theta.segment(D.p+1,D.p));  // xGamma
	double gamma2 = theta(2*D.p+1), sig2 = theta(D.p);
	MatrixXd tmpGZ = gamma2*xb*(std::sqrt(2.*sig2)*D.x).transpose(); // n x ngq
    tmpGZ.colwise() += xg;
	ArrayXd W0 = D.calcW<0,0>(xb,xg,sig2,gamma2).col(0); // note that here we use xb, xg NOT uxb, uxg
	const double lam = calcLambda(W0-D.xi,VectorXd::Constant(D.n,1.0)); // also note here inputs are length-n with all multiplicites==1
	VectorXd piX( 1./(lam*(W0.array() - D.xi) + D.n) );
	const int N = D.n, P = D.p-1;
	double** t_X = Matrix_copy<NR_OFFSET>(D.X.rightCols(P));
	VectorXd myY = D.y;
	double* t_Y = Vector_copy<NR_OFFSET>(myY);
	double* t_d = Vector_copy<NR_OFFSET>(D.d);
	double* t_beta = Vector_copy<NR_OFFSET>(theta.head(P+1));
	double* t_gamma = Vector_copy<NR_OFFSET>(theta.tail(P+2));
	double* t_xb = Vector_copy<NR_OFFSET>(xb); // CHECK
	double* t_xg = Vector_copy<NR_OFFSET>(xg); // CHECK
	double* t_zg = Vector_copy<NR_OFFSET>(D.XY*theta.tail(P+2));
	double** t_tmpGZ = Matrix_copy<NR_OFFSET>(tmpGZ);
	double** t_Wi = Matrix_copy<NR_OFFSET>(W0);
	double* t_w = Vector_copy<NR_OFFSET>(D.w);
	double* t_x = Vector_copy<NR_OFFSET>(D.x);
	double* t_Pi = Vector_copy<NR_OFFSET>(piX);
	std::array<double,2> dumbLam = {0., lam};
    double *lamPtr = &dumbLam[0];
	MatrixXd dumbTmp(MatrixXd::Zero(D.nParam,D.nParam));
	double** d2theta = Matrix_copy<NR_OFFSET>(dumbTmp);
	using rdMap = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>;  // row-major, double Map
    MatrixXd res;
    // call the monstrosity
    if(false){
        spregn_d2logL(N, P, t_X, t_Y, t_d,t_beta, sig2, t_gamma, D.xi, t_xb, t_xg,
		    t_zg, t_tmpGZ, t_Wi, D.ngq, t_w, t_x, t_Pi, lamPtr, d2theta);
            // convert d2theta to an Eigen::Map
	    double* h0 = &d2theta[1][1];
        rdMap H(h0,D.nParam,D.nParam);  // in this case, alignment doesn't matter since the matrix is square and symmetric
        res = H;
    } else {
        Rcpp::Rcout << "would have called spregn_d2logL\n";
        res = MatrixXd::Identity(D.nParam,D.nParam);
    }
	// free them
	freeMat<double,NR_OFFSET>(t_X);  freeMat<double,NR_OFFSET>(t_tmpGZ);
	freeMat<double,NR_OFFSET>(t_Wi); freeMat<double,NR_OFFSET>(d2theta);
	delete[] t_Y;
	delete[] t_d;
	delete[] t_beta;
	delete[] t_gamma;
	delete[] t_xb;
	delete[] t_xg;
	delete[] t_zg;
	delete[] t_w;
	delete[] t_x;
	delete[] t_Pi;
	return res;
}

*/

// [[Rcpp::export]]
Rcpp::List fit_lzspd(Eigen::MatrixXd X,Eigen::VectorXd y,Eigen::VectorXd d,double rate,Eigen::VectorXd theta,
                     int ngq = 20,double TOL = 0.,int MAXIT = 100,int verb = 0,
                     const std::string method = "Brent",bool justBeta = false,const std::string conv = "func") {
    using Rcpp::Named;
    const lzspd D(X,y,d,rate,ngq);
    if(theta.size() != D.nParam || theta(D.p) <= 0.){ // create initial estimate if it wasn't provided/seems funky
        theta = VectorXd::Zero(D.nParam);
        VectorXd beta0 = betaFit(D.y,D.X,true);
        theta.head(D.p) = beta0;
        theta(D.p) = (D.y - D.X*beta0).squaredNorm()/D.n; // residual variance MLE
        theta.tail(D.p+1) = logisticFit(D.d,D.XY,0.,1e-4);
        if(verb > 0) Rcpp::Rcout << "Using initial parameters: " << theta.transpose() << std::endl;
    }
    // Boilerplate more or less identical to MDRM and SPD fitting - interface to common dfpmin routines
    if(TOL < 1.0e-10) TOL = 1.0e-8*sqrt(D.nParam);
    funcMin opt_param;
    try { // dfpmin should no longer throw exceptions, however we should expect the expected which is an unexpected failure
        opt_param = dfpmin(theta,D,VectorXi::Constant(1,-1),method,verb,TOL,TOL,conv);
    } catch(std::exception& _ex_){
        //forward_exception_to_r(_ex_); // NOTE this does not unwind destructors safely, see https://github.com/RcppCore/Rcpp/issues/753
        Rcpp::stop(_ex_.what()); // "What, I say what has gone wrong?" - Foghorn Leghorn
    } catch(std::string errMsg){
        Rcpp::stop(errMsg.c_str());
    } catch(const char* errMsg){
        Rcpp::stop(errMsg);
    } catch(...){
        ::Rf_error("c++ exception (unknown reason!)"); // what's the difference between Rf_error and Rcpp::stop in this context??
    }
    const VectorXd mle = opt_param.arg_min;
    if(opt_param.error != dfpmin_error::NONE){
        Rcpp::CharacterVector errmsg(1);
        errmsg[0] = dfpmin_err_msg::messages[opt_param.error-1];
        return Rcpp::List::create(Named("error") = errmsg,Named("parameters") = Rcpp::wrap(theta));
    }
    // end boilerplate
    VectorXd betaHat = mle.head(D.p), gammaHat = mle.tail(D.p+1);
    double sigmaHat = mle(D.p);
    VectorXd out_grad(D.nParam);
    MatrixXd out_Hess(D.nParam,D.nParam);
    D.df(mle,out_grad);
    D.Hess(mle,out_Hess);
    // other Hessian
    MatrixXd H2 = D.opg(mle);
    return Rcpp::List::create(
        Named("theta") = mle, // for easy reference after fitting
        Named("beta") = betaHat,
        Named("gamma") = gammaHat,
        Named("sigma2") = sigmaHat,
        Named("lambda") = D.lambda(mle),
        Named("pX") = D.pX(mle),
        Named("nX") = D.nx,
        Named("gradient") = out_grad,
        Named("Hessian") = out_Hess,
        Named("Hess2") = H2
    );
}

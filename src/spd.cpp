/*
  Secondary phenotypes fitting semi-parametric algorithm
*/

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <iostream>
#include <cmath>
#include "util.h"
#include "logistic.h"
#include "linemins.h"
#include <chrono> //testing
#include <iostream>


using namespace Eigen;

using Rcpp::Named;
using Rcpp::Rcout;
using std::endl;

using MapVxd = Eigen::Map<Eigen::VectorXd>;
using MapMxd = Eigen::Map<Eigen::MatrixXd>;
using MapVxi = Eigen::Map<Eigen::VectorXi>;
using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using ArrRef = const Eigen::Ref<const Eigen::ArrayXd>&;


// returns a mat1.rows() x mat2.rows() matrix of logistic probabilities
// where element [i,j] is 1/(1+exp(-g0-g1*M1[i,]-g2*M2[j,]))
ArrayXXd calcPi(MatRef mat1,MatRef mat2,VecRef gamma){
	if(mat1.cols() + mat2.cols() + 1 != gamma.size()){
		Rcpp::stop("dimension error calculating pis!");
	}
	VectorXd xg = mat1*gamma.segment(1,mat1.cols()), yg = mat2*gamma.tail(mat2.cols());
	MatrixXd IS = MatrixXd::Constant(xg.size(),yg.size(),-gamma(0)); // gamma0 is constant across all entries
	IS.colwise() -= xg;
	IS.rowwise() -= yg.transpose();
	return 1.0/(1. + IS.array().exp()); // outer product: mX x mY array
}

// fulfilling the empirical likelihood constraint:
inline double flam(double lam,ArrRef z,ArrRef nz,double n){
	return (nz/(lam*z+n)).sum() - 1.;
}

/*
 * Calculate lambda to satisfy the known population-level prevalence @ given parameters
 * ksi has already been subtracted from the inputs z, so we need at least one > 0 and one < 0
 * to have a non-trivial solution. The 'nz' values should be integer (counts of unique X rows)
 * but may be arbitrary real-valued weights
*/
double calcLambda(ArrRef z,ArrRef nz){
	const double N = nz.sum(), eps = 1.e-5;
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
		if(std::abs(fL) < 1.e-5) break;
		mid = 0.5*(aLeft + aRight);
		fL = flam(mid,z,nz,N);
		if(fL*flam(aLeft,z,nz,N) > 0) aLeft = mid;
		else aRight = mid; 
	}
	return mid;
}

// [[Rcpp::export]]
double spd_lambda(Rcpp::NumericVector z0,Rcpp::NumericVector nz0){
	const Eigen::Map<ArrayXd> z(Rcpp::as<Map<ArrayXd>>(z0)), nz(Rcpp::as<Map<ArrayXd>>(nz0));
	if(z.size() != nz.size()) Rcpp::stop("spd_lambda: unequal sizes!");
	return calcLambda(z,nz);
}

/* class for secondary phenotypes model with likelihood and gradient methods */
class spd{
public:
	const MatrixXd X,Y;
	const VectorXd D;
	const double ksi;
	const int verbosity,N,P,Q;
	MatrixXd UX,UY,XY;
	double lambda;
	int n0,n1,mX,mY,errCode,nParam;
	VectorXd nx,ny;
	bool uniqX,zeroLambda;
	spd(MatRef xx,MatRef yy,VecRef dd,double kk,int verb=0) : 
		X(xx),Y(yy),D(dd),ksi(kk),verbosity(verb), N(X.rows()), P(X.cols()), Q(Y.cols()) {
		n0 = (D.array() == 0).count();
		n1 = N-n0;
		if(N != Y.rows() || N != D.size() || n1 != (D.array()==1).count() || n1 == N || ksi <= 0. || ksi >= 1.){
			Rcpp::stop("Bad inputs to data set!");
		}
		VectorXi nX(N), nY(N);
		UX = uniq(X,nX);
		UY = uniq(Y,nY);
		mX = nX.size();
		mY = nY.size();
		nx = nX.cast<double>();
		ny = nY.cast<double>();
		uniqX = (mX < N) ? true : false;
		XY = MatrixXd(N,P+Q+1);
		XY.col(0) = VectorXd::Constant(N,1);
		XY.block(0,1,N,P) = X;
		XY.rightCols(Q) = Y;
		nParam = mY + P*Q + P + Q; // jump sizes, beta, gamma respectively
	}
	double operator()(VecRef) const; // log-likelihood
	double llv2(VecRef) const; // new and improved? version of log-likelihood which does not explode
	void df(VecRef,Eigen::Ref<Eigen::VectorXd>) const; // gradient
	double lambdaF(VecRef) const;
	MatrixXd yhats(VecRef th) const;
	VectorXd unwrap(VecRef x) const { // extracting the "right" alpha vector from the dense storage with other parameters
		VectorXd res(mY);
		res.head(mY-1) = x.head(mY-1);
		res(mY-1) = 0.;
		return res;
	}
	void Hess(VecRef,Ref<MatrixXd>) const; // Hessian
};

static constexpr double max_double{std::numeric_limits<double>::max()}; // auto -> std::initializer list?
static constexpr auto max_base = std::log(max_double); // largest number that can be exponentiated safely (~709.78)
static constexpr auto max_use = 0.5*max_base;

inline double logSumExp(ArrRef z){
	const double xM = z.maxCoeff();
	return (xM > max_base) ? xM + std::log((z - xM).exp().sum()) : std::log(z.exp().sum());
}

// version of 'LogSumExp' specific to the calculations done with DRM term
VectorXd lse(MatRef Y,MatRef mu,VecRef alpha){
	const int N(mu.rows()), M(Y.rows());
	VectorXd res(N);
	ArrayXd tmp(M);
	for(int i=0;i<N;i++){
		tmp = Y*mu.row(i).transpose() + alpha;
		res(i) = logSumExp(tmp);
	}
	return res;
}

// [[Rcpp::export]]
Rcpp::NumericVector lsexp(MatrixXd Y,MatrixXd mu,VectorXd alp){
	auto res = lse(Y,mu,alp);
	return Rcpp::wrap(res);
}

// these are divided by d2 values, which by definition are larger
// so if this blows up, the bottom will blow up as well, and it's save to just make this a large, positive number
VectorXd piNum(MatRef Y,MatRef mu,VecRef alpha,VecRef yg,VecRef xg,const double g0){
	const size_t N = mu.rows(), M = Y.rows();
	const double max_add = 0.8*max_double/static_cast<double>(M);
	VectorXd res(N);
	double gt,E2v,iv,rsum;
	for(size_t i=0;i<N;i++){
		rsum = 0;
		for(size_t j=0;j<M;j++){
			gt = g0 + xg(i) + yg(j);
			E2v = mu.row(i).dot(Y.row(j)) + alpha(j);
			iv = std::exp(E2v + gt)/(1.0 + std::exp(gt));
			rsum += (std::isnan(iv) || std::isinf(iv)) ? max_add : iv;
		}
		res(i) = rsum;
	}
	return res;
}

double spd::operator()(VecRef theta) const { // *negative* log-likelihood
	// extracting parameters
	if(theta.size() != nParam){
		Rcpp::Rcout << "nParam = " << nParam << "; " << "theta.size() = " << theta.size() << '\n';
		Rcpp::stop("loglik: incorrectly sized parameter!");
	}
	const MatrixXd beta( unVectorize(theta.segment(mY - 1,P*Q),Q) );
	const VectorXd alpha( unwrap(theta) );
	const VectorXd gamma( theta.tail(P+Q+1) );
	// preliminary calculations
	const MatrixXd XB(X*beta); // N x Q
	const MatrixXd UXB = uniqX ? UX*beta : XB; // mX x Q.
	const ArrayXd gxy(XY*gamma);
	const ArrayXXd E1( ((XB*UY.transpose()).rowwise() + alpha.transpose()).array().exp() ); // N x M
	const double d1 = E1.rowwise().sum().log().sum();
	const ArrayXXd E2( uniqX ? ((UXB*UY.transpose()).rowwise() + alpha.transpose()).array().exp() : E1 ); // R x M
	const ArrayXXd pis( calcPi(UX,UY,gamma) ); // R x M - could be folded into piNum?
	const ArrayXd d2( E2.rowwise().sum() );  // no. of elements is R (indexes with k)
	const ArrayXd piNum( (E2*pis).rowwise().sum() );  // same no. elements as denoms
	const ArrayXd zz( piNum/d2 - ksi );  // pass in these elements DIFFERENCED with ksi
	// calculate the answer
	double drmLL = ny.dot(alpha) + (Y.cwiseProduct(XB)).sum() - d1;
	double res = drmLL; // apparently declaring lam within a try{} block causes scope issues (it's too local)
	double logis = D.dot(gxy.matrix()) - (1.+gxy.exp()).log().sum();
	const double lam = calcLambda(zz,nx);
	double itt = nx.dot((lam*zz + (double)N).log().matrix());
	res += logis - itt;
	if(verbosity > 4){  // usually TMI
		Rcout << "drm term:  " << drmLL << endl;
		Rcout << "logistic term:  " << logis << endl;
		Rcout << "interaction term:  " << itt << endl;
		Rcout << "lambda and f(lambda):  " << lam << ", " << flam(lam,zz,nx,N) << endl;
	}
	if(std::isnan(res) || std::isinf(res)){ // isnan from cmath for double
		//Rcpp::Rcout << "warning: res = " << res << "; returning 0\n";
		return 0.; // smaller than any negative log-likelihood, so this should ensure the algorithm keeps searching
	}
	return -res/N;
}

double spd::llv2(VecRef theta) const {
	// setup:
	if(theta.size() != nParam){
		Rcpp::Rcout << "nParam = " << nParam << "; " << "theta.size() = " << theta.size() << '\n';
		Rcpp::stop("loglik: incorrectly sized parameter!");
	}
	const MatrixXd beta( unVectorize(theta.segment(mY - 1,P*Q),Q) );
	const VectorXd alpha( unwrap(theta) );
	const VectorXd gamma( theta.tail(P+Q+1) );
	// preliminary calculations:
	const MatrixXd XB(X*beta); // N x Q
	const MatrixXd UXB = uniqX ? UX*beta : XB; // mX x Q.
	const ArrayXd gxy(XY*gamma);
	const VectorXd d1(lse(UY,XB,alpha)); // safely calculate DRM denominator terms
	const ArrayXd d2 = uniqX ? lse(UY,UXB,alpha) : d1;
	const VectorXd Xg(UX*gamma.segment(1,P)), Yg(UY*gamma.tail(Q));
	const ArrayXd piNums = piNum(UY,UXB,alpha,Yg,Xg,gamma(0));
	const ArrayXd zz( piNums/d2.exp() - ksi );  // pass in these elements DIFFERENCED with ksi
	// finally, calculate the answer
	double drmLL = ny.dot(alpha) + (Y.cwiseProduct(XB)).sum() - d1.sum();
	double res = drmLL;
	double logis = D.dot(gxy.matrix()) - (1.+gxy.exp()).log().sum();
	const double lam = calcLambda(zz,nx);
	double itt = nx.dot((lam*zz + static_cast<double>(N)).log().matrix());
	res += logis - itt;
	if(verbosity > 4){  // setting verbosity to 5+ gives this info, which is usually way too much
		Rcout << "drm term:  " << drmLL << endl;
		Rcout << "logistic term:  " << logis << endl;
		Rcout << "interaction term:  " << itt << endl;
		Rcout << "lambda and f(lambda):  " << lam << ", " << flam(lam,zz,nx,N) << endl;
	}
	return -res/N;
}

// [[Rcpp::export]]
double spdLL(VectorXd theta,VectorXd dd,MatrixXd Y,MatrixXd X,double ksi,int verb = 0){
	const spd D(X,Y,dd,ksi,verb);
	return D(theta);
}

// [[Rcpp::export]]
double spdLL2(VectorXd theta,VectorXd dd,MatrixXd Y,MatrixXd X,double ksi,int verb = 0){
	const spd D(X,Y,dd,ksi,verb);
	return D.llv2(theta);
}
	
void spd::df(VecRef theta,Eigen::Ref<Eigen::VectorXd> g) const {  // gradient
	// setup:
	if(g.size() != nParam){Rcpp::stop("Wrong size input gradient.");}
	g.setZero();
	const int gLen = P + Q + 1;
	int a,b;
	const MatrixXd beta( unVectorize(theta.segment(mY - 1,P*Q),Q) );
	const VectorXd alpha( unwrap(theta) );
	const VectorXd gamma( theta.tail(gLen) );
	const MatrixXd XB(X*beta); // N x Q
	const MatrixXd UXB = uniqX ? UX*beta : XB; //  mX x Q.
	// preliminary calculations:
	const VectorXd gxy( XY*gamma );
	const ArrayXXd pis( calcPi(UX,UY,gamma) ); // R x M
	const ArrayXXd E1( ((XB*UY.transpose()).rowwise() + alpha.transpose()).array().exp() ); // N x M
	const ArrayXXd E2( uniqX ? ((UXB*UY.transpose()).rowwise()+alpha.transpose()).array().exp() : E1 ); // R x M
	const ArrayXXd epi( E2*pis );  // M x R
	const ArrayXXd epi2( epi*(1.0-pis) ); // used in dgamma, also M x R
	const ArrayXd piNum( epi.rowwise().sum() );  // length R
	const ArrayXd piNum2( epi2.rowwise().sum() ); // \sum_j E_jk*pi_jk*(1-pi_jk)
	const ArrayXd d1( E1.rowwise().sum() ); // used in alpha, beta (DRM terms) (length N)
	const ArrayXd d2( uniqX ? E2.rowwise().sum() : d1 ); // used in all terms, length R
	const ArrayXd zz( piNum/d2 - ksi );
	const ArrayXXd UE1( E1.matrix()*UY );
	// fill in the 'easy' part first since we abort the rest when lambda = 0
	// alpha
	VectorXd ds = (E1.colwise()/d1).colwise().sum();// DRM d_L/d_alpha
	g.head(mY-1) = ny.head(mY-1) - ds.head(mY-1);
	int offset = mY - 1;
	for(int z = 0; z < P*Q; z++){
		a = z%P; b = z/P; // a -> rows, b -> columns
		g(offset + z) = X.col(a).dot(Y.col(b)-(UE1.col(b)/d1).matrix());  // DRM term
	}
	offset += P*Q;
	VectorXd pii = 1./(1.+(-gxy.array()).exp()); // logistic pis
	g.tail(gLen) = XY.transpose()*(D-pii); // logistic derivative
	double lam = calcLambda(zz,nx);
	if(lam==0.) return;
	const ArrayXd d3( piNum + ((double)N/lam - ksi)*d2 );
	const ArrayXd d4( d2*d3 ); // used in dbeta, dalpha
	const ArrayXXd UE( uniqX? E2.matrix()*UY : UE1);
	const ArrayXXd UEP( epi.matrix()*UY );
	ArrayXd tmpK(mX);
	// alpha
	for(int b=0; b<mY-1; b++){
		tmpK = pis.col(b)*d2-piNum;
		g(b) -= (nx.array()*E2.col(b)*tmpK/d4).sum();
	}
	offset = mY - 1;
	// beta
	const ArrayXXd xc = UX.array().colwise()*nx.array(); // pre-calculate nx * UX.col(a)
	for(int z = 0; z < P*Q; z++){
		int a = z%P, b = z/P; // a -> rows, b -> columns
		g(offset + z) -= (xc.col(a)*(d2*UEP.col(b)-piNum*UE.col(b))/d4).sum(); // interaction term
	}
	offset += P*Q;
	//Rcpp::Rcout << "Calculated gradient of beta: " << g.segment(mY-1,P*Q).transpose() << endl;
	//gamma
	const ArrayXd dgX( piNum2/d3 ); // this vector is only j-dependent, thus it is constant for gamma_X and intercept
	const ArrayXXd UEPP = epi2.matrix()*UY;
	for(int a=0;a<gLen;a++){
		if(a==0){ // the gamma intercept
			g(offset + a) -= nx.dot(dgX.matrix());
		}
		else if(a <= P){  // derivative for X
			g(offset + a) -= (xc.col(a-1)*dgX).sum();
		}
		else{ // derivative for Y
			// tmpK = (epi2.rowwise()*UY.col(a-P-1).transpose().array()).rowwise().sum(); // sum_j u*E*pi*(1-pi)
			g(offset + a) -= (nx.array()*UEPP.col(a - P - 1)/d3).sum();
		}
	}
	//Rcpp::Rcout << "Calculated gradient of gamma: " << g.tail(gLen).transpose() << endl;
	if(is_nan(g) || is_inf(g)){ // is_nan for Eigen class members
		g.setZero(); // triggers false convergence 
		return;
	}
	g *= -1./N;
	return;
}

// [[Rcpp::export]]
Rcpp::NumericVector spdGrad(VectorXd dd,MatrixXd Y,MatrixXd X,double ksi,VectorXd theta,int verb = 0){
	spd D(X,Y,dd,ksi,verb);
	VectorXd gr(D.nParam);
	D.df(theta,gr);
	return Rcpp::wrap(gr);
}

double spd::lambdaF(VecRef theta) const {
	const MatrixXd beta( unVectorize(theta.segment(mY - 1,P*Q),Q) );
	const VectorXd alpha( unwrap(theta) );
	const VectorXd gamma( theta.tail(P+Q+1) );
	const MatrixXd UXB( UX*beta ); // N x Q and mX x Q.
	const ArrayXXd E2(  ((UXB*UY.transpose()).rowwise()+alpha.transpose()).array().exp() ); // R x M
	const ArrayXXd pis( calcPi(UX,UY,gamma) ); // R x M - could be folded into piNum?
	const ArrayXd denoms( E2.rowwise().sum() );  // no. of elements is R (indexes with k)
	const ArrayXd piNum( (E2*pis).rowwise().sum() );  // same no. elements as denoms
	const ArrayXd zz( piNum/denoms - ksi );  // pass in these elements DIFFERENCED with ksi
	return calcLambda(zz,nx);
}

// to estimate beta, we can still use residuals as in DRM model. Function to calculate them:
MatrixXd spd::yhats(VecRef th) const {
	if(th.size() != nParam) Rcpp::stop("Bad input to yhats.");
	const MatrixXd beta( unVectorize(th.segment(mY - 1,P*Q),Q) );
	const VectorXd alpha( unwrap(th) );
	MatrixXd XB = X*beta;
	const MatrixXd E( ((UY*XB.transpose()).colwise() + alpha).array().exp() );  // m x n array
	const ArrayXd d( E.colwise().sum() ); // n denominators, summed over j=1,...,m
	return (E.transpose()*UY).array().colwise()/d;
}

// behold this horrifying monstrosity from the depths of hell
void spd::Hess(VecRef th,Eigen::Ref<Eigen::MatrixXd> H) const {
	H.setZero();
	const int gLen = P + Q + 1, pq = P*Q;
	const MatrixXd beta( unVectorize(th.segment(mY - 1,P*Q),Q) );
	const VectorXd alpha( unwrap(th) );
	const VectorXd gamma( th.tail(P+Q+1) );
	const MatrixXd XB( X*beta );
	const MatrixXd UXB = uniqX ? UX*beta : XB; // N x Q and mX x Q.
	// preliminary calculations:
	const ArrayXd gxy( XY*gamma );
	const ArrayXXd pis( calcPi(UX,UY,gamma) ); // R x M
	const ArrayXXd E1( ((XB*UY.transpose()).rowwise() + alpha.transpose()).array().exp() ); // N x mY
	const ArrayXXd E2( uniqX ? ((UXB*UY.transpose()).rowwise()+alpha.transpose()).array().exp() : E1 ); // mX x mY
	const ArrayXXd epi( E2*pis );  // mX x mY
	const ArrayXXd epi2( epi*(1.0 - pis) ); //  mX x mY
	const ArrayXXd epi3( epi2*(1.0 - 2.0*pis) ); // we need to rowwise() multiply this several times in d2Gamma
	const ArrayXd piNum( epi.rowwise().sum() );  // length mX
	const ArrayXd piNum2( epi2.rowwise().sum() ); // length mX
	const ArrayXd piNum3( epi3.rowwise().sum() ); // used in d/gamma d/gamma terms (related to the skewness of bernoulli variables)
	const ArrayXd d1( E1.rowwise().sum() ); // used in alpha, beta (DRM terms)
	const ArrayXd d11( d1*d1 );
	const ArrayXd d2( uniqX ? E2.rowwise().sum() : d1 ); // used in all terms
	const ArrayXd d22( d2*d2 );
	const ArrayXd zz( piNum/d2 - ksi );
	double lam = calcLambda(zz,nx); // we could skip this since lambda is set by logLik
	const double dif = (double)N/lam - ksi;
	const ArrayXd d3( piNum + dif*d2 );
	const ArrayXd d33( d3*d3 );
	const ArrayXd d4( d2*d3 );
	const ArrayXd d44( d4*d4 );
	const ArrayXXd XA(X), UAY(UY), UAX(UX); // for convenience so it's not necessary to write .array() everywhere
	const ArrayXXd xc = UAX.colwise()*nx.array();  // X counts and columns always together
	const ArrayXXd UE1(E1.matrix()*UY), UEP( epi.matrix()*UY ), UEPP(epi2.matrix()*UY);
	const ArrayXXd UE( uniqX? E2.matrix()*UY : UE1);
	int rowOffset, colOffset,a,b,c,d;
	ArrayXd tmpJ(mY), tmpK(mX), tmpK2(mX), tmpN(N);
	ArrayXd scl(mX), udif1(mX), udif2(mX), udif3(mX), udif4(mX);
	// top corner: d/alpha d/alpha:
	tmpK = piNum*(d3+dif*d2);
	for(int a=0; a < mY-1; a++){
		for(int b=0; b <= a; b++){
			H(a,b) += (E1.col(a)*E1.col(b)/d11).sum(); // DRM term
			tmpK2 = E2.col(a)*E2.col(b);
			scl = nx.array()*tmpK2/d44;
			udif1 = d22*(pis.col(a)*pis.col(b)+dif*(pis.col(a)+pis.col(b))) - tmpK;
			H(a,b) += (scl*udif1).sum();
			if(a==b){
				H(a,b) -= (E1.col(a)/d1).sum(); // DRM term
				udif2 = (piNum - pis.col(a)*d2);
				H(a,b) += (nx.array()*E2.col(a)*udif2/d4).sum(); // interaction term
			} 
		}
	}
	// d/alpha d/beta
	rowOffset = mY - 1, colOffset = 0;
	for(int bb = 0; bb < pq; bb++){ // beta index
		a = bb%P, b = bb/P;  // a is the row and b is the column of beta
		udif3 = d2*UEP.col(b) + UE.col(b)*(dif*d2 + d3); // doesn't depend on alpha
		for(int z = 0; z < mY - 1; z++){ // alpha index
			tmpN = XA.col(a)*E1.col(z)*(UE1.col(b) - UAY(z,b)*d1)/d11;
			double drmba = tmpN.sum();
			H(rowOffset + bb,z) += drmba;
			tmpK = xc.col(a)*E2.col(z)/d4;
			tmpK2 = tmpK/d4;
			// calculating from d/dalpha(d/dbeta) (as opposed to the reverse above) Obviously, these should match!
			const double uzb = UAY(z,b);
			udif1 = d22*pis.col(z)*(UEP.col(b) - uzb*piNum - dif*d2*uzb);
			udif2 =  dif*d22*(pis.col(z)*UE.col(b) + uzb*piNum + UEP.col(b));
			udif3 = piNum*piNum*(uzb*d2 - UE.col(b));
			udif4 = -2.*dif*d2*UE.col(b)*piNum;
			double ab2 = (tmpK2*(udif1+udif2+udif3+udif4)).sum();
			H(rowOffset + bb,z) += ab2;
		}
	}
	// d/alpha d/gamma
	rowOffset = mY + pq - 1;
	for(int b=0; b<mY-1; b++){
		scl = nx.array()*E2.col(b)/d33; // scaling for all cases: c_k/d3_k^2
		udif1 = pis.col(b)*(1.0 - pis.col(b));
		udif2 = pis.col(b) + dif;
		tmpK = nx.array()*E2.col(b)/d33;
		for(int a=0; a<gLen; a++){
			if(a <= P){
				udif3 = udif2*piNum2 - d3*udif1;
				if(a > 0){ // X - columns
					H(rowOffset + a,b) = (tmpK*UAX.col(a-1)*udif3).sum();
				}	
				else{ // the intercept
					H(rowOffset + a,b) = (tmpK*udif3).sum();
				}
			}
			else{ // columns of Y
				int ycol = a - P - 1;
				udif3 = udif2*UEPP.col(ycol) - d3*UAY(b,ycol)*udif1;
				H(rowOffset + a,b) = (tmpK*udif3).sum();
			}
		}
	}
	// d/beta d/beta
	rowOffset = mY - 1; colOffset = mY - 1;
	ArrayXd uuE(mX), uuEp(mX);  // containers for \sum_{j=1}^{m} u_{jb}u_{jd}E_{jk} [\pi_{jk}]
	for(int y=0; y<pq; y++){
		a = y%P, b = y/P; // row/col indices of first beta element
		for(int z=0; z<=y; z++){
			c = z%P, d = z/P; // row/col indices of second beta
			tmpJ = UAY.col(b)*UAY.col(d);
			uuE = (E2.rowwise()*tmpJ.transpose()).rowwise().sum();
			uuEp = (epi.rowwise()*tmpJ.transpose()).rowwise().sum();
			const ArrayXd uuE1 = (E1.rowwise()*tmpJ.transpose()).rowwise().sum(); // a drm term
			double drmT = ((XA.col(a)*XA.col(c)/d11)*(UE1.col(b)*UE1.col(d) - d1*uuE1)).sum(); // DRM term
			tmpK2 = nx.array()*UAX.col(a)*UAX.col(c);  // scaling term
			tmpK = tmpK2/d33;
			udif1 = (dif*(UE.col(b)*UEP.col(d) + UE.col(d)*UEP.col(b) + piNum*uuE) + UEP.col(b)*UEP.col(d) - d3*uuEp);
			double t1 = (tmpK*udif1).sum();
			tmpK = tmpK2*piNum/(d4*d3);
			udif2 = piNum*uuE - UE.col(b)*UE.col(d)*2.*dif;
			double t2 = (tmpK*udif2).sum();
			tmpK = tmpK2*piNum/d4; // for this last term, there's no distribution so where things go isn't too important. Avoid big cancellation?
			udif3 = -piNum*UE.col(b)*UE.col(d)/d4;
			double t3 = (tmpK*udif3).sum();
			// Rcout << "Total beta at entry [" << rowOffset + y << ", " << colOffset + z << "]: " << drmT << " + " << t1 << " + " << t2 << " + " << t3 <<  " = " << drmT + t1 + t2 + t3 << endl;
			H(rowOffset+y,colOffset+z) = drmT + t1 + t2 + t3;
		}
	}
	// d/beta d/gamma
	rowOffset = mY + pq - 1; colOffset = mY - 1;
	for(int bb = 0; bb < pq; bb++){
		a = bb%P, b = bb/P;  // row/col index of beta
		for(int z=0;z<gLen;z++){
			if(z <= P){ // columns of X
				tmpK = xc.col(a)/d33;
				udif1 = piNum2*(UEP.col(b) + dif*UE.col(b)) - d3*UEPP.col(b);
				if(z > 0){
					tmpK = nx.array()*UAX.col(a)*UAX.col(z-1)/d33;
				}
				H(rowOffset + z,colOffset + bb) = (tmpK*udif1).sum();
			}
			else{
				tmpK = xc.col(a)/d33;
				int ycol = z - P - 1;
				tmpJ = UAY.col(b)*UAY.col(ycol);
				udif1 = (epi2.rowwise()*tmpJ.transpose()).rowwise().sum();
				udif2 =  (UEP.col(b)+dif*UE.col(b))*UEPP.col(ycol) - udif1*d3;
				H(rowOffset + z,colOffset + bb) = (tmpK*udif2).sum();
			}
			// Rcout << "Updated coefficients [" << rowOffset+z << ", " << colOffset + bb << "]" << endl;
		}
	}
	// d/gamma d/gamma
	rowOffset = mY - 1 + pq;
	colOffset = rowOffset;
	ArrayXd pii = 1.0/(1.0+(-gxy).exp());
	// MatrixXd t1 = XY.array().colwise()*(pii*(1.0-pii)); // this cannot (easily) be squished into the next line. Not a yuuge matrix tho.
	H.block(rowOffset,colOffset,gLen,gLen) = -XY.transpose()*(XY.array().colwise()*(pii*(1.0-pii))).matrix();  // the logistic Hessian
	double gAdj; // holder for the interaction term gamma adjustment (organizational purposes only)
	scl = nx.array()/d33;
	for(int a = 0; a < gLen; a++){
		for(int b = 0; b <= a; b++){
			if(a <= P){
				if(a == 0){
					tmpK = ArrayXd::Constant(mX,1.0);  // ::Constant(rows,cols,value)
				} else { // a-1 picks out a column of X. n_x is already in 'scl'
					tmpK = XA.col(a-1);
				}
				if(b<=P){ // both X col
					udif1 = piNum2*piNum2 - d3*piNum3; // constant with respect to a,b
					if(b==0){
						tmpK2 = ArrayXd::Constant(mX,1.0);
					} else {
						tmpK2 = XA.col(b-1);
					}
					udif2 = scl*tmpK*tmpK2*udif1;
					gAdj = udif2.sum();
					if(verbosity > 2) Rcout << "Adding " << gAdj << " to gamma position (" << a << ", " << b << ")" << endl;
					H(rowOffset + a,colOffset + b) += gAdj;
				} else { // a-> X, b-> Y. This section should't ever be reached since b <= a within the loop
					Rcout << "Whoops, we shouldn't be here!" << endl;
					int ycol = b - P - 1;
					tmpJ = UAY.col(ycol);
					udif1 = (epi3.rowwise()*tmpJ.transpose()).rowwise().sum(); // \sum u_jb E_jk pi (1-pi) (1-2*pi)
					udif2 = scl*tmpK*(piNum2*UEPP.col(ycol) - d3*udif1);
					gAdj = udif2.sum();
					Rcout << "Adding " << gAdj << " to gamma position (" << a << ", " << b << ")" << endl;
					H(rowOffset + a,colOffset + b) += gAdj;
				}
			} else { // a -> Y
				int ycol = a - P - 1;
				tmpJ = UAY.col(ycol);
				udif1 = (epi3.rowwise()*tmpJ.transpose()).rowwise().sum();
				if(b <= P){ // b -> X
					if(b == 0){ tmpK = ArrayXd::Constant(mX,1,1.0); }
					else{ tmpK = UAX.col(b-1); }
					udif2 = piNum2*UEPP.col(ycol) - d3*udif1;
					gAdj = (scl*tmpK*udif2).sum();
					if(verbosity > 2) Rcout << "Adding " << gAdj << " to gamma position (" << a << ", " << b << ")" << endl;
					H(rowOffset + a,colOffset + b) += gAdj;
				} else { // b -> Y
					int bcol = b - P - 1;
					tmpJ = UAY.col(bcol)*UAY.col(ycol);
					udif1 = (epi3.rowwise()*tmpJ.transpose()).rowwise().sum();
					udif2 = UEPP.col(ycol)*UEPP.col(bcol) - d3*udif1;
					// printRng(udif2,"Y-Y gamma");
					gAdj = (scl*udif2).sum();
					if(verbosity > 2) Rcout << "Adding " << gAdj << " to gamma position (" << a << ", " << b << ")" << endl;
					H(rowOffset + a,colOffset + b) += gAdj;
				}
			}
			// Rcout << "Adjustment for gamma coefficients " << a << " and " << b << ": " << gAdj << endl;
			// Rcout << "Updated coefficients [" << rowOffset + a << ", " << colOffset + b << "]" << endl;
		}
	}
	for(int i=0;i<nParam;i++){ // symmetry
		for(int j=0;j<i;j++){
			if( std::isnan(H(i,j) || std::isinf(H(i,j))) || H(i,j) == 0. ) Rcout << "Warning: entry (" << i << ", " << j << ") is unreal!" << endl;
			H(j,i) = H(i,j);
		}
	}
	H *= -1./N;
	if(verbosity > 2) Rcout << "Finished Hessian calculations! min. is " << H.minCoeff() << " and max is " << H.maxCoeff() << endl;
	return;
}

// convert alpha  parameters to jump sizes F
VectorXd a2f(VecRef alpha){
	const int L = alpha.size();
	VectorXd res(L+1);
	double den = alpha.array().exp().sum() + 1.0;
	res.head(L) = alpha.array().exp()/den;
	res(L) = 1.0/den;
	return res;
}

// [[Rcpp::export]]
Rcpp::NumericVector logiFit(Rcpp::NumericVector dd,Rcpp::NumericMatrix xx,double lambda,double tol){
	const MapMxd D(Rcpp::as<MapMxd>(dd)), X(Rcpp::as<MapMxd>(xx));
	VectorXd res(X.cols());
	try{
		res = logisticFit(D,X,lambda,tol);
	} catch(const char* errMsg) {
		Rcpp::stop(errMsg);
	} catch(...){
		Rcpp::stop("unknown error occurred");
	}
	return Rcpp::wrap(res);
}

// should be a member function? Then can call from the constructor (first time), or re-initialize later
VectorXd initParams(const spd& DB){
	VectorXd b0 = 0.2*vectorize(betaFit(DB.Y,DB.X,true)); // initial beta shrunk towards zero to decrease chance of overflow in initial likelihood calculations
	VectorXd g0(DB.P+DB.Q+1);
	try{
		g0 = logisticFit(DB.D,DB.XY,0.33,1e-4); // 3rd arg is lambda
		Rcpp::Rcout << "g0 = " << g0.transpose() << '\n';
	} catch(const char* errMsg){
		throw(errMsg);
	} catch(...){
		throw("unknown error in logisticFit");
	}
	double meanD = DB.D.sum()/DB.N;
	g0(0) += log(DB.ksi*(1.-meanD)/((1.-DB.ksi)*meanD)); // adjust intercept for biased sampling
	if(DB.verbosity > 1) Rcout << "gamma0 = " << g0.transpose() << endl;
	VectorXd res(0.25*ArrayXd::Random(DB.nParam)); // initial jump sizes
	res.tail(g0.size()) = g0;
	res.segment(DB.mY-1,b0.size()) = b0;
	return res;
}

MatrixXd runBoot(const spd& DB,int nBoot,double TOL,int verb,std::string method="Brent",const std::string conv = "func"){
	const int n = DB.N, pq = DB.P*DB.Q;
	MatrixXd res(DB.P,DB.Q);
	MatrixXd varObs(MatrixXd::Zero(nBoot,pq)); // only store beta, theta since jump sizes vary by bootstrap data set
	MatrixXd XX(n,DB.P),YY(DB.N,DB.Q),bHat(DB.P,DB.Q);
	VectorXd DD(n);
	int nFit = 0,ntry=0;
	Rcout << "Bootstrap #: ";
	while(nFit < nBoot){
		ntry++; if(ntry > 10) break;
		Rcpp::checkUserInterrupt();
		ArrayXi inxs = sample(n,0,n-1);
		for(int i=0;i<n;i++){
			XX.row(i) = DB.X.row(inxs(i));
			YY.row(i) = DB.Y.row(inxs(i));
			DD(i) = DB.D(inxs(i));
		}
		spd tD(XX,YY,DD,DB.ksi,0);
		VectorXd th0(tD.nParam);
		funcMin res;
		try{
			th0 = initParams(tD);
		} catch(...){
			continue; // just ignore failures
		}
		try{
			res = dfpmin(th0,tD,VectorXi::Constant(1,-1),method,verb,TOL,TOL,conv);
			th0 = res.arg_min;
		} catch(const char* e){
			Rcpp::Rcout << "fail: " << e << "!" << std::endl;
			continue;
		} catch(...){
			Rcpp::Rcout << "mystery fail!" << std::endl;
			continue;
		}
		bHat = unVectorize(th0.segment(tD.mY-1,pq),DB.Q);
		MatrixXd resids(tD.Y-tD.yhats(th0));
		MatrixXd residCov(cov(resids));
		VectorXd bAdj(vectorize(bHat*residCov));
		if(!(is_nan(bAdj)||is_inf(bAdj))){  // filter bad rows here, if possible. If any slip through, cov() should get them.
			if(bHat.array().abs().maxCoeff() < 10){ // filter out strange converge combinations
				varObs.row(nFit++) = bAdj.transpose();	
			}
		}
	}
	return varObs;
}

// [[Rcpp::export]]
Rcpp::List fit2pd(Rcpp::NumericMatrix xx,Rcpp::NumericMatrix yy,Rcpp::NumericVector dd, double ksi,double TOL = 0.,int verb = 0,int nBoot = 0,
	std::string method = "Brent",const std::string conv = "func") {
	const MapMxd X(Rcpp::as<MapMxd>(xx));
	const MapMxd Y(Rcpp::as<MapMxd>(yy));
	const MapVxd D(Rcpp::as<MapVxd>(dd));
	// initialize data object
	spd D2(X,Y,D,ksi,verb);
	double MLE;
	int nit;
	if(TOL <= 0.) TOL = 1.e-5*D2.nParam;
	VectorXd th0(D2.nParam);
	int nTry = 0;
	funcMin res;
	while(nTry < 3){  // may need to try a few initial values
		try{
			th0 = initParams(D2);  // good initial values!?
		} catch(const char* errMsg) {
			Rcpp::warning(errMsg);
			return Rcpp::List::create(Named("errMsg") = errMsg);
		} catch(...){
			Rcpp::warning("unknown exception in initParams");
			return Rcpp::List::create(Named("errMsg") = "unknown exception in initParams");
		}
		try{
			res = dfpmin(th0,D2,VectorXi::Constant(1,-1),method,verb,TOL,TOL,conv);
			th0 = res.arg_min;
			MLE = res.min_value;
			nit = res.iterations;
			if(res.error != dfpmin_error::NONE){
    			Rcpp::CharacterVector errmsg(1);
    			errmsg[0] = dfpmin_err_msg::messages[res.error-1];
    			return Rcpp::List::create(Named("error") = errmsg,
								  		Named("parameters") = Rcpp::wrap(th0));
  			}
			nTry = 3;
		} catch(const char* errMsg) {
			Rcpp::warning(errMsg);
			if(++nTry >= 3) return Rcpp::List::create(Named("errMsg") = errMsg);
		} catch(...) {
			Rcpp::warning("unknown exception in initParams");
			if(++nTry >= 3) return Rcpp::List::create(Named("errMsg") = "unknown exception in dfpmin");
		}
	}
	
	MatrixXd resids(D2.Y - D2.yhats(th0));
	MatrixXd residCov(cov(resids));
	VectorXd rGrad(D2.nParam);
	D2.df(th0,rGrad);
	MatrixXd hc(D2.nParam,D2.nParam);
	
	D2.Hess(th0,hc);
	ArrayXd vars = (1./D2.N)*hc.inverse().diagonal();  // don't need negative here since everything is already *(-1)
	MatrixXd bootV; // forward declaration without initialization
	ArrayXd bootVar;
	if(nBoot > 0){
		if(nBoot <= 1) nBoot = 25; // set min. # of bootstrap samples
		bootV = runBoot(D2,nBoot,TOL,verb,method); // matrix of bootstrap betas
		bootVar = cov(bootV).diagonal(); // sample covariances of those betas
	}
	else{
		bootV = MatrixXd(1,1);
		bootV(0,0) = NAN; // gets coerced to NA?
		bootVar = VectorXd::Constant(1,0.);
	}
	ArrayXd fhat = a2f(th0.head(D2.mY-1));
	VectorXd b0 = (D2.UY.array().colwise()*fhat).colwise().sum();
	MatrixXd bhat(D2.P+1,D2.Q); // including estimated intercept b0
	bhat.row(0) = b0.transpose();
	bhat.bottomRows(D2.P) = unVectorize(th0.segment(D2.mY-1,D2.P*D2.Q),D2.Q);
	return Rcpp::List::create(
		Named("alpha") = Rcpp::wrap(th0.head(D2.mY-1)),
		Named("F") = Rcpp::wrap(fhat),
		Named("gamma") = Rcpp::wrap(th0.tail(D2.P+D2.Q+1)),
		Named("beta") = Rcpp::wrap(bhat.bottomRows(D2.P)),
		Named("beta0") = Rcpp::wrap(b0),
		Named("UY") = Rcpp::wrap(D2.UY),
		Named("sigma2") = Rcpp::wrap(residCov),
		Named("logLik") = MLE,
		Named("iterations") = nit,
		Named("gradient") = Rcpp::wrap(rGrad),
		Named("lambda") = D2.lambdaF(th0),
		Named("variances") = Rcpp::wrap(vars),
		Named("beta.se") = Rcpp::wrap(vars.segment(D2.mY-1,D2.P*D2.Q).sqrt()),
		Named("gamma.se") = Rcpp::wrap(vars.tail(D2.P+D2.Q+1).sqrt()),
		Named("bootstrap_values") = Rcpp::wrap(bootV),
		Named("bootstrap.se") = Rcpp::wrap(bootVar.sqrt())
	);
}


// [[Rcpp::export]]
int timeit(VectorXd theta,VectorXd dd,MatrixXd Y,MatrixXd X,double ksi,int ntrial = 100){
	const spd D(X,Y,dd,ksi);
	auto t1 = std::chrono::high_resolution_clock::now();
	double res = 0;
	for(int i=0;i<ntrial;++i){
		res += D(theta);
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	for(int i=0;i<ntrial;++i){
		res += D.llv2(theta);
	}
	auto t3 = std::chrono::high_resolution_clock::now();
	Rcpp::Rcout << "Prevent optimizing away: " << res << std::endl;
	// Note: floating-point versions don't need explicit casting, see https://en.cppreference.com/w/cpp/chrono/duration/duration_cast
	std::chrono::duration<double,std::milli> ts1 = t2 - t1;
	std::chrono::duration<double,std::milli> ts2 = t3 - t2;
	Rcpp::Rcout << "Avg. time of first method (ms): " << ts1.count()/ntrial << std::endl;
	Rcpp::Rcout << "Avg. time of second method: (ms)" << ts2.count()/ntrial << std::endl;
	return 0;
}
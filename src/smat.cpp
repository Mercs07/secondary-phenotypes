// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace Eigen;
using Rcpp::Rcout;
using Rcpp::Named;
using std::endl;

typedef Eigen::Map<Eigen::ArrayXXd> MapAxd;
typedef Eigen::Map<Eigen::VectorXd> MapVxd;
typedef Eigen::Map<Eigen::MatrixXd> MapMxd;
typedef const Eigen::Ref<const Eigen::VectorXd>& VecRef;
typedef const Eigen::Ref<const Eigen::MatrixXd>& MatRef;
typedef const Eigen::Ref<const Eigen::ArrayXd>& ArrRef;

// Kronecker product of two matrices.  Not actually used anywhere since it is more efficient and straightforward to loop
MatrixXd kron(MatRef A,MatRef B){
	const size_t Ar = A.rows(), Ac = A.cols(), Br = B.rows(), Bc = B.cols();
	MatrixXd res(Ar*Br,Ac*Br);
	for(size_t i=0;i<Ar;i++){
		for(size_t j=0;j<Ac;j++){
			res.block(i*Br,j*Bc,Br,Bc) = A(i,j)*B;
		}
	}
	return res;
}

// convert matrix beta to vector gamma
VectorXd b2g(MatRef beta){
	const size_t P = beta.rows(), M = beta.cols();
	VectorXd res((P-1)*M + 1);
	for(size_t j=0;j<M;j++){
		res.segment(j*(P-1),P-1) = beta.col(j).head(P-1);
	}
	res((P-1)*M) = beta(P-1,0);
	return res;
}

// convert vector gamma into the corresponding beta matrix
template<typename T> // does a SegmentReturnType or VectorBlock have size(), operator(), and segment() methods?
MatrixXd g2b(T &g,int M){
	const size_t P = g.size()/M; // number of rows in final matrix *minus* one
	MatrixXd res(P+1,M);
	for(int j=0;j<M;j++){
		res.block(0,j,P,1) = g.segment(j*P,P);
	}
	res.row(P) = VectorXd::Constant(M,g(g.size()-1)).transpose();
	return res;
}

// the 'internal' bigX which should be more efficient by avoiding malloc() at each call.
// the caller is responsible for ensuring that nrow(bx)*[length(x)-1] = ncol(bx)
void bigX(MatrixXd& bx,VecRef x){
	const size_t M = bx.rows(), P = x.size(); // x here is actually one 'longer' than what's referred to in Lin's paper since we pack 's' along with X.
	const int ncol = (P-1)*M + 1; 
	if( bx.cols() != ncol ) Rcpp::stop("Cannot calculate bigX! Wrong number of columns");
	bx.setZero();
	bx.col(ncol-1) = VectorXd::Constant(M,x(P-1));
	const VectorXd XH = x.head(P-1);
	for(size_t i=0;i<M;i++){
		bx.block(i,(P-1)*i,1,P-1) = XH.transpose();
	}
	return;
}

// Capital X as defined in X. Lin, et al. Just a testing function
// [[Rcpp::export]]
Rcpp::NumericMatrix bigX(VectorXd X,unsigned int M){
	int ncol = (X.size()-1)*M + 1; // the last row of beta is constant
	MatrixXd res(M,ncol);
	bigX(res,X);
	return Rcpp::wrap(res);
}

// the estimating equation for beta - caller responsible for checking inputs!
// [[Rcpp::export]]
Rcpp::NumericMatrix betaEE(MatrixXd X,MatrixXd Y,VectorXd sigma,MatrixXd R,VectorXd w,MatrixXd beta){
	MatrixXd yScl = Y.array().rowwise()*(sigma.transpose().array().sqrt());
	MatrixXd Rinv = R.inverse();
	const int N = X.rows(), P = X.cols(), M = Y.cols();
	const int K = (P-1)*M + 1;
	MatrixXd biggX(M,K);
	VectorXd gam = b2g(beta);
	VectorXd res = VectorXd::Zero(K);
	for(int i=0;i<N;i++){
		bigX(biggX,X.row(i));
		res += w(i)*biggX.transpose()*Rinv*(yScl.row(i).transpose() - biggX*gam);
	}
	return Rcpp::wrap(res);
}

// estimating equation for beta with homogeneous effects:
// \sum_i X^T R^{-1} (y/sigma - X_i\beta)
// if homogeneous, we need to ensure that the input beta has a constant bottom row!
VectorXd U2(MatRef X,MatRef yScl,ArrRef w,MatRef R,MatRef beta){
	const int N = X.rows(), P = X.cols(), M = yScl.cols();
	if(yScl.rows() != N){ Rcpp::stop("Failure in beta estimating equation: Y and X sizes don't match."); }
	MatrixXd dif( (yScl - X*beta)*R.inverse() ); // N x M matrix -> the order here doesn't match the equation R(Y-XB) since we aren't doing row-wise calculation.
	MatrixXd res( MatrixXd::Zero(P,M) );
	ArrayXXd tmpA(P,M);
	for(int i=0;i<N;i++){
		res += w(i)*X.row(i).transpose()*dif.row(i);
	}
	VectorXd gam =  b2g(res); // converts from matrix into properly vectorized 'gamma' (for heterogeneous effects, we need a conditional here)
	gam(gam.size()-1) = res.row(P-1).sum(); // replace single value with sum as needed here (for heterogeneous effects?)
	return gam;
}

// extra estimating equation for eta variables - only used for conducting the score test that \eta = 0
VectorXd U22(MatRef X,MatRef yScl,ArrRef w,MatRef R,MatRef beta){
	const int N = X.rows(), P = X.cols()-1, M = yScl.cols();
	if(yScl.rows() != N || beta.cols() != M){ Rcpp::stop("Failure in eta estimating equation: Y and X sizes don't match."); }
	// const MatrixXd dif( (yScl - X*beta)*(R.inverse().rightCols(M-1)) );  // N x (M-1)
	MatrixXd res(yScl - X.leftCols(P)*beta.topRows(P));
	res.colwise() -= X.col(P)*beta(P,0); // "alpha" in last row of beta
	const VectorXd wx = X.col(P).array()*w;
	return R.inverse().bottomRows(M-1)*res.transpose()*wx;
}

// estimating equation for sigma
VectorXd U1(MatRef X,MatRef yScl,ArrRef w,MatRef beta){
	ArrayXXd dif = yScl - X*beta;
	dif = dif*yScl.array() - 1.0;
	dif.colwise() *= w;
	return dif.colwise().sum();
}

// update for beta
VectorXd bNew(MatRef X,MatRef yScl,ArrRef w,MatRef Rinv){
	const size_t N = X.rows(), P = X.cols(), M = yScl.cols();
	const size_t bigXcol = (P-1)*M + 1;
	MatrixXd yr( yScl*Rinv ); // N x M
	MatrixXd XRX = MatrixXd::Zero(bigXcol,bigXcol);
	VectorXd XRY = VectorXd::Zero(bigXcol);
	MatrixXd XX(M,bigXcol); // filled in each i
	for(size_t i=0;i<N;i++){ // these two pieces also constitute the estimating equation, namely [ XRX - XRY = 0 ]
		bigX(XX,X.row(i));
		XRX += w(i)*(XX.transpose()*Rinv*XX);
		XRY += w(i)*(XX.transpose()*yr.row(i).transpose());
	}
	return XRX.inverse()*XRY;
}

VectorXd sNew(MatRef X,MatRef yScl,ArrRef w,MatRef beta,VecRef sigma){
	 // if we are using a common effect model, then the last row of beta MUST be constant. This avoids all the fussing with bigX and turns all matrices into vectors
	const ArrayXXd XB = X*beta;
	const ArrayXd sigRt = 1./sigma.array().sqrt(), sigInv = 1./sigma.array();
	ArrayXXd XBXB = 0.5*((XB*XB).rowwise()/sigma.transpose().array()); // most of the denominator
	XBXB.rowwise() += sigInv.transpose();
	XBXB.colwise() *= w;
	ArrayXd denoms = XBXB.colwise().sum();
	ArrayXXd nums = yScl.array()*(yScl.array() - XB) - 1.0;
	nums.colwise() *= w;
	ArrayXd NN = nums.colwise().sum();
	return sigma + (NN/denoms).matrix(); // update sigma.
}

// this is intended to be an Eigen class (i.e. with size() and sum() methods - it could even be an Rcpp class!
template<typename C>
double Ecov(const C &a1,const C &a2){
	if(a2.size() != a1.size()) Rcpp::stop("Inputs of different sizes are incompatible."); // both Rcpp and Eigen have size() member functions.
	const double n = (double)a1.size();
	const double m1 = a1.sum()/n, m2 = a2.sum()/n; 
	// we need to explicitly tell Eigen to 'view' a1 and a2 as arrays, otherwise it gets confused with classes like 'block', 'colXptr', etc.
	return ((a1.array() - m1)*(a2.array() - m2)).sum()/(n-1.);
}

// more efficient method to calculate a covariance matrix (not just pairwise) let X be a N x P matrix...
// we cannot (easily) templatize this since it involves operator* which is very different for Eigen classes array vs. matrix
MatrixXd Mcov(MatRef X){
	const double n = X.rows();
	const VectorXd colSums = X.array().colwise().sum(); // 
	ArrayXXd crosProd = X.transpose()*X - (1./n)*(colSums*(colSums.transpose())); // P x P matrix of cross-products
	return (1./(n-1))*crosProd; // divide colwise and rowwise by crosProd.diagonal().sqrt() to get correlation matrix.
}

// updating the correlation matrix based on residuals
// question: should we include weights here? Yes, for consistency with rest of method
MatrixXd rNew(MatRef residuals,ArrRef w,std::string method = "unstructured"){
	MatrixXd resid(residuals.array().colwise()*w);
	const size_t M = resid.cols();
	MatrixXd eC(Mcov(resid));
	VectorXd vars(eC.diagonal());
	//MatrixXd D1 = vars.asDiagonal(); // the .asDiagonal() method is of class "diagonalWrapper" NOT "MatrixBase" and doesn't have an array() method.
	ArrayXd sds = vars.array().sqrt(); // standard deviations.
	if(method == "unstructured") {
		ArrayXXd CM(eC);
		CM.colwise() /= sds;
		CM.rowwise() /= sds.transpose();
		return CM;
	}
	else if(method == "CS") { // compound-symmetric method
		// collect all off-diagonal cross products
		ArrayXXd CP = resid.transpose()*resid;
		ArrayXd cSums = resid.array().colwise().sum();
		double cpSum = 0.;
		// here's a na{\"i}ve way to estimate compound symmetry: use mean of all the (off-diagonal) pairwise correlations
		for(size_t i=1;i<M;i++){
			for(size_t j=0;j<i;j++){
				cpSum += eC(i,j)/sqrt(vars(i)*vars(j));
			}
		}
		double alphaHat = 2.*cpSum/(double)(M*(M-1)); // mean correlation among the observations.
		MatrixXd CSMat = MatrixXd::Constant(M,M,alphaHat);
		CSMat.diagonal() = VectorXd::Constant(1.0,M);
		return CSMat;
	} else if(method=="independent") {
		return MatrixXd::Identity(M,M);  // last resort is independence, just a placeholder
	}
	else {
		Rcpp::stop("Invalid correlation matrix method. Pass one of: 'unstructured', 'CS'.");
	}
}

// the matrix 'A' as defined in Roy et al. (2003) - derivative of EE for eta w/r/to beta, sigma2 parameters
MatrixXd calcA(MatRef X,ArrRef w,MatRef R,VecRef sigma2,MatRef beta){
	const int P = X.cols()-1, N = X.rows(), M = sigma2.size();
	if(beta.rows() - 1 != P || w.size() != N || R.rows() != M) Rcpp::stop("calcA: Bad dimensions of inputs.");
	const MatrixXd XB(X*beta);
	ArrayXd sInv = 1./sigma2.array();
	const MatrixXd LR = (R.inverse()).bottomRows(M-1);
	const MatrixXd LRP = LR*(sInv.matrix().asDiagonal());
	MatrixXd A1(MatrixXd::Zero(M-1,M)); // d_sigma / d_eta
	MatrixXd A2(MatrixXd::Zero(M-1,M*P+1)); // d_gamma / d_eta
	MatrixXd BigX(M,M*P+1);
	const VectorXd G = b2g(beta);
	for(int i=0;i<N;i++){
		bigX(BigX,X.row(i));
		A1 += w(i)*X(i,P)*LRP*(BigX*G).asDiagonal();
		A2 += w(i)*X(i,P)*LR*BigX;
	}
	MatrixXd A(A1.rows(),A1.cols()+A2.cols());
	A.leftCols(A1.cols()) = 0.5*A1;
	A.rightCols(A2.cols()) = A2;
	return A;
}

// standard error estimates based on estimating equations and sandwich estimator
// returns two matrices, H and G (H is "hessian" and G is the outer product of gradients)
// multiply H.transpose()*G.inverse()*H to get sandwich estimate of covariance
// new addition: to calculate the score test, we calculate BIG G which includes the estimating equation for etas adding another (M-1) rows and cols
std::vector<Eigen::MatrixXd> seEst(MatRef X,MatRef yScl,ArrRef w,MatRef beta,VecRef sigma,MatRef R){
	const size_t P = X.cols(), M = yScl.cols(), N = X.rows();
	const size_t bp = (P-1)*M + 1;  // Matrix H has blocks M x M, M x (P-1)*M + 1, [(P-1)*M+1] x [(P-1)*M+1]
	MatrixXd H11(MatrixXd::Zero(M,M)), H12(MatrixXd::Zero(M,bp)), H21(MatrixXd::Zero(bp,M)), H22(MatrixXd::Zero(bp,bp));
	MatrixXd UU(MatrixXd::Zero(M + bp + M-1,M + bp + M - 1));
	VectorXd uu(M + bp + M - 1); // container for observation-wise estimating equation values (including eta)
	MatrixXd biggX(M,bp);
	MatrixXd Rinv(R.inverse());
	ArrayXd sInv = 1./sigma.array();
	MatrixXd RP = Rinv*sInv.matrix().asDiagonal();
	VectorXd XG(M),tmp(M); // containers for bigX*gamma and Psi^{-1}*X*gamma
	MatrixXd XB = X*beta; // used for sigma est. eq.
	MatrixXd res(yScl - XB);
	MatrixXd LR = res*Rinv.rightCols(M-1); // ingredient for eta EE (n x m-1)
	const VectorXd gg(b2g(beta));
	for(size_t i=0;i<N;i++){
		bigX(biggX,X.row(i));
		XG = biggX*gg;
		tmp = w(i)*(sInv + 0.5*sInv*XG.array()*XG.array());
		H11 += tmp.asDiagonal(); // this matrix is really diagonal!
		H12 += w(i)*XG.asDiagonal()*biggX;
		H21 += 0.5*w(i)*biggX.transpose()*RP*XG.asDiagonal();
		H22 += w(i)*biggX.transpose()*Rinv*biggX;
		uu.head(M) = w(i)*(yScl.row(i).array()*(yScl.row(i) - XB.row(i)).array() - 1.);
		uu.segment(M,bp) = w(i)*biggX.transpose()*Rinv*((yScl.row(i) - XB.row(i)).transpose());
		uu.tail(M-1) = w(i)*X(i,P-1)*LR.row(i);
		UU += uu*uu.transpose(); // sum of outer product of gradients is one approximation to information
	}
	MatrixXd HH(M+bp,M+bp);
	HH.block(0,0,M,M) = H11;
	HH.block(0,M,M,bp) = H12;
	HH.block(M,0,bp,M) = H21;
	HH.block(M,M,bp,bp) = H22;
	std::vector<Eigen::MatrixXd> ret(2);
	ret[0] = HH;  ret[1] = UU;
	return ret;
}

// wrapper for calculating sandwich estimator of beta and gamma.
// [[Rcpp::export]]
Rcpp::NumericMatrix seWrap(MatrixXd X,MatrixXd Y,VectorXd w,MatrixXd beta,VectorXd gam,MatrixXd R){
	const MatrixXd yScl = Y.array().rowwise()/gam.array().transpose();
	const int dimG1 = Y.cols()*X.cols() + 1; // no. of parameters in beta, sigma
	std::vector<MatrixXd> GH = seEst(X,yScl,w,beta,gam,R);
	MatrixXd G = GH[1].block(0,0,dimG1,dimG1);
	MatrixXd covEst = (GH[0]).transpose()*G.inverse()*GH[0];
	return Rcpp::wrap(covEst); // just returning the information matrix for inspection purposes.
}

// update the parameters beta and gamma given current correlation matrix R, X, Y, weights w, and params beta and sigma
// old testing function... not used anywhere else
/*
Rcpp::List updateParams(Rcpp::NumericMatrix xx,Rcpp::NumericMatrix yy,Rcpp::NumericVector ww,Rcpp::NumericMatrix bb,Rcpp::NumericMatrix rr,Rcpp::NumericVector ss,bool common = true){
	const MatrixXd X = Rcpp::as<Eigen::MatrixXd>(xx);
	const MatrixXd Y = Rcpp::as<Eigen::MatrixXd>(yy);
	const ArrayXd w = Rcpp::as<Eigen::ArrayXd>(ww);
	const MatrixXd beta = Rcpp::as<MatrixXd>(bb);
	const MatrixXd R = Rcpp::as<MatrixXd>(rr);
	const MatrixXd Rinv = R.inverse();
	VectorXd sigma = Rcpp::as<VectorXd>(ss);
	const int N = X.rows(), M = Y.cols(), P = X.cols();
	// const int bigXcol = (P-1)*M + 1; // number of unique parameters since the last ROW of beta is constrained to be constant.
	if(w.size() != N || bb.rows() != P || bb.cols() != M || rr.rows() != M || rr.rows() != rr.cols() || ss.size() != M){
		Rcpp::stop("Data is incompatibly-sized.");
	}
	if(common){
		if(beta.col(M-1) != VectorXd::Constant(M,beta(0,M-1))){
			Rcpp::stop("Common-effect model needs the last column of beta to be constant.");
		}
	}
	const MatrixXd yScl = Y.array().rowwise()/sigma.array().transpose().sqrt();
	VectorXd bTmp(bNew(X,yScl,w,Rinv));  // needed as a matrix to update sigma
	MatrixXd bOut = g2b(bTmp,M);
	VectorXd sigNew = sNew(X,Y,w,bOut,sigma); // use updated beta in the update for sigma
	const int MAXIT = 10;
	int nit = 0;
	while( (sigNew-sigma).squaredNorm() > 0.001 && nit < MAXIT ){
		sigma = sigNew;
		sigNew = sNew(X,Y,w,bOut,sigma);
		// Rcout << "sigma updated to: \n" << sigNew.transpose() << endl;
		nit++;
	}
	// update working correlation matrix:
	MatrixXd residual = yScl - X*bOut; // the 'y' whose expectation is X\beta is actually the scaled Y, a.k.a. Y*
	MatrixXd Rhat = rNew(residual,w,"unstructured");
	// optionally, print values of the estimating equations that we're trying to zero
	bool verb = true;
	if(verb){
		MatrixXd rinvv = Rhat.inverse();
		VectorXd u1val = U1(X,yScl,w,bOut);
		// VectorXd u2val = U2(bOut,rinvv,X,yScl,w); // last bool argument is for homogeneous effects
		Rcout << "Estimating equation U1 (sigma) values: " << u1val.transpose() << endl;
		// Rcpp::Rcout << "Estimating equation U2 (beta) values: " << u2val.transpose() << endl;
		Rcout << "The beta update of size " << bTmp.size() << ":\n" << bTmp.transpose() << endl;
		Rcout << "The matrix version of beta:\n" << bOut << endl;
		Rcout << "sigma updated to: \n" << sigNew.transpose() << endl;
		Rcout << "working correlation matrix:\n" << Rhat << endl;
	}
	return Rcpp::List::create(
		Rcpp::Named("betahat") = Rcpp::wrap(bOut),
		Rcpp::Named("sigmahat") = Rcpp::wrap(sigNew),
		Rcpp::Named("Rhat") = Rcpp::wrap(Rhat)
	);
}
*/


// compute the cross-product matrix efficiently
// the rankUpdate method maps X -> X + alpha*A*A^T for scalar alpha
inline MatrixXd AtA(MatRef A){
	const int p(A.cols());
	return MatrixXd(p,p).setZero().selfadjointView<Lower>().rankUpdate(A.adjoint());
}

// compute regression coefficients beta using LLT decomposition.
MatrixXd betaFit(MatRef Y,MatRef X){
	const Eigen::LLT<MatrixXd> thellt(AtA(X)); // compute the Cholesky decomposition of X^{T}X
	return thellt.solve(X.adjoint()*Y); // is constructor initialization better than assignment for Eigen classes?
}

// [[Rcpp::export]]
Rcpp::List smatFit(MatrixXd X,MatrixXd Y,VectorXd ww,std::string corrStruct = "unstructured",int verbose = 0){
	const ArrayXd w(ww);
	const int N = X.rows(), P = X.cols(), M = Y.cols();
	if(N != Y.rows() || N != w.size()){ Rcpp::stop("X, Y and w must have the same number of rows"); }
	MatrixXd beta( betaFit(Y,X) );
	double alphaHat = beta.row(P-1).sum()/M; // approximation to the common effect
	beta.row(P-1) = VectorXd::Constant(M,alphaHat).transpose();
	if(verbose > 0) Rcout << "The initial beta parameters:\n" << beta << endl;
	// initialize sigma to sample variances of residuals
	MatrixXd resid(Y-X*beta);
	const MatrixXd covMat(Mcov(resid));
	VectorXd sigma(covMat.diagonal());
	const ArrayXd sds = sigma.array().sqrt();
	MatrixXd R(covMat.array()/((sigma*sigma.transpose()).array().sqrt())); // initial correlation matrix
	MatrixXd Rinv = R.inverse();
	MatrixXd yScl(Y.array().rowwise()/sds.transpose());
	const int MAXIT = 25;
	const double TOL = 1.e-5;
	double currDist = U1(X,yScl,w,beta).squaredNorm() + U2(X,yScl,w,R,beta).squaredNorm(), paramDist = 0.;
	if(verbose > 0) Rcout << "Initial distance: " << currDist << endl;
	int cntr = 0;
	VectorXd bVec((P-1)*M + 1); // a container to hold the vector version of beta
	VectorXd sigOld(M);
	MatrixXd betaOld(P,M);
	while(currDist > TOL && cntr < MAXIT){
		// store values of old parameters before updating:
		betaOld = beta;
		sigOld = sigma;
		// first step: update beta (closed form)
		bVec = bNew(X,yScl,w,Rinv);
		beta = g2b(bVec,M);
		// next update sigma using new beta values, and yScl using new sigma:
		sigma = sNew(X,yScl,w,beta,sigma); 
		yScl = Y.array().rowwise()/sigma.transpose().array().sqrt();
		// last update the working correlation matrix and inverse
		resid = yScl - X*beta;
		R = rNew(resid,w,corrStruct);
		Rinv = R.inverse();
		// finally, update the convergence metric(s)
		currDist = U1(X,yScl,w,beta).squaredNorm() + U2(X,yScl,w,R,beta).squaredNorm();
		paramDist = (beta-betaOld).squaredNorm() + (sigma-sigOld).squaredNorm();
		cntr++;
		if(verbose > 0){
			Rcout << "On iteration " << cntr << ", beta is\n " << beta << "\nSigma is " << sigma.transpose() << "\nThe working correlation: \n" << R << "\nThe distance is " << currDist << endl << endl;
			Rcout << "change in parameters: " << paramDist << endl;
			Rcout << "sigma EE: " << U1(X,yScl,w,beta).transpose() << "\nbeta EE: " << U2(X,yScl,w,R,beta).transpose() << "\n\n" << endl;
		}
	}
	const int dimG1 = P*M + 1;
	std::vector<MatrixXd> HG = seEst(X,yScl,w,beta,sigma,R); // HG[0] = H, HG[1] = G
	MatrixXd H(HG[0]), G(HG[1]);
	MatrixXd covHat = (H.transpose()*G.block(0,0,dimG1,dimG1).inverse()*H).inverse();
	VectorXd seHat = covHat.diagonal().array().sqrt();
	VectorXd gamSE = seHat.tail(dimG1-M);
	MatrixXd betaSE = g2b(gamSE,M);
	VectorXd sigSE = seHat.head(M);
	// the score test:
	VectorXd etaEE = U22(X,yScl,w,R,beta);
	MatrixXd A = calcA(X,w,R,sigma,beta);
	MatrixXd AHinv = A*(H.inverse());
	MatrixXd SIG = G.bottomRightCorner(M-1,M-1) - AHinv*G.topRightCorner(dimG1,M-1) + (AHinv*G.topLeftCorner(dimG1,dimG1) - G.bottomLeftCorner(M-1,dimG1))*AHinv.transpose();
	double scoreObs = etaEE.dot(SIG.inverse()*etaEE);
	return Rcpp::List::create(Rcpp::Named("beta") = beta,Named("sigma") = sigma,
		Named("beta.se") = betaSE,Named("sigma.se") = sigSE,Named("R") = R,
		Named("score.obs") = scoreObs, Named("A") = A, Named("U2") = etaEE,
		Named("SIG") = Rcpp::wrap(SIG));
}
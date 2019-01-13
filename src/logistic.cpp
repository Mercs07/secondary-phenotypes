// logistic regression
// likelihood, gradient, and Hessian

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <cmath>

using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using ArrRef = const Eigen::Ref<const Eigen::ArrayXd>&;

using namespace Eigen;

static VectorXd sign(VecRef V){ // by default, code sign(0) as 0
	ArrayXd res(V.size());
	for(int j=0;j<V.size();j++){
		res(j) = V(j)==0 ? 0 : std::abs(V(j))/V(j);
	}
	return res;
}

double logisLL(VecRef gam,MatRef X,VecRef D,const double lam){
	const ArrayXd pis = 1./(1.+(-(X*gam).array()).exp());
	return (1.0/X.rows())*((D.array()*pis.log()).sum() + ((1.0-D.array())*(1.0-pis).log()).sum() - lam*gam.array().abs().sum());
}

VectorXd logisG(VecRef gam,MatRef X,VecRef D,const double lam){
	const VectorXd pis = 1./(1.+(-(X*gam).array()).exp());
	VectorXd absG = lam*gam.array()/(gam.array().abs());
	return (1.0/X.rows())*(X.transpose()*(D-pis) -  lam*sign(gam)); // elementwise division x/|x| gives sign(x)
}

void logisH(VecRef gam,MatRef X,MatRef D,MatrixXd& H){
	const ArrayXd pis = 1./(1.+(-(X*gam).array()).exp());
	H = -(1.0/X.rows())*X.transpose()*(X.array().colwise()*(pis*(1.0-pis))).matrix();
}

/* 	logistic regression with an optional regularizing penalty (L1-norm)
	use a small (positive) penalty term lambda to ensure we don't get inflated
	initial estimates
*/
VectorXd logisticFit(VecRef D,MatRef X,double lambda = 0.,const double GTOL = 1.0e-3){
	if((D.array() == 1).count() + (D.array()==0).count() != D.size()){ throw("Bad inputs to logistic fit!"); }  // check 0-1 inputs for D
	if(lambda < 0) lambda = 0;
	const int P = X.cols(), MAXIT = 100;
	const ArrayXd ecv(X.transpose()*D);
	VectorXd g(P), dir(P), tstTh(P);  
	// VectorXd theta(ecv/(ecv.abs().maxCoeff()));  // good initial values?
	VectorXd theta(0.1*VectorXd::Random(P));     // good initial values?
	double LL, oldLL = logisLL(theta,X,D,lambda),gDist = 1.0;
	MatrixXd H(P,P);
	g = logisG(theta,X,D,lambda);
	logisH(theta,X,D,H);
	int it2;
	ArrayXd oldSign(P),newSign(P),zeroed = ArrayXd::Constant(P,1.0);
	const double EPS = 1.0e-5;  // we don't need to get too close to the optimum since this is just an initial estimate
	for(int its = 0;its<MAXIT;its++){
		// Rcout << "Iteration " << its << ": LL = " << oldLL << ", dist = " << gDist << endl;
		if(gDist < GTOL) return theta;
		dir = H.inverse()*g;
		dir = dir.array()*zeroed; // zero out any coefficients which have reached zero
		// Rcout << "Current direction: " << dir.head(std::min(5,P)).transpose() << endl;
		tstTh = theta - dir;
		LL = logisLL(tstTh,X,D,lambda);
		while(std::isnan(LL) || std::isinf(LL)){ dir*= 0.5; tstTh += dir; } // initial bounds to avoid overflow
		it2 = 0;
		while(LL < oldLL + EPS){ // primitive line search by interval halving: require at least an EPS-size increase in objective function
			dir *= 0.5;
			tstTh += dir;
			LL = logisLL(tstTh,X,D,lambda);
			if(std::isnan(LL)) continue; // wait till we get a real number to compare against
			it2++;
			if(it2 > 25){  // continuing a search in this direction is fruitless.
				// Rcout << "Inner it " << it2 <<", abandoning theta..." << tstTh.head(std::min(P,3)).transpose() << endl;
				tstTh = 0.1*tstTh; // shrink towards zero (re-initialize, effectively) without changing sign
				break;
			};
		}
		// check if sign of coefficient has crossed zero: if so, we need to zero it out instead of letting it become negative
		if(its > 1){
			oldSign = sign(theta); newSign = sign(tstTh);
			for(int j=0;j<P;j++){
				if(oldSign(j) != newSign(j)){
					tstTh(j) = 0.;
					zeroed(j) = 0;
				}
			}
		}
		theta = tstTh;
		// Rcout << "At end of round " << its << ", theta is now " << theta.head(std::min(5,P)).transpose() << endl;
		oldLL = logisLL(theta,X,D,lambda);
		g = logisG(theta,X,D,lambda);
		g = g.array()*zeroed; // need to calculate gradient keeping in mind coefficients which were 'bumped out' of model
		logisH(theta,X,D,H);
		gDist = g.squaredNorm();
	}
	Rcpp::warning("Could not fit logistic model. Sad!");
	return VectorXd::Zero(P);  // default, still not a bad starting point
}
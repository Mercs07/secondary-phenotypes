#ifndef logistic_helpers
#define logistic_helpers
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <cmath>

using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using ArrRef = const Eigen::Ref<const Eigen::ArrayXd>&;

double logisLL(VecRef gam,MatRef X,VecRef D,const double lam);

VectorXd logisG(VecRef gam,MatRef X,VecRef D,const double lam);

void logisH(VecRef gam,MatRef X,MatRef D,Eigen::MatrixXd& H);

VectorXd logisticFit(VecRef D,MatRef X,double lambda,const double GTOL);

#endif
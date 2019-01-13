//#pragma once

#ifndef rcppeigen_utils

#define rcppeigen_utils
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

/*
 * Various utility functions which are not specific to a particular model
 */

using namespace Eigen;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;

// helper functions to check whether a given Eigen object has any NaN or Inf
// unfortunately, we can't templatize this across both matrices and arrays, since an Array<> doesn't have an .array() method (perhaps it should be built in as a no-op)
template<typename Derived>
inline bool is_inf(const Eigen::MatrixBase<Derived>& x){
   return ! ( ((x - x).array() == (x - x).array()).all() );
}

template<typename Derived>
inline bool is_nan(const Eigen::MatrixBase<Derived>& x){
   return !( (x.array() == x.array()).all() );
}

// same methods but for Array<> - derived inputs
template<typename Derived>
inline bool is_inf(const Eigen::ArrayBase<Derived> &x){
	return !((x-x) == (x-x)).all();
}

template<typename Derived>
inline bool is_nan(const Eigen::ArrayBase<Derived> &x){
	return !(x==x).all();
}

// test for infinite or NAN
template<typename Derived>
bool any_nan(const Eigen::MatrixBase<Derived>& x){
  return is_nan(x) || is_inf(x);
}

template<typename Derived>
bool any_nan(const Eigen::ArrayBase<Derived>& x){
  return is_nan(x) || is_inf(x);
}

template<int T> // T is -1 for Matrix (Eigen::Dynamic) or 1 (VectorXd) (or another constexpr)
void printrng(Eigen::Matrix<double,Eigen::Dynamic,T>& M,const char* name){
  Rcpp::Rcout << "Range of " << name << ": " << M.minCoeff() << ", " << M.maxCoeff() << std::endl;
}


// set the values of x to zero at locations specified by ii. The purpose of this is to
// fit a model with some parameter(s) fixed at a certain value
void setZeros(Eigen::Ref<Eigen::VectorXd>,const Eigen::Ref<const Eigen::VectorXi>&);

// sample 'size' integers uniformly distributed on 1,...,N (inclusive!) for selecting nonparametric bootstrap samples
// note: Rcpp includes the R distributions, which can be called as in R and return an Rcpp::NumericVector
ArrayXi sample(int N,int lo,int hi);

/*take an input matrix (each row is an observation) and a same-length location vector.
 return the matrix of unique elements and set the location vector so uniq.row(loc[i])=Y.row(i)
 in the context of algorithm-running this function returns U and sets nU as a side-effect
 uniqCnt gets 'trimmed' at the end.
 */
MatrixXd uniq(MatRef Y,VectorXi& uniqCnt,Eigen::Ref<Eigen::VectorXi> inxMap);

// overloaded version which does not bother with tracking indices
MatrixXd uniq(MatRef Y,VectorXi& uniqCnt);

// sample covariance matrix
MatrixXd cov(MatRef X);

// considered using an Eigen::Map to just 'view' the same data as a vector; however, if we'd like to
// put several matrices into a single vector, they do need to be copied to maintain contiguous storage
// so this futzing around is necessary even if everything is const
VectorXd vectorize(MatRef M);

MatrixXd unVectorize(VecRef vec,const int ncols);

// the rankUpdate method maps X -> X + alpha*A*A^T for scalar (double) alpha
// note that this returns a selfadjointView, not a 'regular' matrix.
inline MatrixXd AtA(MatRef A){
  const int p(A.cols());
  return MatrixXd(p,p).setZero().selfadjointView<Lower>().rankUpdate(A.adjoint());
}

// compute OLS beta using LLT decomposition.
MatrixXd betaFit(MatRef Y,MatRef X,bool add_intercept);

inline Eigen::MatrixXd cbind1(MatRef X){
    Eigen::MatrixXd res(X.rows(),X.cols()+1);
    res.col(0) = VectorXd::Constant(X.rows(),1.0);
    res.rightCols(X.cols()) = X;
    return res;
}

Eigen::MatrixXd cbind2(MatRef X,MatRef Y,bool intercept);

#endif // end rcppeigen_utils
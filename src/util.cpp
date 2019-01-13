// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppEigen.h>
#include "util.h" // header-only functions that *other* functions in here depend on

/*
 * Various utility functions which are not specific to the particular model
 * implemented in mdrm/spd
 */

using namespace Eigen;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;

// set the values of x to zero at locations specified by ii (SORTED)
void setZeros(Eigen::Ref<Eigen::VectorXd> x,const Eigen::Ref<const Eigen::VectorXi>& ii){
  const int L = x.size() - 1; int ix;
  for(int i=0;i<ii.size();i++){
    ix = ii(i);
    if(ix >= 0 && ix < L) x(ix) = 0.; // skip out of range
  }
}

// sampling N items from the range [lo,...,hi], inclusive of endpoints, *with* replacement (this is just for bootstrap where we always sample w/replacement)
ArrayXi sample(int N,int lo,int hi){
	ArrayXd sample = (Rcpp::as<ArrayXd>(Rcpp::runif(N,lo,hi+1)));
	return sample.template cast<int>();
}

/*take an input matrix (each row is an observation) and a same-length location vector.
 return the matrix of unique elements and set the location vector so uniq.row(loc[i])=Y.row(i)
 in the context of algorithm-running this function returns U and sets nU as a side-effect
 uniqCnt gets 'trimmed' at the end.
 */
MatrixXd uniq(MatRef Y,VectorXi& uniqCnt,Eigen::Ref<Eigen::VectorXi> inxMap){
  const int N = Y.rows(), P = Y.cols();
  uniqCnt.setZero();
  MatrixXd U = MatrixXd::Random(N,P);
  int curry = 0; bool match;
  for(int i=0;i<N;i++){
    match = false;
    for(int j=0;j<curry;j++){
      if(Y.row(i)==U.row(j)){ // we could put an epsilon-close criterion here to make a fuzzy comparison
        uniqCnt(j)++;
        match = true;
        inxMap(i) = j;
        break;
      }
    }
    if(!match){
      U.row(curry) = Y.row(i);
      inxMap(i) = curry;
      uniqCnt(curry++)++;
    }
  }
  uniqCnt.conservativeResize(curry);
  return U.block(0,0,curry,P);
}

// overloaded version which does not bother with tracking indices
MatrixXd uniq(MatRef Y,VectorXi& uniqCnt){
	const int N = Y.rows();
	uniqCnt.setZero();
	MatrixXd U  = MatrixXd::Random(N,Y.cols());  // NOTE: for 'safety', do a random initialization
	int curry = 0; bool match;
	for(int i=0;i<N;i++){
		match = false;
		for(int j=0;j<curry;j++){
			if(Y.row(i)==U.row(j)){ // we could put an epsilon-close comparison here to make a fuzzy comparison
				uniqCnt(j)++;
				match = true;
				break;
			}
		}
		if(!match){
			U.row(curry) = Y.row(i);
			uniqCnt(curry++)++; // get the index for nU first, then increment it (postfix)
		}
	}
	uniqCnt.conservativeResize(curry);
	return U.topRows(curry);
}

// sample covariance matrix
MatrixXd cov(MatRef X){
  const int N = X.rows();
  MatrixXd sscp( X.transpose()*X );
  VectorXd xmu = (1./N)*X.colwise().sum();
  sscp -= N*xmu*xmu.transpose();
  return (1./(N-1))*sscp;
}

VectorXd sign(VecRef V){ // by default, code sign(0) as 0
	ArrayXd res(V.size());
	for(int j=0;j<V.size();j++){
		res(j) = V(j)==0 ? 0 : std::abs(V(j))/V(j);
	}
	return res;
}

VectorXd vectorize(MatRef M){
  const int P = M.rows(), Q = M.cols();
  if(Q==1) return M;
  VectorXd res(P*Q);
  for(int i=0;i<Q;i++){
    res.segment(i*P,P) = M.col(i);
  }
  return res ;
}

MatrixXd unVectorize(VecRef vec,const int ncols){
  const size_t r = vec.size()/ncols;
  if(r <= 1) return vec.transpose();
  MatrixXd mat(r,ncols);
  for(int i=0;i<ncols;i++){
    mat.col(i) = vec.segment(i*r,r);
  }
  return mat;
}

MatrixXd betaFit(MatRef Y,MatRef X,bool add_intercept = false){
  MatrixXd X1;
  if(add_intercept){
    X1 = MatrixXd(X.rows(),X.cols()+1); // initial estimate needs X and a column of ones
	  X1.col(0) = VectorXd::Constant(X.rows(),1.0);  // intercept
	  X1.rightCols(X.cols()) = X;
  } else {
    X1 = X;
  }
  const Eigen::LLT<MatrixXd> thellt(AtA(X1));
  return thellt.solve(X1.adjoint()*Y).bottomRows(X.cols());
}

Eigen::MatrixXd cbind2(MatRef X,MatRef Y,bool intercept = false){
    const int N = X.rows(), P = X.cols() + Y.cols() + (intercept ? 1 : 0);
    if(N != Y.rows()) Rcpp::stop("cbind2: incorrect sizes of X and Y");
    MatrixXd res(N,P);
    res.rightCols(Y.cols()) = Y;
    if(intercept){
        res.col(0) = VectorXd::Constant(X.rows(),1.0);
        res.block(0,1,N,X.cols()) = X;
    } else {
        res.leftCols(X.cols()) = X;
    }
    return res;
}

// line minimization routines. There are two approaches, here labeled
// 'brent_linemin' (Brent's method) and 'cubic_linemin' (using cubic approximation)
// former seems more stable for high-dimensional problems
// this version is for use with Rcpp, but the only depedency is printing traces/errors while fitting,
// i.e. Rcpp::Rcout


#ifndef line_min_methods

#define line_min_methods
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "util.h"
#include <cmath> // std::abs
#include <limits>
#include <sstream>
#include <type_traits> // is_arithmetic

using VecRef = const Eigen::Ref<const Eigen::VectorXd>&;
using MatRef = const Eigen::Ref<const Eigen::MatrixXd>&;
using refVec = Eigen::Ref<Eigen::VectorXd>;

// *****
// helpers
// *****

// this is decidedly *not* the standard 'sign' function but is a tuft of vestigial kruft from NR
template<typename T>
inline T SIGN(const T a,const T b){
  static_assert(std::is_arithmetic<T>::value,"SIGN expects a single, numeric argument.");
	return b > 0. ? std::abs(a) : -std::abs(a);
}

// T is typically a double in this context: bookkeeping for functions 'brent' and 'mnbrak' below
// initial value of a is discarded while d is untouched.
template<typename T>
inline void SHIFT(T& a,T& b,T& c,const T& d){
	a=b; b=c; c=d;
}

// calculation of an opaque 'test' value which occurs in the cubic linesearch
inline double calc_test(VecRef x,VecRef y){
  Eigen::ArrayXd z = x.array().abs()/y.array().abs().max(1.0);
  return z.maxCoeff();
}

// track errors and deal with them sensibly
// the functions which might "throw" these errors are cubic_linemin and brent.
// maybe better to combine these to a static map?
enum dfpmin_error : unsigned int {
  NONE,
  MAX_ITERATION,
  POSITIVE_SLOPE,
  NON_FINITE_VALUE,
  BRENT_ERROR,
  CUBIC_ERROR
};

struct dfpmin_err_msg {
  static const char* messages[];
};
const char* dfpmin_err_msg::messages[] = {
  "too many iterations in dfpmin",
  "roundoff error in cubic linesearch (positive slope)",
  "NaN/infinite value in parameters",
  "error in Brent linemin routine",
  "error in Cubic linemin routine"
};

// the return value of dfpmin
struct funcMin{
  dfpmin_error error;
  int iterations;
  double min_value;
  Eigen::VectorXd arg_min,gradient;
};

// bracket an set of initial values with opposite signs for use in Brent's methods
// the 'UnaryOp' argument may be passed as a lambda, etc. but must have overloaded double operator()(double)
template<typename UnaryOp>
void mnbrak(double *ax,double *bx,double *cx,double *fa,double *fb,double *fc,UnaryOp func){
  constexpr double GOLD = 1.618034, GLIMIT = 100., TINY = 1.0e-20;
  double ulim,u,r,q,fu,dum;
  *fa = func(*ax);
  *fb = func(*bx);
  if (*fb > *fa) {  // switch direction in this case
    SHIFT(dum,*ax,*bx,dum); // equivalent to swap() in this case
    SHIFT(dum,*fb,*fa,dum);
  }
  *cx = (*bx) + GOLD*(*bx - *ax);
  *fc = func(*cx);
  while (*fb > *fc) { // potential for infinite loop is right here?
    r = (*bx - *ax)*(*fb - *fc);
    q = (*bx - *cx)*(*fb - *fa);
    u = (*bx) - ((*bx - *cx)*q - (*bx - *ax)*r)/(2.0*SIGN(std::max(std::abs(q-r),TINY),q-r));
    ulim = (*bx) + GLIMIT*(*cx-*bx);
    if ((*bx-u)*(u-*cx) > 0.0){
      fu = func(u);
      if (fu < *fc){
        *ax = *bx;
        *bx = u;
        *fa = *fb;
        *fb = fu;
        return;
      } else if (fu > *fb) {
        *cx = u;
        *fc = fu;
        return;
      }
      u = (*cx) + GOLD*(*cx - *bx);
      fu = func(u);
    } else if ((*cx - u)*(u - ulim) > 0.0){
      fu = func(u);
      if (fu < *fc){
        SHIFT(*bx,*cx,u,*cx + GOLD*(*cx - *bx));
        SHIFT(*fb,*fc,fu,func(u));
      }
    } else if ((u - ulim)*(ulim - *cx) >= 0.0){
      u = ulim;
      fu = func(u);
    } else {
      u = (*cx) + GOLD*(*cx - *bx);
      fu = func(u);
    }
    SHIFT(*ax,*bx,*cx,u);
    SHIFT(*fa,*fb,*fc,fu);
  }
}

// first 3 args are bounds attained from mnbrak
template<typename UnaryOp>
double brent(const double ax,const double bx,const double cx,const double tol,
  double* xmin,UnaryOp func,dfpmin_error& err){
  constexpr size_t ITMAX{100u};
  constexpr double CGOLD{0.3819660}, ZEPS{1.0e-10};
  err = dfpmin_error::NONE;

  const double f0 = func(bx);
  double d{0.}, e{0.}, etemp;
  double u, v{bx}, w{bx}, x{bx};  // search points
  double fu, fv{f0}, fw{f0}, fx{f0}; // hold function value @ search points
  double p,q,r,tol1,tol2,xm; // other miscellaneous bookkeeping
  double a = std::min(ax,cx), b = std::max(ax,cx);

  for(size_t iter = 0;iter < ITMAX;iter++) {
    xm = 0.5*(a + b);
    tol1 = tol*std::abs(x) + ZEPS;
    tol2 = 2.0*tol1;
    if (std::abs(x-xm) <= (tol2 - 0.5*(b-a))) {
      *xmin = x;
      return fx;
    }
    if (std::abs(e) > tol1){
      r = (x - w)*(fx - fv);
      q = (x - v)*(fx - fw);
      p = (x - v)*q - (x - w)*r;
      q = 2.0*(q - r);
      if (q > 0.0) p = -p;
      q = std::abs(q);
      etemp = e;
      e = d;
      if(std::abs(p) >= std::abs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x)){
        d = CGOLD*(e = (x >= xm ? a-x : b-x));
      } else {
        d = p/q;
        u = x + d;
        if (u-a < tol2 || b-u < tol2)
          d = SIGN(tol1,xm-x);
      }
    } else {
      e  = (x >= xm) ? a - x : b - x;
      d = CGOLD*e;
    }
    u = (std::abs(d) >= tol1 ? x + d : x + SIGN(tol1,d));
    fu = func(u);
    if(fu <= fx){
      if (u >= x) a = x; else b = x;
      SHIFT(v,w,x,u);
      SHIFT(fv,fw,fx,fu);
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) {
        v = w;
        w = u;
        fv = fw;
        fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u;
        fv = fu;
      }
    }
  }
  err = dfpmin_error::BRENT_ERROR;
  *xmin = x; // caller of brent needs access to this
  return fx;
}

// BFGS update of inverse Hessian approximation
// update of Hessian matrix in quasi-Newton algorithm. Preserves positive definiteness
// 'dth' is change in parameters (delta theta), 'dgr' is change in gradient
void BFGS(Eigen::Ref<Eigen::MatrixXd> H,VecRef dth,VecRef dgr){
  const Eigen::VectorXd Hy(dgr.adjoint()*H.selfadjointView<Eigen::Lower>());
  const double rho = 1./dgr.dot(dth);
  const double k = rho*rho*dgr.dot(Hy) + rho;
  H.selfadjointView<Eigen::Lower>().rankUpdate(dth,Hy,-rho); // rank 2 update with distinct u and v. Note capitalization format
  H.selfadjointView<Eigen::Lower>().rankUpdate(dth,k); // rank 'K' update with a K-column matrix u (and alpha scalar)
}

/*
 * Arguments: xold are the current parameter values, and fold is the current objective function value
 * g is the gradient at xold
 * p is the candidate search direction, but usually we need to contract it towards xold to avoid overshooting
 * x holds the updated point upon return (the goal is func(x) < func(xold))
 */
template <class T>
dfpmin_error cubic_linemin(VecRef xold, const double fold, VecRef g,refVec p,refVec x,double& f,
  double stpmax,const T& func) {

  constexpr double alpha{1.0e-4}, TOLX{std::pow(std::numeric_limits<double>::epsilon(),0.67)};
  constexpr int maxit = 200; // provide some guard against infinite loop

  const double p_norm = p.norm();
  if(stpmax <= 0.) stpmax = 1.0;
  if(p_norm > stpmax) p *= stpmax/p_norm;
  const double slope = g.dot(p);
  if(slope >= 0.) return dfpmin_error::POSITIVE_SLOPE;

  double rhs1,rhs2,tmplam,a,lambda = 1.0,alam2 = 0.0,b,disc,f2 = 0.0;
  const double min_lambda = TOLX/calc_test(p,xold);
  
  for(int i=0;i<maxit;++i){
    x = xold + lambda*p;
    f = func(x);
    if(lambda < min_lambda && i > 5){
      Rcpp::Rcout << "lambda issues at i = " << i << " and lambda = " << lambda << "!\n";
      x = xold;
      return dfpmin_error::CUBIC_ERROR;
    } else if(f <= fold + alpha*lambda*slope){ // what we want: sufficient function decrease
      return dfpmin_error::NONE;
    } else {  // keep searching: calculate a polynomial interpolation to guess where a local min. might be
      if(lambda == 1.0) tmplam = -slope/(2.0*(f-fold-slope)); // first time through; quadratic approximation
      else {                                                // otherwise, do a cubic interpolation when we have sufficient data
        double l2 = lambda*lambda, al2 = alam2*alam2;
        rhs1 = f - fold - lambda*slope;
        rhs2 = f2 - fold - alam2*slope;
        a = (rhs1/l2 - rhs2/al2)/(lambda-alam2);
        b = (-alam2*rhs1/l2 + lambda*rhs2/al2)/(lambda-alam2);
        if (a == 0.0) tmplam = -slope/(2.0*b);
        else {
          disc = b*b - 3.0*a*slope;
          if (disc < 0.0) tmplam = 0.5*lambda;
          else if (b <= 0.0) tmplam = (-b + sqrt(disc))/(3.0*a);
          else tmplam = -slope/(b + sqrt(disc));
        }
        if(tmplam > 0.5*lambda) tmplam = 0.5*lambda;
      }
    }
    alam2 = lambda;
    f2 = f;
    lambda = std::max(tmplam,0.1*lambda); // limit the rate of shrinkage to 90% per round
  }
  return dfpmin_error::MAX_ITERATION; // final return path is an error path
}

template<typename T>
class brent_linemin{
public:
  brent_linemin(const T& _funct,double _ftol = 0.) : funct(_funct) {
    ftol = _ftol <= 0. ? param.size()*1.e-7 : _ftol;
  }
  double f1dim(double alpha) const;
  // this updates the 'I/O' parameter p0 and returns new objective function value
  double linemin(VectorXd& p0,VecRef search_dir);
private:
  const T funct; // needs operator(VectorXd) and grad() method
  VectorXd param,search_dir;
  static constexpr double TOL = 2.0e-4; // no non-static constexpr, please
  double fv,xmin,ftol;
  double ax,bx,cx,fa,fb,fc;
};

template<typename T>
inline double brent_linemin<T>::f1dim(double alpha) const {
    const VectorXd v(param + alpha*search_dir);
    return funct(v);
}

template<typename T>
double brent_linemin<T>::linemin(VectorXd& p,VecRef dir0){
  param = p;
  search_dir = dir0;
  ax = 0.; bx = 1e-3; cx = 2.0;   // re-set the marker variables
  mnbrak(&ax,&bx,&cx,&fa,&fb,&fc,[&](double a) -> double{ return this->f1dim(a); }); // now ax,bx,cx are (hopefully) filled with something good
  
  dfpmin_error e;
  fv = brent(ax,bx,cx,TOL,&xmin,[&](double a) -> double{ return this->f1dim(a); },e); // xmin is the alpha value which minimized in the search direction
  if(e != dfpmin_error::NONE) Rcpp::stop("Error in Brent linemin routine.");
  double max_dir = search_dir.array().abs().maxCoeff();
  if (max_dir >= 50) xmin = 1.0/max_dir; // lower bound of 0.02, hmmm....
  param += search_dir*xmin;
  p = param;
  return funct(p);
}

/*
 * A simple class to hold the logic of performing line minimization with BFGS updating
 * The line minimization routines are factored out since they're conceptually separate and can be viewed as a black box from this function.
 * So, the point of this is to provide a consistent interface vis-a-vis the other line search approach;
 * then either one can be harnessed by the same dfpmin() code.
 * and might be parallelized easily
 * 
 * Ultimately, this should be refactored to have a second template argument which is the line minimization class instance,
 * so that anything can be plugged in so long as it has some basic API, ex. a search() method which takes the
 * current point + direction and returns a new point.
 */
template<typename T>
funcMin dfpmin(VectorXd& p0,const T& funcd,const Eigen::Ref<const Eigen::VectorXi>& zc,const std::string& method = "Brent",
      int verb = 0,double ftol = 0.,double gtol = 0.,const std::string& conv_method = "func"){
  using namespace Eigen;
  using Rcpp::Rcout;
  constexpr double EPS = 1.0e-10;
  const int ITMAX = 200, n = p0.size();
  bool zeros = zc(0) >= 0;     // do we need to futz around with setting things to zero? (if fitting a constrained model)
  if(zeros) setZeros(p0,zc);

  funcMin R;
  R.error = dfpmin_error::NONE;
  dfpmin_error e;
  brent_linemin<T> BL(funcd,ftol);
  double max_step = std::min(1.,p0.array().abs().maxCoeff());
  int its, psz = std::min(10,n);

  double test, fret = funcd(p0), fp = funcd(p0); // starting value of the objective function
  MatrixXd hessian( MatrixXd::Identity(n,n) );
  VectorXd g(n), oldg(n), hdg(n), p(p0), pnew(p0), pdiff(n);
  funcd.df(p,g); // update g
  if(zeros) setZeros(g,zc);
  VectorXd xi{-g};  // initial candidate search direction
  for(its = 0;its < ITMAX;++its){
    if(verb > 0){
      Rcout << "\n*****\n   Iteration " << its << '\n';
      if(verb > 1){ Rcout << "tail(theta) = " << p.tail(psz).transpose() << '\n'; }
      Rcout << "objective function value: " << fret << '\n';
      Rcout << "gradient norm: " << g.norm() << std::endl;
    }
    if(method == "Brent"){
      fret = BL.linemin(pnew,xi);
    } else {
      e = cubic_linemin(p,fp,g,xi,pnew,fret,max_step,funcd);
    }
    if(any_nan(p)) R.error = dfpmin_error::NON_FINITE_VALUE;
    if(e != dfpmin_error::NONE) R.error = e;
    if(R.error != dfpmin_error::NONE) break;
    // which convergence criterion to use is not adequately explained.
    if (conv_method=="func"){
      test = ftol*(std::abs(fret) + std::abs(fp)+EPS);
      if(2.0*std::abs(fret-fp) <= test) {
        if(verb > 0) Rcout << "\nfret - fp = " << 2.0*std::abs(fret - fp) << ", under convergence criterion of " << test << std::endl;
        break;
      }
    } else {
      test = calc_test(g,p)/std::max(1.0,std::abs(fret));
      if(test < gtol){
        if(verb > 0) Rcpp::Rcout << "Test value " << test << " under gradient tolerance of " << gtol << std::endl;
        break;
      }
    }
    fp = fret;
    if(zeros) setZeros(pnew,zc); // this also sets pdiff(zc) zero
    pdiff = pnew - p;
    p = pnew;
    oldg = g;
    funcd.df(p,g);
    if(zeros) setZeros(g,zc);

    BFGS(hessian,xi,g-oldg); // update (inverse) Hessian approximation
    xi = -g.adjoint()*hessian.selfadjointView<Eigen::Lower>(); // new search direction -f'/f''
  }
  if(its >= ITMAX) R.error = dfpmin_error::MAX_ITERATION;
  R.iterations = its;
  R.arg_min = p; R.min_value = fret;
  if(R.error == dfpmin_error::NONE){
    funcd.df(p,g);
    R.gradient = g;
  }
  return R;
}

#endif // end line_min_methods
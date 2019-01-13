#include <cmath> // M_PI among other math constants
#include <utility>
#include <vector>
#include <type_traits>

#ifndef __GAUHER__
#define __GAUHER__
/* 
    Calculating weights for Gaussian quadrature
    x and w had better have (at least) n elements allocated (but we don't check and this 
    will not throw a std::out_of_range() to interoperate with raw pointers)
    Q: if n < 6 (so m <= 3) and we don't get through the initial if..else stuff, is the quadrature valid?
    NOTE that these returned weights are calibrated according to the so-called """Physicists' polynomials"""
    see https://en.wikipedia.org/wiki/Hermite_polynomials so that they do NOT directly give expected values of
    standard normal functionals.

    Note - can we make this constexpr if N pre-specified? gcc supposed to have constexpr math implementations
*/
template<typename T>
std::pair<std::vector<T>,std::vector<T>> gauher(const size_t n) {
    static_assert(std::is_floating_point<T>::value,"types in gauher must be floating-point");
    std::vector<T> w(n),x(n);
    double p1,p2,p3,pp,z,z1;
    constexpr double EPS = 3.0e-14, PIM4 = 1.0/std::pow(M_PI,0.25);
    constexpr size_t MAXIT = 10;
    const size_t m = (n+1)/2; // due to symmetry we need only go halfway
    const double sqrt2n = std::sqrt(2.0*n);
    for (size_t i=0;i<m;i++){
        if (i == 0) {
            z = std::sqrt(2.0*n+1) - 1.85575*std::pow(2.0*n+1,-0.16667);
        } else if (i == 1){
            z -= 1.14*std::pow(n,0.426)/z;
        } else if (i == 2){
            z = 1.86*z - 0.86*x[0];
        } else if (i == 3){
            z = 1.91*z - 0.91*x[1];
        } else {
            z = 2.0*z - x[i-2];
        }
        for(size_t its=0; its<MAXIT; ++its){
            p1 = PIM4;
            p2 = 0.0;
            for(size_t j=0;j<n;j++){
                p3 = p2;
                p2 = p1;
                p1 = z*std::sqrt(2.0/(j+1))*p2 - std::sqrt(static_cast<double>(j)/static_cast<double>(j+1))*p3;
			}
            pp = sqrt2n*p2;
			z1 = z;
            z = z1 - p1/pp;
            if(std::abs(z-z1) <= EPS) break;
			if(its == MAXIT){throw("gauher exceeded max iterations!");} // error state
        }
        x[i] = z;
        x[n-1-i] = -z;  // n-1, n-2, ..., n-m
        w[i] = 2.0/(pp*pp);
        w[n-1-i] = w[i];
    }
    return std::make_pair(w,x);
}

#endif
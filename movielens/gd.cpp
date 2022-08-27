// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::export]]
List gd(int u_n, int m_n, int factor_n, NumericVector y, 
        double L_rate, int epochs){
  arma::mat P(factor_n, u_n, fill::randu)
  arma::mat Q(factor_n, m_n, fill::randu)
  
  for(int i = 0; i < epochs; i++){
    for(int j = 0; j < y.size(); j++){
      double err = arma::dot(P.col(j), Q.col(j)) - y(j);
      P.col(J) -= 
    }
  }
}

/*** R
timesTwo(42)
*/

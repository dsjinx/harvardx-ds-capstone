// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace arma;

// [[Rcpp::export]]
Rcpp::List gd(Rcpp::NumericVector U_i, 
              Rcpp::NumericVector M_j, 
              Rcpp::NumericVector y, 
              int u_n, int m_n, int factor_n, 
              double L_rate, int epochs){
  
  mat P(factor_n, u_n, fill::randu);
  mat Q(factor_n, m_n, fill::randu);
  
  for(int i = 0; i < epochs; i++){
    for(int j = 0; j < y.size(); j++){
      double err = dot(P.col(U_i(j)), Q.col(M_j(j))) - y(j);
      
      //nabla_P = err * Q.col(M_j(j)) + P.col(U_i(j));
      //nabla_Q = err * P.col(U_i(j)) + Q.col(M_j(j));
      
      P.col(U_i(j)) -= L_rate * (err * Q.col(M_j(j)) + P.col(U_i(j)));
      Q.col(M_j(j)) -= L_rate * (err * P.col(U_i(j)) + Q.col(M_j(j)));
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("P") = P,
                            Rcpp::Named("Q") = Q);
}

/*** R
timesTwo(42)
*/

// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace arma;

// [[Rcpp::export]]
Rcpp::List gd(Rcpp::NumericVector U_i, 
              Rcpp::NumericVector M_j, 
              Rcpp::NumericVector y, 
              int u_n, int m_n, int factor_n, 
              double L_rate, double lambda, int epochs){
  arma_rng::set_seed(3);
  mat P(factor_n, u_n, fill::randu);
  arma_rng::set_seed(4);
  mat Q(factor_n, m_n, fill::randu);
  
  for(int i = 0; i < epochs; i++){
    for(int j = 0; j < y.size(); j++){
      int ui = U_i(j) - 1;
      int mj = M_j(j) - 1;
      double err = dot(P.col(ui), Q.col(mj)) - y(j);
      
      vec nabla_P_temp = err * Q.col(mj) + lambda * P.col(ui);
      vec nabla_Q_temp = err * P.col(ui) + lambda * Q.col(mj);
      
      P.col(ui) -= L_rate * nabla_P_temp;
      Q.col(mj) -= L_rate * nabla_Q_temp;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("P") = P,
                            Rcpp::Named("Q") = Q);
}

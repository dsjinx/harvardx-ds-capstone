// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace arma;

// [[Rcpp::export]]
double gdtune(mat P, mat Q, 
               Rcpp::NumericVector ytr,
               Rcpp::NumericVector ytst,
               Rcpp::NumericVector Uitr,
               Rcpp::NumericVector Uitst,
               Rcpp::NumericVector Mjtr,
               Rcpp::NumericVector Mjtst,
               double L_rate, double lambda, 
               int epochs){
  vec err(ytst.size());
  double se;
  
  for(int i = 0; i < epochs; i++){
    for(int j = 0; j < ytr.size(); j++){
      int ui = Uitr(j) - 1;
      int mj = Mjtr(j) - 1;
      double err = dot(P.col(ui), Q.col(mj)) - ytr(j);
      
      vec nabla_P_temp = err * Q.col(mj) + lambda * P.col(ui);
      vec nabla_Q_temp = err * P.col(ui) + lambda * Q.col(mj);
      
      P.col(ui) -= L_rate * nabla_P_temp;
      Q.col(mj) -= L_rate * nabla_Q_temp;
    }
  }
  
  for(int k = 0; k < ytst.size(); k++){
    int ui = Uitst(k) - 1;
    int mj = Mjtst(k) - 1;
    err(k) = dot(P.col(ui), Q.col(mj)) - ytst(k);
  }
  
  se = dot(err, err);

  return se;
}

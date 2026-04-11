data {
  int<lower=0> N; // number of observations
  int<lower=0> K; // number of betas
  matrix[N, K] X; // predictor matrix (standardized)
  vector[N] y;    // log(Salary)
}

parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
}

model {
  // Priors
  alpha ~ normal(16, 2);
  beta  ~ normal(0, 1);
  sigma ~ normal(0, 1);

  // Likelihood
  y ~ normal(X * beta + alpha, sigma);
}

generated quantities {
  // Posterior predictive distribution
  vector[N] y_rep;    // posterior predictive distribution data - result
  vector[N] log_lik;  // pointwise log-likelihood
  vector[N] residual; // residuals on log scale

  for (n in 1:N) {
    real mu_n = alpha + X[n] * beta;
    y_rep[n]   = normal_rng(mu_n, sigma);
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma);
    residual[n] = y[n] - mu_n;
  }
}

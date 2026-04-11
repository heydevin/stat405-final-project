data {
  int<lower=0> N; // number of observations
  int<lower=0> K; // number of betas
  int<lower=0> J; // number of positions
  matrix[N, K] X; // predictor matrix (standardized)
  vector[N] y;    // log(Salary)
  array[N] int<lower=1, upper=J> pos_id;  // position index for each player
}

parameters {
  // Hyperparameters (group-level)
  real mu_a; // mean of position intercepts
  real<lower=0> tau; // SD of position intercepts

  // Group-level parameters
  vector[J] alpha; // position-specific intercepts

  // Population-level parameters
  vector[K] beta; // shared regression coefficients
  real<lower=0> sigma; // residual SD
}

model {
  // Hyperpriors
  mu_a ~ normal(16, 2);
  tau  ~ normal(0, 1);

  // Group-level priors (partial pooling)
  alpha ~ normal(mu_a, tau);

  // Priors on coefficients
  beta  ~ normal(0, 1);
  sigma ~ exponential(1);

  // Likelihood
  for (n in 1:N) {
    y[n] ~ normal(alpha[pos_id[n]] + X[n] * beta, sigma);
  }
}

generated quantities {
  // Posterior predictive distribution
  vector[N] y_rep;    // posterior predictive distribution data - result
  vector[N] log_lik;  // pointwise log-likelihood
  vector[N] residual; // residuals on log scale

  for (n in 1:N) {
    real mu_n = alpha[pos_id[n]] + X[n] * beta;
    y_rep[n]   = normal_rng(mu_n, sigma);
    log_lik[n] = normal_lpdf(y[n] | mu_n, sigma);
    residual[n] = y[n] - mu_n;
  }
}

functions {
  matrix get_changepoint_matrix(vector t, vector t_change, int N, int S) {
    // Assumes t_change are sorted.
    matrix[N, S] A;
    row_vector[S] a_row;
    int cp_idx;

    // Start with an empty matrix.
    A = rep_matrix(0, N, S);

    // Fill in each row of A.
    for (i in 1:N) {
      cp_idx = 1;
      // Start with an empty row.
      a_row = rep_row_vector(0, S); 
      while ((cp_idx <= S) && (t[i] >= t_change[cp_idx])) {
        a_row[cp_idx] = 1;
        cp_idx = cp_idx + 1;
      }
      A[i] = a_row;
    }
    return A;
  }

  // Logistic trend functions

  vector logistic_gamma(real k, real m, vector delta, vector t_change, int S) {
    vector[S] gamma;  // adjusted offsets, for piecewise continuity
    vector[S + 1] k_s;  // actual rate in each segment
    real m_pr;
    
    // Compute the rate in each segment
    k_s = append_row(k, k + cumulative_sum(delta));

    // Piecewise offsets
    m_pr = m; // The offset in the previous segment
    for (i in 1:S) {
      gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1]);
      m_pr = m_pr + gamma[i];  // update for the next segment
    }
    return gamma;
  }

  real logistic_trend(
    real k,
    real m,
    vector delta, // ???
    real t,
    real cap,
    row_vector A, //row
    vector t_change,
    int S
  ) {
    vector[S] gamma;
    gamma = logistic_gamma(k, m, delta, t_change, S);
    return cap * inv_logit((k + A * delta) * (t - (m + A * gamma)));
  }

  // Linear trend function

  real linear_trend(
    real k,
    real m,
    vector delta,
    real t,
    row_vector A,
    vector t_change
  ) {
      // why diff?
    return (k + dot_product(A, delta)) * t + (m + dot_product(A, -(t_change .* delta)));
  }
}

data {
  int N;                // Number of data points 
  int T;                // Number of time periods
  int<lower=1> K;       // Number of regressors
  int<lower=1> C;       // Number of categories
  vector[N] t;          // Time
  vector[N] cap;        // Capacities for logistic trend
  vector[N] y;          // Time series
  
  int category[N];
  matrix[N,K] X;        // Regressors
  vector[K] sigmas;     // ?Scale on seasonality prior
  int trend_indicator;  // 0 for linear, 1 for logistic
  vector[K] s_a;        // Indicator of additive features
  vector[K] s_m;        // Indicator of multiplicative features
  int S;                // Number of changepoints
  real<lower=0> tau;    // Scale on changepoints prior
  vector[S] t_change;   // Times of trend changepoints
}

transformed data {
  matrix[N, S] A;
  A = get_changepoint_matrix(t, t_change, N, S);
}

parameters {
  real k_mu;                  // Base trend growth rate
  real m_mu;                  // Trend offset
  vector[S] delta_mu;         // Trend rate adjustments
  real k[C];
  real m[C];
  vector[S] delta[C];   // Trend rate adjustments
  real<lower=0> sigma_obs;  // Observation noise
  vector[K] beta;           // Regressor coefficients
}

transformed parameters {
  
  vector[N] y_hat;
  for (i in 1:N){
    print("k, m, delta, t_change", k, m, delta, t_change)
    if (trend_indicator == 0) {
        y_hat[i] = logistic_trend(k[category[i]], m[category[i]], delta[category[i]],
                                  t[i], cap[i], A[i], t_change, S);
    }else{
        y_hat[i] = linear_trend(k[category[i]], m[category[i]], delta[category[i]],
                                  t[i], A[i], t_change);
    }
    y_hat[i] *= (1 + X[i] * (beta .* s_m));
    y_hat[i] += X[i] * (beta .* s_a);
    // indexing matrix considered inefficient
  }
}

model {
  //priors
  k_mu ~ normal(0, 5);
  k ~ normal(k_mu, 5);
  m_mu ~ normal(0, 5);
  m ~ normal(m_mu, 5);
  delta_mu ~ double_exponential(0, tau);
  for (c in 1:C)
    delta[c] ~ double_exponential(delta_mu, tau);
  sigma_obs ~ normal(0, 0.5);
  beta ~ normal(0, sigmas);

  // Likelihood
  y ~ normal(y_hat, sigma_obs);
  // aim to vectorise sampling gradient
}

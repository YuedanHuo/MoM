# Gibbs sampler for bivariate normal

def Gibbs_normal(mu = np.zeros(2), w_1 = 1., w_2 = 1., rho = 0.5, n_iter = 1000):
  X_1 = np.zeros((n_iter,1))
  X_2 = np.zeros((n_iter,1))
  X_1[0] = 10
  X_2[0] = 10 # set it to see the burn in
  for i in range(1,n_iter):
    X_1[i] = mu[0]+ np.random.normal((X_2[i-1]-mu[1])*rho/w_1, (w_1-rho**2/w_1)**.5)
    X_2[i] = mu[1] + np.random.normal((X_1[i]-mu[0])*rho/w_2, (w_2-rho**2/w_2)**.5)
  return X_1, X_2

rho = 0.99 # have tried 0.5, 0.99, 0.999
K = [5,6,7,8,9,10,11,12]
n_estimators = 1000
n_samples = 5000
obm_var = np.empty(n_estimators)
obm_var_sqr = np.empty(n_estimators)
is_estimates = np.empty(n_estimators)
is_estimates_sqr = np.empty(n_estimators)
CI_MoM = np.empty((len(K),n_estimators,2))
CI_MoM_sqr = np.empty((len(K),n_estimators,2))
CI_MoM_large = np.empty((len(K),n_estimators,2))
CI_MoM_sqr_large = np.empty((len(K),n_estimators,2))
CI_IS = np.empty((len(K),n_estimators,2))
CI_IS_sqr = np.empty((len(K),n_estimators,2))

for i in tqdm(range(n_estimators)):
  X_1, _ = Gibbs_normal(rho=rho, n_iter = n_samples)
  X_1 = X_1.reshape(-1)

  #consider the second moment
  X_1_sqr = X_1**2

  obm_var[i] = obm_variance(X_1, math.floor(len(X_1)**.5)) # consider batch size = n*1/2
  obm_var_sqr[i] = obm_variance(X_1_sqr, math.floor(len(X_1_sqr)**.5))
  is_estimates[i] = np.mean(X_1)
  is_estimates_sqr[i] = np.mean(X_1_sqr)

  for k in range(len(K)):
    _,CI_MoM[k,i,:] = MoM_CI(np.ones(len(X_1)), X_1, K[k])
    _,CI_MoM_sqr[k,i,:]= MoM_CI(np.ones(len(X_1_sqr)), X_1_sqr, K[k])
    _,CI_MoM_large[k,i,:] = MoM_CI_largeK(np.ones(len(X_1)), X_1, K[k]*8)
    _,CI_MoM_sqr_large[k,i,:] = MoM_CI_largeK(np.ones(len(X_1_sqr)), X_1_sqr, K[k]*8)

    bound = -stats.norm.ppf(.5**(K[k]))
    CI_IS[k,i,0] = is_estimates[i] - bound*np.sqrt(obm_var[i]/n_samples)
    CI_IS[k,i,1] = is_estimates[i] + bound*np.sqrt(obm_var[i]/n_samples)
    CI_IS_sqr[k,i,0] = is_estimates_sqr[i] - bound*np.sqrt(obm_var_sqr[i]/n_samples)
    CI_IS_sqr[k,i,1] = is_estimates_sqr[i] + bound*np.sqrt(obm_var_sqr[i]/n_samples)


K = np.array(K)
coverage_rate_MoM = np.zeros(len(K))
coverage_rate_MoM_sqr = np.zeros(len(K))
coverage_rate_IS = np.zeros(len(K))
coverage_rate_IS_sqr = np.zeros(len(K))
coverage_rate_MoM_large = np.zeros(len(K))
coverage_rate_MoM_sqr_large = np.zeros(len(K))
for i in range(n_estimators):
  for k in range(len(K)):
    if CI_MoM[k,i,0] <= 0 <= CI_MoM[k,i,1]:
      coverage_rate_MoM[k] += 1
    if CI_MoM_sqr[k,i,0] <= 1 <= CI_MoM_sqr[k,i,1]:
      coverage_rate_MoM_sqr[k] += 1
    if CI_IS[k,i,0] <= 0 <= CI_IS[k,i,1]:
      coverage_rate_IS[k] += 1
    if CI_IS_sqr[k,i,0] <= 1 <= CI_IS_sqr[k,i,1]:
      coverage_rate_IS_sqr[k] += 1
    if CI_MoM_large[k,i,0] <= 0 <= CI_MoM_large[k,i,1]:
      coverage_rate_MoM_large[k] += 1
    if CI_MoM_sqr_large[k,i,0] <= 1 <= CI_MoM_sqr_large[k,i,1]:
      coverage_rate_MoM_sqr_large[k] += 1
coverage_rate_MoM = coverage_rate_MoM/n_estimators
coverage_rate_MoM_sqr = coverage_rate_MoM_sqr/n_estimators
coverage_rate_IS = coverage_rate_IS/n_estimators
coverage_rate_IS_sqr = coverage_rate_IS_sqr/n_estimators
coverage_rate_MoM_large = coverage_rate_MoM_large/n_estimators
coverage_rate_MoM_sqr_large = coverage_rate_MoM_sqr_large/n_estimators



fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].plot(K, coverage_rate_MoM, label = 'MoM confidence interval')
ax[0].plot(K, coverage_rate_IS, label = 'OBM confidence interval')
ax[0].plot(K, coverage_rate_MoM_large, label ='MoM quantile confiedence interval')
ax[0].plot(K, 1-.5**(K-1), label ='intended probability') 
ax[0].set_xlabel("K")
ax[0].set_ylabel("Coverage rate of first moment")
ax[0].set_title(f"Coverage Rate of First Moment: ρ = {rho}, Including Burn-in")
ax[0].legend()

ax[1].plot(K, coverage_rate_MoM_sqr, label = 'MoM confidence interval')
ax[1].plot(K, coverage_rate_IS_sqr, label = 'OBM confidence interval')
ax[1].plot(K, coverage_rate_MoM_sqr_large, label ='MoM quantile confiedence interval')
ax[1].plot(K, 1-.5**(K-1), label ='intended probability')
ax[1].set_xlabel("K")
ax[1].set_ylabel("Coverage rate of second moment")
ax[1].set_title(f"Coverage Rate of Second Moment: ρ = {rho}, Including Burn-in")

ax[1].legend()
plt.show()
import math
n_estimators = 1000
n_samples = 15000
Ks = [5,6,7,8,9,10,11,12]
CI_Large = np.empty((n_estimators,len(Ks),2,4))
CI_small = np.empty((n_estimators,len(Ks),2,4))
CI_is = np.empty((n_estimators,len(Ks),2,4))

# confidence interval for removing data point 1
for i in range(1000):
  theta = thetas_list[i]
  sigma_sqr = sigma_sqrs_list[i]
  log_weights = 0.5*np.log(sigma_sqr) + ((y.to_numpy()[0] - X_scaled.to_numpy()[0] @ theta.T) ** 2)/(2*sigma_sqr)
  log_weights = log_weights - np.max(log_weights)
  weight = np.exp(log_weights)
  weight = weight/np.sum(weight)
  is_estimator = np.sum(weight[:, np.newaxis] * theta, axis=0)



  for j,k in enumerate(Ks):
    bound = - stats.norm.ppf(.5**(k))
    for w in range(4):
      _, CI = MoM_CI_largeK(log_weights, theta[:,w], k*8, log_weight=True)
      CI_Large[i,j,0,w] = CI[0]
      CI_Large[i,j,1,w] = CI[1]

      _, CI_L = MoM_CI(weight, theta[:,w], k)
      CI_small[i,j,0,w] = CI_L[0]
      CI_small[i,j,1,w] = CI_L[1]

      #### correction
      obm_chain = theta[:,w]*weight*n_samples #times n_samples since we normalize it already by dividing by the sum of weights
      var_is = obm_variance(obm_chain, math.floor(n_samples**.5))
      CI_is[i,j,0,w] = is_estimator[w] + bound*np.sqrt(var_is/n_samples)
      CI_is[i,j,1,w] = is_estimator[w] - bound*np.sqrt(var_is/n_samples)


# calculate the coverage rate
rate_mom_large = np.zeros((len(Ks), 4))
rate_mom = np.zeros((len(Ks), 4))
rate_is = np.zeros((len(Ks), 4))
for w in range(4):
  for j,k in enumerate(Ks):
    for n in range(n_estimators):
      if CI_Large[n,j,0,w] <= theta_true[w] <= CI_Large[n,j,1,w]:
        rate_mom_large[j,w] += 1
      if CI_small[n,j,0,w] <= theta_true[w] <= CI_small[n,j,1,w]:
        rate_mom[j,w] += 1
      if CI_is[n,j,0,w] >= theta_true[w] >= CI_is[n,j,1,w]:
        rate_is[j,w] += 1

rate_is = rate_is/n_estimators
rate_mom_large = rate_mom_large/n_estimators
rate_mom = rate_mom/n_estimators

# plot the coverage rate against K
for w in range(4):
  plt.plot(Ks,rate_mom_large[:,w], label = 'MoM_large')
  plt.plot(Ks,rate_mom[:,w], label = 'MoM')
  plt.plot(Ks,rate_is[:,w], label = 'IS')
  plt.plot(Ks, 1-.5**(np.array(Ks)-1), label ='intended probability' )
  plt.title('coverage rate for removing data point 1 in coord %d'%w)
  plt.xlabel('K')
  plt.ylabel('coverage rate')
  plt.legend()
  plt.show()

# plot the length of confidence interval against K
length_mom_large = np.mean(abs(CI_Large[:,:,1,:] - CI_Large[:,:,0,:]), axis = 0)
length_mom = np.mean(abs(CI_small[:,:,1,:] - CI_small[:,:,0,:]), axis = 0)
length_is = np.mean(abs(CI_is[:,:,1,:] - CI_is[:,:,0,:]), axis = 0)
for w in range(4):
  plt.plot(Ks,length_mom_large[:,w], label = 'MoM_large')
  plt.plot(Ks,length_mom[:,w], label = 'MoM')
  #plt.plot(Ks, length_is[:,w], label = 'IS')
  plt.title('length of confidence interval for removing data point 14 in coord %d'%w)
  plt.xlabel('K')
  plt.ylabel('length')
  plt.legend()
  plt.show()


# experiment on removing data point 14
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

n_estimators = 1000
n_samples = 10000
K = 5
Ks = [5,6,7,8,9,10,11,12]
IS = np.zeros((n_estimators, 4))
IS_vars = np.zeros((n_estimators, 4))
MoM = np.zeros((len(K),n_estimators, 4))
CI_IS = np.zeros((len(K),n_estimators,2,4))
CI_MoM = np.zeros((len(K),n_estimators,2,4))
CI_MoM_large = np.zeros((len(K),n_estimators,2,4))
ess = np.empty(n_estimators)
max_weight = np.empty(n_estimators)

for i in range(1000):
  theta = thetas_list[i] # 
  sigma_sqr = sigma_sqrs_list[i]
  log_weights = 0.5*np.log(sigma_sqr) + ((y[13] - X_scaled[13] @ theta.T) ** 2)/(2*sigma_sqr)
  log_weights = log_weights - np.max(log_weights)
  weights = np.exp(log_weights)
  weights = weights/np.sum(weights)

  ess[i] = 1/np.sum(weights**2)

  IS[i,:] = np.sum(weights[:, np.newaxis] * theta, axis=0)
  for w in range(4):
    chains = weights * theta[:,w]
    IS_vars[i,w] = obm_variance(chains, math.floor(len(chains)**.5))

  for j, K in enumerate(Ks):
    bound = - stats.norm.ppf(.5**(k))
    for w in range(4):
      MoM[j,i,w], CI_MoM[j,i,:,w] = MoM_CI(weights, theta[:, w], K)
      _, CI_MoM_large = MoM_CI_largeK(log_weights, theta[:,w], k*8, log_weight=True)
      IS_vars[i,w] = obm_variance(chains, math.floor(len(chains)**.5))
      CI_IS[j,i,0] = IS[i] + bound*np.sqrt(IS_vars[i]/n_samples)
      CI_IS[j,i,1] = IS[i] - bound*np.sqrt(IS_vars[i]/n_samples)

      
# calculate the 'ture' value
#run 12 chains
X_14 = X_scaled.drop(13, axis=0)
y_14 = y.drop(13, axis=0)

thetas_14 = np.empty((72000,4))
for i in range(3):
  results = parallel_gibbs(X_14.to_numpy(), y_14.to_numpy(), n_chains=4, n_iterations=6100)
  j = 0
  for result in results:
    thetas = np.array(result[0])
    thetas = thetas[100:] # cut the burn in
    thetas_14[(i*4+j)*6000:(i*4+j+1)*6000] = thetas
    j+=1
theta_14 = thetas_14.mean(axis=0)
 
 
# plot rate of coverage
rate_is = np.zeros((len(K), 4))
rate_mom = np.zeros((len(K), 4))
rate_mom_large = np.zeros((len(K), 4))
for w in range(4):
  for j,k in enumerate(K):
    for n in range(n_estimators):
      if CI_IS[j,n,0,w] >= theta_14[w] >= CI_IS[j,n,1,w]:
        rate_is[j,w] += 1
      if CI_MoM[j,n,0,w] <= theta_14[w] <= CI_MoM[j,n,1,w]:
        rate_mom[j,w] += 1
      if CI_MoM_large[j,n,0,w] <= theta_14[w] <= CI_IS[j,n,1,w]
        rate_mom_large[j,w] += 1

rate_is = rate_is/n_estimators
rate_mom = rate_mom/n_estimators
rate_mom = rate_mom_large/ n_estimators

for w in range(4):
  plt.plot(K,rate_mom_large[:,w], label = 'MoM_large')
  plt.plot(K,rate_mom[:,w], label = 'MoM')
  plt.plot(K, rate_is[:,w], label='IS')
  plt.plot(K, 1-.5**(np.array(K)), label ='intended probability' )
  plt.title('coverage rate for removing data point 14 in coord %d'%w)
  plt.xlabel('K')
  plt.ylabel('coverage rate')
  plt.legend()
  plt.show()


# violin plot of distribution of estimators
for w in range(4):
  df = pd.DataFrame()
  df = pd.DataFrame({
    'IS': IS[:,w],
    'MoM': MoM[0,:,w] # take K = 5
    })
  sns.violinplot(df)
  plt.axhline(theta_14[w])
  plt.show()
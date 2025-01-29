# experiment on removing data point 14
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

n_estimators = 1000
n_samples = 15000
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

# compute the estimators
# removing data point 1
for i in range(1000):
  theta = thetas_list[i]
  sigma_sqr = sigma_sqrs_list[i]
  log_weights = 0.5*np.log(sigma_sqr) + ((y.to_numpy()[0] - X_scaled.to_numpy()[0] @ theta.T) ** 2)/(2*sigma_sqr)
  log_weights = log_weights - np.max(log_weights)
  weights = np.exp(log_weights)
  weights = weights/np.sum(weights)

  max_weight[i] = np.max(weights)
  ess[i] = 1/np.sum(weights**2)
  ess_mom = get_ess_mom(weights, K)

  IS[i,:] = np.sum(weights[:, np.newaxis] * theta, axis=0)

  weight_split = np.array_split(weights, K)
  theta_split = np.array_split(theta, K, axis = 0)
  mean = np.empty((4,K))
  for k in range (K):
    weight_split[k] = weight_split[k]/np.sum(weight_split[k])
    mean[:,k] = np.sum(weight_split[k][:, np.newaxis] * theta_split[k], axis=0)
  MoM[i,:] = np.median(mean, axis = 1)


# plot the violin plot
for w in range(4):
  df = pd.DataFrame()
  df = pd.DataFrame({
    'IS': IS[:,w],
    'MoM': MoM[:,w]
    })
  sns.violinplot(df)
  plt.axhline(theta_true[w])
  plt.title('violin plot of removing data point 1')
  plt.show()

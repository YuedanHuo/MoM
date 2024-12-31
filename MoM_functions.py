import numpy as np

def MoM_CI(weight,test_function, num_group):
  '''
  return MoM estimators and the corresponding confidence interval
  '''
  assert len(weight) == len(test_function)

  confidence_interval = []
  weight_split = np.array_split(weight, num_group) # consider this for now
  test_function_split = np.array_split(test_function, num_group)

  mean = []

  for i in range(num_group):
    mean.append(np.sum(np.multiply(test_function_split[i],weight_split[i]))/np.sum(weight_split[i]))

  theta_mom = np.median(mean)

  #get the confidence interval
  mean = np.sort(mean)
  confidence_interval.append(mean[0])
  confidence_interval.append(mean[num_group-1])

  return theta_mom, confidence_interval

def MoM_CI_largeK(weight,test_function, num_group, log_weight = True):
  '''
  return MoM estimators and the corresponding quantile confidence interval
  '''
  assert len(weight) == len(test_function)

  confidence_interval = []
  weight_split = np.array_split(weight, num_group) # consider this for now
  test_function_split = np.array_split(test_function, num_group)

  mean = []

  upper = 3*num_group//4
  lower = num_group//4

  for i in range(num_group):
    if log_weight:
      weight_split[i] = np.exp(weight_split[i]-np.max(weight_split[i]))
      weight_split[i] = weight_split[i]/np.sum(weight_split[i])
      mean.append(np.sum(np.multiply(test_function_split[i],weight_split[i])))
    else:
      weight_split[i] = weight_split[i]/np.sum(weight_split[i])
      mean.append(np.sum(np.multiply(test_function_split[i],weight_split[i])))

  theta_mom = np.median(mean)

  #get the confidence interval
  mean = np.sort(mean)
  confidence_interval.append(mean[lower-1])
  confidence_interval.append(mean[upper-1])

  return theta_mom, confidence_interval
# normal examples

epsilons = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #only consider epsilon from 0 to less than 1 otherwise the variance is not bounded and no CLT
K = [5,6,7,8,9,10,11,12]
n_estimations = 10000
n_samples = 10000
CI_mom = np.empty((len(epsilons),len(K),n_estimations,2 )) 
CI_mom_large = np.empty((len(epsilons),len(K),n_estimations,2 ))
CI_is = np.empty((len(epsilons),len(K),n_estimations,2 ))
theta_is = np.empty((len(epsilons),n_estimations)) 
theta_mom = np.empty((len(epsilons),n_estimations))
var_is = np.empty((len(epsilons),n_estimations))
confidence_level = np.empty(len(K))

for i in range(len(epsilons)):
  epsilon = epsilons[i]
  for j in range(n_estimations):
    samples = np.random.normal(0,1/(1+epsilon)**0.5,n_samples)
    test_functions = samples**2 # consider second moment

    # normalize the weight in log space to prevent overflow
    log_weight = (epsilon*samples**2)/2
    weight = np.exp(log_weight - np.max(log_weight))
    weight = weight / np.sum(weight)

    theta_is[i,j] = np.sum(test_functions*weight)/np.sum(weight)
    var_is[i,j] = get_asym_var(test_functions,weight,theta_is[i,j])
    theta_mom[i,j], _ = MoM_CI(weight,test_functions,num_group=K[k])

    for k in range(len(K)):
      confidence_level[k] = 1 - 2**(-K[k]+1)
      bound = stats.norm.ppf((1 - confidence_level[k]) / 2)
      _, CI_mom[i,k,j,:] = MoM_CI(weight,test_functions,num_group=K[k])
      _, CI_mom_large[i,k,j,:] = MoM_CI_largeK(weight,test_functions,num_group=K[k]*8)
      CI_is[i,k,j,:] = [theta_is[i,j]+bound*np.sqrt(var_is[i,j]/n_samples),theta_is[i,j]-bound*np.sqrt(var_is[i,j]/n_samples)]


# violin plots of estimators
df = pd.DataFrame()  # Initialize an empty DataFrame for each n
for i in range(len(epsilons)):
    is_ = theta_is[i,:]  # Extract IS data
    mom_ = theta_mom[i,:]  # Extract MoM data

    # Create a DataFrame for the current epsilon
    df_e = pd.DataFrame({
        'IS with epsilon = %.2f' % epsilons[i]: is_,
        'IS-MoM with epsilon = %.2f' % epsilons[i]: mom_,
    })

        # Concatenate the new DataFrame with the existing one and assign it back to df
    df = pd.concat([df, df_e], axis=0)

plt.figure(figsize=(100, 10))
sns.violinplot(data=df, inner='quartile')
plt.title('Violin Plot of Estimators Distributions for n=%d moment' % n)  # Title for each plot
plt.axhline(n-1, color='black', linestyle='--', label ='True Value')  # Add a horizontal line at
plt.ylabel('Values')  # Y-axis label
plt.show()  # Display the plot

# plot rate of coverage
coverage_rate_is = np.zeros((len(epsilons), len(K)))
coverage_rate_mom = np.zeros((len(epsilons), len(K)))
coverage_rate_mom_large = np.zeros((len(epsilons), len(K)))
for i in range(len(epsilons)):
  for k in range(len(K)):
    for j in range(n_estimations):
      if CI_is[i,k,j,0] <= 1 and CI_is[i,k,j,1] >= 1:
        coverage_rate_is[i,k] += 1
      if CI_mom[i,k,j,0] <= 1 and CI_mom[i,k,j,1] >= 1:
        coverage_rate_mom[i,k] += 1
      if CI_mom_large[i,k,j,0] <= 1 and CI_mom_large[i,k,j,1] >= 1:
        coverage_rate_mom_large[i,k] += 1
coverage_rate_is = coverage_rate_is / n_estimations
coverage_rate_mom = coverage_rate_mom / n_estimations
coverage_rate_mom_large = coverage_rate_mom_large / n_estimations

fig,ax = plt.subplots(len(epsilons),1, figsize = (10,100))
for i in range(len(epsilons)):
  ax[i].plot(K, coverage_rate_is[i,:], label = 'IS CI')
  ax[i].plot(K, coverage_rate_mom[i,:], label = 'MoM CI')
  ax[i].plot(K, coverage_rate_mom_large[i,:], label = 'MoM-quantile CI')
  ax[i].plot(K, confidence_level, label = 'Intended Probability')
  ax[i].set_title('epsilon = '+str(epsilons[i]))
  ax[i].set_xlabel('K')
  ax[i].set_ylabel('Coverage rate')
  ax[i].legend()
# set up data 
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import invgamma, invwishart
import numpy as np


data = sm.datasets.stackloss.load_pandas().data
X = data.drop('STACKLOSS', axis=1)
X.insert(0, 'Ones', 1) # add a column of ones for the intercept
y = data['STACKLOSS']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.iloc[:, 1:])  # Do not scale the intercept column
X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])  # Add the intercept back
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Gibbs sampler
def inverse_gamma(a, b):
  return 1/np.random.gamma(a, 1/b)

def gibbs_chain(X,y,n_iterations = 100):
    '''
   generate samples from full posterior by Gibbs sampler of depth 200

    '''
    X = X.to_numpy()
    y = y.to_numpy()

    # Initialize from the prior
    thetas = []
    SIGMAs =[]
    sigma_sqrs = []

    #sigma_sqr = inverse_gamma(a=1, b=1)  # sigma_square ~ IG(a,b), a=b=1
    sigma_sqr = inverse_gamma(a=1, b=1) # we don't have to change this one
    SIGMA = invwishart.rvs(df=4 , scale= np.diag([90000, 9, 9, 9])) # SIGMA ~ IW(v,R)
    theta = np.random.multivariate_normal(mean=np.zeros(4), cov=SIGMA)  # theta ~ N(0,SIGMA)

    #calculate X transpose X
    XtX = X.T @ X


    for i in range(n_iterations):
        # Step 1: Update theta
        inv_SIGMA = np.linalg.inv(SIGMA)
        cov = np.linalg.inv(XtX / sigma_sqr + inv_SIGMA)
        mean = cov @ (X.T @ y / sigma_sqr)
        theta = np.random.multivariate_normal(mean=mean, cov=cov)

        # Step 2: Update sigma_sqr
        #alpha = 1 + 0.5 * y.shape[0]
        alpha = 1 + 0.5 * y.shape[0]
        residual = y - X @ theta
        #beta = 2 / (residual.T @ residual + 2)
        beta = 2 / (residual.T @ residual + 2)
        sigma_sqr = invgamma.rvs(a=alpha, scale=1/beta)

        # Step 3: updata SIGMA
        R = 4*np.diag([90000, 9, 9, 9]) + np.outer(theta, theta)  # Update scale matrix
        SIGMA = invwishart.rvs(df=4 + 1, scale=R)  # Update SIGMA with new scale and updated degrees of freedom

        thetas.append(theta)
        sigma_sqrs.append(sigma_sqr)
        SIGMAs.append(SIGMA)


    return thetas, sigma_sqrs, SIGMAs

#run chains in parallel
def parallel_gibbs(X, y, n_chains=4, n_iterations=100):
    '''
    Run multiple Gibbs sampler chains in parallel.

    Parameters:
    - X: The design matrix.
    - y: The target vector.
    - n_chains: The number of parallel Gibbs chains to run.
    - n_iterations: The number of iterations for each Gibbs chain.

    Returns:
    - List of (theta, sigma_sqr, SIGMA) tuples for each chain.
    '''
    results = Parallel(n_jobs=n_chains)(delayed(gibbs_chain)(X, y, n_iterations) for _ in range(n_chains))

    return results

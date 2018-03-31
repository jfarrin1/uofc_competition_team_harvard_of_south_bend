
# coding: utf-8

###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


###

ticker_df = pd.read_csv('ticker_data.csv')
factor_df = pd.read_csv('factor_data.csv')


###

ticker_df = ticker_df.sort_values(by=['ticker','timestep'])



mean = []
stdev = []
sharpe = []

return_array = np.zeros((2519,1000))
z = 0

for i in range(1,2520000,2520):
    return_array[:,z] = ticker_df['returns'][i:i+2519]
    mean.append(ticker_df['returns'][i:i+2519].mean())
    stdev.append(ticker_df['returns'][i:i+2519].std())
    sharpe.append(mean[z]/stdev[z])
    z += 1


plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


plt.plot(stdev,mean,'o')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')



###

def rand_weights(n):
    k = np.random.rand(n)
    return k/sum(k)


###

def random_portfolio(returns):
    '''Returns the mean and standard deviation of returns'''
    
    p = np.asmatrix(np.mean(returns,axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
        
    mu = w*p.T
    sigma = np.sqrt(w*C*w.T)
    
    return mu, sigma



###

n_portfolios = 500
means, stds = np.column_stack([
    random_portfolio(return_array) 
    for _ in range(n_portfolios)
])


###

plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')
plt.show()


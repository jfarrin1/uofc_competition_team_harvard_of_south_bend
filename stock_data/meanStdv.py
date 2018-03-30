
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


ticker_df = pd.read_csv('ticker_data.csv')
factor_df = pd.read_csv('factor_data.csv')


# In[7]:


print(factor_df.head())


# In[36]:


ticker_df = ticker_df.sort_values(by=['ticker','timestep'])

#tick_dict = {}
#keys = np.arange(0,1001,1)


mean = []
stdev = []

return_array = np.zeros((2519,1000))
z = 0

for i in range(1,2520000,2520):
    return_array[:,z] = ticker_df['returns'][i:i+2519]
    z += 1
    mean.append(ticker_df['returns'][i:i+2519].mean())
    stdev.append(ticker_df['returns'][i:i+2519].std())
    
#print(mean)
#print(stdev)

plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#plt.plot(ticker_df['timestep'][:2520],ticker_df['returns'][:2520])

#print(ticker_df.head())

plt.plot(stdev,mean,'o')
plt.xlabel('Volatility')
plt.ylabel('Expected Return')

print(return_array)


# In[ ]:


def rand_weights(n):
    k = np.random.rand(n)
    return k/sum(k)


# In[56]:


def random_portfolio(returns):
    '''Returns the mean and standard deviation of returns'''
    
    p = np.asmatrix(np.mean(returns,axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
        
    mu = w*p.T
    sigma = np.sqrt(w*C*w.T)
    
    return mu, sigma


# In[57]:


return_vec = np.random.randn(4,100)
print(return_vec)


# In[58]:


#array1 = np.array([[1,2],[3,4]])

#print(array1[1,0])

# row then column


# In[53]:


#array2 = np.zeros((2,4))


#array2[:,1] = [1,2]
#print(array2)


# In[54]:


n_portfolios = 500
means, stds = np.column_stack([
    random_portfolio(return_array) 
    for _ in range(n_portfolios)
])


# In[55]:


plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')
plt.show()


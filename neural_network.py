from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

 
ticker_df = pd.read_csv('stock_data/ticker_data.csv')
factor_df = pd.read_csv('stock_data/factor_data.csv')

ticker_df = ticker_df.iloc[1000:]
ticker_df.sort_values(['ticker','timestep'], inplace=True)
factor_df = factor_df.iloc[1:]

ticker_df.set_index('timestep', inplace=True)
factor_df.set_index('timestep', inplace=True)
stock_df = ticker_df.join(factor_df, how='left')

stock_df.sort_values(['ticker', 'index'], inplace=True)
stock_df = stock_df.as_matrix()

ticker = stock_df[:,4]
features = stock_df[:,6:]

#ticker = np.genfromtxt('stock_data/ticker_data.csv', delimiter=",")
#features_raw = np.genfromtxt('stock_data/factor_data.csv', delimiter=",")
#ticker = ticker_df.as_matrix()
#ticker = ticker[ticker[:,5].argsort()]
print(ticker[2520:2560])
print('=========')
print(features[2520:2560][:])
#features = np.empty(0, dtype=float)
#for ix in range(1000):
#	features = np.append(features,features_raw[2:][:])
#	for i in range(999):
#		features.insert(features[i][:],i)
print(len(features))
print(len(ticker))
features_train = features[0:1512000]
ticker_train = ticker[0:1512000]
features_test = features[1512000:]
ticker_test = ticker[1512000:]
nn = MLPRegressor(hidden_layer_sizes=(100,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto', learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True, random_state=0, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08) 

n = nn.fit(features_train, ticker_train)

predicted_returns = nn.predict(features_test)

output = np.subtract(predicted_returns,ticker_test)
output = np.absolute(output)

print(output.mean())



#plt.plot(features_train, ticker_train, 'r', features_test, predicted_returns)
#plt.show()
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.scatter(features_train, ticker_train, s=1, c='b', marker="s", label='real')
#ax1.scatter(features_test, predicted_returns, s=10, c='r', marker="o", label='NN Prediction')
#plt.show()

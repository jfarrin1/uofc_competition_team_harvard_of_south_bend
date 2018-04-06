from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt

 
'''ticker_df = pd.read_csv('stock_data/ticker_data.csv')
factor_df = pd.read_csv('stock_data/factor_data.csv')
ticker_df.set_index('timestep', inplace=True)
factor_df.set_index('timestep', inplace=True)
stock_df = ticker_df.join(factor_df, how='left')

returns = stock_df[['returns']]
stock_df.drop('returns')
stock_df = stock_df.as_matrix()
'''

ticker = np.genfromtxt('stock_data/ticker_data_test.csv', delimiter=",")
features = np.genfromtxt('stock_data/factor_data.csv', delimiter=",")

features_train = features[2:1512,:9]
ticker_train = ticker[2:1512,4]
features_test = features[1512:,:9]
ticker_test = ticker[1512:, 4]
nn = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam')

n = nn.fit(features_train, ticker_train)

predicted_returns = nn.predict(features_test)





#plt.plot(features_train, ticker_train, 'r', features_test, predicted_returns)
#plt.show()
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.scatter(features_train, ticker_train, s=1, c='b', marker="s", label='real')
#ax1.scatter(features_test, predicted_returns, s=10, c='r', marker="o", label='NN Prediction')
#plt.show()

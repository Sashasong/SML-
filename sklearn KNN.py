import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
data = np.loadtxt('knn.txt')


index = [i for i in range(len(data))]
np.random.shuffle(index)
data = data[index]

[m,n] = np.shape(data)
X = data[:,0:n-1]
Y = data[:,-1]
print('data is ready')

scaler = MinMaxScaler() 
scaler.fit(X)  
X = scaler.transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 42)

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsClassifier

model1 = KNeighborsRegressor(n_neighbors=25)
model1.fit(X_train, Y_train.astype('int'))
y1_pred = model1.predict(X_test)
score1 = model1.score(X_test, Y_test.astype('int'))

model2 = KNeighborsRegressor(n_neighbors=5, weights = 'distance')
model2.fit(X_train, Y_train.astype('int'))
y2_pred = model2.predict(X_test)
score2 = model2.score(X_test, Y_test.astype('int'))

model3 = KNeighborsRegressor(n_neighbors=5, radius = 500)
model3.fit(X_train, Y_train.astype('int'))
y3_pred = model3.predict(X_test)
score3 = model3.score(X_test, Y_test.astype('int'))
print (score1, score2, score3)


print('MAE= %s' %mean_absolute_error(Y_test.astype('int'),y2_pred))
print('R2 = %f' %r2_score(Y_test.astype('int'),y2_pred))


y_t_train = Y_train
y_p_train = model2.predict(X_train)
y_t_test = Y_test
y_p_test = y2_pred

x1 = [x for x in range(len(y_t_train))]
x2 = [x for x in range(len(y_p_test))]
fig,axs = plt.subplots(2,1,figsize=(18,12))

axs[0].plot(x1,y_t_train,label='Real_SML_train',color='steelblue',
            alpha=0.7)
axs[0].plot(x1,y_p_train,label='Predict_SML_train',color='tomato',
            alpha=0.7,linestyle = ':',linewidth=2)
axs[0].legend()

axs[1].plot(x2,y_t_test,label='Real_SML_test',color='steelblue',alpha=0.7)
axs[1].plot(x2,y_p_test,label='Predict_SML_test',color='lightcoral',alpha=0.7,linestyle = ':',linewidth=2)
axs[1].legend()

axs[1].set_xlabel('Number of sites',fontsize=15)
axs[1].set_ylabel('Thickness of SML(cm)',fontsize=15,y=1)

plt.show()

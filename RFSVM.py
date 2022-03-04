from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy as np


data = np.loadtxt('knn.txt')

index = [i for i in range(len(data))]
np.random.shuffle(index)
data = data[index]

[m,n] = np.shape(data)
X = data[:,0:n-1]
Y = data[:,-1]
print('data is ready')

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = StandardScaler() 
scaler.fit(X)  
X = scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3)


from sklearn.svm import SVR
svr = SVR(kernel ='rbf',degree = 3,coef0 = 0.0,
		tol = 0.001,C = 1.0, epsilon = 0.1,shrinking = True,cache_size = 200,		
		verbose = False,max_iter = -1 )
svr.fit(X_train,Y_train)
y_pred3 = svr.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,explained_variance_score
print('模型得分:{:.2f}'.format(svr.score(X_test,Y_test)))
print('MAE= %s' %mean_absolute_error(Y_test,y_pred3))
print('R2 = %f' %r2_score(Y_test,y_pred3))


rfc_param = RandomForestRegressor(n_estimators=500,max_depth=10,max_features=5,min_samples_leaf=10)
rfc_param.fit(X_train,Y_train)

pred_1 = rfc_param.predict(X_test)
score_r = rfc_param.score(X_test, Y_test)
print("Random Forest:{}".format(score_r))
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,explained_variance_score
print('MAE= %s' %mean_absolute_error(Y_test,pred_1))
print('R2 = %f' %r2_score(Y_test,pred_1))


from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=200,min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3)
gbr.fit(X_train,Y_train)
pred_2 = gbr.predict(X_test)
score_g = gbr.score(X_test, Y_test)
print("GradientBoostingRegressor:{}".format(score_g))
print('MAE= %s' %mean_absolute_error(Y_test,pred_2))
print('R2 = %f' %r2_score(Y_test,pred_2))


from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd



n_folds = 6  
model_names = ['SVR','RF', 'GBR']  
model_dic = [svr,rfc_param, gbr]  
cv_score_list = []  
pre_y_list = []  

for model in model_dic:  
    scores = cross_val_score(model, X, Y, cv=n_folds) 
    cv_score_list.append(scores) 
    pre_y_list.append(model.fit(X, Y).predict(X))  
    

n_samples, n_features = X.shape  
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score] 
model_metrics_list = []  

for i in range(3): 
    tmp_list = []  
    for m in model_metrics_name:  
        tmp_score = m(Y, pre_y_list[i]) 
        tmp_list.append(tmp_score)  
    model_metrics_list.append(tmp_list)  
    
    
df1 = pd.DataFrame(cv_score_list, index=model_names) 
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2']) 

print ('samples: %d \t features: %d' % (n_samples, n_features))  
print (70 * '-')  
print ('cross validation result:') 
print (df1)  
print (70 * '-')  
print ('regression metrics:')  
print (df2)  
print (70 * '-') 
print ('short name \t full name')  
print ('ev \t explained_variance')
print ('mae \t mean_absolute_error')
print ('mse \t mean_squared_error')
print ('r2 \t r2')
print (70 * '-') 


plt.figure()  
plt.plot(np.arange(X.shape[0]), Y, color='dimgray', label='true SML',linewidth = '1.5')  
color_list = ['orange', 'steelblue', 'coral', 'y']  
linestyle_list = ['-', '.', 'o', 'v'] 
linewidth_list = ['1','1','0.5','1']

for i, pre_y in enumerate(pre_y_list):  
    plt.plot(np.arange(X.shape[0]), pre_y_list[i], color_list[i], label=model_names[i]) 
plt.title('Result Comparison')  
plt.legend(loc='upper right')  
plt.ylabel('Real and predicted SML')  
plt.show('dpi = 600') 

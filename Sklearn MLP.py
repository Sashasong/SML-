from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
print (__doc__)

data = np.loadtxt ('SML.txt')

index = [i for i in range (len(data))]
np.random.shuffle(index)
data = data[index]

[m,n] = np.shape(data)
input = data [:,0:n-1]
output = data[:,-1]

scaler = StandardScaler()
scaler.fit(input)
input = scaler.transform(input)

input_train, input_test, output_train, output_test = train_test_split(input, output, test_size = 0.3, random_state = 0)
clf = MLPRegressor(
    hidden_layer_sizes = (20,40,10),activation = 'relu',
    solver = 'sgd', alpha = 1e-4, batch_size = 400,
    learning_rate = 'constant', learning_rate_init=1e-4,
    power_t=0.5, max_iter=5000, shuffle=True,
    random_state=1, tol=0.0001, verbose=False,
    warm_start=False, momentum=0.9,
    nesterovs_momentum=True,
    early_stopping=False,beta_1=0.9,
    beta_2=0.999, epsilon=1e-06
)

clf.fit(input_train, output_train)
y_pred = clf.predict(input_test)
print('模型得分:{:.2f}'.format(clf.score(input_test,output_test)))
print('MAE= %s' %mean_absolute_error(output_test,y_pred))

pp = r2_score(output_test,y_pred)
print('R2 = %f' %pp

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd

      
n_folds = 6
model_names = ['MLP']
model_dic = [clf]
cv_score_list = []
pre_y_list = []
      
for model in model_dic:
      scores = cross_val_score(model, input, output, cv=n_folds)
      cv_score_list.append(scores)
       pre_y_list.append(model.fit(input, output).predict(input))
      
n_samples, n_features = input.shape
model_metrics_name = [ mean_absolute_error, r2_score]   
model_metrics_list = []
      
for i in range(1): 
      tmp_list = []
      for m in model_metrics_name:
          tmp_score = m(output, pre_y_list[i])
          tmp_list.append(tmp_score) 
      model_metrics_list.append(tmp_list)
      
df1 = pd.DataFrame(cv_score_list, index=model_names)   
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['mae', 'r2'])
      
print ('samples: %d \t features: %d' % (n_samples, n_features))
print (70 * '-')  
print ('cross validation result:') 
print (df1) 
print (70 * '-')  
print ('regression metrics:')  
print (df2)  
print (70 * '-') 
print ('short name \t full name')  
print ('mae \t mean_absolute_error')
print ('r2 \t r2')
print (70 * '-')      
      
y_p_train = chouchou(clf.predict(input_train))
y_t_train = output_train
y_p_test = y_pred
y_t_test = output_test

x_train = [x for x in range(len(y_t_train))]
x_test = [x for x in range(len(y_p_test))]
fig,axs = plt.subplots(2,1,figsize=(18,12))

axs[0].plot(x_train,y_t_train,label='Real_SML_train',color='steelblue',
            alpha=0.7)
axs[0].plot(x_train,y_p_train,label='Predict_SML_train',color='tomato',
            alpha=0.7,linestyle = ':',linewidth=2)
axs[0].legend()

axs[1].plot(x_test,y_t_test,label='Real_SML_test',color='steelblue',alpha=0.7)
axs[1].plot(x_test,y_p_test,label='Predict_SML_test',color='lightcoral',alpha=0.7,linestyle = ':',linewidth=2)
axs[1].legend()

axs[1].set_xlabel('Number of sites',fontsize=15)
axs[1].set_ylabel('Thickness of SML(cm)',fontsize=15,y=1)

print('Programe is OK!')
     

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
    hidden_layer_sizes = (20,30,10),activation = 'relu',
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

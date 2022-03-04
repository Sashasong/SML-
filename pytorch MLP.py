import os
import torch
import matplotlib
import numpy as np
import pandas as pd
import torch.nn as nn
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
       
        self.fc1 = nn.Sequential(
            nn.Linear(5, 20),
            nn.ReLU())
        
        self.fc2 = nn.Sequential(
            nn.Linear(20, 40)
            )
       
        self.fc3 = nn.Sequential(
            nn.Linear(40, 10)
            )
        
        self.fc4 = nn.Sequential(
            nn.Linear(10, 1)
            )
        
    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        y = self.fc3(y)
        y = self.fc4(y)
        return y
         
def train(data, testData, mean, std):

    
    save_path = "C:/Users/86131/Desktop/chouchou1"

    epochs = 1000
    batchSize = 499
    learning_rate = 0.001
    tempLoss = 0
    losss = []
    
    model = Model()
    criterion = nn.L1Loss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        i = 0
        tempLoss = 0
        np.random.shuffle(data)
        
        x_train_data = data[:,0:5]
        y_train_target = data[:,5:]
    
        while i+batchSize < data.shape[0]:
        
            input = x_train_data[i:i+batchSize, :]
            input = torch.from_numpy(input)
            label = y_train_target[i:i+batchSize]
            label = torch.from_numpy(label)
            
            output = model(input)  
        
            loss = criterion(output, label)
            
            losss.append(loss.item())
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            i = i + batchSize
            
        if epoch % 10 == 0:

            torch.save(model, save_path + '/models/model_{}'.format(epoch))
            print("epoch:  ", epoch)
            #temp =  test(testData, '/models/model_{}'.format(epoch), mean, std)
    
    plt.plot(np.arange(len(losss)), losss, color='steelblue',linewidth='2')
   
    plt.savefig(os.getcwd()+"/loss.png",dpi=600)
    
def test(testData, modelName, mean, std):
    x_test_data = testData[:,0:5]
    y_test_target = testData[:,5:]
    
    model_path = os.getcwd()  +  modelName
    model = torch.load(model_path)
    model.eval()
    
    batchSize = 8
    i = 0
    outputs = []
    criterions = []
    criterion = nn.L1Loss()
    
    with torch.no_grad():
    
        while i < testData.shape[0]:
            input = x_test_data[i, :]
            input = torch.from_numpy(input)
            label = y_test_target[i]
            label = torch.from_numpy(label)
            
            output = model(input)
            outputs.append(output.item())
            
            loss = criterion(output, label)
            criterions.append(loss.item())
            
            i = i + 1
            
    print("mae:  ", np.mean(criterions))
    
    x = np.arange(0, testData.shape[0])

    y_test_target = y_test_target * std + mean
    outputs = np.array(outputs)


    out = outputs * std + mean
    out[out<0]=0
    

    
    outputs = np.reshape(outputs, (1, testData.shape[0]))
    plt.clf()
    plt.plot(x, y_test_target, color='r')
    plt.plot(x, out, color='b')
    plt.savefig(os.getcwd() + "/testResult.png")  
 
    return np.mean(criterions)

if __name__ == "__main__": 

    data=[]
    with open('C:/Users/86131/Desktop/chouchou1/chouchou.csv', 'r',encoding='utf-8-sig') as f_input:
        for line in f_input:        
            data.append(list(line.strip().split(',')))

    dataset = pd.DataFrame(data)
    datasets = np.array(dataset).astype(np.float32)
    
    mean = datasets.mean(axis=0)
    datasets -= mean
    std = datasets.std(axis=0)
    datasets /= std 
    
    testData = datasets[500:, :]
    #np.random.shuffle(datasets)

    
    
    trainData = datasets[0:500, :]
    
    
    
    train(trainData, testData, mean[-1], std[-1])
    
    modelName = "/models/model_9990"

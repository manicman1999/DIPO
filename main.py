# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:44:23 2018

@author: mtm916
"""

#Imports
import numpy as np
from keras.utils import to_categorical
import random
import matplotlib.pyplot as plt
from math import floor

#Get Data
class Data(object):
    
    def __init__(self):
        
        self.steps = []
        self.day = []
        self.temp = []
        self.precip = []
        self.avg = -1
    
    def add(self, steps, day, temp, precip):
        
        if(steps != ''):
            self.steps.append(int(steps))
        elif len(self.steps) > 1:
            self.steps.append(-1)
        else:
            return
        
        self.day.append(day)
        
        if(temp != ''):
            self.temp.append(float(temp))
        else: #If not temp value, use the day before
            self.temp.append(self.temp[-1])
        
        if(precip != ''):
            self.precip.append(float(precip))
        else: #If no precipitation, assume none
            self.precip.append(0)
    
    def average(self):
        
        if self.avg > 0:
            return self.avg
        
        av = 0.0
        n = 0.0
        
        for i in range(len(self.steps)):
            if self.steps[i] != -1:
                n = n + 1
                av = av + self.steps[i]
        
        self.avg = float(av / n)
        
        return self.avg
    
    def prepare(self, n = -1):
        
        if n == -1:
            n = random.randint(100, len(self.steps) - 1)
        
        if self.steps[n] == -1:
            return self.prepare()
        
        if n < 8 or n > len(self.steps) - 1:
            print('N out of bounds!')
            return
        
        firsthalf = np.array(self.steps[n-8:n-1])
        
        firsthalf[firsthalf == -1] = self.average()
        
        secondhalf = to_categorical(self.day[n], num_classes = 7)
        
        output = [np.concatenate((firsthalf, secondhalf), axis = 0), self.steps[n]]
        
        output[0] = np.append(output[0], self.temp[n])
        output[0] = np.append(output[0], self.precip[n])
        
        return output
        
        
        


matt = Data()
aidan = Data()

with open('data/steps.csv') as file:
    dat = file.read().split('\n')
    
    for i in range(2, len(dat)):
        datx = dat[i].split(',')
        matt.add(datx[1], int(datx[3]), datx[4], datx[5])
        aidan.add(datx[2], int(datx[3]), datx[4], datx[5])
    
print(matt.average())

#Model
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

#Model Build
class Predictor (object):
    
    def __init__(self, steps = 0):
        
        #Learning Rate And Optimizer
        self.LR = 0.005
        self.OP = Adam(lr = self.LR, decay = 1e-7)
        
        #Init Self.D
        self.D = None
        self.discriminator()
        
        #Steps
        self.steps = steps
        
    def discriminator(self):
        
        if self.D == None:
            self.D = Sequential()
            
            # Input Size
            # Past 7 Days Of Steps (7, linear)
            # Day Of The Week (7, one-hot)
            # Temperature (1, linear)
            # Precipitation (1, linear)
            # Total (16 | L7 + OH7)
            self.D.add(Dense(32, activation = 'relu', input_shape = [16]))
            self.D.add(Dropout(0.1))
            self.D.add(Dense(32, activation = 'relu'))
            self.D.add(Dense(32, activation = 'relu'))
            self.D.add(Dense(32, activation = 'relu'))
            self.D.add(Dropout(0.1))
            self.D.add(Dense(1, activation = 'relu'))
        
        self.D.compile(optimizer = self.OP, loss = 'mean_squared_logarithmic_error')
    
    def train(self, dataob = None, batch = 256):
        
        if dataob == None:
            print("Don't forget a 'Data' object!")
            return
        
        #Organize Data
        train_data = []
        label_data = []
        temp = []
        
        for i in range(batch):
            temp = dataob.prepare()
            train_data.append(temp[0])
            label_data.append([temp[1]])
            
        loss = self.D.train_on_batch(np.array(train_data), np.array(label_data))
        
        if self.steps % 10 == 0.1:
            print("Loss: " + str(loss))
        
        self.steps = self.steps + 1
        
        if self.steps % 100 == 0:
            self.save(floor(self.steps / 10000))
            print("Saved!")
        
    def evaluate(self, dataob = None, n1 = -1):
        
        if dataob == None:
            print("Don't forget a 'Data' object!")
            return
        
        if n1 == -1:
            n1 = random.randint(8, 100)
        
        temp = dataob.prepare(n1)
        out = self.D.predict(np.array([temp[0]]))[0]
        return (temp[1], out[0])
    
    def save(self, num = ''):
        dis_json = self.D.to_json()

        with open("Models/dis.json", "w") as json_file:
            json_file.write(dis_json)

        self.D.save_weights("Models/dis"+str(num)+".h5")
        
    def load(self, num = ''):
        
        steps1 = self.steps
        
        self.__init__(steps1)

        #Discriminator
        dis_file = open("Models/dis.json", 'r')
        dis_json = dis_file.read()
        dis_file.close()
        
        self.D = model_from_json(dis_json)
        self.D.load_weights("Models/dis"+str(num)+".h5")
        
        self.discriminator()
        





pred = Predictor(1)

zo = [0, 0]

while(pred.steps < 5000):
    pred.train(matt)
    pred.train(aidan)
    
    if pred.steps % 200 == 0:
        
        x = []
        y = []
        
        x2 = []
        y2 = []
        
        tempx = 0
        tempy = 0
        
        for i in range(8, 100):
            (tempx, tempy) = pred.evaluate(matt, n1 = i)
            x.append(tempx)
            y.append(tempy)
            
            #(tempx, tempy) = pred.evaluate(matt)
            #x2.append(tempx)
            #y2.append(tempy)
            
        
        plt.plot([0, 10000, 30000], [0, 10000, 30000])
        plt.scatter(x, y)
        #plt.scatter(x2, y2)
        
        
        # calc the trendline
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x,p(x),"r--")
        # the line equation:
        print("\n\n\ny = %.6fx + (%.6f)"%(z[0],z[1]))
        
        plt.show()
        
        zo = z



today = [6128, 11603, 4071, 3128, 5789, 7624, 5647,
         0, 0, 0, 0, 1, 0, 0, 22.0, 4.0]

print(pred.D.predict(np.array([today])))


import tensorflowjs as tfjs

tfjs.converters.save_keras_model(pred.D, 'C:/Users/mtm916/Desktop/PySpace/DIPO/ModelJS/')










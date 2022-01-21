#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 18:45:59 2022

@author: ksankara
"""

import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout 

tr_data = pd.read_csv('Stock_data.csv')  
tr_data_proc = tr_data.iloc[:, 1:2].values  
data_dates = tr_data.iloc[:, 0].values 

# PREPROCESS DATA
N=60
from sklearn.preprocessing import MinMaxScaler  
scaler = MinMaxScaler(feature_range = (0, 1))

tr_data_scaled = scaler.fit_transform(tr_data_proc)  
tr_data_scaled.shape
print(tr_data_proc)

features_set = []  
labels = []  
for i in range(N, 1280):  
    features_set.append(tr_data_scaled[i-N:i, 0])
    labels.append(tr_data_scaled[i, 0])
    
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1)) 

## LSTM TRAINING

model = Sequential()  
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1))) 
model.add(Dropout(0.2))  
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50))  
model.add(Dropout(0.2))  
model.add(Dense(units = 1))  
model.compile(optimizer = 'adam', loss = 'mean_squared_error')  
model.fit(features_set, labels, epochs = 100, batch_size = 32)  


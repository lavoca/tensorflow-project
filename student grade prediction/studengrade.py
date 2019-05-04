import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping

data = pd.read_csv('student-mat.csv',sep=';')
data = data[['G3','G2','G1','studytime','failures','absences']]
predict = 'G3'

x_train  = np.array(data.drop(columns=[predict])) 
y_train  = np.array(data[predict])



model = Sequential()



x_train_cols = x_train.shape[1]

model.add(Dense(150 , activation = 'relu' , input_shape = (x_train_cols, )))
model.add(Dense(150 , activation = 'relu'))
model.add(Dense(150 , activation = 'relu'))
model.add(Dense(1))


model.compile(optimizer='adam' , loss='mean_squared_error')

early_stopping = EarlyStopping(patience=3)

model.fit(x_train,y_train,validation_split=0.2,epochs=50,callbacks=[early_stopping])












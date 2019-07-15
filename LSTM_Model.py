#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.model_selection import train_test_split
import keras.backend as K
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
#pd.set_option('display.max_rows', None)
data=pd.read_csv('data_processed.csv')
T=data['Event_label'].shift(-1)
M=data['Displacement_processed'].shift(-1)
data['goal_label']=T
data['goal_displacement']=M
data = pd.DataFrame.dropna(data)
X =np.array(data[['Time_series_processed','Streamwise_velocity_processed','Displacement_processed']])
Y = data['goal_displacement']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f
# create model
model = Sequential()
#model.add(Dense(1, input_dim=1, init='uniform', activation='relu')) #input layer
model.add(LSTM(60, input_shape=(1, 3), return_sequences=True,activation='relu'))
#model.add(Dropout(0.1))
model.add(LSTM(20, return_sequences=False,activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(1, init='uniform', activation='tanh'))#output layer
# Optimizers
RMSProp = optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=None,decay=0)
# Compile model
model.compile(optimizer='RMSprop',loss='mean_squared_error', )

# Fit the model
history=model.fit(X_train, y_train, validation_data=(X_test,y_test),
                  nb_epoch=100, batch_size=1)
# evaluate the model
# 随机数参数
pred_test_y = model.predict(X_test)

from sklearn.metrics import mean_squared_error
pred_acc = mean_squared_error(y_test, pred_test_y)
print('pred_acc',pred_acc)

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图形大小
plt.figure(figsize=(8, 4), dpi=80)
#plt.plot(range(len(Y[801:1000])), Y[801:1000], ls='-.',lw=2,c='g',label='ORi_Value')
plt.plot(range(len(y_test)), y_test, ls='-.',lw=2,c='r',label='True_Value')
plt.plot(range(len(pred_test_y)), pred_test_y, ls='-',lw=2,c='b',label='Predict_Value')
plt.savefig('The_Prediction_of_Displacement.svg',format="svg")
# 绘制网格
plt.grid(alpha=0.4, linestyle=':')
plt.legend()
#plt.xlabel('S_V') #设置x轴的标签文本
plt.ylabel('label') #设置y轴的标签文本
# 展示
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Loss_with_Epoch.svg',format="svg")
plt.show()

#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.ensemble import *
import itertools
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
#pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt

data = pd.read_csv('E0002.csv',header=None,names=['Column1','streamwise_velocity','Column3','Time_series','Column5','displacement'])
data_pre = data[['Time_series','streamwise_velocity','displacement']]
streamwise_velocity = data['streamwise_velocity']
Time_series= data['Time_series']
displacement = data['displacement']
def time_process():#输入输出长度相同，时间插值且步长一致
    Time_Gap = []
    for i in range(1,len(Time_series)):
        Time_Gap.append((Time_series[i]-Time_series[0])/10**6)#就算和第一个时刻的时间差
    Time_step = Time_Gap[-1]/(len(Time_series)-1)#时间序列的一个步长
    time_series_new=np.arange(0,(len(Time_series))*Time_step,Time_step)#等长插值
    #这里的kk是一个完整的插值，其长度应该是和time-series的长度一样
    '''plt.figure()
    plt.plot(Time_series[0:2000]/10**6)
    plt.plot(time_series_new[0:2000])
    plt.show()'''
    return time_series_new

def streamwise_velocity_process():#用于对流速数据的处理
    def three_sigma(data_series):
        rule = (data_series.mean() - 3 * data_series.std() > data_series)\
               | (data_series.mean() + 3 * data_series.std() < data_series)#3simga法则
        index_st = np.arange(data_series.shape[0])[rule]#找出异常值位置
        outliers = data_series.iloc[index_st]
       # plt.figure()
        #plt.plot(data_series[index_st], 'r*', label='Before Process')
       #print('异常值的索引为： ',index)
        for i in index_st:
            data_series[i] = (data_series[i-1]+data_series[i+1])/2#将异常值用相邻数据的均值替换
        #plt.plot(data_series[index_st], 'b*',label = 'After Process')
        #plt.title('Anomaly Process by Three Simga Rule')
       # plt.legend( )
        #plt.show()
        return data_series.tolist()
    def three_sigma_pic(data,threshold = .997):#当且仅当用来做图
        d = pd.DataFrame(streamwise_velocity)
        streamwise_velocity['isAnomaly'] = d > d.quantile(threshold)
        d.insert(1, 'isAnomaly', streamwise_velocity['isAnomaly'])
       # sns.relplot(x=range(0, len(streamwise_velocity) - 1), y="streamwise_velocity",
                   # hue="isAnomaly", style="isAnomaly", data=d)
        #plt.show()
    st = three_sigma(streamwise_velocity)    #输出异常数值的数值
    some= three_sigma_pic(streamwise_velocity)
    return st
    #这里需要检测下st的长度是多少
time_series_new = time_process()
streamwise_velocity_new=streamwise_velocity_process()
data_new= pd.concat([pd.DataFrame(time_series_new),pd.DataFrame(streamwise_velocity_new),
                     displacement-(-3.1)],axis=1)#合并
data_new.columns = ['Time_series_pro','Streamwise_velocity_pro','Displacement_pro']
#时间和速度都已经改了
#先将步长求均值然后把第一个值给替换掉，然后再把所有第一个值给抽出来。
def displacement_process():

    def step_ave(x, y):  # 对x取y个步长形成一个新的x，对y步长取均值
        step_gap = []
        for i in range(0, int(len(x) / y)):
            m = sum(x[i * y:(i + 1) * y]) / y
            step_gap.append(m)
        #step_average=list(itertools.chain.from_iterable(itertools.repeat(element, y) for element in step_gap))
        #df_ave = pd.DataFrame(step_average,columns=['step_average'])
        return step_gap
    data_time = step_ave(data_new['Time_series_pro'],100)
    data_st = step_ave(data_new['Streamwise_velocity_pro'],100)
    data_dis = step_ave(data_new['Displacement_pro'],100)
    def event_detection(X):
        slope = []
        for i in range(0, len(X) - 1):
            slope.append(X[i + 1] - X[i])  # 斜率
        df = pd.DataFrame(slope, columns=['slope'])
        df['event_label'] = ''
        slope_sta = 0.2
        df.loc[abs(df['slope']) >= slope_sta, 'event_label'] = 1
        df.loc[abs(df['slope']) < slope_sta, 'event_label'] = 0
        return df
    df_test = event_detection(data_dis)
    data_final = pd.concat([pd.DataFrame(data_time), pd.DataFrame(data_st),pd.DataFrame(data_dis),
                         df_test], axis=1)  # 合并
    data_final.columns = ['Time_series_processed', 'Streamwise_velocity_processed',
                        'Displacement_processed','Slpoe','Event_label']
    data_final.drop(data_final.index[len(data_final) - 1])
    return data_final

data_processed = displacement_process()
print(data_processed)

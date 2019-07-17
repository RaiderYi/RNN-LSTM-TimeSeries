# RNN-LSTM-TimeSeries
## This is for my master project in UoG 
LSTM TimeSeries prediction 时间序列预测

### Data—Preprocess   数据预处理
数据预处理包含：时间序列的插值，流速的异常点检测与替换，移动事件的检测，对事件进行标签处理。  
Data preprocessing includes: Interpolation of time series, Detection and replacement of abnormal points of streamwise_ve, Detection of moving events, and label the events.

### Model-Building   模型建立
模型采用三层LSTM模型，优化器和LOSS函数参看代码 训练100次.维基百科介绍[长短期记忆](https://en.wikipedia.org/wiki/Long_short-term_memory)  
The Model in this project is built on LSTM model,here is the wiki link [LSTM-Wiki](https://en.wikipedia.org/wiki/Long_short-term_memory)
The parameters of the model are lr=0.001,nb_epoch=100, batch_size=1. 

### 效果图
![损失](https://github.com/RaiderYi/RNN-LSTM-TimeSeries/blob/master/PIC/Loss_with_Epoch.svg)

![预测](https://github.com/RaiderYi/RNN-LSTM-TimeSeries/blob/master/PIC/The_Prediction_of_Displacement.svg)

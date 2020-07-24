from tensorflow.keras.layers import Dense, Input, LSTM, Conv2D, MaxPooling2D, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def data_preprocessing(df):
    new = pd.DataFrame()
    
    df = df.loc[:,['ObservationDate','Confirmed', 'Deaths','Recovered']]
    
    df = df.groupby(['ObservationDate'], as_index='ObservationDate').sum()
    df = df.diff().fillna(0)

    new = (df-df.mean())/df.std()
    
    return new

def createTrainData(xData, step, Flag):
    # Flags 1:LSTM , 2:CNN
    
    xData = np.array(xData, dtype=np.float32)
    m = np.arange(len(xData) - step)
    
    x, y = [], []
    for i in m:
        a = xData[i:(i+step)]
        x.append(a)
    if Flag == 1:
        xBatch = np.reshape(np.array(x), (len(m), step, xData.shape[1]))
    else:
        xBatch = np.reshape(np.array(x), (len(m), step, 3, 1))
        
    for i in m+1:
        a = xData[i:(i+step)]
        y.append(a[-1])
        
    yBatch = np.reshape(np.array(y), (len(m), xData.shape[1]))
    
    return xBatch, yBatch


df = pd.read_csv('C:/Users/student/Downloads/covid_19_data.csv')
data = data_preprocessing(df)

# LSTM
nInput = 1
nOutput = data.shape[1]
nStep = 20
nHidden = 50

# CNN
nFeature = data.shape[1]
nChannel = 1 

# LSTM
lstm_x, y = createTrainData(data, nStep, 1)

LSTM_Input = Input(batch_shape=(None, nStep, nOutput))
xLstm = LSTM(nHidden, activation='relu')(LSTM_Input)
LSTM_Output = Dense(nOutput)(xLstm)
LSTM_model = Model(inputs = LSTM_Input, outputs = LSTM_Output)

# CNN
cnn_x, y = createTrainData(data, nStep, 2)

CNN_Input = Input(batch_shape = (None, nStep, nFeature, nChannel))
xConv1 = Conv2D(filters=30, kernel_size=(8,2), strides=1, padding = 'same', activation='relu')(CNN_Input)
xPool1 = MaxPooling2D(pool_size=(2,2), strides=1, padding='valid')(xConv1)
xConv2 = Conv2D(filters=10, kernel_size=(8,2), strides=1, padding = 'same', activation='relu')(xPool1)
xPool2 = MaxPooling2D(pool_size=(2,2), strides=1, padding='valid')(xConv2)
xFlat = Flatten()(xPool2)
CNN_Output = Dense(1, activation='linear')(xFlat)
CNN_model = Model(inputs = CNN_Input, outputs = CNN_Output)


merge = concatenate([LSTM_model.output, CNN_model.output])
merge_Output = Dense(32, activation='relu')(merge)
merge_Output = Dense(16, activation='relu')(merge_Output)
merge_Output = Dense(nOutput)(merge_Output)


model = Model([LSTM_Input, CNN_Input], merge_Output)

model.compile(loss='mse', optimizer=Adam(lr=0.01))
history = model.fit([lstm_x, cnn_x], y, epochs = 300, batch_size = 300)


nFuture = 10
if len(data) > 100:
    lastData = np.copy(data[-100:]) 
else:
    lastData = np.copy(data)
dx = np.copy(lastData)
estimate = [dx[-1]]

for i in range(nFuture):

    lstm_px = dx[-nStep:].reshape(1, nStep, nFeature)
    cnn_px = dx[-nStep:].reshape(-1, nStep, nFeature, nChannel)
    
    yHat = model.predict([lstm_px,cnn_px])
    
    estimate.append(yHat.reshape(-1))

    dx = np.insert(dx, len(dx), yHat, axis=0)


plt.figure(figsize=(20, 7))
plt.plot(history.history['loss'], color='red')
plt.title("Loss History")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


ax1 = data.index[-len(lastData):]
ticks=[ax1[i] for i in range(len(ax1)) if i%30 == 0]
ax2 = np.arange(len(lastData), len(lastData) + len(estimate))
plt.figure(figsize=(8, 3))
plt.plot(ax1, lastData[:, :1], 'b-o', color='blue', markersize=3, label='Confirmed', linewidth=1)
plt.plot(ax1, lastData[:, 1:2], 'b-o', color='green', markersize=3, label='Deaths', linewidth=1)
plt.plot(ax1, lastData[:, -1], 'b-o', color='magenta', markersize=3, label='Recovered', linewidth=1)
plt.plot(ax2, estimate, 'b-o', color='red', markersize=3, label='Estimate')
plt.xticks(ticks, fontsize=10)
plt.axvline(x=ax1[-1],  linestyle='dashed', linewidth=1)
plt.legend()
plt.show()


# =============================================================================
# Model: "model_2"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_2 (InputLayer)            [(None, 20, 3, 1)]   0                                            
# __________________________________________________________________________________________________
# conv2d (Conv2D)                 (None, 20, 3, 30)    510         input_2[0][0]                    
# __________________________________________________________________________________________________
# max_pooling2d (MaxPooling2D)    (None, 19, 2, 30)    0           conv2d[0][0]                     
# __________________________________________________________________________________________________
# conv2d_1 (Conv2D)               (None, 19, 2, 10)    4810        max_pooling2d[0][0]              
# __________________________________________________________________________________________________
# input_1 (InputLayer)            [(None, 20, 3)]      0                                            
# __________________________________________________________________________________________________
# max_pooling2d_1 (MaxPooling2D)  (None, 18, 1, 10)    0           conv2d_1[0][0]                   
# __________________________________________________________________________________________________
# lstm (LSTM)                     (None, 50)           10800       input_1[0][0]                    
# __________________________________________________________________________________________________
# flatten (Flatten)               (None, 180)          0           max_pooling2d_1[0][0]            
# __________________________________________________________________________________________________
# dense (Dense)                   (None, 3)            153         lstm[0][0]                       
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 1)            181         flatten[0][0]                    
# __________________________________________________________________________________________________
# concatenate (Concatenate)       (None, 4)            0           dense[0][0]                      
#                                                                  dense_1[0][0]                    
# __________________________________________________________________________________________________
# dense_2 (Dense)                 (None, 32)           160         concatenate[0][0]                
# __________________________________________________________________________________________________
# dense_3 (Dense)                 (None, 16)           528         dense_2[0][0]                    
# __________________________________________________________________________________________________
# dense_4 (Dense)                 (None, 3)            51          dense_3[0][0]                    
# ==================================================================================================
# Total params: 17,193
# Trainable params: 17,193
# Non-trainable params: 0
# __________________________________________________________________________________________________
# 
# =============================================================================

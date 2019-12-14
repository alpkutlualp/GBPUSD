
# coding: utf-8

# ## Data Import and Package Uploads

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import matplotlib
get_ipython().magic(u'matplotlib inline')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[3]:


path='/Users/alpandthemachine/Downloads/GBPUSD_Ticks_23.06.2016-23.06.2016-2.csv'
df=pd.read_csv(path)
df.head()


# ## Data Preprocessing and Visualization

# In[4]:


df['Gmt time']=pd.to_datetime(df['Gmt time'])
df.head()


# In[5]:


df=df.set_index('Gmt time')
df.head()


# In[6]:


dfba=df.ix[:,0:2]
dfvol=df.ix[:,2:4]

dfprod=pd.DataFrame()
dfprod.insert(0,'AskIntervalSum',dfba['Ask'].multiply(dfvol['AskVolume']))
dfprod.insert(1,'BidIntervalSum',dfba['Bid'].multiply(dfvol['BidVolume']))

dfba_res=dfba.resample('30s').ohlc()
dfba_avg=dfba.resample('30s').mean()

dfvol_res=dfvol.resample('30s').sum()
dfprod_res=dfprod.resample('30s').sum()
dfprod_res.head()


# In[48]:


dfvol_res.head()


# In[7]:


dfprices=pd.DataFrame()
dfprices.insert(0,'TWAP OHLC Bid',(dfba_res.ix[:,4]+dfba_res.ix[:,5]+dfba_res.ix[:,6]+dfba_res.ix[:,7])/4.0)
dfprices.insert(1,'TWAP OHLC Ask',(dfba_res.ix[:,0]+dfba_res.ix[:,1]+dfba_res.ix[:,2]+dfba_res.ix[:,3])/4.0)
dfprices.insert(2,'TWAP OHLC Mid',(dfprices['TWAP OHLC Bid']+dfprices['TWAP OHLC Ask'])/2.0)
dfprices.head()


# In[8]:


np.argwhere(np.isnan(dfprices['TWAP OHLC Mid']))


# In[9]:


dfprices.dropna()
dfprices=dfprices.dropna()
np.argwhere(np.isnan(dfprices['TWAP OHLC Mid']))


# In[11]:


plt.plot(dfprices.index.values[0:np.shape(dfprices)[0]],dfprices.ix[0:np.shape(dfprices)[0],'TWAP OHLC Mid'])
plt.xlabel('Time')
plt.ylabel('GBPUSD')
plt.title('FX Rate Progression on 24 June 2016')
plt.grid(True)


# In[12]:


train=int(np.shape(dfprices)[0]*0.70)
TWAP_OHLC_Mid_train=dfprices['TWAP OHLC Mid'][0:train]
TWAP_OHLC_Mid_test=dfprices['TWAP OHLC Mid'][train:np.shape(dfprices)[0]]


# In[14]:


train


# In[15]:


np.shape(dfprices)[0]


# In[16]:


TWAP_OHLC_Mid_test.shape


# ## ARIMA: A Classical Time Series Model

# In[17]:


levels=TWAP_OHLC_Mid_train
levels_acf=acf(levels,nlags=10)
levels_pacf=pacf(levels,nlags=10)

plt.bar(range(11),levels_acf)
plt.axhline(y=0,linestyle='--',color='black')
plt.axhline(y=-1.96/np.sqrt(len(levels)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(levels)),linestyle='--',color='black')
plt.ylim((-1.1, 1.1)) 
plt.title('ACF of GBPUSD Levels')

#The acf is declining slowly and smoothly as expected below.


# In[18]:


plt.bar(range(11),levels_pacf)
plt.axhline(y=0,linestyle='--',color='black')
plt.axhline(y=-1.96/np.sqrt(len(levels)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(levels)),linestyle='--',color='black')
plt.ylim((-1.1, 1.1)) 
plt.title('PACF of GBPUSD Levels')
#The pacf shows strong correlation close to 1 at lag 1 as expected.


# In[19]:


diffs=TWAP_OHLC_Mid_train.diff(1)
cfdiffs=diffs[1:len(diffs)]
lag_acf=acf(cfdiffs,nlags=10)

plt.bar(range(11),lag_acf)
plt.axhline(y=0,linestyle='--',color='black')
plt.axhline(y=-1.96/np.sqrt(len(levels)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(levels)),linestyle='--',color='black')
plt.ylim((-1.1, 1.1)) 
plt.title('ACF of GBPUSD Diffs')


# In[21]:


lag_pacf=pacf(cfdiffs,nlags=10)
plt.bar(range(11),lag_pacf)
plt.axhline(y=0,linestyle='--',color='black')
plt.axhline(y=-1.96/np.sqrt(len(levels)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(levels)),linestyle='--',color='black')
plt.ylim((-1.1, 1.1)) 
plt.title('PACF of GBPUSD Diffs')


# In[23]:


model=ARIMA(TWAP_OHLC_Mid_train,order=(2,1,1))
model_arima=model.fit()
ARIMA_forecasts=model_arima.forecast(steps=756)[0]


# In[24]:


ARIMA_preds_diff = model_arima.fittedvalues.cumsum()
ARIMA_preds_diff.head()


# In[25]:


ARIMA_preds = pd.Series(TWAP_OHLC_Mid_train.ix[0], index=TWAP_OHLC_Mid_train.index)
ARIMA_preds = ARIMA_preds.add(ARIMA_preds_diff,fill_value=0)
ARIMA_preds.head()


# In[26]:


plt.plot(pd.Series(ARIMA_preds))
plt.plot(TWAP_OHLC_Mid_train)


# In[27]:


plt.plot(pd.Series(ARIMA_forecasts))
plt.plot(TWAP_OHLC_Mid_test.reset_index()[['TWAP OHLC Mid']])


# In[73]:


plt.plot(pd.Series(ARIMA_forecasts)[0:30])
plt.plot(TWAP_OHLC_Mid_test.reset_index()[['TWAP OHLC Mid']][0:30])


# In[28]:


TWAP_OHLC_Mid_train.shape, ARIMA_preds.shape, TWAP_OHLC_Mid_test.shape, ARIMA_forecasts.shape


# In[29]:


#rmse_train = np.sqrt(mean_squared_error(Y_train_inv_plot, ARIMA_forecasts))
rmse_train= np.sqrt(mean_squared_error(TWAP_OHLC_Mid_train, ARIMA_preds))
rmse_test = np.sqrt(mean_squared_error(TWAP_OHLC_Mid_test, pd.Series(ARIMA_forecasts)))
rmse_train, rmse_test


# In[30]:


one_step_forecasts=[]
for i in range(len(TWAP_OHLC_Mid_test)):
    model_one_step=ARIMA(dfprices['TWAP OHLC Mid'][0:train+i],order=(1,1,1))
    one_step_fit=model_one_step.fit()
    one_step_forecasts.append(one_step_fit.forecast(steps=1)[0][0])


# In[86]:


plt.plot(one_step_forecasts,'bo',label='ARIMA Forecasts')
plt.plot(TWAP_OHLC_Mid_test.reset_index()[['TWAP OHLC Mid']],'r+', label='TWAP Test Data')
plt.xlabel('30 second intervals')
plt.ylabel('GBPUSD')
plt.title('ARIMA One Step Ahead GBP/USD Forecasts vs. Test Data')
plt.legend(loc="upper right")
plt.grid(True)


# In[88]:


rmse_test = np.sqrt(mean_squared_error(TWAP_OHLC_Mid_test, one_step_forecasts))
rmse_test


# In[38]:


TWAP_OHLC=TWAP_OHLC_Mid_test.reset_index()
TWAP_test=np.array(TWAP_OHLC['TWAP OHLC Mid'])


# In[39]:


mean_abs_ARIMA=((one_step_forecasts-TWAP_test)/TWAP_test)*100.0


# In[40]:


TWAP_OHLC=TWAP_OHLC_Mid_test.reset_index()
TWAP_test=np.array(TWAP_OHLC['TWAP OHLC Mid'])
mean_abs_ARIMA=((one_step_forecasts-TWAP_test)/TWAP_test)*100.0
mean_abs_ARIMA_=abs(mean_abs_ARIMA)
mean_abs_ARIMA_


# In[41]:


mean_abs_ARIMA=((one_step_forecasts-TWAP_test)/TWAP_test)*100.0
mean_abs_ARIMA_=abs(mean_abs_ARIMA)
np.mean(mean_abs_ARIMA_)


# In[52]:


bs_ARIMA=np.zeros(len(one_step_forecasts))
if one_step_forecasts[0]>dfprices['TWAP OHLC Mid'][-1]:
    bs_ARIMA[0]=1.0
else:
    bs_ARIMA[0]=-1.0
for i in range(len(one_step_forecasts)-1):
    if one_step_forecasts[1+i]>TWAP_test[i]:
        bs_ARIMA[i+1]=1.0
    else:
        bs_ARIMA[i+1]=-1.0


# In[54]:


prof_ARIMA=one_step_forecasts-TWAP_test


# In[55]:


np.sum(prof_ARIMA)


# In[59]:


prof_ARIMA_cumul=np.cumsum(prof_ARIMA)
prof_ARIMA


# In[50]:


np.sum((one_step_forecasts[0]>dfprices['TWAP OHLC Mid'][-1]))


# In[64]:


len(seq_test)


# In[280]:


n_features, n_steps_in


# ### MLP

# In[66]:


from numpy import array
from keras.models import Sequential
from keras.layers import Dense
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# In[67]:


raw_seq=dfprices['TWAP OHLC Mid']


# In[68]:


seq_len=len(raw_seq)
train_size=int(seq_len*0.70)
test_size= seq_len-train_size
raw_train=dfprices['TWAP OHLC Mid'][0:train_size]


# In[69]:


forecasts_mlp=[]
# choose a number of time steps
n_steps = 1
# split into samples
# define model
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=n_steps))
model.add(Dense(50, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model


# In[70]:


for i in range(test_size):
    X, y = split_sequence(raw_seq[0:train_size+i], n_steps)
    model.fit(X, y, epochs=30,batch_size=512,verbose=1)
# demonstrate prediction
    x_input = raw_seq[train_size-1+i]
    x_input = x_input.reshape((1, n_steps))
    yhat = model.predict(x_input, verbose=0)
    forecasts_mlp.append(yhat)


# In[71]:


forecasts_mlp=np.array(forecasts_mlp)


# In[72]:


forecasts_mlp=np.reshape(forecasts_mlp,-1)


# In[73]:


forecasts_mlp.shape


# In[74]:


TWAP_test.shape


# In[89]:


raw_test.shape


# In[90]:


raw_test=raw_seq[train_size:]


# In[91]:


raw_test.shape


# In[92]:


forecasts_mlp


# In[93]:


raw_test=pd.Series(raw_test)
raw_test.reset_index();
raw_test=np.array(raw_test)
raw_test


# In[84]:


plt.plot(forecasts_mlp,'bo', label='MLP Forecasts')
plt.plot(raw_test,'r+',label='Test Data')
plt.xlabel('30 second intervals')
plt.ylabel('GBPUSD')
plt.title('Multi-Layer Perceptron One-Step Ahead GBP/USD Forecasts vs. Test Data')
plt.legend(loc="upper right")
plt.grid(True)


# In[94]:


rmse_test = np.sqrt(mean_squared_error(forecasts_mlp, raw_test))
rmse_test


# In[95]:


mean_abs_mlp=((forecasts_mlp-TWAP_test)/TWAP_test)*100.0
mean_abs_mlp_=abs(mean_abs_mlp)
mean_abs_mlp_


# In[96]:


np.mean(mean_abs_mlp_)


# ## CNN 

# In[97]:


# univariate cnn example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
 
# define input sequence


# In[98]:


raw_seq=dfprices['TWAP OHLC Mid']
seq_len=len(raw_seq)
train_size=int(seq_len*0.70)
test_size= seq_len-train_size
raw_train=dfprices['TWAP OHLC Mid'][0:train_size]


# In[99]:


n_steps = 3
# split into samples
#X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
#X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[100]:


forecasts_cnn=[]


# In[101]:


test_size


# In[422]:


a=[0,1,2,3,4,5,6,7,8,9]
a[-2],a[9],a[0:10]


# In[102]:


for i in range(test_size):
    X, y = split_sequence(raw_seq[0:train_size+i], n_steps)
    X=X.reshape((X.shape[0], X.shape[1], n_features))
    model.fit(X, y, epochs=30,batch_size=512,verbose=1)
# demonstrate prediction
    x_input = raw_seq[train_size-3+i:train_size+i]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    forecasts_cnn.append(yhat)


# In[118]:


forecasts_cnn=np.reshape(forecasts_cnn,-1)
plt.plot(forecasts_cnn,'bo', label="CNN Forecasts")
plt.plot(raw_test,'r+', label="Test Data")
plt.xlabel('30 second intervals')
plt.ylabel('GBPUSD')
plt.title('Convolutional Neural Network One-Step Ahead GBP/USD Forecasts vs. Test Data')
plt.legend(loc="upper right")
plt.grid(True)


# In[104]:


rmse_test = np.sqrt(mean_squared_error(forecasts_cnn, raw_test))
rmse_test


# In[105]:


mean_abs_cnn=((forecasts_cnn-TWAP_test)/TWAP_test)*100.0
mean_abs_cnn_=abs(mean_abs_cnn)
np.mean(mean_abs_cnn_)


# ## CNN LSTMs

# In[111]:


# univariate cnn lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# In[112]:


raw_seq = dfprices['TWAP OHLC Mid']
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[113]:


seq_len=len(raw_seq)
train_size=int(seq_len*0.70)
test_size= seq_len-train_size


# In[447]:





# In[114]:


forecasts_CNN_LSTM=[]


# In[115]:


for i in range(test_size):
    n_steps=4
    n_seq=2
    X, y = split_sequence(raw_seq[0:train_size+i], n_steps)
    n_steps=2
    X=X.reshape((X.shape[0], n_seq, n_steps, n_features))
    model.fit(X, y, epochs=30, batch_size=512,verbose=1)
    # demonstrate prediction
    x_input = raw_seq[train_size-4+i:train_size+i]
    x_input = x_input.reshape((1, n_seq, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    forecasts_CNN_LSTM.append(yhat)


# In[116]:


forecasts_CNN_LSTM=np.reshape(forecasts_CNN_LSTM,-1)
plt.plot(forecasts_CNN_LSTM,'bo', label='CNN-LSTM Forecasts')
plt.plot(raw_test,'r+',label="Test Data")
plt.xlabel('30 second intervals')
plt.ylabel('GBPUSD')
plt.title('CNN-LSTM One-Step Ahead GBP/USD Forecasts vs. Test Data')
plt.legend(loc="upper right")
plt.grid(True)


# In[121]:


rmse_test = np.sqrt(mean_squared_error(forecasts_CNN_LSTM, raw_test))
rmse_test


# In[122]:


mean_abs_cnn_lstm=((forecasts_CNN_LSTM-TWAP_test)/TWAP_test)*100.0
mean_abs_cnn_lstm_=abs(mean_abs_cnn_lstm)
np.mean(mean_abs_cnn_lstm_)


# # ConvLSTM

# In[123]:


# univariate convlstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import ConvLSTM2D
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# In[124]:


raw_seq = dfprices['TWAP OHLC Mid']
# define model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[125]:


forecasts_ConvLSTM=[]


# In[126]:


for i in range(test_size):
    n_steps=4
    n_seq=2
    X, y = split_sequence(raw_seq[0:train_size+i], n_steps)
    n_steps=2
    X=X.reshape((X.shape[0], n_seq, 1, n_steps, n_features))
    model.fit(X, y, epochs=30, batch_size=512,verbose=1)
    # demonstrate prediction
    x_input = raw_seq[train_size-4+i:train_size+i]
    x_input = x_input.reshape((1, n_seq, 1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    forecasts_ConvLSTM.append(yhat)


# In[127]:


forecasts_ConvLSTM=np.reshape(forecasts_ConvLSTM,-1)
plt.plot(forecasts_ConvLSTM,'bo',label='ConvLSTM Forecasts')
plt.plot(raw_test,'r+',label="Test Data")
plt.xlabel('30 second intervals')
plt.ylabel('GBPUSD')
plt.title('ConvLSTM One-Step Ahead GBP/USD Forecasts vs. Test Data')
plt.legend(loc="upper right")
plt.grid(True)


# In[128]:


rmse_test = np.sqrt(mean_squared_error(forecasts_ConvLSTM, raw_test))
rmse_test


# In[129]:


mean_abs_conv_lstm=((forecasts_ConvLSTM-TWAP_test)/TWAP_test)*100.0
mean_abs_conv_lstm_=abs(mean_abs_conv_lstm)
np.mean(mean_abs_conv_lstm_)


# ## Bidirectional LSTM

# In[134]:


# univariate bidirectional lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
 
# split a univariate sequence
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# In[135]:


# define input sequence
raw_seq = dfprices['TWAP OHLC Mid']
# choose a number of time steps
# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model


# In[136]:


forecasts_bidi=[]


# In[137]:


for i in range(test_size):
    n_steps=3
    X, y = split_sequence(raw_seq[0:train_size+i], n_steps)
    n_features=1
    X=X.reshape((X.shape[0], X.shape[1], n_features))
    model.fit(X, y, epochs=30, batch_size=512,verbose=1)
    # demonstrate prediction
    x_input = raw_seq[train_size-3+i:train_size+i]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    forecasts_bidi.append(yhat)


# In[146]:


forecasts_bidi=np.reshape(forecasts_bidi,-1)
plt.plot(forecasts_bidi,'bo',label='Bidirectional LSTM Forecasts')
plt.plot(raw_test,'r+',label="Test Data")
plt.xlabel('30 second intervals')
plt.ylabel('GBPUSD')
plt.title('Bidirectional LSTM One-Step Ahead GBP/USD Forecasts vs. Test Data')
plt.legend(loc="upper right")
plt.grid(True)


# In[140]:


rmse_test = np.sqrt(mean_squared_error(forecasts_bidi, raw_test))
rmse_test


# In[141]:


mean_abs_bidi=((forecasts_bidi-TWAP_test)/TWAP_test)*100.0
mean_abs_bidi_=abs(mean_abs_bidi)
np.mean(mean_abs_bidi_)


# # Vanilla LSTM

# In[142]:


# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# In[143]:


n_steps=1
n_features=1
model = Sequential()
model.add(LSTM(5, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[144]:


forecasts_van5=[]


# In[145]:


for i in range(test_size):
    n_steps=1
    X, y = split_sequence(raw_seq[0:train_size+i], n_steps)
    n_features=1
    X=X.reshape((X.shape[0], X.shape[1], n_features))
    model.fit(X, y, epochs=30, batch_size=512,verbose=1)
    # demonstrate prediction
    x_input = raw_seq[train_size-1]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    forecasts_van5.append(yhat)


# In[151]:


forecasts_van5=np.reshape(forecasts_van5,-1)
plt.plot(forecasts_van5,'bo',label='Test Data')
plt.plot(raw_test,'r+',label="Vanilla LSTM Forecasts")
plt.xlabel('30 second intervals')
plt.ylabel('GBPUSD')
plt.title('Vanilla LSTM One-Step Ahead GBP/USD Forecasts vs. Test Data')
plt.legend(loc="lowerright")
plt.grid(True)


# In[147]:


rmse_test = np.sqrt(mean_squared_error(forecasts_van5, raw_test))
rmse_test


# In[495]:


plt.plot(forecasts_van5[7:],'bo')
plt.plot(raw_test[7:],'r+')


# In[528]:


mean_abs_van5=((forecasts_van5-TWAP_test)/TWAP_test)*100.0
mean_abs_van5_=abs(mean_abs_van5)
np.mean(mean_abs_van5_)


# ## Vanilla LSTM - 50 Neurons

# In[496]:


n_steps=1
n_features=1
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[497]:


forecasts_van50=[]


# In[498]:


for i in range(test_size):
    n_steps=1
    X, y = split_sequence(raw_seq[0:train_size+i], n_steps)
    n_features=1
    X=X.reshape((X.shape[0], X.shape[1], n_features))
    model.fit(X, y, epochs=30, batch_size=512,verbose=1)
    # demonstrate prediction
    x_input = raw_seq[train_size-1]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    forecasts_van50.append(yhat)


# In[499]:


forecasts_van50=np.reshape(forecasts_van50,-1)
plt.plot(forecasts_van5,'bo')
plt.plot(raw_test,'r+')


# In[529]:


rmse_test = np.sqrt(mean_squared_error(forecasts_van50, raw_test))
rmse_test


# In[530]:


rmse_test = np.sqrt(mean_squared_error(forecasts_van50, raw_test))
rmse_test
mean_abs_van50=((forecasts_van50-TWAP_test)/TWAP_test)*100.0
mean_abs_van50_=abs(mean_abs_van50)
np.mean(mean_abs_van50_)


# ## Stacked LSTM - 50 Neurons

# In[531]:


n_steps=1
n_features=1
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[532]:


forecasts_stacked=[]


# In[533]:


for i in range(test_size):
    n_steps=1
    X, y = split_sequence(raw_seq[0:train_size+i], n_steps)
    n_features=1
    X=X.reshape((X.shape[0], X.shape[1], n_features))
    model.fit(X, y, epochs=30, batch_size=512,verbose=1)
    # demonstrate prediction
    x_input = raw_seq[train_size-1]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    forecasts_stacked.append(yhat)


# In[ ]:


rmse_test = np.sqrt(mean_squared_error(forecasts_van50, raw_test))
rmse_test
mean_abs_van50=((forecasts_van50-TWAP_test)/TWAP_test)*100.0
mean_abs_van50_=abs(mean_abs_van50)
np.mean(mean_abs_van50_)


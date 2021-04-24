import streamlit as st
import pandas as pd
from PIL import Image
import math
import numpy as np
from datetime import date
from plotly import graph_objs as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import model_from_json

st.title('Stock Forecast App Using GRU Model')

#image=Image.open("C:/Users/win10/webapp/Stock_Market_Changes.png")
#st.image(image,use_column_width=True)

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ('BBRI.JK', 'BBNI.JK')
st.sidebar.title('Data Aktual')
selected_stock = st.sidebar.selectbox('Choose stock market dataset ', stocks)

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

    
#data_load_state = st.sidebar.text('Loading data...')
data = load_data(selected_stock)

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def test_model(df,nama_model):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(nama_model)
    print("Loaded model from disk")

    data=df.filter({'Close'})
    dataset= data.values
    training_data_len = math.ceil(len(dataset)*0.8)

    #scaling model
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)

    #test
    test_data=scaled_data[training_data_len-60:,:]
    x_test=[]
    y_test=dataset[training_data_len:,:]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i,0])

    x_test=np.array(x_test)
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

    #GRU
    predictions=loaded_model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    #visualization
    train=data[:training_data_len]
    valid=data[training_data_len:]
    valid['Predictions']=predictions
    plt.figure(figsize=(16,8))
    plt.title('Model GRU')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('close price (RP)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train','Val','Predictions'], loc='upper right')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def prediksi(df,nama_model):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(nama_model)
    print("Loaded model from disk")


    data=df.filter({'Close'})
    dataset= data.values
    training_data_len = math.ceil(len(dataset)*0.8)

    #scaling model
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)

    #test
    test_data=scaled_data[training_data_len-60:,:]

    x_input=test_data[211:].reshape(1,-1)
    x_input.shape

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()


    df1=df.reset_index()['Close']
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    # demonstrate prediction for next 10 days

    lst_output=[]
    n_steps=100
    i=0
    while(i<30):
        
        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = loaded_model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = loaded_model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
        

    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)

    plt.plot(day_new,scaler.inverse_transform(df1[1160:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

def dataframe_prediksi(df):
    nama_model="modelBRI.h5"
    prediksi(df,nama_model)

def dataframe_test_model(df):
    nama_model="modelBRI.h5"
    test_model(df,nama_model)
   

plot_raw_data()
    
# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input history 5 years stock market dataset for predictions", type=["csv"])
if uploaded_file is not None:
    uploaded_file = pd.read_csv(uploaded_file)
    testing_tombol=st.sidebar.button('Testing Model')
    if testing_tombol:
        dataframe_test_model(uploaded_file)
    predict_tombol=st.sidebar.button('Predict for 30 days in the future')
    if predict_tombol:
        dataframe_prediksi(uploaded_file)




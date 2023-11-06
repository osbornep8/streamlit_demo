import streamlit as st
import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

@st.cache_data
def get_data(filename):
    taxi_data = pd.read_parquet(filename)
    return taxi_data 
@st.cache_data
def set_model_params():
    X = taxi_data[[input_feature]].to_numpy()
    X = X.reshape(-1,1)
    y = taxi_data[['trip_distance']].to_numpy()
    y = y.reshape(-1,1)
    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, test_size=0.2)
    min_sample_split = math.ceil(0.05*X_train[:][1])
    

    # regr.fit(X,y)
    # prediction = regr.predict(y)
    return X_train, y_train, X_test, y_test, min_sample_split

with header:
    st.title("Welcome to my first Data Science Project in Streamlit")

with dataset:
    st.header("NYC Taxi Dataset")
    st.text("Data Source: ")
    taxi_data = get_data('data/yellow_tripdata_2023-08.parquet')
    # st.write(taxi_data.head())
    # st.write(taxi_data.describe())
    st.subheader("Pick-up Location ID Distribution on NYC Dataset")
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)
with features:
    st.header("Features I Created")
    st.markdown("* **First feature:** I created the first feature becuz of this.. I used the following logic..")
    st.markdown("* **Second feature:** I created the second feature based on... I calculated it using..")
with model_training:
    st.header("Time to Train The Model!")
    st.text('Here you get to choose the hyperparamets of the model and see how the performance goes!')
    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider('What should be the max depth of the model?', min_value=10, max_value=100, value=10, step=10)
    n_estimators = sel_col.selectbox('How many trees should there be?', options=[10,50,100,'No limit'], index=0)
    sel_col.text('Here is a list of features to choose from: ')
    sel_col.table(taxi_data.columns)
    input_feature = sel_col.text_input('Which feature should be used as Input Feature?', 'PULocationID')

    X_train, y_train, _, _, min_sample_split = set_model_params()

    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth, min_samples_split=min_sample_split)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    # X = taxi_data[[input_feature]].to_numpy()
    # X = X.reshape(-1,1)
    # y = taxi_data[['trip_distance']].to_numpy()
    # y = y.reshape(-1,1)
    # st.write(X.shape, y.shape)
    
    regr.fit(X_train,y_train)
    prediction = regr.predict(y_train)
    # y, prediction = set_model(max_depth, n_estimators)

    disp_col.subheader('Mean Absolute Error of the model is: ')
    disp_col.write(mean_absolute_error(y_train,prediction))
    disp_col.subheader('Mean Squared Error of the model is: ')
    disp_col.write(mean_squared_error(y_train,prediction))
    disp_col.subheader('R sqaured score of the model is: ')
    disp_col.write(r2_score(y_train,prediction))
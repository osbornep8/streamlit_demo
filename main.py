import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

@st.cache_data
def get_data(filename):
    taxi_data = pd.read_parquet(filename)
    return taxi_data 

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
    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No limit'], index=0)
    sel_col.text('Here is a list of features to choose from: ')
    sel_col.table(taxi_data.columns)
    input_feature = sel_col.text_input('Which feature should be used as Input Feature?', 'PULocationID')

    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]].to_numpy()
    X = X.reshape(-1,1)
    y = taxi_data[['trip_distance']].to_numpy()
    y = y.reshape(-1,1)
    st.write(X.shape, y.shape)

    regr.fit(X,y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean Absolute Error of the model is: ')
    disp_col.write(mean_absolute_error(y,prediction))
    disp_col.subheader('Mean Squared Error of the model is: ')
    disp_col.write(mean_squared_error(y,prediction))
    disp_col.subheader('R sqaured score of the model is: ')
    disp_col.write(r2_score(y,prediction))
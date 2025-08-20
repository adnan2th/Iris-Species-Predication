import streamlit as st
import joblib
import pandas as pd
import numpy as np


Model = joblib.load("irise.model")
Iris_data =pd.read_csv("Iris.csv")

st.set_page_config(
    page_title="Iris Species Detection",  # Browser tab title
    page_icon="🌸",  # Emoji as icon
    layout="wide"
)



st.title("Iris Species Detection App")
st.header("Know Find it?")

# Create a slider
col1, col2 = st.columns(2)

with col1:

    sepal_lenght = st. slider('Sepal length (cm)', float(Iris_data['SepalLengthCm'].min()), float(Iris_data['SepalLengthCm'].max()), float(Iris_data['SepalWidthCm'].mean()))
    petal_lenght = st. slider('Petal length (cm)', float(Iris_data['PetalLengthCm'].min()), float(Iris_data['PetalLengthCm'].max()), float(Iris_data['PetalWidthCm'].mean()))

with col2:
    sepal_Width = st. slider('Sepal Width (cm)', float(Iris_data['SepalWidthCm'].min()), float(Iris_data['SepalWidthCm'].max()), float(Iris_data['SepalWidthCm'].mean()))
    petal_Width = st. slider('Petal Width (cm)', float(Iris_data['PetalWidthCm'].min()), float(Iris_data['PetalWidthCm'].max()), float(Iris_data['PetalWidthCm'].mean()))


st.button("Predict")
    # Make prediction
Input_data =(sepal_lenght,petal_lenght,sepal_Width,petal_Width)
input_data_as_numpy_array = np.asarray(Input_data)

# Reshape  NumPy array as we predicting only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = Model.predict(input_data_reshaped)
    
# Show result
st.success(f"This is likely a **{prediction}** iris")


if prediction == "Iris-setosa":
        st.image("D:\FYP\Other Projects\Iris Specie\\51376589362_b92e27ae7a_b.jpg",  width=300)

elif prediction == "Iris-versicolor":
        st.image("D:\FYP\Other Projects\Iris Specie\Iris-versicolor-Blue-Flag-Iris-Flower-scaled.jpg",  width=300)

else:
    st.image("D:\FYP\Other Projects\Iris Specie\images.jpeg", width=300)


import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle

loaded_model = pickle.load(open('linear_model.pkl', 'rb'))


def predict(model, data):
  predictions = model.predict(data)

  return predictions

image1 = Image.open('picture1.png').resize((500,500))
image2 = Image.open('picture2.png').resize((500,500))
image3 = Image.open('picture3.png')
image4 = Image.open('picture4.jpg').resize((500,300))


def run():

  st.sidebar.image(image4)
  st.sidebar.title('Page navigator')
  option = st.sidebar.selectbox(
     'selection',
     ('App', 'Description'))
  
  st.sidebar.write(
        "This web application is Kaggle's implementation from Body Fat Prediction Dataset : https://www.kaggle.com/fedesoriano/body-fat-prediction-dataset"
    )
  
  if option == 'Description':
    st.title('**_Model Description page_**')
    st.write(
        '       The model was generated using a SimpleLinearRegression algorithm with data that has already been engineered and extracted.'
    )
    st.write(
        '       With helping of Pycaret for feature selection and selecting a baseline model, also pipeline creating with sklearn.'
    )
    
    col1, col2 = st.columns(2)

    with col1:
      st.caption('Model Status & test prediction instance')
      st.image(image1)

    with col2:
      st.caption('Feature Important')
      st.image(image2)

    
    st.write('**_Strengths_** : The model was good at predicting the overall outcome range and simple to interpretion by feature selection.')
    st.write('**_Weakness_** : Because the dataset was generated from only male samples, the model was biased due to Stratified sampling method, and the RMSE was approximately 4% which the result could be shifted up to one division in some predictions when comparing to the previous chart.')




  else:
    st.image(image3)
    st.title('**_Body Fat Percentage Prediction_**')
    st.write('Observing and fill the feature below to get the expected body fat percentage.')
    weight_kg = st.number_input('Weight : in kilograms')
    height_cm = st.number_input('Height : in cm')
    Abdomen = st.number_input('Abdomen : in cm')
    Wrist = st.number_input('Wrist : in cm')

    bmi = 0

    input_name = ['Abdomen',	'Weight_kg',	'BMI',	'Wrist',	'Height_cm']
    input_value =  [Abdomen,weight_kg,bmi,Wrist,height_cm]

    dict_list = dict(zip(input_name,input_value))
    dataframe = pd.DataFrame([dict_list])

    if st.button('Predicting..'):
      dataframe['BMI'] = weight_kg / ((height_cm / 100) ** 2)
      output = predict(loaded_model, dataframe)
      st.success('Your approximately body fat percentage is : {:.2f} %'.format(output[0]))


if __name__ == '__main__':
    run()








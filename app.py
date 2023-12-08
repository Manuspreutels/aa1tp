# app.py
import streamlit as st
import joblib
import pandas as pd
import os
from funciones_y_clases import variables_prediccion, feat_eng, NeuralNetwork, input_features, WindDirToXDir, WindDirToYDir, code_mesesY, code_mesesX, code_lluvia, pipeline_predict, r_squared

# Se carga el pipeline del modelo.
path_dir=os.path.dirname(os.path.abspath(__file__))
path=os.path.join(path_dir, 'pipeline.joblib')
pipe = joblib.load(path)

st.title('Predicción de lluvia')

input_texto = ["WindDir3pm", "WindDir9am", "WindGustDir", 'RainToday', 'Date']
input_number = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Cloud9am', 'Cloud3pm']

def get_user_input():
    """
    esta función genera los inputs del frontend de streamlit para que el usuario pueda cargar los valores.
    Además, contiene el botón para hacer el submit y obtener la predicción.
    No hace falta hacerlo así, las posibilidades son infinitas.
    """
    input_dict = {}

    with st.form(key='my_form'):
        #input_dict["Date"] = st.date_input(f"Ingrese la fecha", key='"date"')
        i = 0
        for feat in input_texto:
            input_value = st.text_input(f"Ingrese el valor de {feat}", None, key=f"text{i}")
            input_dict[feat] = input_value
            i += 1

        i = 0
        for feat in input_number:
            input_value = st.number_input(f"Ingrese el valor de {feat}", value=0.0, step=0.01, key=f"number{i}")
            input_dict[feat] = input_value
            i += 1
       
        submit_button = st.form_submit_button(label='Submit')

    return pd.DataFrame([input_dict]), submit_button


user_input, submit_button = get_user_input()


# When the 'Submit' button is pressed, perform the prediction
if submit_button:
    # Predict wine quality
    prediction = pipeline_predict(pipe, user_input)
    prediction_value = prediction[0]

    # Display the prediction
    st.header("Lluvia predecida")
    st.write(prediction_value)
    


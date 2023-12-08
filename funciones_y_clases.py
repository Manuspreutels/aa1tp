import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

code_lluvia = {"Yes":1, "No":0}

code_mesesX = {
    1: np.cos(np.pi/4),
    2: np.cos(np.pi/4),
    3: np.cos(3*np.pi/4),
    4: np.cos(3*np.pi/4),
    5: np.cos(3*np.pi/4),
    6: np.cos(5*np.pi/4),
    7: np.cos(5*np.pi/4),
    8: np.cos(5*np.pi/4),
    9: np.cos(7*np.pi/4),
    10: np.cos(7*np.pi/4),
    11: np.cos(7*np.pi/4),
    12: np.cos(np.pi/4)
}
code_mesesY = {
    1: np.sin(np.pi/4),
    2: np.sin(np.pi/4),
    3: np.sin(3*np.pi/4),
    4: np.sin(3*np.pi/4),
    5: np.sin(3*np.pi/4),
    6: np.sin(5*np.pi/4),
    7: np.sin(5*np.pi/4),
    8: np.sin(5*np.pi/4),
    9: np.sin(7*np.pi/4),
    10: np.sin(7*np.pi/4),
    11: np.sin(7*np.pi/4),
    12: np.sin(np.pi/4)
}



WindDirToXDir = {
    'N': 0,
    'S': 0,
    'E': 1,
    'W': -1,
    'NW': 2**0.5,
    'NE': -2**0.5,
    'SE': 2**0.5,
    'SW': -2**0.5,
    'NNW': np.cos(5*np.pi/8),
    'NNE': np.cos(3*np.pi/8),
    'ENE': np.cos(1*np.pi/8),
    'WNW': np.cos(7*np.pi/8),
    'ESE': np.cos(-1*np.pi/8),
    'SSE': np.cos(-3*np.pi/8),
    'SSW': np.cos(-5*np.pi/8),
    'WSW': np.cos(-7*np.pi/8)
}
WindDirToYDir = {
    'N': 1,
    'S': -1,
    'E': 0,
    'W': 0,
    'NW': 2**0.5,
    'NE': 2**0.5,
    'SE': -2**0.5,
    'SW': -2**0.5,
    'NNW': np.sin(3*np.pi/8),
    'NNE': np.sin(3*np.pi/8),
    'ENE': np.sin(1*np.pi/8),
    'WNW': np.sin(1*np.pi/8),
    'ESE': np.sin(-1*np.pi/8),
    'SSE': np.sin(-3*np.pi/8),
    'SSW': np.sin(-3*np.pi/8),
    'WSW': np.sin(-1*np.pi/8)
}

variables_prediccion = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',  'WindSpeed9am', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Cloud9am', 'Cloud3pm', 'RainToday', 'HumidityChange', 'SeasonX', 'SeasonY', 'WindDir3pmX', 'WindDir3pmY', 'WindDir9amX', 'WindDir9amY', 'WindGustDirX', 'WindGustDirY']
 
input_features = ['Date', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 	'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Cloud9am', 'Cloud3pm', 'RainToday']

def feat_eng(df):
    df.columns = df.columns.str.replace(' ', '_')
    df["Date"] = pd.to_datetime(df['Date'])
    df['HumidityChange'] = df['Humidity3pm'] - df['Humidity9am']

    df['SeasonX'] = df["Date"].dt.month.replace(code_mesesX)
    df['SeasonY'] = df["Date"].dt.month.replace(code_mesesY)

    df['WindDir3pmX'] = df['WindDir3pm'].replace(WindDirToXDir)
    df['WindDir3pmY'] = df['WindDir3pm'].replace(WindDirToYDir)
    df['WindDir9amX'] = df['WindDir9am'].replace(WindDirToXDir)
    df['WindDir9amY'] = df['WindDir9am'].replace(WindDirToYDir)
    df['WindGustDirX'] = df['WindGustDir'].replace(WindDirToXDir)
    df['WindGustDirY'] = df['WindGustDir'].replace(WindDirToYDir)
    df['RainToday'] = df["RainToday"].replace(code_lluvia)
    df2 = df[variables_prediccion]
    return df2

def r_squared(y_true, y_pred):
    r2 = 1 - tf.reduce_sum(tf.square(y_true - y_pred)) / tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return r2

class NeuralNetwork(BaseEstimator, TransformerMixin):
    def __init__(self, n_layers, n_units, n_out, activation, loss, metrics):
        model = Sequential()
        for i in range(n_layers):
            model.add(Dense(n_units[i], activation='sigmoid')) # capas densas con activacion sigmoide
        # capa de salida
        model.add(Dense(n_out, activation='sigmoid'))
        # compilar
        model.compile(optimizer='adam', loss=loss, metrics=metrics)
        self.model = model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=0)
        return self

    def transform(self, X):
        return None
    
    def predict(self, X):
        self.model.predict(X)
        return self

def pipeline_fit(pipeline, x_tr, y_tr):
  trainnew = feat_eng(x_tr)
  pipeline.fit(trainnew, y_tr)

def pipeline_predict(pipeline, x):
  xnew = feat_eng(x)
  pipeline.predict(xnew)

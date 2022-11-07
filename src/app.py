from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo de análisis de tweets"
    # 1. Endpoint que devuelva la predicción de los nuevos datos enviados mediante argumentos en la llamada
@app.route('/prediccion', methods=['GET'])
def predict():
    model = pickle.load(open('sentiment_model','rb'))

    text = request.args.get('texto', None)
    prediction = model.predict([text])
    
    return "La predicción del sentimiento del tweet es: " + str(prediction) + ", siendo 0 positivo y 1 negativo."

    # if text is None:
    #     return "No se ha introducido el texto."
    # else:
    #     prediction = model.predict([text])
    #     return "La predicción del sentimiento del tweet es: " + prediction

# 2. Endpoint que reentrene de nuevo el modelo 
@app.route('/mejorprediccion', methods=['GET'])
def retrain():
    connection = sqlite3.connect('thebridge_tweets.db')
    cursor = connection.cursor()
    select_books = "SELECT tweet FROM tweets"
    result = cursor.execute(select_books).fetchall()
    names = [description[0] for description in cursor.description]
    connection.close()
    
    df = pd.DataFrame(result, columns=names)
    
    

    X = df.drop(columns=['sales'])
    y = df['sales']

    model = pickle.load(open('sentiment_model','rb'))
    model.fit(X,y)
    pickle.dump(model, open('data/advertising_model_v1','wb'))

    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')

    return "New model retrained and saved as advertising_model_v1. The results of MAE with cross validation of 10 folds is: " + str(abs(round(scores.mean(),2)))
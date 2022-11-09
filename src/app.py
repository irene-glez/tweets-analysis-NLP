from flask import Flask, request, jsonify
import os
import pickle


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

# 0.Home
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


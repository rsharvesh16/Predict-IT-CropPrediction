from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from flask import jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()

app = Flask(__name__, static_folder="templates/assets")

def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        features = [float(request.form['Nitrogen_value']),
                    float(request.form['Phosphorus_value']),
                    float(request.form['Potassium_value']),
                    float(request.form['Temperature_value']),
                    float(request.form['Humidity_value']),
                    float(request.form['Ph_value']),
                    float(request.form['Rainfall_value'])]
        single_pred = np.array(features).reshape(1, -1)

        # Make a prediction using the loaded model
        loaded_model = load_model('model1-randomforest.pkl')
        prediction = loaded_model.predict(single_pred)
        print(prediction)
        # Display the predicted crop on the result page
        return render_template('predict.html', prediction = prediction[0])
    return render_template('predict.html')  # This line ensures predict.html is rendered when method is GET

# Redirect to the predict.html page when the predict button is clicked


@app.route('/chat', methods=['POST', 'GET'])
def chat():
    if request.method == 'POST':
        input_text = request.form['input_text']

        input_prompt = """If the question is related to soil or crop or nitrogen, postassium, phosphorous,
        rainfall, soil ph, or humidity or temperature, answer related to that
        else dont answer. You have to answer to the question in three or four lines and you have to answer only in english."""

        response = generate_response(input_text, input_prompt)

        return render_template('chat.html', input_text=input_text, response=response)

    return render_template('chat.html')


def generate_response(input_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content([input_text, "", prompt])
    return response.text

if __name__ == '__main__':
    app.run(debug=True)

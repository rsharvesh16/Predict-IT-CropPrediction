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
       
        # Display the predicted crop on the result page
        return render_template('predict.html', prediction = prediction[0])
    return render_template('predict.html')  # This line ensures predict.html is rendered when method is GET

# Redirect to the predict.html page when the predict button is clicked


@app.route('/chatbot', methods = ['GET', 'POST'])
def ai_chatbot():
    os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    if request.method=='POST':
        input_text = request.form['input_text']
        print(input_text)
        uploaded_file = request.files['uploaded_file']

        model = genai.GenerativeModel('gemini-pro-vision')
        image_parts = [{"mime_type": uploaded_file.mimetype, "data": uploaded_file.read()}]
        print(image_parts)
        response = model.generate_content([input_text, image_parts[0], ""])
        print(response)

        return render_template('chatbot.html', input_text = input_text, response=response.text)
    return render_template('chatbot.html')


if __name__ == '__main__':
    app.run(debug=True)

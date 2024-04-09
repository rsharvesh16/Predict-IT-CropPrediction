from flask import Flask, render_template, request, redirect, url_for, Markup
import pickle
import numpy as np
from flask import jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import sqlite3
from model import predict_image
import content

load_dotenv()

app = Flask(__name__, static_folder="")

def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

def create_db():
    conn = sqlite3.connect('input_data.db')
    cursor = conn.cursor()

    # Create a table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS input_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            temperature REAL,
            humidity REAL,
            ph_value REAL,
            rainfall REAL,
            predicted_crop TEXT
        )
    ''')

    conn.commit()
    conn.close()

# Call the function to create the database and table
create_db()

# Define the home route
@app.route('/')
@app.route('/home')
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
        predicted_crop = prediction[0]

        store_input_data(features, predicted_crop)

        # Display the predicted crop on the result page
        return render_template('predict.html', prediction=predicted_crop)
    return render_template('predict.html')  # This line ensures predict.html is rendered when method is GET

# Redirect to the predict.html page when the predict button is clicked
def store_input_data(features, predicted_crop):
    conn = sqlite3.connect('input_data.db')
    cursor = conn.cursor()

    # Insert input values into the table
    cursor.execute('''
        INSERT INTO input_data (nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall, predicted_crop)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', tuple(features + [predicted_crop]))
    
    conn.commit()
    conn.close()


@app.route('/detect', methods=['GET','POST'])
def detect():
    return render_template('detect.html')


@app.route('/chat', methods=['POST', 'GET'])
def chat():
    if request.method == 'POST':
        input_text = request.form['input_text']

        input_prompt = """You are an expert in understanding soil, agriculture, climate, crops and plant diseases.
        If we provide or ask you anything related to soil or crop or nitrogen, postassium, phosphorous,
        rainfall, soil ph, or humidity or temperature or if any question related to Plant Disease, answer related to that
        else Dont Answer and Say - "I am Optimized only for Agricultural and Plant Disease, Please ask any quereies related to that, Thank You, Team -TechTriad. You have to answer to the question in two or three or four lines and you have to answer only in english and it should be easily understandable."""

        # Handle text input
        response = generate_response(input_text, input_prompt)
        return render_template('chat.html', input_text=input_text, response=response)

    return render_template('chat.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            file = request.files['img']
            if file:
                # Save the file to a temporary location or process it directly
                # Example: file.save('uploads/' + secure_filename(file.filename))
                img = file.read()
                prediction = predict_image(img)
                res = Markup(content.disease_dic.get(prediction))
                if res:

                    return render_template('result.html', result=res)
                else:
                    return render_template('result.html', result="Unknown result")
            else:
                return render_template('result.html', result="No file uploaded")
        except Exception as e:
            print(e)  # Log the exception for debugging purposes
            return render_template('result.html', result="Error occurred")
    return render_template("result.html")


def generate_response(input_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content([input_text, "", prompt])
    return response.text

def get_gemini_response(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        image_parts = [{
            "mime_type" : uploaded_file.content_type,
            "data" : bytes_data
        }]
        return image_parts
    else:
        raise FileNotFoundError("No file Found")



if __name__ == '__main__':
    app.run(debug=True)

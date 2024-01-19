from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from flask import jsonify

app = Flask(__name__, static_folder="templates/assets")

def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/Predict', methods=['POST', 'GET'])
def Predict():
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
        return render_template('predict.html', prediction=prediction[0])
    return render_template('predict.html')  # This line ensures predict.html is rendered when method is GET

# Redirect to the predict.html page when the predict button is clicked
@app.route('/predict_button_clicked')
def predict_button_clicked():
    return render_template('predict.html')

@app.route('/predict.html')  # Add this route for predict.html
def predict():
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)

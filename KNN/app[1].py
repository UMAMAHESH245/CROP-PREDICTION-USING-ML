# app.py

from flask import Flask, render_template, request
import pickle
# Create a Flask web application instance
app = Flask(__name__)

# Load the pre-trained KNN model using pickle
with open('knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)


# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')


# Define a route to handle crop prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve the input values from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pH = float(request.form['pH'])
        rainfall = float(request.form['rainfall'])

        # Use the loaded KNN model to make a prediction
        input_values = [[N, P, K, temperature, humidity, pH, rainfall]]
        prediction = knn_model.predict(input_values)

        # Determine the predicted crop label
        crop_dict = {
            1: "Rice",
            2: "Maize",
            3: "Jute",
            4: "Cotton",
            5: "Coconut",
            6: "Papaya",
            7: "Orange",
            8: "Apple",
            9: "Muskmelon",
            10: "Watermelon",
            11: "Grapes",
            12: "Mango",
            13: "Banana",
            14: "Pomegranate",
            15: "Lentil",
            16: "Black Gram",
            17: "Mung Bean",
            18: "Moth Beans",
            19: "Pigeon Peas",
            20: "Kidney Beans",
            21: "Chickpea",
            22: "Coffee"
        }
        predicted_crop = crop_dict[prediction[0]]

        return render_template('result.html', crop_name=predicted_crop)


if __name__ == '__main__':
    app.run(debug=True)

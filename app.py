from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve user input from the form
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    gender = request.form['gender']
    work_type = request.form['work_type']
    residence_type = request.form['residence_type']
    smoking_status = request.form['smoking_status']

    # Load the trained model
    with open('RandomForest.pkl', 'rb') as file:
        predictor = pickle.load(file)

    # Preprocess user input
    user_data = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'female': gender == "female",
        'male': gender == "male",
        'Other': gender == "other",
        'government_work': work_type == "government",
        'private_work': work_type == "private",
        'self_employed': work_type == "self-employed",
        'children_work': work_type == "children",
        'never_worked': work_type == "never worked",
        'rural_resident': residence_type == "rural",
        'urban_resident': residence_type == "urban",
        'formerly_smoked': smoking_status == "formerly smoked",
        'never_smoked': smoking_status == "never smoked",
        'smokes': smoking_status == "smokes"
    }

    # Construct input data with the correct number of features
    features = ['age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi',
                'female', 'male', 'Other', 'government_work', 'private_work', 'self_employed',
                'children_work', 'never_worked', 'rural_resident', 'urban_resident', 'formerly_smoked',
                'never_smoked', 'smokes']
    X_test = np.array([user_data[feature] for feature in features]).reshape(1, -1)

    # Make predictions
    predictions = predictor.predict(X_test)

    # Determine prediction result
    if predictions[0] == 1:
        result = "You are at risk of a stroke!"
    else:
        result = "You are not at risk of a stroke."

    # Render the result page with the prediction result
    return render_template('result.html', result = result)

if __name__ == '__main__':
    app.run(debug=True)

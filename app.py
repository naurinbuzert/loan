from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('logistic.pkl')


# ---------------- HOME PAGE ----------------
@app.route('/')
def home():
    return render_template('home.html')


# ---------------- LOAN PAGE ----------------
@app.route('/loan')
def loan_page():
    return render_template('loan.html')


# ---------------- PREDICTION ----------------
@app.route('/predict', methods=['POST'])
def predict():
    data = [
        int(request.form['Age']),
        int(request.form['Gender']),
        int(request.form['Education']),
        float(request.form['Income']),
        int(request.form['Emp_exp']),
        int(request.form['Home_ownership']),
        int(request.form['loan_amnt']),
        int(request.form['loan_intent']),
        float(request.form['loan_int_rate']),
        float(request.form['loan_percent_income']),
        int(request.form['cb_person_cred_hist_length']),
        float(request.form['credit_score']),
        int(request.form['previous_loan_defaults_on_file'])
    ]

    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        result = "Loan Approved"
    else:
        result = "Loan Not Approved"

    return render_template('loan.html', prediction_text=result)


# ---------------- CONTACT PAGE ----------------
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    success = False

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        print("New Contact Message:")
        print("Name:", name)
        print("Email:", email)
        print("Message:", message)

        success = True

    return render_template('contact.html', success=success)


if __name__ == "__main__":
    app.run(debug=True)
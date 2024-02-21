from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('credit_score_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data
    Num_Bank_Accounts = float(request.form['Num_Bank_Accounts'])
    Annual_Income = float(request.form['Annual_Income'])
    Interest_Rate= float(request.form['Interest_Rate'])
    Monthly_Inhand_Salary = float(request.form['Monthly_Inhand_Salary'])
    Num_Credit_Card = float(request.form['Num_Credit_Card'])
    Delay_from_due_date =float(request.form['Delay_from_due_date'])
    Num_of_Delayed_Payment = float(request.form['Num_of_Delayed_Payment'])
    Num_Credit_Inquiries = float(request.form['Num_Credit_Inquiries'])
    Outstanding_Debt = float(request.form['Outstanding_Debt'])
    Total_EMI_per_month = float(request.form['Total_EMI_per_month'])
    Changed_Credit_Limit = float(request.form['Changed_Credit_Limit'])
    Monthly_Balance = float(request.form['Monthly_Balance'])
    Credit_Mix = request.form['Credit_Mix']
    Credit_Utilization_Ratio = float( request.form['Credit_Utilization_Ratio'])
    Num_of_Loan= request.form['Num_of_Loan']
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()

    Credit_Mix_encoded = label_encoder.fit_transform([Credit_Mix])
    


    # Define the top features used in the model
    top_features = [Num_Bank_Accounts, Annual_Income, Interest_Rate, Monthly_Inhand_Salary,
                  Num_Credit_Card, Delay_from_due_date, Num_of_Delayed_Payment,
                  Num_Credit_Inquiries, Outstanding_Debt, Total_EMI_per_month,
                  Changed_Credit_Limit, Monthly_Balance, Credit_Mix_encoded,
                  Credit_Utilization_Ratio, Num_of_Loan]
               
    input_data = [float(i) for i  in top_features]
    data = np.array(input_data).reshape(1,-1)
    prediction = model.predict(data)[0]
    
    if prediction == 'Poor':
        result = f"You are not eligible for loan and your credit score is {prediction}"
    elif prediction == 'Standard':
        result = f"You are eligible for loan and your credit score is {prediction}"
    elif prediction == 'Good':
        result = f"You are eligible for a loan and your credit score is {prediction}"
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
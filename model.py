import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv(r"C:\Users\LENOVO\final_credit_data.csv")



X=data.drop(['Credit_Score'],axis=1)
y=data['Credit_Score']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)
from imblearn.over_sampling import SMOTE
# Assuming X_train contains your feature vectors and y_train contains the corresponding labels

# Instantiate SMOTE
smote = SMOTE(random_state=42)

# Resample the dataset
x_train_samp, y_train_samp = smote.fit_resample(X_train, y_train)

# Check the distribution of classes after resampling
unique, counts = np.unique(y_train_samp, return_counts=True)
print(dict(zip(unique, counts)))
X_train,X_test,y_train,y_test=train_test_split(x_train_samp, y_train_samp,test_size=0.20,random_state=42)

# Define the Random Forest classifier with n_estimators=[10]
top_features= ['Interest_Rate', 'Outstanding_Debt', 'Credit_Mix',
       'Delay_from_due_date', 'Num_Credit_Inquiries', 'Changed_Credit_Limit',
       'Num_Bank_Accounts', 'Monthly_Balance', 'Credit_Utilization_Ratio',
       'Num_of_Loan', 'Num_Credit_Card', 'Total_EMI_per_month',
       'Monthly_Inhand_Salary', 'Num_of_Delayed_Payment', 'Annual_Income']

X_train_selected = X_train[top_features ]
X_test_selected = X_test[top_features ]

model = RandomForestClassifier()

# Train the classifier with your data
model.fit(X_train_selected, y_train) 
model_predictions=model.predict(X_test_selected)
# Evaluate the model
accuracy = accuracy_score(y_test, model_predictions)
print("Accuracy:", accuracy)
pickle.dump(model,open('model.pkl','wb'))#save trained model.
import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

st.header('Churn Prediction using XGBoost Classifier')
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Remove identifier columns
df.drop(columns='customerID', inplace=True)

# Replace empty values with NaN
df['TotalCharges'] = df['TotalCharges'].replace('', np.nan)

# Convert the column to numeric, replacing non-numeric values with NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Change Total Charges datatype into float
df['TotalCharges'] = df['TotalCharges'].astype(float)

# Fill null value in 'TotalCharges' with the mean
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

# Loop over each column in the DataFrame where dtype is 'object'
for col in df.select_dtypes(include=['object']).columns:
    # Initialize a LabelEncoder object
    label_encoder = preprocessing.LabelEncoder()
    # Fit the encoder to the unique values in the column
    label_encoder.fit(df[col].unique())
    # Transform the column using the encoder
    df[col] = label_encoder.transform(df[col])

# Select the features (X) and the target variable (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train the XGBoost classifier
xgb = XGBClassifier(gamma=0.2, learning_rate=0.1, max_depth=3, n_estimators=200)
xgb.fit(X_train, y_train)

st.subheader('User Input for Churn Prediction')

gender = st.radio("What's your Gender ?", ('Female', 'Male'))
gender_mapping = {'Female': 0, 'Male': 1}
gender_encoded = gender_mapping.get(gender, 0)

SeniorCitizen = st.radio("Are you a Senior Citizen ?", ('No', 'Yes'))
SeniorCitizen_mapping = {'No': 0, 'Yes': 1}
SeniorCitizen_encoded = SeniorCitizen_mapping.get(SeniorCitizen, 0)

Partner = st.radio("Do you have a partner ?", ('No', 'Yes'))
Partner_mapping = {'No': 0, 'Yes': 1}
Partner_encoded = Partner_mapping.get(Partner, 0)

Dependent = st.radio("Do you have a dependent ?", ('No', 'Yes'))
Dependent_mapping = {'No': 0, 'Yes': 1}
Dependent_encoded = Dependent_mapping.get(Dependent, 0)

Tenure = st.number_input('Number of months you have stayed with the company')

PhoneService = st.radio("Do you have a phone service or not ?", ('No', 'Yes'))
PhoneService_mapping = {'No': 0, 'Yes': 1}
PhoneService_encoded = PhoneService_mapping.get(PhoneService, 0)

MultipleLines = st.radio("Do you have multiple lines or not ?", ('No phone service', 'No', 'Yes'))
MultipleLines_mapping = {'No phone service': 1, 'No': 0, 'Yes': 2}
MultipleLines_encoded = MultipleLines_mapping.get(MultipleLines, 0)

InternetService = st.radio("What type of Internet Service do you have?", ('DSL', 'Fiber optic', 'No'))
InternetService_mapping = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
InternetService_encoded = InternetService_mapping.get(InternetService, 0)

OnlineSecurity = st.radio("Do you have online security or not ?", ('No internet service', 'No', 'Yes'))
OnlineSecurity_mapping = {'No internet service': 1, 'No': 0, 'Yes': 2}
OnlineSecurity_encoded = OnlineSecurity_mapping.get(OnlineSecurity, 0)

OnlineBackup = st.radio("Do you have online backup or not ?", ('No internet service', 'No', 'Yes'))
OnlineBackup_mapping = {'No internet service': 1, 'No': 0, 'Yes': 2}
OnlineBackup_encoded = OnlineBackup_mapping.get(OnlineBackup, 0)

DeviceProtection = st.radio("Do you have device protection or not ?", ('No internet service', 'No', 'Yes'))
DeviceProtection_mapping = {'No internet service': 1, 'No': 0, 'Yes': 2}
DeviceProtection_encoded = DeviceProtection_mapping.get(DeviceProtection, 0)

TechSupport = st.radio("Do you have tech support or not ?", ('No internet service', 'No', 'Yes'))
TechSupport_mapping = {'No internet service': 1, 'No': 0, 'Yes': 2}
TechSupport_encoded = TechSupport_mapping.get(TechSupport, 0)

StreamingTV = st.radio("Do you have streaming TV or not ?", ('No internet service', 'No', 'Yes'))
StreamingTV_mapping = {'No internet service': 1, 'No': 0, 'Yes': 2}
StreamingTV_encoded = StreamingTV_mapping.get(StreamingTV, 0)

StreamingMovies = st.radio("Do you have streaming Movies or not ?", ('No internet service', 'No', 'Yes'))
StreamingMovies_mapping = {'No internet service': 1, 'No': 0, 'Yes': 2}
StreamingMovies_encoded = StreamingMovies_mapping.get(StreamingMovies, 0)

Contract = st.radio("What is your contract term ?", ('Month-to-month', 'One year', 'Two year'))
Contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
Contract_encoded = Contract_mapping.get(Contract, 0)

PaperlessBilling = st.radio("Do you have paperless billing nor not ?", ('No', 'Yes'))
PaperlessBilling_mapping = {'No': 0, 'Yes': 1}
PaperlessBilling_encoded = PaperlessBilling_mapping.get(PaperlessBilling, 0)

PaymentMethod = st.radio("What is your payment method ?", ('Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'))
PaymentMethod_mapping = {'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1, 'Electronic check': 2, 'Mailed check': 3}
PaymentMethod_encoded = PaymentMethod_mapping.get(PaymentMethod, 0)

MonthlyCharges = st.number_input('Amount Charged to you each month :')

TotalCharges = st.number_input('Total Amount Charged to you :')

X_new = [[gender_encoded, SeniorCitizen_encoded, Partner_encoded, Dependent_encoded, Tenure, PhoneService_encoded,
          MultipleLines_encoded, InternetService_encoded, OnlineSecurity_encoded, OnlineBackup_encoded,
          DeviceProtection_encoded, TechSupport_encoded, StreamingTV_encoded, StreamingMovies_encoded,
          Contract_encoded, PaperlessBilling_encoded, PaymentMethod_encoded, MonthlyCharges, TotalCharges]]

if st.button('Predict Churn'):
    y_pred_prob = xgb.predict_proba(X_new)
    churn_probability = y_pred_prob[0, 1] * 100
    st.write(f'Churn Probability: {churn_probability:.2f}%')

    if churn_probability >= 50:
        st.write('Based on the input, the model predicts that you are likely to churn.')
    else:
        st.write('Based on the input, the model predicts that you are unlikely to churn.')


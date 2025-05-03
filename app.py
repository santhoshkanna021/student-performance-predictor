import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Title of the app
st.title("🎓 Student Performance Predictor")
st.write("Predict a student's average score based on their inputs")

# Load dataset
df = pd.read_csv("data/StudentsPerformance.csv")

# Create a new column for average score
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df_encoded.drop('average_score', axis=1)
y = df_encoded['average_score']

# Feature scaling (recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_scaled, y)

# --- Streamlit user inputs ---
st.header("📋 Input Student Information")

# Input fields
math_score = st.slider("Math Score", 0, 100, 70)
reading_score = st.slider("Reading Score", 0, 100, 70)
writing_score = st.slider("Writing Score", 0, 100, 70)
gender = st.selectbox("Gender", df['gender'].unique())
race = st.selectbox("Race/Ethnicity", df['race/ethnicity'].unique())
parent_edu = st.selectbox("Parental Education", df["parental level of education"].unique())
lunch = st.selectbox("Lunch", df["lunch"].unique())
test_prep = st.selectbox("Test Preparation", df["test preparation course"].unique())

# Create a new input row as a DataFrame
input_dict = {
    'math score': [math_score],
    'reading score': [reading_score],
    'writing score': [writing_score],
    'gender': [gender],
    'race/ethnicity': [race],
    'parental level of education': [parent_edu],
    'lunch': [lunch],
    'test preparation course': [test_prep]
}
input_df = pd.DataFrame(input_dict)

# Encode the input in the same way as training data
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

# Scale the input
input_scaled = scaler.transform(input_encoded)

# Predict
if st.button("Predict Average Score"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🎯 Predicted Average Score: {prediction:.2f}")

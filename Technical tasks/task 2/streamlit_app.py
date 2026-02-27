import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessors we saved earlier
model = joblib.load('income_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("📊 Adult Income Prediction Dashboard")
st.write("Adjust the demographic parameters in the sidebar to predict if a person's income exceeds $50K/year.")

# 1. Create a sidebar for user inputs
st.sidebar.header("User Demographics")

def get_user_input():
    age = st.sidebar.slider("Age", 17, 90, 39)
    workclass = st.sidebar.selectbox("Workclass", ["Private", "State-gov", "Federal-gov", "Self-emp-not-inc", "Local-gov"])
    education = st.sidebar.selectbox("Education", ["Bachelors", "HS-grad", "11th", "Masters", "Some-college"])
    marital_status = st.sidebar.selectbox("Marital Status", ["Never-married", "Married-civ-spouse", "Divorced"])
    occupation = st.sidebar.selectbox("Occupation", ["Adm-clerical", "Exec-managerial", "Prof-specialty", "Sales", "Craft-repair"])
    sex = st.sidebar.radio("Sex", ["Male", "Female"])
    hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)

    # Bundle everything into a dictionary (using default values for unlisted features to keep the UI clean)
    user_data = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": 189778, 
        "education": education,
        "education.num": 10, 
        "marital.status": marital_status,
        "occupation": occupation,
        "relationship": "Not-in-family",
        "race": "White",
        "sex": sex,
        "capital.gain": 0,
        "capital.loss": 0,
        "hours.per.week": hours_per_week,
        "native.country": "United-States"
    }
    return pd.DataFrame(user_data, index=[0])

# Store the input into a dataframe
input_df = get_user_input()

# Display the chosen parameters on the main page
st.subheader("Selected Profile")
st.write(input_df)

# 2. Add a prediction button
if st.button("Predict Income", type="primary"):

    # Match the preprocessing steps from the training phase
    query_encoded = pd.get_dummies(input_df)

    # Ensure the columns match exactly what the model expects, filling missing ones with 0
    query_encoded = query_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale the data
    query_scaled = scaler.transform(query_encoded)

    # Run the model
    prediction = model.predict(query_scaled)

    st.markdown("---")
    st.subheader("Prediction Result:")

    # Display the result
    if prediction[0] == 1:
        st.success("✅ **>50K**: This profile is predicted to make over $50,000 annually.")
    else:
        st.error("❌ **<=50K**: This profile is predicted to make under $50,000 annually.")

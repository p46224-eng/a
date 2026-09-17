import streamlit as st
import joblib
import pandas as pd

# Load the saved model
lr_model = joblib.load('linear.sav')

st.title('Sales Prediction App')
st.write('Enter the advertising budgets for TV, Radio, and Newspaper to predict sales.')

# Input fields for features
tv = st.number_input('TV Advertising Budget', min_value=0.0, value=100.0, step=0.1)
radio = st.number_input('Radio Advertising Budget', min_value=0.0, value=20.0, step=0.1)
newspaper = st.number_input('Newspaper Advertising Budget', min_value=0.0, value=10.0, step=0.1)

# Create a DataFrame for prediction
# Ensure the order of columns matches the training data ('TV', 'Radio', 'Newspaper')
input_data = pd.DataFrame([{'TV': tv, 'Radio': radio, 'Newspaper': newspaper}])

if st.button('Predict Sales'):
    prediction = lr_model.predict(input_data)[0]
    st.success(f'Predicted Sales: {prediction:.2f}')

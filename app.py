import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load your pre-trained model
# Load your pre-trained model
model = load_model("C:\\Users\\SUYASH\\stock_price_prediction_app\\my_model.keras")
# Update path if needed

# Load sample stock data
df = pd.read_csv("C:\\Users\\SUYASH\\stock_price_prediction_app\\stock_data.csv")
# Update the path if needed

# Function to preprocess user input and predict stock direction
def predict_stock_direction(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    prediction = model.predict(scaled_features)
    return (prediction > 0.5).astype(int)  # 1 = Up, 0 = Down

# Title of the app
st.title('Stock Price Direction Prediction')

# Input fields for stock data (user can input values)
st.subheader('Input Stock Data')

open_price = st.number_input('Open Price', min_value=0.0, value=100.0)
high_price = st.number_input('High Price', min_value=0.0, value=110.0)
low_price = st.number_input('Low Price', min_value=0.0, value=95.0)
close_price = st.number_input('Close Price', min_value=0.0, value=105.0)
volume = st.number_input('Volume', min_value=0, value=2000000)

# When the user clicks the "Predict" button
if st.button('Predict'):
    # Prepare the input features (same as during model training)
    user_input = np.array([[open_price, high_price, low_price, close_price, volume]])
    
    # Predict the stock direction (up/down)
    prediction = predict_stock_direction(user_input)
    
    # Display the result
    if prediction == 1:
        st.success('The stock price is predicted to go UP tomorrow!')
    else:
        st.error('The stock price is predicted to go DOWN tomorrow!')

 # Step 1: Load Your Neural Network Model and Define Prediction Function
import tensorflow as tf
import numpy as np

# Load your trained neural network model
# Replace this with code to load your specific model
model = tf.keras.models.load_model('crops.h5')

# Function to make predictions using the model
def make_prediction(input_data):
    # Preprocess input data as needed (e.g., convert to NumPy array)
    input_data_array = np.array(input_data)
    # Make predictions using the loaded model
    predictions = model.predict(input_data_array)
    # Return the predictions
    return predictions

# Step 2: Define Your Streamlit App Code
import streamlit as st


# Define label mapping dictionary
label_mapping = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
    5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
    10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
    15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
    19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}

# Function to create Streamlit app
def main():

    # Set background color of Streamlit app to white
    st.set_page_config(page_title="CropMe", page_icon="🌾", initial_sidebar_state="expanded")


    # Set app title
    st.title('Predict the Best Crop for your farm!')
    # Display the image using a URL
    image_url = "https://raw.githubusercontent.com/KaushikMreddy/TalkCrops_Capstone/main/crop-image.jpg" # Replace this with your image URL
    st.image(image_url, use_column_width=True)

    

    # Add a brief description
    st.write('This App allows you to find your ideal crop for your farmland')

    # Add input fields for user to enter data
    st.header('Enter Input Data')
    # Example input fields (replace with your specific input fields)
    N = st.slider('N:', min_value=0, max_value=200, value=0)
    P = st.slider('P:', min_value=0, max_value=200, value=0)
    K = st.slider('K:', min_value=0, max_value=200, value=0)
    temperature = st.slider('Temperature (Celcius):', min_value=5.0, max_value=50.0, value=0.0, step=0.01)
    humidity = st.slider('Humidity %:', min_value=0.0, max_value=100.0, value=0.0, step=0.01)
    ph = st.slider('PH:', min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rainfall = st.slider('Rainfall (mm):', min_value=0.0, max_value=300.0, value=0.0, step=0.01)


    # Make predictions when user clicks the 'Predict' button
    if st.button('Predict'):
        # Collect input data into a list or array
        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        # Make predictions using the model
        predictions = make_prediction(input_data)
        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(predictions)
        # Find the corresponding label using the label mapping dictionary
        predicted_label = label_mapping[predicted_class_index]
        # Display the predicted label
        st.header('Predicted Crop:')
        st.subheader(predicted_label.upper())

# Step 3: Run Your Streamlit App
if __name__ == '__main__':
    main()
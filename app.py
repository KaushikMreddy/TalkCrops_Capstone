
import streamlit as st
from joblib import load
import numpy as np
import matplotlib.pyplot as plt

# Load your trained Random Forest model
rf_classifier = load('crops.joblib')

# Function to make predictions using the model
def make_prediction(input_data):
    predictions = rf_classifier.predict(input_data)
    return predictions

# Define label mapping dictionary
label_mapping = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
    5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
    10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
    15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
    19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}


def plot_spider_graph(inputs):
    # Calculate angles for each category
    categories = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'PH', 'Rainfall']
    values_list = [int(value) for value in inputs]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values = values_list + [values_list[0]] 
    angles += angles[:1]

    # Create the spider graph
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='skyblue', alpha=0.6)
    ax.set_yticklabels([])  # Hide radial ticks
    plt.xticks(angles[:-1], categories, color='grey', size=10)  # Set category labels
    st.pyplot(fig)  # Display the spider graph

# Function to create Streamlit app
def main():
    # Set background color of Streamlit app to white
    st.set_page_config(page_title="CropMe", page_icon="🌾", initial_sidebar_state="expanded")

    # Set app title
    st.title('Predict the Best Crop for your farm!')
    # Display the image using a URL
    image_url = "https://raw.githubusercontent.com/KaushikMreddy/TalkCrops_Capstone/main/crop-image.jpg"
    st.image(image_url, use_column_width=True)


    # Add a brief description
    st.write('Using a Random Forest model, our app recommends the optimal crop for planting based on soil nutrient levels (NPK), rainfall, humidity, pH, and temperature. With a 99% accuracy rate, users can confidently make informed decisions about their crop selection.')

    st.write('A Random Forest model is an ensemble learning method that builds multiple decision trees during training and outputs the mode of the classes for classification problems or the average prediction for regression problems.')


    # Add input fields for user to enter data in the sidebar
    with st.sidebar:
        st.header('Enter Input Data')
        #Set default values for the sliders
        default_N = 70
        default_P = 75
        default_K = 100
        default_temperature = 25.0
        default_humidity = 50.0
        default_ph = 6.0
        default_rainfall = 100.0
        
        N = st.slider('Nitrogen (kg/ha):', min_value=0, max_value=140, value = default_N )
        P = st.slider('Phosphorous (kg/ha):', min_value=5, max_value=145,value = default_P )
        K = st.slider('Potassium (kg/ha):', min_value=5, max_value=205,value = default_K)
        temperature = st.slider('Temperature (Celcius):', min_value=8.0, max_value=50.0, step=0.01,value = default_temperature)
        humidity = st.slider('Humidity %:', min_value=0.0, max_value=100.0, step=0.01,value = default_humidity)
        ph = st.slider('PH:', min_value=0.0, max_value=14.0, step=0.1,value = default_ph)
        rainfall = st.slider('Rainfall (mm):', min_value=20.0, max_value=300.0, step=0.01,value = default_rainfall)

    # Make predictions when user clicks the 'Predict' button
    if st.sidebar.button('Predict'):
        # Check if all input values are zero
        if N == P == K == temperature == humidity == ph == 0 and rainfall == 20:
            st.header('Just:')
            st.subheader('Suffer')
        else:
            # Collect input data into a list or array
            input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
            # Make predictions using the model
            predictions = make_prediction(input_data)
            # st.write(input_data[0][0])
            # plot_spider_graph(input_data[0])
            pred_val = int(predictions)
            # Find the corresponding label using the label mapping dictionary
            predicted_label = label_mapping[pred_val]
            # Display the predicted label
            st.markdown(
    f"""
    <div style="background-color:#f0f0f0;padding:10px;border-radius:10px">
        <h2 style="color:#008000;text-align:center;">Predicted Crop</h2>
        <h3 style="color:#4d4dff;text-align:center;">Plant {predicted_label.upper()} and Prosper</h3>
    </div>
    <br>
    <strong>Model Accuracy:</strong> Our Random Forest model achieves a remarkable accuracy rate of 99%.
    """,
    unsafe_allow_html=True
)
# Run the app
if __name__ == '__main__':
    main()

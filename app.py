 
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


# Define the crop options for the dropdown menu
crop_options = ["apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton", "grapes", "jute", "kidneybeans",
                "lentil", "maize", "mango", "mothbeans", "mungbean", "muskmelon", "orange", "papaya", "pigeonpeas",
                "pomegranate", "rice", "watermelon"]

# Define label mapping dictionary
label_mapping = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
    5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
    10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
    15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
    19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}

avg_crop_values = {
  "apple": {"N": 66.1, "P": 91.4, "K": 78.0, "temperature": 43.5, "humidity": 92.5, "ph": 42.8, "rainfall": 37.5},
  "banana": {"N": 71.4, "P": 56.9, "K": 18.3, "temperature": 59.7, "humidity": 80.5, "ph": 42.7, "rainfall": 35.0},
  "blackgram": {"N": 28.6, "P": 46.6, "K": 4.9, "temperature": 64.1, "humidity": 74.0, "ph": 51.0, "rainfall": 22.6},
  "chickpea": {"N": 14.3, "P": 46.6, "K": 36.6, "temperature": 26.7, "humidity": 17.0, "ph": 52.8, "rainfall": 26.6},
  "coconut": {"N": 14.3, "P": 6.9, "K": 11.0, "temperature": 55.9, "humidity": 95.5, "ph": 42.6, "rainfall": 52.8},
  "coffee": {"N": 71.4, "P": 17.2, "K": 11.0, "temperature": 51.7, "humidity": 61.0, "ph": 48.3, "rainfall": 52.4},
  "cotton": {"N": 85.7, "P": 32.8, "K": 11.0, "temperature": 45.3, "humidity": 79.9, "ph": 49.3, "rainfall": 28.0},
  "grapes": {"N": 14.3, "P": 91.4, "K": 97.6, "temperature": 51.0, "humidity": 83.0, "ph": 42.2, "rainfall": 21.6},
  "jute": {"N": 57.1, "P": 32.8, "K": 24.4, "temperature": 48.3, "humidity": 80.9, "ph": 48.2, "rainfall": 58.3},
  "kidneybeans": {"N": 14.3, "P": 46.6, "K": 11.0, "temperature": 36.2, "humidity": 21.5, "ph": 41.1, "rainfall": 28.0},
  "lentil": {"N": 14.3, "P": 46.6, "K": 11.0, "temperature": 47.0, "humidity": 44.0, "ph": 49.1, "rainfall": 13.2},
  "maize": {"N": 57.1, "P": 32.8, "K": 11.0, "temperature": 44.6, "humidity": 65.1, "ph": 44.7, "rainfall": 21.7},
  "mango": {"N": 14.3, "P": 19.0, "K": 11.0, "temperature": 63.0, "humidity": 50.0, "ph": 40.8, "rainfall": 31.4},
  "mothbeans": {"N": 14.3, "P": 32.8, "K": 11.0, "temperature": 65.4, "humidity": 52.5, "ph": 48.0, "rainfall": 27.2},
  "mungbean": {"N": 14.3, "P": 32.8, "K": 11.0, "temperature": 65.3, "humidity": 85.5, "ph": 47.9, "rainfall": 20.0},
  "muskmelon": {"N": 71.4, "P": 6.9, "K": 24.4, "temperature": 54.0, "humidity": 92.5, "ph": 45.7, "rainfall": 7.4},
  "orange": {"N": 14.3, "P": 6.9, "K": 3.7, "temperature": 42.5, "humidity": 92.5, "ph": 42.9, "rainfall": 36.6},
  "papaya": {"N": 36.1, "P": 40.0, "K": 24.4, "temperature": 66.7, "humidity": 92.5, "ph": 48.2, "rainfall": 48.2},
  "pigeonpeas": {"N": 14.3, "P": 46.6, "K": 11.0, "temperature": 54.7, "humidity": 50.0, "ph": 42.9, "rainfall": 48.1},
  "pomegranate": {"N": 14.3, "P": 6.9, "K": 18.3, "temperature": 38.4, "humidity": 90.1, "ph": 45.0, "rainfall": 35.8},
  "rice": {"N": 56.8, "P": 32.8, "K": 18.3, "temperature": 43.2, "humidity": 82.5, "ph": 45.9, "rainfall": 80.2},
  "watermelon": {"N": 71.4, "P": 6.9, "K": 24.4, "temperature": 51.0, "humidity": 85.0, "ph": 46.1, "rainfall": 16.5}
}


def plot_spider_graph(crop_name):
    crop_values = avg_crop_values[crop_name]
    categories = list(crop_values.keys())
    values_list = [crop_values[category] for category in categories]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values = values_list + [values_list[0]] 
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.6)
    ax.set_yticklabels([]) 
    plt.xticks(angles[:-1], categories, color='black', size=10)

    # Annotate each point with its corresponding value
    for angle, value, category in zip(angles, values_list, categories):
        ax.text(angle, value, f'{value}', ha='center', va='center')

    st.pyplot(fig)


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
        <h3 style="color:#4d4dff;text-align:center;">Plant {predicted_label.upper()} in your field</h3>
    </div>
    <br>
    <strong>Model Accuracy:</strong> Our Random Forest model achieves a remarkable accuracy rate of 99%.
    <br>
    """,
    unsafe_allow_html=True
)
     # Add dropdown menu at the bottom to select crop
    selected_crop = st.selectbox("Select a Crop", crop_options)

    if selected_crop:
        plot_spider_graph(selected_crop)
    else:
        st.write("Select a crop to display its attributes.")

# Run the app
if __name__ == '__main__':
    main()

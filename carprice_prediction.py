import streamlit  as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and preprocessing steps
model = joblib.load(r"C:\Users\THIRU\OneDrive\Desktop\car_project\final_random_forest_model1.pkl")
label_encoders = joblib.load(r"C:\Users\THIRU\OneDrive\Desktop\car_project\label_encoder.pkl")
scalers = joblib.load(r"C:\Users\THIRU\OneDrive\Desktop\car_project\min_max.pkl")  # Assuming this is your loaded scaler for inverse transformation

# Load dataset for filtering and identifying similar data
data = pd.read_csv(r"C:\Users\THIRU\OneDrive\Desktop\car_project\cleaned_cardata.csv")

# Set pandas option to handle future downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

# Features used for training
features = ['Mileage', 'Model_year', 'Kilometer_Driven', 'Engine_displacement', 'Fuel_type', 'Model',
            'Transmission', 'Owner_No.', 'Body_type', 'City', 'Max_power', 'Car_Age', 'Mileage_normalized']

# Function to preprocess input data
def preprocess_input(df):
    df['Car_Age'] = 2024 - df['Model_year']
    df['Mileage_normalized'] = df['Mileage'] / df['Car_Age']
    return df
# Streamlit Application
st.set_page_config(page_title="Car Price Prediction", page_icon=":red_car:", layout="wide")
st.title("Car Price Prediction")

# Sidebar for user inputs
st.sidebar.header('Input Car Features')
# Streamlit Application

#car image path
car_image_path="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRRsQfjGnjd8gBWy8GX8MceHN-yLdABrM8uzw&s"
# Display the image of the car (after the output)
st.image(car_image_path, caption="Sample Car Image",use_column_width=True)

# Set background colors
input_background_color = "lightcoral"  # Light maroon color
result_background_color = "#FFF8E7"  # Cosmic latte or beige color

st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container {{
        background-color: {result_background_color};
    }}
    .stButton>button {{
        background-color: lightblue;
        color: white;
    }}
    .result-container {{
        text-align: center;
        background-color: {result_background_color};
        padding: 10px;  /* Reduced padding */
        border-radius: 10px;
        width: 70%;  /* Reduced width to decrease container size */
        margin: 0 auto;  /* Center container with auto margin */
    }}
    .prediction-title {{
        font-size: 28px;
        color: maroon;
    }}
    .prediction-value {{
        font-size: 36px;
        font-weight: bold;
        color: maroon;
    }}
    .info {{
        font-size: 18px;
        color: grey;
          }}
    .sidebar .sidebar-content {{
        background-color: {input_background_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Get user inputs in a defined order
selected_Model = st.sidebar.selectbox('1. Car Model', data['Model'].unique())

filtered_data = data[data['Model'] == selected_Model]
Body_type = st.sidebar.selectbox('2. Body Type', filtered_data['Body_type'].unique())
Fuel_type = st.sidebar.selectbox('3. Fuel Type', filtered_data['Fuel_type'].unique())
Transmission = st.sidebar.selectbox('4. Transmission Type', filtered_data['Transmission'].unique())
ModelYear = st.sidebar.number_input('5. Model Year', min_value=1980, max_value=2024, value=2015)
OwnerNo = st.sidebar.number_input('6. Number of Previous Owners', min_value=0, max_value=10, value=1)
KilometersDriven = st.sidebar.number_input('7. Kilometers Driven', min_value=0, max_value=500000, value=10000)

# Adjust mileage slider
min_mileage = np.floor(filtered_data['Mileage'].min())
max_mileage = np.ceil(filtered_data['Mileage'].max())
Mileage = st.sidebar.slider('8. Mileage (kmpl)', min_value=float(min_mileage), max_value=float(max_mileage), value=float(min_mileage), step=0.5)
City = st.sidebar.selectbox('9. City', data['City'].unique())
Max_power = st.sidebar.number_input('10. Max Power (bhp)', min_value=0, max_value=1000, value=100)
Engine_displacement = st.sidebar.number_input('11. Engine Displacement (cc)', min_value=0, max_value=10000, value=1000)

# Create a DataFrame for user input
user_input_data = {
    'Fuel_type': [Fuel_type],
    'Body_type': [Body_type],
    'Kilometer_Driven': [KilometersDriven],
    'Transmission': [Transmission],
    'Owner_No.': [OwnerNo],
    'Model': [selected_Model],
    'Model_year': [ModelYear],
    'City': [City],
    'Mileage': [Mileage],
    'Max_power': [Max_power],
    'Engine_displacement': [Engine_displacement],
    'Car_Age': [2024 - ModelYear],
    'Mileage_normalized': [Mileage / (2024 - ModelYear)]
}

user_df = pd.DataFrame(user_input_data)

# Ensure the columns are in the correct order and match the trained model's features
user_df = user_df[features]

# Preprocess user input data
user_df = preprocess_input(user_df)

# Apply label encoding
for column in ['Fuel_type', 'Body_type', 'Transmission', 'Model', 'City']:
    if column in user_df.columns and column in label_encoders:
        user_df[column] = user_df[column].apply(lambda x: label_encoders[column].transform([x])[0])

# Button to trigger prediction
if st.sidebar.button('Predict'):
    if user_df.notnull().all().all():
        try:
            # Make prediction
            predicted_price = model.predict(user_df)

            # Inverse transform the predicted price to get it back to the original scale
            predicted_price_norm = scalers.inverse_transform([[predicted_price[0]]])[0][0]

            # Display the predicted price
            st.markdown(f"""
                  <div class="result-container">
                    <h2 class="prediction-title">Predicted Car Price</h2>
                    <p class="prediction-value">₹{predicted_price_norm:,.2f}</p>  <!-- Use normalized value -->
                    <p class="info">Car Age: {user_df['Car_Age'][0]} years</p>
                    <p class="info">Efficiency Score: {user_df['Mileage_normalized'][0]:,.2f} km/year</p>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        missing_fields = [col for col in user_df.columns if user_df[col].isnull().any()]
        st.error(f"Missing fields: {', '.join(missing_fields)}. Please fill all required fields.")
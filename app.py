import streamlit as st
import numpy as np
import pickle

# Load models and scaler
crop_model = pickle.load(open('model.pkl', 'rb'))
npk_model = pickle.load(open('npk_model.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate",
    15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Streamlit app
st.title("🌾 Crop and NPK Recommendation System")

st.write("Enter the environmental conditions to get the top 3 recommended crops and ideal NPK levels.")

# Input fields
temperature = st.number_input("🌡️ Temperature (°C)", value=25.0)
humidity = st.number_input("💧 Humidity (%)", value=50.0)
ph = st.number_input("🧪 pH level", value=6.5)
rainfall = st.number_input("☔ Rainfall (mm)", value=100.0)

if st.button("Predict"):
    try:
        # Prepare input
        input_data = np.array([[temperature, humidity, ph, rainfall]])
        standardized_features = ms.transform(input_data)

        # Crop prediction
        crop_probabilities = crop_model.predict_proba(standardized_features)[0]
        top_3_indices = np.argsort(crop_probabilities)[-3:][::-1]
        top_3_crops = [(crop_dict.get(idx + 1, 'Unknown Crop'), crop_probabilities[idx] * 100) for idx in top_3_indices]

        # NPK prediction
        npk_prediction = npk_model.predict(standardized_features)[0]
        predicted_N, predicted_P, predicted_K = npk_prediction

        # Display results
        st.subheader("🌱 Top 3 Recommended Crops")
        for crop, confidence in top_3_crops:
            st.write(f"- {crop} (Confidence: {confidence:.2f}%)")

        st.subheader("🧬 Recommended NPK Levels")
        st.write(f"Nitrogen (N): {predicted_N:.2f}")
        st.write(f"Phosphorus (P): {predicted_P:.2f}")
        st.write(f"Potassium (K): {predicted_K:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

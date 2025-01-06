import streamlit as st
import sqlite3
import serial
import asyncio
import time
import numpy as np
import pickle

# Establish a serial connection
def init_serial(port='COM4', baudrate=9600):
    try:
        return serial.Serial(port, baudrate, timeout=1)
    except serial.SerialException as e:
        st.error(f"Error opening serial port: {e}")
        return None

ser = init_serial()

# SQLite database setup
conn = sqlite3.connect('soil_data_db.sqlite')

def create_table():
    with conn:
        s = conn.cursor()
        s.execute(''' 
            CREATE TABLE IF NOT EXISTS soil_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nitrogen REAL,
                phosphorus REAL,
                potassium REAL,
                temperature REAL
            );
        ''')
        conn.commit()

def add_sensor_data(nitrogen, phosphorus, potassium, temperature):
    with conn:
        s = conn.cursor()
        s.execute(''' 
            INSERT INTO soil_data (nitrogen, phosphorus, potassium, temperature)
            VALUES (?, ?, ?, ?);
        ''', (nitrogen, phosphorus, potassium, temperature))
        conn.commit()

def get_latest_sensor_data():
    s = conn.cursor()
    return s.execute('SELECT * FROM soil_data ORDER BY id DESC LIMIT 1').fetchone()

def delete_record_by_id(record_id):
    with conn:
        s = conn.cursor()
        s.execute('DELETE FROM soil_data WHERE id = ?;', (record_id,))
        conn.commit()

create_table()

# Global buffer to store last 3 sensor readings
sensor_buffer = []

# Function to parse the raw sensor data and add it to buffer
def parse_raw_data(data):
    try:
        # Convert the raw data into an integer
        value = int(data)
        
        # Add the new value to the buffer
        sensor_buffer.append(value)
        
        # Keep only the last 3 readings (for Nitrogen, Phosphorus, Potassium)
        if len(sensor_buffer) > 3:
            sensor_buffer.pop(0)

        # If we have all 3 values in the buffer, return them
        if len(sensor_buffer) == 3:
            nitrogen, phosphorus, potassium = sensor_buffer
            temperature = 25.0  # Assume a default temperature value
            return nitrogen, phosphorus, potassium, temperature
        else:
            return None
    except Exception as e:
        st.error(f"Error parsing raw data: {e}, Raw: {data}")
        return None

# Stream serial data
async def stream_serial_data(duration=60):
    if not ser:
        st.warning("Serial port not initialized.")
        return

    end_time = time.time() + duration
    st.info("Streaming data for 1 minute...")

    while time.time() < end_time:
        try:
            if ser.in_waiting > 0:
                raw_data = ser.readline().decode('utf-8').strip()
                st.write(f"Raw Data: {raw_data}")

                # Parse and save data
                parsed = parse_raw_data(raw_data)
                if parsed:
                    add_sensor_data(*parsed)
                    st.write(f"Data saved: Nitrogen: {parsed[0]}, Phosphorus: {parsed[1]}, Potassium: {parsed[2]}, Temperature: {parsed[3]}")
                else:
                    st.warning(f"Invalid data format: {raw_data}")
            await asyncio.sleep(0.1)
        except Exception as e:
            st.error(f"Error reading serial data: {e}")
            break

# Load the model and scalers
model = pickle.load(open('./random_forest_model.pkl', 'rb'))
sc = pickle.load(open('./standscaler.pkl', 'rb'))
ms = pickle.load(open('./minmaxscaler.pkl', 'rb'))

# Streamlit UI
def main():
    st.title('Soil Sensor Data Manager and Crop Prediction')

    # Soil Data Stream Section
    st.header("Stream Soil Sensor Data")
    if st.button("Stream Data for 1 Minute"):
        asyncio.run(stream_serial_data())

    st.header("Retrieve Latest Sensor Data")
    if st.button("Retrieve Latest Data"):
        latest = get_latest_sensor_data()
        if latest:
            st.success(f"Latest Data: Nitrogen: {latest[1]}, Phosphorus: {latest[2]}, Potassium: {latest[3]}, Temperature: {latest[4]}")
            delete_record_by_id(latest[0])  # Remove the record after retrieval
            st.write("Previous record deleted.")
        else:
            st.warning("No data available.")

    # Crop Prediction Section
    st.header("Crop Prediction")

    # Fetch the latest sensor data from the database
    latest_data = get_latest_sensor_data()
    if latest_data:
        nitrogen = latest_data[1]
        phosphorus = latest_data[2]
        potassium = latest_data[3]
        temperature = latest_data[4]
        
        st.write(f"Nitrogen: {nitrogen}")
        st.write(f"Phosphorus: {phosphorus}")
        st.write(f"Potassium: {potassium}")
        st.write(f"Temperature: {temperature}")

        if st.button('Predict Crop'):
            # Use sensor data for N, P, K, and temperature values
            N = nitrogen
            P = phosphorus
            K = potassium
            temperature = temperature

            # Add placeholders for missing features (e.g., pH or other features that the model expects)
            # Assuming pH is missing, set it to a default value, for example, 7.0
            pH = 7.0  # Example placeholder
            extra_feature_1 = 0.0  # Another placeholder for missing features if needed
            extra_feature_2 = 0.0 
            # Combine the features into a list
            feature_list = [N, P, K, temperature, pH, extra_feature_1,extra_feature_2]

            # Reshape the features to fit the model input shape
            single_pred = np.array(feature_list).reshape(1, -1)

            # Scale the input features
            scaled_features = ms.transform(single_pred)

            # Now transform using StandardScaler if necessary
            final_features = sc.transform(scaled_features)

            # Predicting probabilities
            probabilities = model.predict_proba(final_features)[0]

            # Sorting and selecting top 3 indices
            top_3_indices = np.argsort(probabilities)[-3:][::-1]

            # Mapping indices to crop names
            crop_dict = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            result_image = {
            1: "static/crops/Rice.jpeg", 2: "static/crops/Maize.jpeg", 3: "static/crops/Jute.jpeg",
            4: "static/crops/Cotton.jpeg", 5: "static/crops/Coconut.jpeg", 6: "static/crops/Papaya.jpeg",
            7: "static/crops/Orange.jpeg", 8: "static/crops/Apple.jpeg", 9: "static/crops/Muskmelon.jpeg",
            10: "static/crops/Watermelon.jpeg", 11: "static/crops/Grapes.jpeg", 12: "static/crops/Mango.jpeg",
            13: "static/crops/Banana.jpeg", 14: "static/crops/Pomegranate.jpeg", 15: "static/crops/Lentil.jpeg",
            16: "static/crops/Blackgram.jpeg", 17: "static/crops/Mungbean.jpeg", 18: "static/crops/Mothbeans.jpeg",
            19: "static/crops/Pigeonpeas.jpeg", 20: "static/crops/Kidneybeans.jpeg", 21: "static/crops/Chickpea.jpeg",
            22: "static/crops/Coffee.jpeg"
        }

            crop_description = {
                "Rice": "Rice is predominantly cultivated in Asia, especially in regions like China, India, and Southeast Asia. It thrives in clayey, loamy soils that retain water well.",
                "Maize": "Maize, or corn, is widely grown in the United States, Brazil, and China. It prefers well-drained, fertile loamy soils rich in organic matter.",
                "Jute": "Jute is mainly cultivated in Bangladesh and India, particularly in the Ganges Delta. It grows best in alluvial soil with a high clay content.",
                "Cotton": "Cotton is densely grown in India, the United States, and China. It favors deep, well-drained sandy loam soils with a slightly acidic to neutral pH.",
                "Coconut": "Coconut palms are most commonly found in tropical coastal regions like Indonesia, the Philippines, and India. They grow well in sandy, loamy, or alluvial soils that are well-drained.",
                "Papaya": "Papaya is extensively grown in India, Brazil, and Mexico. It thrives in well-drained sandy loam or alluvial soils rich in organic matter.",
                "Orange": "Oranges are primarily cultivated in Brazil, the United States (especially Florida), and China. They prefer well-drained sandy loam soils rich in organic content.",
                "Apple": "Apples are widely grown in temperate regions like the United States, China, and Europe. They thrive in well-drained loamy soils with a slightly acidic pH.",
                "Muskmelon": "Muskmelon is grown in warmer climates such as in India and the United States. It prefers sandy loam soils that are well-drained and rich in organic matter.",
                "Watermelon": "Watermelon is cultivated in warm regions like China, Turkey, and the United States. It grows best in sandy loam soils with good drainage and moderate organic content.",
                "Grapes": "Grapes are primarily cultivated in Mediterranean climates like those found in Italy, Spain, and France. They thrive in well-drained loamy or sandy loam soils with good fertility.",
                "Mango": "Mangoes are extensively grown in India, Thailand, and Mexico. They favor well-drained alluvial or loamy soils rich in organic matter.",
                "Banana": "Bananas are widely cultivated in tropical regions such as India, Brazil, and Ecuador. They thrive in well-drained loamy soils with high organic content.",
                "Pomegranate": "Pomegranates are primarily grown in India, Iran, and the Mediterranean. They prefer well-drained sandy or loamy soils with a neutral to slightly alkaline pH.",
                "Lentil": "Lentils are commonly grown in India, Turkey, and Canada. They prefer well-drained loamy soils with a neutral to slightly acidic pH.",
                "Blackgram": "Blackgram is mainly cultivated in India and Myanmar. It thrives in well-drained loamy soils with moderate fertility.",
                "Mungbean": "Mungbean is widely grown in India, China, and Southeast Asia. It prefers well-drained loamy soils with good fertility.",
                "Mothbeans": "Mothbeans are grown in India and parts of Africa. They thrive in well-drained sandy loam soils with a moderate organic content.",
                "Pigeonpeas": "Pigeonpeas are primarily cultivated in India and parts of Africa. They thrive in well-drained sandy loam soils with moderate fertility.",
                "Kidneybeans": "Kidney beans are widely grown in India, the United States, and Brazil. They prefer well-drained, fertile loamy soils with a neutral pH.",
                "Chickpea": "Chickpeas are cultivated mainly in India, Turkey, and Australia. They prefer well-drained, fertile loamy soils.",
                "Coffee": "Coffee is predominantly grown in tropical regions like Brazil, Colombia, and Ethiopia. It thrives in slightly acidic, well-drained soils with high organic content."
            }

            top_crops = [crop_dict[i+1] for i in top_3_indices]
            top_crops_info = [crop_description[crop] for crop in top_crops]

            # Display the predicted crops and their descriptions
            
            for idx in top_3_indices:
                crop_name = crop_dict.get(idx , "Unknown Crop")
                st.write(f"Crop : {crop_name}")
                st.write(crop_description.get(crop_name, "Description not available."))
                image_path = result_image[idx]
                st.image(image_path, caption=crop_name, use_column_width=True)
if __name__ == "__main__":
    main()

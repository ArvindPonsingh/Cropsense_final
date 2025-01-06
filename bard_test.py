import google.generativeai as palm

# Directly specify the API key
palm_api_key = "AIzaSyBGLCbvBh7WsnoacXW7RhWgQ4uLM6fftHI"  # Replace with your actual API key

# Create a config
palm.configure(api_key=palm_api_key)
model = palm.GenerativeModel(model_name="gemini-1.5-flash-8b-exp-0924")

# Generate some text
def classify_soil_and_advice(nitrogen, phosphorus, potassium, soil_moisture):
    prompt = (
        f"Acting as a Smart Agricultural Advisor, classify soil quality and assess erosion risk "
        f"based on the following parameters:\n"
        f"- Nitrogen: {nitrogen} mg/kg\n"
        f"- Phosphorus: {phosphorus} mg/kg\n"
        f"- Potassium: {potassium} mg/kg\n"
        f"- Soil Moisture: {soil_moisture}%\n\n"
        f"**Soil Quality Classification Criteria**:\n"
        f"- Good Quality: Nitrogen > 50 mg/kg, Phosphorus > 30 mg/kg, Potassium > 200 mg/kg, Soil Moisture 20–50%\n"
        f"- Moderate Quality: Nitrogen 20–50 mg/kg, Phosphorus 15–30 mg/kg, Potassium 100–200 mg/kg, Soil Moisture < 20% or > 50%\n"
        f"- Poor Quality: Nitrogen < 20 mg/kg, Phosphorus < 15 mg/kg, Potassium < 100 mg/kg, Soil Moisture < 10% or > 70%\n\n"
        f"**Erosion Risk Classification Criteria**:\n"
        f"- Low Risk: Good soil quality and soil moisture 20–50%\n"
        f"- Moderate Risk: Moderate soil quality or soil moisture extremes (10–20% or 50–70%)\n"
        f"- High Risk: Poor soil quality or soil moisture < 10% or > 70%\n\n"
        f"Provide a structured response in the following format:\n"
        f"**Soil Quality Classification**: [Good/Moderate/Poor]\n"
        f"**Erosion Risk**: [Low/Moderate/High]\n\n"
        f"**Analysis**:\n"
        f"- Nitrogen (N): [Value] mg/kg - [Comment]\n"
        f"- Phosphorus (P): [Value] mg/kg - [Comment]\n"
        f"- Potassium (K): [Value] mg/kg - [Comment]\n"
        f"- Soil Moisture: [Value]% - [Comment]\n\n"
        f"**Recommendations**:\n"
        f"1. [Nitrogen improvement strategy, if required]\n"
        f"2. [Phosphorus improvement strategy, if required]\n"
        f"3. [Potassium improvement strategy, if required]\n"
        f"4. [Moisture management or erosion prevention strategy]\n"
    )
    response = model.generate_content(prompt)
    return response.text


print(classify_soil_and_advice(55, 35, 210, 40))
# Expected Output: Good soil quality, Low erosion risk, Recommendations for maintaining quality.
# print(classify_soil_and_advice(30, 25, 150, 60))
# # Expected Output: Moderate soil quality, Moderate erosion risk, Recommendations for improvement.
# print(classify_soil_and_advice(10, 8, 50, 5))
# Expected Output: Poor soil quality, High erosion risk, Recommendations for enrichment and erosion prevention.

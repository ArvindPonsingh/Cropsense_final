# Crop Sense - Smart Crop and Fertilizer Recommendation System 🌱

CropSense is an intelligent agricultural advisory system designed to assist farmers and agricultural enthusiasts in optimizing crop health and soil management. This project leverages real-time field data, including key soil parameters such as **Nitrogen (N)**, **Phosphorus (P)**, **Potassium (K)**, and **Soil Moisture**, to classify soil quality and assess erosion risk. The application provides **actionable recommendations** to improve soil quality and mitigate risks, enhancing sustainable farming practices.

The system is built using **Flask** for the backend and features an **interactive frontend** for data input, allowing users to analyze multiple soil samples. It incorporates modern web design principles, ensures seamless interaction, and integrates API features to fetch data about **nearby markets** or **farming resources**. CropSense is a step toward smarter farming through data-driven decision-making.

---

## 📌 **Features**

- **✔ Predicts the best crop based on soil and environmental factors.**  

- **✔ Recommends the optimal fertilizer for a given crop and soil composition.**  

- **✔ Uses Random Forest and Decision Tree Classifier models.**  

- **✔ REST API powered by Flask & Python.**  

- **✔ Frontend built using React.js with Axios for API calls.**  

- **✔ Supports CORS for cross-origin communication.**  

---

## 🛠️ ** Tech Stack**

- **✔ Frontend**\
     React.js\
     Axios\
     CSS

- **✔ Backend**\
   Flask\
   Python\
   scikit-learn\
   NumPy\
   Pickle

- **✔ Machine Learning Models**\
   Random Forest\
   Decision Tree

---

### 🌟 Why CropSense?  
CropSense aims to simplify and enhance decision-making for small-scale farmers, supporting **sustainable agriculture** through **technology-driven solutions**.

---

![Screenshot 2025-03-05 135906](https://github.com/user-attachments/assets/a039c595-ebdb-4d81-8a7b-0595e12b931f)

## **📂 Folder Structure**
   ```bash
      Crop-Sense-React/
      │── backend/             # Flask Backend (Python)
      │   ├── app.py           # Main Flask Application
      │   ├── models/          # Machine Learning Models
      │   ├── static/          # Images & Static Files
      │   ├── requirements.txt # Python Dependencies
      │
      │── frontend/            # React Frontend
      │   ├── src/
      │   │   ├── components/  # React Components
      │   │   ├── pages/
      │   │   ├── App.js       # Main React App
      │   │   ├── index.js     # Entry Point
      │   ├── public/          # Public Assets
      │   ├── package.json     # React Dependencies
      │
      │── README.md            # Documentation
      │── .gitignore           # Ignore unnecessary files
   ```
---

## **🚀 Installation & Setup** 
   1️⃣ Clone the Repository
   ```sh
         git clone https://github.com/your-username/crop-sense.git
         cd crop-sense
   ```
   2️⃣ Backend Setup (Flask)
   ```sh
         cd backend
         pip install -r requirements.txt
         python app.py
   ```
   Ensure flask-cors is installed to avoid CORS issues:
   ```sh
         pip install flask-cors
   ```
   3️⃣ Frontend Setup (React.js)
   ```sh
         cd frontend
         npm install
         npm start
   ```
---
## 🌾Crop Recommendation ScreenShots 
![Screenshot 2025-03-05 140016](https://github.com/user-attachments/assets/c113e6c9-3585-4430-97df-2d981e71dbea)
![Screenshot 2025-03-05 140106](https://github.com/user-attachments/assets/c7be16d1-3fab-4462-b26f-8afc3c73c29f)

## 🥫Fertilizer Recommendation ScreenShots 
![Screenshot 2025-03-05 140026](https://github.com/user-attachments/assets/4821059c-125b-4fc3-a07b-f38aa0bad65a)
![Screenshot 2025-03-05 140132](https://github.com/user-attachments/assets/627fdfae-955e-44e4-9984-9b9437af14be)



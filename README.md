🧠 Brain Tumor Detection using Deep Learning and Grad-CAM (Flask Web App)

This project implements a Brain Tumor Classification system using Deep Learning (MobileNetV2) with 95–98% accuracy.
A user-friendly Flask Web App allows users to upload MRI brain scans and view:

✔ Prediction (Tumor / No Tumor)
✔ Confidence Score
✔ Grad-CAM Heatmap showing the region that influenced the model


---

📌 Features

🔬 MobileNetV2-based CNN trained for high accuracy

🩺 Brain MRI Tumor Detection (Binary Classification)

🔥 Grad-CAM Visualization for explainable AI

🌐 Flask Web App with HTML, CSS, JavaScript UI

📁 Image preprocessing using OpenCV

📊 Model metrics and training graphs

🎯 Real-time prediction from uploaded MRI images



---

🛠️ Tech Stack / Tools

Python

TensorFlow / Keras

MobileNetV2

OpenCV

Flask

HTML, CSS, JavaScript



---

📂 Project Structure

Brain-Tumor-Detection/
│
├── models/
│   └── mobilenetv2_brain_tumor.h5
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   └── index.html
│
├── app.py
├── gradcam.py
├── preprocess.py
├── requirements.txt
└── README.md


---

🚀 How to Run the Project

1️⃣ Install Dependencies

pip install -r requirements.txt

2️⃣ Run Flask App

python app.py

3️⃣ Open in Browser

http://127.0.0.1:5000/

Upload your Brain MRI image — the app will show:

Tumor / No Tumor

Confidence score

Grad-CAM heatmap



---

📊 Model Performance

Model: MobileNetV2

Accuracy: 95–98%

Uses Transfer Learning + Fine-tuning

Grad-CAM for model explainability



---

🎯 Future Enhancements

Deploy on Render / AWS / Azure

Add CT Scan / Multiple tumor types classification

Build REST API for mobile integration



---

🤝 Contributions

Pull requests are welcome! Feel free to improve UI, model accuracy, or add new features.


---

⭐ Show Support

If you like this project, don’t forget to star ⭐ the repository!


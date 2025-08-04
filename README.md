# 🚗 AutoVision – Real-Time Vehicle Detection & Traffic Analysis

AutoVision is a real-time vehicle detection and traffic analysis system built using **YOLOv8**, **Streamlit**, and **OpenCV**. It can detect vehicles from live camera feed or uploaded videos, analyze traffic density, and provide evaluation metrics like **Precision**, **Recall**, **F1 Score**, and **Accuracy** — all with a user-friendly dashboard and downloadable reports.

---

## 📸 Features

- ✅ Real-time object detection (cars, bikes, buses, trucks, vans, bicycles)
- 📊 Traffic density classification: Low / Medium / High
- 🔬 Evaluation metrics (Precision, Recall, F1, Accuracy)
- 📈 Visualizations for F1 Score over time and traffic density distribution
- 📁 Exportable CSV report of the detection session
- 🎥 Supports webcam or uploaded video file as input

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Backend**: Python
- **Computer Vision**: YOLOv8 (Ultralytics), OpenCV
- **Data Analysis**: Pandas, Matplotlib, NumPy

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vaibhav0809git/Real-time-vehicle-detection-using-Yolo
cd autovision

###create environment

# python -m venv venv
#source venv/bin/activate        # For Linux/macOS
#venv\Scripts\activate           # For Windows

##Need to install dependecies such as

# pip install streamlit opencv-python ultralytics pandas matplotlib numpy

##to run it :
#streamlit run autovision.py



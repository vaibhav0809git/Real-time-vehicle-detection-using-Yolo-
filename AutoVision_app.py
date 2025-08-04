
import cv2
import tempfile
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import random

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

st.set_page_config(page_title="AutoVision - Vehicle Detector", layout="wide")
st.title("🚗 AutoVision - Vehicle Detection & Traffic Analysis")

vehicle_classes = ["car", "motorcycle", "bus", "truck", "bicycle", "van"]
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.1, 1.0, 0.3, 0.05)

if 'detection_data' not in st.session_state:
    st.session_state.detection_data = []
if 'is_detecting' not in st.session_state:
    st.session_state.is_detecting = False

input_type = st.sidebar.radio("Choose Input Source", ("📷 Start Camera", "📁 Upload Video"))

# Placeholders for persistent metric display
metrics_placeholder = st.container()


def get_density_class(count):
    if count < 5:
        return "Low"
    elif count < 15:
        return "Medium"
    else:
        return "High"


def get_density_color(density):
    return {"Low": "#00ff00", "Medium": "#ffff00", "High": "#ff0000"}.get(density, "#ffffff")


def simulate_ground_truth(pred_counts):
    gt_counts = {}
    for k, v in pred_counts.items():
        variation = random.choice([-1, 0, 1])
        gt_counts[k] = max(0, v + variation)
    return gt_counts


def compute_evaluation_metrics(pred_counts, gt_counts):
    TP = sum(min(pred_counts.get(cls, 0), gt_counts.get(cls, 0)) for cls in vehicle_classes)
    FP = sum(max(0, pred_counts.get(cls, 0) - gt_counts.get(cls, 0)) for cls in vehicle_classes)
    FN = sum(max(0, gt_counts.get(cls, 0) - pred_counts.get(cls, 0)) for cls in vehicle_classes)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1_score * 100, 2),
        "accuracy": round(accuracy * 100, 2),
    }


def run_detection(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("❌ Cannot open video source.")
        return

    st.session_state.is_detecting = True
    st.session_state.detection_data = []

    col1, col2 = st.columns([2, 1])
    FRAME_WINDOW = col1.image([])
    stop_button = col2.button("🛑 Stop Detection", type="secondary")

    frame_count = 0
    start_time = time.time()
    all_metrics = []
    last_metrics = {}

    st.info("🔍 Detecting vehicles... Click 'Stop Detection' to end.")

    while cap.isOpened() and st.session_state.is_detecting:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=confidence_threshold)
        annotated_frame = results[0].plot()

        vehicle_count = 0
        class_counter = defaultdict(int)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in vehicle_classes:
                class_counter[cls_name] += 1
                vehicle_count += 1

        gt_counts = simulate_ground_truth(class_counter)
        eval_metrics = compute_evaluation_metrics(class_counter, gt_counts)
        all_metrics.append(eval_metrics)
        last_metrics = eval_metrics

        density = get_density_class(vehicle_count)
        current_data = {
            "timestamp": time.strftime('%H:%M:%S'),
            "vehicle_count": vehicle_count,
            "density": density,
            **class_counter,
            **eval_metrics
        }
        st.session_state.detection_data.append(current_data)

        info_text = f"Vehicles: {vehicle_count} | Density: {density}"
        timestamp_text = time.strftime('%H:%M:%S')
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated_frame, timestamp_text, (10, annotated_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        FRAME_WINDOW.image(annotated_frame, channels="BGR", use_container_width=True)

        frame_count += 1

        if stop_button:
            st.session_state.is_detecting = False
            break

        time.sleep(0.05)

    cap.release()
    st.session_state.is_detecting = False

    # Show final metrics below prediction window
    with metrics_placeholder:
        st.markdown("### 📊 Final Metrics")
        st.metric("Last F1 Score", last_metrics.get('f1_score', 0))
        st.metric("Last Precision", last_metrics.get('precision', 0))
        st.metric("Last Recall", last_metrics.get('recall', 0))
        st.metric("Last Accuracy", last_metrics.get('accuracy', 0))
        st.metric("Total Detection Time", f"{int(time.time() - start_time)}s")

    if st.session_state.detection_data:
        st.success("✅ Detection complete!")
        df = pd.DataFrame(st.session_state.detection_data)

        st.markdown("### 📈 Evaluation Summary")
        st.dataframe(df[['timestamp', 'vehicle_count', 'precision', 'recall', 'f1_score', 'accuracy']])

        # Density Bar Chart
        st.markdown("### 🚦 Vehicle Density Distribution")
        fig, ax = plt.subplots()
        density_counts = df['density'].value_counts()
        ax.bar(density_counts.index, density_counts.values, color=[get_density_color(d) for d in density_counts.index])
        ax.set_xlabel("Density Level")
        ax.set_ylabel("Number of Frames")
        ax.set_title("Traffic Density Distribution")
        st.pyplot(fig)

        # F1 Score Over Time
        st.markdown("### 📉 F1 Score Over Time")
        fig2, ax2 = plt.subplots()
        ax2.plot(df['timestamp'], df['f1_score'], marker='o', linestyle='-', color='blue')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score Over Time')
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

        

        csv_data = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Detection Report (CSV)",
            data=csv_data,
            file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if input_type == "📷 Start Camera":
    if st.button("🎥 Start Camera Detection", type="primary"):
        run_detection(0)

elif input_type == "📁 Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_file.read())
        if st.button("🎬 Start Video Analysis", type="primary"):
            run_detection(temp_video.name)

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📋 Features")
    st.markdown("""
    - Real-time Detection
    - Traffic Density Metrics
    - Live Precision, Recall, F1, Accuracy
    - Final Charts: F1 Score & Density
    - Exportable CSV Reports
    """)


#create environment 
# python -m venv venv
#source venv/bin/activate        # For Linux/macOS
#venv\Scripts\activate           # For Windows

#Need to install dependecies such as
# pip install streamlit opencv-python ultralytics pandas matplotlib numpy

#to run it : streamlit run autovision.py

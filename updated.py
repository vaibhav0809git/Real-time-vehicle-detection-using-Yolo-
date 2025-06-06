import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict
import time

# Load YOLOv8 model (tiny version or your custom model)
model = YOLO("yolov8n.pt")

st.set_page_config(page_title="AutoVision - Vehicle Detector", layout="wide")
st.title("🚗 AutoVision - Vehicle Detection & Traffic Analysis")

input_type = st.sidebar.radio("Choose Input Source", ("📷 Start Camera", "📁 Upload Video"))
FRAME_WINDOW = st.image([])

# Class names (from COCO dataset used in YOLOv8)
vehicle_classes = ["car", "motorcycle", "bus", "truck", "bicycle", "van"]

def get_density_class(count):
    if count < 5:
        return "Low"
    elif count < 15:
        return "Medium"
    else:
        return "High"

def run_detection(source=0):
    cap = cv2.VideoCapture(source)
    log_data = []

    if not cap.isOpened():
        st.error("❌ Cannot open video source.")
        return

    st.info("🔍 Detecting vehicles... Press Stop to end.")

    FRAME_WINDOW.image([])  # Clear previous image

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated_frame = frame.copy()
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id]
    if cls_name in vehicle_classes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, cls_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        # Count vehicles
        vehicle_count = 0
        class_counter = defaultdict(int)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in vehicle_classes:
                class_counter[cls_name] += 1
                vehicle_count += 1

        density = get_density_class(vehicle_count)

        # Log data
        log_data.append({
            "timestamp": time.strftime('%H:%M:%S'),
            "elapsed_min": int((time.time() - start_time) // 120),  # 2-minute interval block
            "vehicle_count": vehicle_count,
            "density": density,
            **class_counter
        })

        # Overlay info
        info_text = f"Total Vehicles: {vehicle_count} | Density: {density}"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)

        FRAME_WINDOW.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()

    # Save log
    df = pd.DataFrame(log_data)
    df.to_csv("vehicle_logs.csv", index=False)

    st.success("✅ Detection complete.")
    st.download_button("⬇️ Download CSV Log", data=df.to_csv(index=False), file_name="vehicle_logs.csv", mime="text/csv")

    # Show density chart by 2-min intervals
    grouped = df.groupby("elapsed_min")["vehicle_count"].mean().reset_index()
    grouped["time_block"] = grouped["elapsed_min"].apply(lambda x: f"{x*2}-{x*2+2} min")

    st.subheader("📊 Vehicle Density (Average Every 2 Minutes)")
    st.line_chart(data=grouped.set_index("time_block")["vehicle_count"])

# CAMERA
if input_type == "📷 Start Camera":
    cam_index = st.sidebar.selectbox("Select Camera Index", [0, 1, 2, 3])
if st.button("Start Detection from Camera"):
    run_detection(cam_index)


# VIDEO UPLOAD
elif input_type == "📁 Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_file.read())
        if st.button("Start Detection from Uploaded Video"):
            run_detection(temp_video.name)

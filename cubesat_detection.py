
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
import os

# ======= PAGE CONFIG =======
st.set_page_config(
    page_title="CubeSat Object Detection Using YOLOv8",
    page_icon="🛰️",
    layout="wide", initial_sidebar_state="expanded")

# page title
st.markdown("""<h1 style="color:white;text-align:center;">  CubeSat_1U_3U_6U Classification and Detection 🛰️</h1>""", unsafe_allow_html= True)

# dark mode theme
st.markdown("""<style>/* Background & text color */
            body { background-color: #0E1117;color: white;}
            /* Streamlit container tweaks */.stApp 
            {background-color: #0E1117;}
            h1, h2, h3, h4, h5, h6 {color: #FFA500;  
            /* orange accent */font-family: 'Segoe UI', sans-serif;}
            p {color: #E0E0E0;}
            /* Button styling */div.stButton >
            button {background-color: #5353ec;
            color: white; border-radius: 12px;font-weight: bold;}
            div.stButton > button:hover {background-color: #FFA500;
            color: #000;}</style>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

tab_1, tab_2 = st.tabs(["📊 presintation", "📈 Model_page"])

with tab_1:
    # ======= CUSTOM STYLES =======
    st.markdown("""
    <style>
        body {background-color: #0E1117; color: #FFFFFF;}
        h1, h2 {color: #F9A825; text-align: center;}
        h3 {color: #FFB300; margin-bottom: 0.4em;}
        .slide-box {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 2.5rem;
            margin-top: 1.5rem;
            box-shadow: 0px 3px 12px rgba(255, 186, 8, 0.15);
            animation: fadeIn 0.8s ease-in-out;}
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(15px);}
            to {opacity: 1; transform: translateY(0);}}
        .list-box {
            background-color: rgba(255, 255, 255, 0.07);
            border-left: 4px solid #FFB300;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;}
        .point {
            font-size: 18px;
            line-height: 1.6em;}
        .footer {
            text-align: center;
            color: #BDBDBD;
            margin-top: 40px;}
    </style>""", unsafe_allow_html=True)



    # ======= SLIDES =======

    slides = [
    {
        "emoji": "🤖",
        "title": "CubeSat Object Detection Project",
        "content": """
        <div class="list-box">
        - Objective: Detect and classify CubeSats (1U, 3U, 6U) using YOLOv8. <br>
        - Frameworks: <b>Ultralytics YOLOv8</b>, <b>Roboflow</b>, and <b>PyTorch</b>. <br>
        - Executed in Google Colab with integrated visualization of results. <br>
        - Project automates satellite detection for educational and research applications.
        </div>
        <p style='text-align:center; color:#A5D6A7;'>🚀 Smart vision for small satellites!</p>
        """
    },

    {
        "emoji": "📸",
        "title": "Dataset Overview",
        "content": """
        <div class="list-box">
        - Source: <a href='https://app.roboflow.com/s-i-m-o/cubesat-1u-3u-6u/browse' target='_blank'>Roboflow – CubeSat 1U–3U–6U</a> dataset. <br>
        - Classes: <b>1U CubeSat</b>, <b>3U CubeSat</b>, <b>6U CubeSat</b>. <br>
        - Task: Object detection with bounding boxes. <br>
        - Version: 1, exported in YOLOv8 format. <br>
        - Downloaded and preprocessed using Roboflow API.
        </div>
        <p style='text-align:center; color:#A5D6A7;'>🧠 Clean, annotated, and ready for model training.</p>
        """
    },

    {
        "emoji": "🧩",
        "title": "Dataset Details",
        "content": """
        <div class="list-box">
        - Total Images: (from Roboflow Version 1). <br>
        - Image Resolution: 640×640 pixels. <br>
        - Annotation Format: YOLOv8 TXT (class, x_center, y_center, width, height). <br>
        - Data Split: Train 70%, Validation 20%, Test 10%. <br>
        - Labeling & Export handled through Roboflow platform.
        </div>
        <p style='text-align:center; color:#A5D6A7;'>📊 Well-balanced dataset for accurate model generalization.</p>
        """
    },

    {
        "emoji": "🧪",
        "title": "Preprocessing & Setup",
        "content": """
        <div class="list-box">
        - Data automatically prepared and split by Roboflow. <br>
        - Resized all images to 640×640 for YOLOv8 compatibility. <br>
        - Augmentations applied: rotation, flipping, exposure & blur. <br>
        - Downloaded dataset directly via Roboflow API key.
        </div>
        <p style='text-align:center; color:#A5D6A7;'>🔧 Robust preprocessing ensures model stability.</p>
        """
    },

    {
        "emoji": "🤖",
        "title": "Model Training Configuration",
        "content": """
        <div class="list-box">
        - Model Architecture: <b>YOLOv8n</b> (Nano model). <br>
        - Training Command: <br>
          <code>yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=25 imgsz=640</code> <br>
        - validting Command: <br>
          <code>yolo task=detect mode=valid model=yolov8n.pt data=data.yaml epochs=25 imgsz=640</code> <br>
        - Predicting Command: <br>
          <code>yolo task=detect mode=predict model=yolov8n.pt data=data.yaml epochs=25 imgsz=640</code> <br>
        - Epochs: 25 <br>
        - Image Size: 640×640 <br>
        </div>
        <p style='text-align:center; color:#A5D6A7;'>💻 Efficient and lightweight training pipeline.</p>
        """
    },

    {
        "emoji": "📈",
        "title": "Model Evaluation & Results",
        "content": """
        <div class="list-box">
        - Evaluation performed using YOLOv8 validation mode. <br>
        - Confusion matrices generated for class-wise accuracy. <br>
        - Key metrics: <b>Precision, Recall, mAP@0.5, mAP@0.5:0.95</b>. <br>
        - Results visualized in <b>results.png</b> and <b>confusion_matrix.png</b>. <br>
        - Model stored as <b>best.pt</b> in the training directory.
        </div>
        <p style='text-align:center; color:#A5D6A7;'>✅ High accuracy and stable convergence observed.</p>
        """
    },

    {
        "emoji": "🚀",
        "title": "Conclusion & Future Work",
        "content": """
        <div class="list-box">
        - The YOLOv8n model successfully identifies CubeSat types (1U, 3U, 6U). <br>
        - Results demonstrate strong generalization despite small dataset size. <br>
        - Future Goals: <br>
          • Integrate real-time video detection. <br>
          • Add more CubeSat categories and variants. <br>
          • we plan to optimize the YOLOv8 model using NVIDIA TensorRT,
           converting it from the original .pt format to an .engine file.
           This conversion significantly enhances computational efficiency,
           achieving up to 36 FPS — nearly doubling real-time performance. <br>
          • Additionally, we aim to implement a feature that allows users to select a specific CubeSat,
           automatically retrieve its TLE (Two-Line Element) data from up to two years in the past,
           and use predictive algorithms to forecast its future orbital position.
           The predicted trajectory will be visualized dynamically, 
           illustrating the CubeSat’s path along its orbit in real time. <br>
          • Add more CubeSat categories and variants. <br>
          • Deploy model via Flask.
        </div>
        <p style='text-align:center; color:#A5D6A7;'>🌌 Towards intelligent satellite recognition systems.</p>
        """
    }]

    # ======= STATE =======
    if "slide_index" not in st.session_state:
        st.session_state.slide_index = 0

    slide = slides[st.session_state.slide_index]

    # ======= DISPLAY =======
    st.markdown(f"<h1>{slide['emoji']} {slide['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div class='slide-box'>{slide['content']}</div>", unsafe_allow_html=True)

    # ======= NAVIGATION =======
    if "slide_index" not in st.session_state:
        st.session_state["slide_index"] = 0

    current_slide = st.session_state["slide_index"]

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅️ Previous", key=f"prev_{current_slide}"):
            st.session_state["slide_index"] = max(0, st.session_state["slide_index"] - 1)
            st.rerun()

    with col2:
        st.markdown(
            f"<p style='text-align:center; color:#999;'>Slide {st.session_state.slide_index+1}/{len(slides)}</p>",
            unsafe_allow_html=True)

    with col3:
        if st.button("➡️ Next", key=f"next_{current_slide}"):
            st.session_state["slide_index"] = min(len(slides) - 1, st.session_state["slide_index"] + 1)
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    # show result loss function and map50
    st.markdown("""<h1 style="color:white;text-align:center;"> 
    Results</h1>""", unsafe_allow_html= True)
    # read img
    image_path = "results.PNG"
    img_cv = cv2.imread(image_path)  # BGR
    # convert to RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    # convert to PIL Image
    img_pil = Image.fromarray(img_rgb)
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(img_pil, use_container_width=True)

   
with tab_2:
    st.markdown("<h1 style='color:white; text-align:center;'>Model</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:red; font-size:18px;'>"
        "Now you can upload an image or video containing 1U, 3U, or 6U CubeSats to detect them.</p>",
        unsafe_allow_html=True)

    # Load model
    model = YOLO("best.pt")

    # Confidence & NMS sliders
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    nms_threshold = st.slider("Non-Maximum Suppression", 0.0, 1.0, 0.4)

    # Upload multiple files
    uploaded_files = st.file_uploader(
        "Select photos or videos",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
        accept_multiple_files=True)

    # Define helper function
    def display_results(results):
        st.write("Results after detection:")
        cols = st.columns(2)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            with cols[i % 2]:
                st.write(f"Object {i + 1}:")
                st.write(f"• Class: {classes[i]}")
                st.write(f"• Confidence: {confs[i]:.2f}")
                st.write(f"• Bounding Box: {boxes[i]}")

    def process_image(temp_path):
        image = Image.open(temp_path)
        st.image(image, caption="Before Detection", use_container_width=True)
        image_np = np.array(image)

        results = model(image_np, conf=confidence_threshold, iou=nms_threshold)
        annotated_frame = results[0].plot()
        st.image(annotated_frame, caption="After Detection", use_container_width=True)
        display_results(results)


    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            st.write(f"📂 Processing: {uploaded_file.name}")

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            # Process only images for now
            if file_extension in ["jpg", "jpeg", "png"]:
                process_image(temp_path)
            elif file_extension in ["mp4", "mov", "avi"]:
                st.video(temp_path)

            st.markdown("<hr>", unsafe_allow_html=True)

    # About section
    st.markdown("### About this App")
    st.info("""
        An application for object detection using the YOLOv8 model. 
        Upload an image or video to analyze and identify CubeSats in it.
        You can adjust the confidence and NMS thresholds from the sidebar.
    """)



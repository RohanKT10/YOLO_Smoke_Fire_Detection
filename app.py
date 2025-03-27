import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(page_title="Fire & Smoke Detection", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>Fire & Smoke Detection</h1>",
    unsafe_allow_html=True
)
st.markdown("""
      This application utilizes **YOLO (You Only Look Once)** to detect fire and smoke in images and videos. 
      """)
st.write("""
### Features:
- Upload an **image or video** for real-time detection.
- Uses a *custom-trained YOLOv8 model for fire and smoke detection** for accurate detection.
- Provides an option to **download** the processed video with bounding boxes.
- This can help in early **fire hazard detection** for safety applications.

###  How It Works:
1. Upload an **image or video** file.
2. The system **processes** the input and applies YOLO detection.
3. A **processed video** with bounding boxes is available for **download**.

Upload an image or video below to get started!
""")

# Upload file (images: png, jpg, jpeg; videos: mp4, avi, mov, mkv)
uploaded_file = st.file_uploader("Choose an image or video file",
                                 type=["png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv"])


# Load YOLO model 
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO('runs/detect/train/weights/best.pt', task="detect")


model = load_model()

if uploaded_file is not None:
    file_type = uploaded_file.type
    if file_type.startswith("image"):
        # Process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run detection
        results = model.predict(image)
        annotated_image = results[0].plot()

        # Save processed image
        output_image_path = "processed_image.jpg"
        cv2.imwrite(output_image_path, annotated_image)

        st.image(annotated_image, channels="BGR", caption="Detection Result")

        # Provide download button for image
        with open(output_image_path, "rb") as img_file:
            st.download_button("Download Processed Image", img_file, file_name="processed_image.jpg")

    elif file_type.startswith("video"):
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file.")
        else:
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Define output video writer
            output_video_path = "processed_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            frame_count = 0
            st.info("Processing video, please wait...")
            progress_bar = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

                frame_count += 1
                progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()
            st.success("Video processing complete! You can download the processed video below.")

            # Provide a download button for processed video
            with open(output_video_path, "rb") as video_file:
                st.download_button("Download Processed Video", video_file, file_name="processed_video.mp4")

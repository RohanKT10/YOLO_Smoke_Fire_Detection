# Fire & Smoke Detection using YOLOv8

## Overview
This project is a **Fire & Smoke Detection** system using a **custom-trained YOLOv8 model**. The application allows users to upload **images or videos**, processes them using **YOLO**, and provides annotated outputs highlighting detected fire and smoke. This tool can be useful for **early fire hazard detection** in safety applications.

## Features
- **Supports image and video processing**
- **Uses a custom YOLOv8 model** for fire and smoke detection
- **Displays bounding boxes** around detected fire and smoke regions
- **Download processed files** with detected annotations
- **Real-time video processing with progress tracking**

## How It Works
1. Upload an **image or video file** (Supported formats: `png, jpg, jpeg, mp4, avi, mov, mkv`)
2. The YOLOv8 model processes the input and detects fire/smoke.
3. The processed image/video is displayed with bounding boxes.
4. Users can **download** the processed file.
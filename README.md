# 🚦 Smart Automation System for Traffic Management

A high-performance **Computer Vision** based Traffic Management and Accident Detection System. This project uses **OpenCV** and the **YOLOv3** (You Only Look Once) deep learning model to monitor real-time traffic, detect accidents, and log violations.

---

## ✨ Features

- **🚗 Real-time Vehicle Detection**: Identifies cars, buses, trucks, and motorbikes with high accuracy using YOLOv3.
- **🚨 Accident Detection**: Automatically detects vehicle collisions using Intersection over Union (IoU) algorithms and logs the event.
- **📸 Automatic Violation Logging**: Captures and saves screenshots of accidents and red-light violations in the `rule_break/` directory.
- **🚥 Traffic Violation Alerts**: Monitors virtual lines to identify red-light jumpers.
- **💻 Modern Web Dashboard**: A sleek, translucent, and responsive web interface for system monitoring and reporting.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.x, OpenCV (DNN Module), NumPy.
- **Frontend**: HTML5, Vanilla CSS3 (Custom Glassmorphism Design).
- **Model**: YOLOv3 (Darknet weights).

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python installed, then run the following to install dependencies:

```bash
pip install opencv-python numpy
```

### 2. Download YOLOv3 Model Files
Due to size limitations, the AI model weights are not included in the repository. Please download them into the project root:

- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) (240MB)
- [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true)
- [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true)

### 3. Usage

#### Run the Detection System:
```bash
python VehicleDetect.py
```
*Press `q` to exit the video windows.*

#### View the Web Dashboard:
Simply open `index.html` in any modern web browser or use a live server.

---

## 📁 Project Structure

- `VehicleDetect.py`: Core logic for AI detection and accident monitoring.
- `index.html`: Main landing page for the traffic management platform.
- `styles.css`: Custom CSS for the dashboard.
- `side2.mp4`: Sample traffic video for testing.
- `rule_break/`: Auto-generated folder for violation/accident screenshots.

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Contributors
- **CodeBeeny**

---
*Created for the SIH (Smart India Hackathon) Project.*

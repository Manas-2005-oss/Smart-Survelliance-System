
# Smart Surveillance System

AI-powered real-time surveillance system for detecting violence and weapons using Deep Learning and Computer Vision.

---

# Overview

The Smart Surveillance System is an intelligent AI-based security monitoring solution designed to detect suspicious activities such as violence and weapon presence in real-time video streams.

This project combines:

*  Violence Detection using CNN-GRU
*  Weapon Detection using YOLOv8
*  Real-time Webcam Monitoring
*  Deep Learning & Computer Vision
*  Automated Alert Detection

The system analyzes live video frames continuously and identifies potential threats to improve surveillance and public safety.

---

# Features

✅ Real-time video surveillance

✅ Violence detection using Deep Learning

✅ Weapon detection using YOLOv8

✅ Live webcam support

✅ Video upload support

✅ Frame-by-frame video analysis

✅ Intelligent alert generation

✅ User-friendly interface

✅ AI-powered threat monitoring

---

#  Technologies Used

| Technology                  | Purpose                 |
| --------------------------- | ----------------------- |
| Python                      | Core Programming        |
| OpenCV                      | Video Processing        |
| PyTorch                     | Deep Learning Framework |
| YOLOv8                      | Weapon Detection        |
| CNN-GRU                     | Violence Detection      |
| NumPy                       | Numerical Computation   |
| Flask / Streamlit (if used) | Web Interface           |

---

#  Installation

## 1️⃣ Clone Repository

git clone https://github.com/Manas-2005-oss/Smart-Survelliance-System.git

## 2️⃣ Navigate to Project Folder

cd Smart-Survelliance-System

## 3️⃣ Create Virtual Environment

python -m venv .venv

## 4️⃣ Activate Virtual Environment

### Windows

.venv\Scripts\activate

### Linux / Mac


source .venv/bin/activate

## 5️⃣ Install Dependencies

pip install -r requirements.txt

---

#  Running the Project

## Run Webcam Surveillance

python camera_test.py

## Run Main Application

python main.py

---

#  Working Methodology

## Step 1: Video Capture

The system captures video either from:

* Live webcam
* CCTV feed
* Uploaded video files

---

## Step 2: Frame Extraction

The captured video is divided into individual frames using OpenCV.

---

## Step 3: Frame Preprocessing

Frames are resized, normalized, and converted into a suitable format for deep learning models.

---

## Step 4: Violence Detection

The CNN-GRU model analyzes temporal patterns and detects violent activities.

---

## Step 5: Weapon Detection

YOLOv8 identifies weapons such as:

* Guns
* Knives
* Dangerous objects

---

## Step 6: Alert Generation

When suspicious activity is detected:

* Alerts are triggered
* Detection labels are displayed
* Threat monitoring is activated

---

#  Model Information

## Violence Detection Model

| Parameter | Details                 |
| --------- | ----------------------- |
| Model     | CNN-GRU                 |
| Task      | Violence Detection      |
| Framework | PyTorch                 |
| Input     | Video Frames            |
| Output    | Violence / Non-Violence |

---

## Weapon Detection Model

| Parameter | Details               |
| --------- | --------------------- |
| Model     | YOLOv8                |
| Task      | Weapon Detection      |
| Framework | Ultralytics           |
| Input     | Images / Video Frames |
| Output    | Bounding Boxes        |

---

#  Output Screenshots

##  Home Screen

<img src="Outputs/<img width="1785" height="885" alt="Screenshot 2026-04-06 195500" src="https://github.com/user-attachments/assets/b371a7ff-b3cf-4a9c-bfb7-60d19e55aa7b" />
" width="900">

---

##  Violence Detection Output

<img src="Outputs/<img width="1814" height="881" alt="Screenshot 2026-04-06 195652" src="https://github.com/user-attachments/assets/75c188f9-2b5c-47b3-a1a5-6a2b935314a9" />
" width="900">

---

##  Weapon Detection Output

<img src="Outputs/<img width="1760" height="984" alt="Screenshot 2026-04-06 195828" src="https://github.com/user-attachments/assets/16dc1a2b-6d43-4a2b-b5ef-296529fe0b65" />
" width="900">

---

##  Real-Time Webcam Detection

<img src="Output/<img width="1012" height="935" alt="Screenshot 2026-04-06 195811" src="https://github.com/user-attachments/assets/7ca7ebc2-89d6-4a74-a8a0-e510206e95bd" />
" width="900">

---

#  Future Enhancements

* Face Recognition Integration
* Cloud-based Surveillance
* SMS / Email Alert System
* Multi-camera Monitoring
* Suspicious Activity Tracking
* AI Audio Threat Detection
* Mobile Application Integration

---

#  Applications

* Smart City Surveillance
* School & College Security
* Airport Monitoring
* Public Safety Systems
* Railway Station Monitoring
* Shopping Mall Surveillance
* Crime Prevention Systems

---

#  Advantages

✅ Improves public safety

✅ Reduces manual monitoring effort

✅ Real-time threat detection

✅ Fast response generation

✅ Intelligent AI-based monitoring

---

#  Limitations

* Requires GPU for faster inference
* Performance depends on dataset quality
* Detection accuracy may vary in low lighting conditions

---

#  Dataset

Dataset used for:

* Violence Detection
* Weapon Detection

 Due to GitHub storage limitations, datasets are not uploaded to this repository.

You can add your dataset download links here.

Example:

Dataset Link: https://your-dataset-link.com

---

#  Author

## Manas Ippalpalli

AI & Full Stack Developer

* GitHub: [https://github.com/Manas-2005-oss](https://github.com/Manas-2005-oss)
* LinkedIn: Add your LinkedIn profile here

---

# ⭐ Support

If you like this project:

⭐ Star the repository

  Fork the repository

  Contribute to improvements

---

#  License

This project is developed for educational and research purposes.

---

#  Conclusion

The Smart Surveillance System demonstrates the power of Artificial Intelligence and Computer Vision in modern security systems. By integrating violence detection and weapon detection models, the system can intelligently monitor real-time video feeds and identify dangerous situations automatically.

This project highlights how deep learning can enhance public safety, reduce manual surveillance effort, and provide faster response mechanisms for threat detection in real-world environments.

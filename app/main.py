from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
import shutil
import os
import cv2
from app.analyzer import analyze_video

app = FastAPI()

# Create folders
os.makedirs("uploads", exist_ok=True)

# ---------------------
# Detection control
# ---------------------
detection_running = False


# ---------------------
# Home Route
# ---------------------
@app.get("/")
def home():
    return FileResponse("static/index.html")


# ---------------------
# Video Upload API
# ---------------------
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_video(file_path)

    return result


# ---------------------
# Start Detection
# ---------------------
@app.get("/start_detection")
def start_detection():
    global detection_running
    detection_running = True
    return {"status": "Detection Started"}


# ---------------------
# Stop Detection
# ---------------------
@app.get("/stop_detection")
def stop_detection():
    global detection_running
    detection_running = False
    return {"status": "Detection Stopped"}


# ---------------------
# Live Camera Stream
# ---------------------
def generate_frames():

    global detection_running

    cap = cv2.VideoCapture(0)

    from collections import deque
    import torch
    from model import CNN_GRU_Violence
    from ultralytics import YOLO

    device = torch.device("cpu")

    violence_model = CNN_GRU_Violence().to(device)
    violence_model.load_state_dict(torch.load("models/violence_model.pth", map_location=device))
    violence_model.eval()

    yolo_model = YOLO("yolov8n.pt")

    CLIP_LEN = 16
    FRAME_SIZE = 160

    frame_buffer = deque(maxlen=CLIP_LEN)

    while True:

        # Stop webcam if detection OFF
        if not detection_running:
            cap.release()
            break

        success, frame = cap.read()

        if not success:
            break

        # ---------------------
        # WEAPON DETECTION
        # ---------------------
        weapon_detected = False

        results = yolo_model(frame, conf=0.25, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = yolo_model.names[cls]

                if label in ["knife", "scissors"]:
                    weapon_detected = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

                    cv2.putText(frame,label,(x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        # ---------------------
        # VIOLENCE DETECTION
        # ---------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb,(FRAME_SIZE,FRAME_SIZE))

        tensor = torch.from_numpy(resized).float()/255.0
        tensor = tensor.permute(2,0,1)

        frame_buffer.append(tensor)

        label_text = "NORMAL"
        color = (0,255,0)

        if len(frame_buffer) == CLIP_LEN:

            clip = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)

            with torch.no_grad():

                probs = torch.softmax(violence_model(clip),dim=1)
                violence_prob = probs[0,1].item()

                if violence_prob > 0.75:
                    label_text = "VIOLENCE"
                    color = (0,0,255)

        # ---------------------
        # COMBINE
        # ---------------------
        if weapon_detected and label_text == "VIOLENCE":
            label_text = "🚨 HIGH ALERT"
            color = (0,0,255)

        elif weapon_detected:
            label_text = "⚠️ WEAPON"
            color = (0,140,255)

        cv2.rectangle(frame,(10,10),(350,80),color,-1)

        cv2.putText(frame,label_text,(20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        # Encode frame
        _, buffer = cv2.imencode('.jpg',frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ---------------------
# Stream Route
# ---------------------
@app.get("/video_feed")
def video_feed():

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
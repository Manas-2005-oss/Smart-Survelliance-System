import cv2
import torch
from model import CNN_GRU_Violence
from collections import deque
import time
import os
from ultralytics import YOLO

CLIP_LEN = 16
SMOOTHING_WINDOW = 5
FRAME_SIZE = 160

VIOLENCE_THRESHOLD = 0.45
VIOLENCE_CONFIRM_FRAMES = 2


def main():

    device = torch.device("cpu")

    model_path = "models/violence_model.pth"

    if not os.path.exists(model_path):
        print("Model not found. Train first.")
        return

    model = CNN_GRU_Violence().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("Model loaded:", model_path)

    # ---------------------
    # YOLO Weapon Model
    # ---------------------
    yolo_model = YOLO("yolov8n.pt")  # Using the smallest YOLOv8 model for speed
    print("YOLO model loaded for weapon detection.")

    cap = cv2.VideoCapture(0)

    frame_buffer = deque(maxlen=CLIP_LEN)
    prob_buffer = deque(maxlen=SMOOTHING_WINDOW)

    violence_counter = 0

    prev_time = time.time()

    print("Smart Surveillance Live (press 'q' to quit)")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ---------------------
        # Weapon Detection
        # ---------------------
        weapon_detected = False

        results = yolo_model(frame, conf=0.2, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = yolo_model.names[cls]
                conf = float(box.conf[0])
                print("Detected:", label, "Confidence:", conf)  # DEBUG

                if label in ["scissors", "knife"]:
                    weapon_detected = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText (frame, f"{label} (weapon)",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        label_text = "NORMAL"
        box_color = (0,255,0)
        conf_text = ""

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb,(FRAME_SIZE,FRAME_SIZE))

        tensor = torch.from_numpy(resized).float()/255.0
        tensor = tensor.permute(2,0,1)

        frame_buffer.append(tensor)

        if len(frame_buffer) == CLIP_LEN:

            clip = torch.stack(list(frame_buffer))
            clip = clip.unsqueeze(0).to(device)

            with torch.no_grad():

                logits = model(clip)
                probs = torch.softmax(logits,dim=1)

                violence_prob = probs[0,1].item()

            prob_buffer.append(violence_prob)

            if len(prob_buffer) == SMOOTHING_WINDOW:

                avg_prob = sum(prob_buffer)/SMOOTHING_WINDOW

                if avg_prob >= VIOLENCE_THRESHOLD:
                    violence_counter += 1
                else:
                    violence_counter = 0

                if violence_counter >= VIOLENCE_CONFIRM_FRAMES:

                    label_text = "VIOLENCE"
                    box_color = (0,0,255)
                    conf_text = f"{avg_prob:.2f}"

                else:

                    label_text = "NORMAL"
                    box_color = (0,255,0)
                    conf_text = f"{1-avg_prob:.2f}"

        # ---------------------
        # Combine Weapon + Violence
        # ---------------------
        if weapon_detected:
            label_text += " + WEAPON"

        cv2.rectangle(frame,(10,10),(320,95),box_color,-1)

        cv2.putText(frame,label_text,(20,45),
                    cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),3)

        if conf_text:
            cv2.putText(frame,f"Conf: {conf_text}",(20,75),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        curr_time = time.time()
        fps = 1.0/(curr_time-prev_time+1e-6)
        prev_time = curr_time

        cv2.putText(frame,f"FPS: {fps:.1f}",
                    (frame.shape[1]-120,frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        cv2.imshow("Smart Surveillance System",frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    main()
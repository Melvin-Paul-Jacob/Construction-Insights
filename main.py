import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


SCALE = 4
VIDEO_PATH = "test.mp4"
PPE_MODEL = "weights/ppe.pt"
PERSON_MODEL = "weights/person.onnx"
CONF_PERSON = 0.5
CONF_PPE = 0.3
SNAPSHOT_DIR = "violations"
REPORT_DIR = "reports"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

PPE_CLASSES = ['Hardhat', 'Mask', 'NO-Hardhat',
               'NO-Mask', 'NO-Safety Vest', 'Person',
               'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

HARDHAT_CLASS = PPE_CLASSES.index("Hardhat")
VEST_CLASS = PPE_CLASSES.index("Safety Vest")
MACHINERY_CLASS = PPE_CLASSES.index("machinery")
VEHICLE_CLASS = PPE_CLASSES.index("vehicle")

PROXIMITY_THRESHOLD = 150  # pixels

person_model = YOLO(PERSON_MODEL, task="detect")
ppe_model = YOLO(PPE_MODEL, task="detect")
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

# reports
worker_report = []
ppe_report = []
violation_report = []
proximity_report = []

saved_events = set()

while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, None, fx=1 / SCALE, fy=1 / SCALE,
                       interpolation=cv2.INTER_LINEAR)
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    track_results = person_model.track(frame, persist=True, conf=CONF_PERSON,
                                       tracker="bytetrack.yaml", verbose=False)

    ppe_results = ppe_model.predict(frame, conf=CONF_PPE, verbose=False)
    annotated = frame.copy()

    ppe_boxes = []
    equipment_boxes = []

    for box in ppe_results[0].boxes:
        cls_id = int(box.cls.item())
        xyxy = box.xyxy.cpu().numpy()[0]
        ppe_boxes.append((cls_id, xyxy))
        if cls_id in [MACHINERY_CLASS, VEHICLE_CLASS]:
            equipment_boxes.append(xyxy)

    worker_count = 0
    if track_results[0].boxes.id is not None:
        boxes = track_results[0].boxes.xyxy.cpu().numpy()
        track_ids = track_results[0].boxes.id.cpu().numpy().astype(int)
        worker_count = len(track_ids)
        for track_id, person_box in zip(track_ids, boxes):
            px1, py1, px2, py2 = person_box
            helmet = False
            vest = False
            for cls_id, ppe_box in ppe_boxes:
                x1, y1, x2, y2 = ppe_box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                inside_person = (px1 <= cx <= px2 and py1 <= cy <= py2)

                if not inside_person: continue
                if cls_id == HARDHAT_CLASS: helmet = True
                if cls_id == VEST_CLASS: vest = True

            ppe_report.append(
                {
                    "timestamp": timestamp,
                    "worker_id": track_id,
                    "helmet": helmet,
                    "vest": vest
                }
            )

            violations = []

            if not helmet:
                violations.append("No Helmet")

            if not vest:
                violations.append("No Vest")

            # save snapshot only once
            for violation in violations:

                event_key = (
                    int(timestamp),
                    track_id,
                    violation
                )

                if event_key not in saved_events:

                    saved_events.add(event_key)

                    fname = (
                        f"{SNAPSHOT_DIR}/"
                        f"{int(timestamp)}_"
                        f"{track_id}_"
                        f"{violation.replace(' ','_')}.jpg"
                    )

                    cv2.imwrite(fname, frame)

                violation_report.append(
                    {
                        "timestamp": timestamp,
                        "worker_id": track_id,
                        "violation": violation
                    }
                )

            person_center = np.array([(px1 + px2)/2, (py1 + py2)/2])
            for eq_box in equipment_boxes:
                ex1, ey1, ex2, ey2 = eq_box
                equipment_center = np.array([(ex1 + ex2)/2, (ey1 + ey2)/2])
                dist = np.linalg.norm(person_center - equipment_center)
                if dist < PROXIMITY_THRESHOLD:
                    proximity_report.append(
                        {"timestamp": timestamp,
                         "worker_id": track_id,
                         "event": "Unsafe Proximity"
                        }
                    )

                    cv2.putText(annotated,"UNSAFE PROXIMITY",
                                (int(px1), int(py1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

            color = (0, 255, 0)
            if not helmet or not vest:
                color = (0, 0, 255)

            cv2.rectangle(annotated, 
                          (int(px1), int(py1)), 
                          (int(px2), int(py2)),
                          color, 2)

            cv2.putText(annotated,
                        f"ID:{track_id} H:{helmet} V:{vest}",
                        (int(px1), int(py1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    worker_report.append({"timestamp": timestamp,
                          "worker_count": worker_count})

    cv2.putText(annotated, f"Workers: {worker_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    cv2.imshow("Safety Monitoring", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Create reports folder
os.makedirs(REPORT_DIR, exist_ok=True)

# Save reports
pd.DataFrame(worker_report).to_csv(REPORT_DIR+"/worker_count.csv", index=False)
pd.DataFrame(ppe_report).to_csv(REPORT_DIR+"/ppe_compliance.csv",index=False)
pd.DataFrame(violation_report).to_csv(REPORT_DIR+"/violations.csv",index=False)
pd.DataFrame(proximity_report).to_csv(REPORT_DIR+"/unsafe_proximity.csv",index=False)
print("Reports saved.")

cap.release()
cv2.destroyAllWindows()
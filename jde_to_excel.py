import os
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort

SEQ_PATH = r"C:\dsp\train\MOT16-11\img1"   
YOLO_WEIGHTS = "yolov5s.pt"
OUTPUT_VIDEO = "tracked_output.mp4"
EXCEL_FILE = "tracking_results.xlsx"
DEVICE = "cpu"
SAVE_TRAJECTORY_PLOT = True
MAX_FRAMES = None  

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=YOLO_WEIGHTS, force_reload=True)
model.to(DEVICE).eval()

tracker = DeepSort(max_age=30, n_init=2,
                   nms_max_overlap=1.0, max_cosine_distance=0.4)

img_files = sorted([f for f in os.listdir(SEQ_PATH) if f.endswith('.jpg')])
if MAX_FRAMES is not None:
    img_files = img_files[:MAX_FRAMES]

first_img = cv2.imread(os.path.join(SEQ_PATH, img_files[0]))
height, width = first_img.shape[:2]
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

tracking_data = []

for frame_idx, img_file in enumerate(img_files):
    img_path = os.path.join(SEQ_PATH, img_file)
    img = cv2.imread(img_path)

    results = model(img[..., ::-1])  
    detections = results.xyxy[0].cpu().numpy() 

    dets_for_tracker = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        dets_for_tracker.append(((x1, y1, x2-x1, y2-y1), conf, int(cls)))

    tracks = tracker.update_tracks(dets_for_tracker, frame=img)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb() 
        x1, y1, x2, y2 = map(int, ltrb)

        tracking_data.append({
            'Frame': frame_idx + 1,
            'ID': track_id,
            'X1': x1,
            'Y1': y1,
            'X2': x2,
            'Y2': y2,
            'Width': x2 - x1,
            'Height': y2 - y1,
            'CenterX': (x1 + x2) // 2,
            'CenterY': (y1 + y2) // 2,
            'Class': results.names[int(det[5])] if len(det) > 5 else "unknown",
            'Confidence': float(det[4]) if len(det) > 4 else -1
        })

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(img)
    cv2.imshow('Tracking', img)
    cv2.waitKey(1)

out.release()
cv2.destroyAllWindows()

df = pd.DataFrame(tracking_data)
df.to_excel(EXCEL_FILE, index=False, engine="openpyxl")
print(f"[INFO] Tracking results saved to {EXCEL_FILE}")

if SAVE_TRAJECTORY_PLOT and not df.empty:
    plt.figure(figsize=(10, 6))
    for track_id in df['ID'].unique():
        track = df[df['ID'] == track_id]
        plt.plot((track['X1'] + track['X2']) / 2,
                 (track['Y1'] + track['Y2']) / 2,
                 label=f'ID {track_id}')
    plt.gca().invert_yaxis()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Object Trajectories')
    plt.legend()
    plt.show()

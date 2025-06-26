import cv2
import torch
import numpy as np
from collections import deque
from model.faceDetector.s3fd import S3FD
from landmark_loconet import loconet
from dlhammer.dlhammer import bootstrap
import collections
import mediapipe as mp

# Load config
default_config = {'cfg': './configs/multi.yaml'}
cfg = bootstrap(default_cfg=default_config, print_cfg=False)

# Load face detector and model
face_detector = S3FD(device='cuda' if torch.cuda.is_available() else 'cpu')
model = loconet(cfg=cfg, n_channel=4, layer=1)
model.loadParameters('pretrained/LoCoNet_LASER.model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Sliding window parameters
FACE_WINDOW = cfg['MODEL']['CLIP_LENGTH']  # Use config value directly, do not multiply
face_buffer = collections.deque(maxlen=FACE_WINDOW)
landmark_buffer = collections.deque(maxlen=FACE_WINDOW)

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes = face_detector.detect_faces(rgb_frame, conf_th=0.9)
    print(f"Detected {len(bboxes)} faces")
    drew_box = False
    buffering = False
    if len(bboxes) == 0:
        # No faces detected at all
        cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        for bbox in bboxes:
            x1, y1, x2, y2, conf = bbox
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                print("Empty face crop, skipping.")
                continue
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            face_crop_resized = cv2.resize(face_crop_gray, (112, 112))
            face_tensor = torch.from_numpy(face_crop_resized).unsqueeze(0).float()  # (1, 112, 112)
            face_tensor = face_tensor.repeat(4, 1, 1)  # (4, 112, 112)
            face_buffer.append(face_tensor)
            # Landmark extraction
            results = mp_face.process(face_crop_rgb)
            if results.multi_face_landmarks:
                landmarks = np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark])
                # Pad or trim to 82 landmarks (LoCoNet expects 82)
                if landmarks.shape[0] < 82:
                    pad = np.zeros((82 - landmarks.shape[0], 2))
                    landmarks = np.vstack([landmarks, pad])
                elif landmarks.shape[0] > 82:
                    landmarks = landmarks[:82]
                landmark_buffer.append(landmarks)
            else:
                # If no landmarks, append zeros
                landmark_buffer.append(np.zeros((82, 2)))
            print(f"Face buffer length: {len(face_buffer)}, Landmark buffer length: {len(landmark_buffer)}")
            if len(face_buffer) < FACE_WINDOW or len(landmark_buffer) < FACE_WINDOW:
                buffering = True
                continue
            visual_feature = torch.stack(list(face_buffer), dim=0).unsqueeze(0).to(device)  # (1, FACE_WINDOW, 4, 112, 112)
            landmark_np = np.stack(list(landmark_buffer))  # (FACE_WINDOW, 82, 2)
            landmark = torch.from_numpy(landmark_np).unsqueeze(0).unsqueeze(0).float().to(device)  # (1, 1, FACE_WINDOW, 82, 2)
            audio_feature = torch.zeros(1, 1, 96, 64, device=device)
            print(f"Running inference with visual_feature shape: {visual_feature.shape}, landmark shape: {landmark.shape}")
            with torch.no_grad():
                pred_score = model.model.forward_evaluation(audio_feature, visual_feature, landmark, None, None, False)
            label = 'Speaking' if pred_score[0] >= 0 else 'Silent'
            color = (0, 255, 0) if label == 'Speaking' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            drew_box = True
    if buffering:
        cv2.putText(frame, "Buffering...", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow('Active Speaker Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

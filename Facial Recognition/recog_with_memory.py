
import os
import json
import time

import cv2
import numpy as np

from memory_db import MemoryDB
from emotion_utils import EmotionEstimator, compute_comfort

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    raise RuntimeError("Please install insightface: pip install insightface") from e

SIM_THR = 0.35
GALLERY_JSON = "gallery.json"
OWNER_NAME = os.environ.get("OWNER_NAME", "owner")
CAM_INDEX = int(os.environ.get("CAM_INDEX", "0"))
EMO_BACKEND = os.environ.get("EMO_BACKEND", "auto")

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))

def load_gallery(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_gallery(path: str, d: dict):
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

def crop_face(bgr_image, bbox, expand=0.15):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = bgr_image.shape[:2]
    dx = int((x2 - x1) * expand)
    dy = int((y2 - y1) * expand)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)
    face = bgr_image[y1:y2, x1:x2].copy()
    return face

def main():
    mem = MemoryDB("theradog_memory.sqlite")

    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    emo = EmotionEstimator(backend=EMO_BACKEND)
    print(f"[INFO] Emotion backend: {emo.backend_name}")

    gallery = load_gallery(GALLERY_JSON)
    print(f"[INFO] Loaded gallery: {list(gallery.keys())}")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Set CAM_INDEX or check permissions.")

    print("[INFO] Press 'e' to enroll largest face as OWNER_NAME, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = app.get(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('e'):
            if faces:
                areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
                idx = int(np.argmax(areas))
                f = faces[idx]
                emb = f.normed_embedding.astype(float).tolist()
                gallery[OWNER_NAME] = {"embedding": emb}
                save_gallery(GALLERY_JSON, gallery)
                mem.ensure_person(OWNER_NAME)
                print(f"[ENROLL] Saved {OWNER_NAME} to gallery and DB.")
            else:
                print("[ENROLL] No face to enroll.")

        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            emb = f.normed_embedding.astype(np.float32)
            best_name = "Unknown"
            best_sim = -1.0
            for name, rec in gallery.items():
                gemb = np.array(rec["embedding"], dtype=np.float32)
                sim = cosine_sim(emb, gemb)
                if sim > best_sim:
                    best_sim, best_name = sim, name
            if best_sim < SIM_THR:
                best_name = "Unknown"

            face_crop = crop_face(frame, f.bbox, expand=0.18)
            try:
                emo_label, val, aro, conf = emo.predict(face_crop)
            except Exception as e:
                print(f"[WARN] Emotion backend error: {e}")
                emo_label, val, aro, conf = "neutral", 0.0, 0.2, 0.0

            comfort = compute_comfort(val, aro)

            person_name = best_name if best_name != "Unknown" else None
            if person_name is not None:
                person_id = mem.ensure_person(person_name)
                profile = mem.get_profile(person_id)

                episode = {
                    "person_id": person_id,
                    "timestamp": time.time(),
                    "distance": profile["pref_distance"],
                    "angle": profile["pref_angle"],
                    "speed": min(profile["speed_cap"], 0.3),
                    "emotion_label": emo_label,
                    "valence": float(val),
                    "arousal": float(aro),
                    "comfort": float(comfort),
                    "conf": float(conf),
                }
                mem.upsert_episode(**episode)
                mem.update_profile_from_episode(
                    person_id=person_id,
                    valence=val,
                    arousal=aro,
                    comfort=comfort,
                    confidence=conf
                )

            color = (0, 255, 0) if best_name != "Unknown" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{best_name}  sim:{best_sim:.2f}  emo:{emo_label} v:{val:+.2f} a:{aro:.2f} conf:{conf:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            bar_w = 120
            v_mid = x1 + (x2 - x1)//2
            v_len = int((val / 1.0) * (bar_w//2))
            cv2.line(frame, (v_mid, y2 + 14), (v_mid + v_len, y2 + 14), (255, 255, 255), 3)
            cv2.putText(frame, "V", (v_mid - bar_w//2 - 14, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            a_len = int(aro * bar_w)
            cv2.line(frame, (x1, y2 + 30), (x1 + a_len, y2 + 30), (255, 255, 255), 3)
            cv2.putText(frame, "A", (x1 - 14, y2 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        cv2.imshow("TheraDog: Face+Emotion", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

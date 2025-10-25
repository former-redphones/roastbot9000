from retinaface import RetinaFace
import numpy as np
import json
import cv2

scan_lock = False

img = cv2.imread("vision/avg_guy.png")

def numpy_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

faces = RetinaFace.detect_faces(img)
print(faces)
if len(faces) >= 1:
    # x1, y1, x2, y2 = faces.values()['face_1']['facial_area']
    face =  max(faces.values(), key=lambda f: (
        (f["facial_area"][2] - f["facial_area"][0]) *
        (f["facial_area"][3] - f["facial_area"][1])
    ))

    face_converted = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
                  for k, v in face.items()}

    with open('vision/avg_guy.json', 'w') as file:
        json.dump(face_converted, file, default=numpy_encoder, indent=4)

    x1, y1, x2, y2 = face['facial_area']
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 5)
    cv2.imshow('Guy', img)
    cv2.waitKey(0)
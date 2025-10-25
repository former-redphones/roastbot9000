from backend.face_normalizer import compute_landmark_differences
from retinaface import RetinaFace
from agent import RoastingAI
from align_face import align_face
import numpy as np
import pyttsx3
import asyncio
import random
import json
import cv2
import re

scan_lock = False

roaster = RoastingAI()

def PLACEHOLDER_ROAST():
    return "placeholder burn, (gottem)"

def rotate_landmarks(landmarks, angle, center):
    rotated = {}
    rad = np.radians(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    cx, cy = center
    for key, (x, y) in landmarks.items():
        x0 = x - cx
        y0 = y - cy

        x_rot = x0 * cos_a - y0 * sin_a
        y_rot = x0 * sin_a + y0 * cos_a

        rotated[key] = (x_rot + cx, y_rot + cy)
    return rotated

async def process_snapshot(img):
    global scan_lock
    print("Processing...")
    faces = await asyncio.to_thread(RetinaFace.detect_faces, img)
    if len(faces) >= 1:
        face = max(faces.values(), key=lambda f: (
            (f["facial_area"][2] - f["facial_area"][0]) *
            (f["facial_area"][3] - f["facial_area"][1])
        ))

        if face['score'] < 0.75:
            print("Low confidence, discarding")
            scan_lock = False
            return
        
        for landmark in face['landmarks'].values():
            cv2.circle(img, (int(landmark[0]), int(landmark[1])), 5, (0, 255, 255), -1)

        # cv2.circle(img, (int(face['landmarks']['right_eye'][0]), int(face['landmarks']['right_eye'][1])), 10, (255, 255, 255), -1)

        x1, y1, x2, y2  = face['facial_area']
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 5)
        cv2.imshow('Snapshot', img)
        diff = compute_landmark_differences(faces)
        roast = await asyncio.to_thread(roaster.promptAI, diff)
        print()
        print(roast)
        print("\n\n")
        print(roast['Roast'])
        print()
        tts = pyttsx3.init()
        tts.say(roast['Roast'])
        await asyncio.to_thread(tts.runAndWait)
        tts.stop()

        aligned_face = align_face(face, img)
        cv2.imshow('Aligned Snapshot', aligned_face)

    else:
        print("Error! No face found!")
    await asyncio.sleep(5)
    scan_lock = False

async def main():
    capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frames_with = 0
    frames_without = 0
    global scan_lock

    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        processed_frame = frame.copy()

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        
        if len(faces) >= 1:
            if not scan_lock:
                frames_with += 1
                frames_without = 0

            for (x, y, w, h) in faces:
                color = (255,255,255)
                if scan_lock:
                    color = (0,0,255)
                if frames_with > 50:
                    color = (0,255,0)
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 5)

        elif not scan_lock:
            frames_without += 1
            if frames_without >= 5:
                frames_with = 0
        
        if frames_with >= 100 and not scan_lock and len(faces) >= 1:
            print("DETECTING!")
            cv2.imshow('Snapshot', frame)
            asyncio.create_task(process_snapshot(frame))
            frames_with = 0
            scan_lock = True

        # Display the captured frame
        cv2.imshow('Face Tracking', processed_frame)

        # Press 'esc' or close window to exit the loop
        if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty("Face Tracking", cv2.WND_PROP_VISIBLE) < 1:
            break
            
        await asyncio.sleep(0)
        
    cv2.destroyAllWindows()
    for task in asyncio.all_tasks():
        task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
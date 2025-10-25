import cv2
import numpy as np

def align_face(face_dict, img):
    landmarks = face_dict['landmarks']
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    angle = angle - 180
    
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    
    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    
    x1, y1, x2, y2 = face_dict['facial_area']
    aligned_face = rotated_img[y1:y2, x1:x2]
    
    return aligned_face

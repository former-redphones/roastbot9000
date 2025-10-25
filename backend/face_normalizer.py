AVERAGE_FACE = {
    "score": 0.9992073178291321,
    "facial_area": [57, 85, 218, 301],
    "landmarks": {
        "right_eye": [97.83584594726562, 164.92042541503906],
        "left_eye": [176.42105102539062, 164.36297607421875],
        "nose": [137.58441162109375, 204.99790954589844],
        "mouth_right": [107.23731994628906, 247.01039123535156],
        "mouth_left": [170.84552001953125, 246.24623107910156]
    }
}

AVERAGE_FACE_NORMALIZED = {
    "right_eye": [25, 37],
    "left_eye": [74, 37],
    "nose": [50, 56],
    "mouth_right": [31, 75],
    "mouth_left": [71, 74]
}

def compute_landmark_differences(faces_dict):
    result = {}

    for face_id, face_data in faces_dict.items():
        diffs = {
            "score": round(face_data.get("score", 0), 5),
            "facial_area": face_data.get("facial_area", None)
        }

        landmarks = face_data.get("landmarks", {})
        face_box = face_data.get("facial_area", None)

        if not face_box:
            continue

        box_width = face_box[2] - face_box[0]
        box_height = face_box[3] - face_box[1]

        landmark_diffs = {}

        for key, avg_coords in AVERAGE_FACE_NORMALIZED.items():
            input_coords = landmarks.get(key)

            if input_coords:

                input_norm_x = ((input_coords[0] - face_box[0]) / box_width) * 100
                input_norm_y = ((input_coords[1] - face_box[1]) / box_height) * 100

                diff_x = int(round(input_norm_x - avg_coords[0]))
                diff_y = int(round(input_norm_y - avg_coords[1]))

                landmark_diffs[key] = [diff_x, diff_y]

        diffs["landmark_differences"] = landmark_diffs
        result[face_id] = diffs

    return result

if __name__ == "__main__":
    example_faces = {
        "face_1": {
            "score": 0.9993,
            "facial_area": [155, 81, 434, 443],
            "landmarks": {
                "right_eye": [157.8, 109.6],
                "left_eye": [274.9, 151.7],
                "nose": [103.4, 299.9],
                "mouth_right": [328.3, 138.7],
                "mouth_left": [220.2, 274.5]
            }
        }
    }

    diff_dict = compute_landmark_differences(example_faces)
    print(diff_dict)

from flask import Flask, request, render_template, jsonify
import face_recognition
import cv2
import numpy as np

app = Flask(__name__)

# Load face encodings and names
obama_image = face_recognition.load_image_file("9921103118.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("9921103163.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

pratham_image = face_recognition.load_image_file("9921103103.jpg")
pratham_face_encoding = face_recognition.face_encodings(pratham_image)[0]

known_face_encodings = [obama_face_encoding, biden_face_encoding, pratham_face_encoding]
known_face_names = ["9921103118", "9921103163", "9921103103"]

video_capture = None

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/identify', methods=['POST'])
def identify():
    global video_capture
    name = request.form['name']
    
    if not video_capture:
        video_capture = cv2.VideoCapture(0)
    print("hello")
    process_this_frame = True
    while True:
        ret, frame = video_capture.read()

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            if name == request.form['name']:
                video_capture.release()
                cv2.destroyAllWindows()
                return jsonify({'Enroll': name})
            
        process_this_frame = not process_this_frame

if __name__ == '__main__':
    app.run(debug=True)

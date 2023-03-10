import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

face_detection = mp_face_detection.FaceDetection(model_selection=1)

img = cv2.imread('pessoas.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = face_detection.process(img_rgb)

if results.detections:
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        h, w, c = img.shape
        x, y, w, h = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Detecção de face', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

face_detection.close()

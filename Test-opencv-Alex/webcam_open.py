import cv2

xml_haar_cascade = ('/Users/Alex/Desktop/python/projeto/haarcascade_frontalface_default.xml')

faceClassifier = cv2.CascadeClassifier(xml_haar_cascade)


capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while not cv2.waitKey(20) & 0xFF == ord("q"):
    ret, frame_color = capture.read()
    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    faces = faceClassifier.detectMultiScale(gray)

    cv2.imshow('color', frame_color)

    for x, y, w, h in faces:
        cv2.rectangle(frame_color, (x,y), (x + w, y + h),(255,255,0),2)

        cv2.imshow('color', frame_color)
import cv2 as cv

webcam = cv.VideoCapture(0)
classificador_rosto = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

if webcam.isOpened():
    validacao,  frame = webcam.read()
    while validacao:
        validacao,  frame = webcam.read()
        key = cv.waitKey(5)
        quadro_cinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = classificador_rosto.detectMultiScale(quadro_cinza, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.imshow('Rostos detectados', frame)    
        if key == 27:  # esc
            break
    cv.imwrite("Fotinha.png", frame)

webcam.realease()
cv.destroyAllWindows()

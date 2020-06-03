import numpy as np
import imutils
import cv2
import pafy
global fc
fc = 0

def Save_face(frame, i, w, h):
    global fc
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    dFace = frame[startY:endY, startX:endX]
    cv2.imwrite("faces/"+str(fc)+".jpg", dFace)
    fc += 1



prototxt = "models/deploy.prototxt"
model = "models/res10_300x300_ssd_iter_140000.caffemodel"
ConfidenceScale = 0.5
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("model loaded...")

url = "https://www.youtube.com/watch?v=fy-abdzbto0"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
vs = cv2.VideoCapture()  # Youtube
vs.open(best.url)
#vs = cv2.VideoCapture(0)  # Youtube
#vs = cv2.VideoCapture("../videos/2020-05-16 12-53-05 PTZ12301-1.avi") #video local


C_frame = 0
while True:
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    count = 0
    OldCount = 0
    NoChange = True
    C_frame += 1
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > ConfidenceScale:
            if count == OldCount:
                if C_frame >= 5:
                    C_frame = 0
                    print("A: Guardando Frame..."+str(C_frame) +" - - F:")
                    Save_face(frame, i, w, h)
                else:
                     print("NO GUARDAR")
            else:
                print("B: Guardando Frame..." +str(C_frame) +" - - F:")
                Save_face(frame, i, w, h)
                OldCount = count
                C_frame = 0

            count += 1
            #Save_face(frame, i, w, h)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Recortar Rostro y capturar encodigns
            #dFace = frame[startY:endY, startX:endX]
            #cv2.imshow("Frame", dFace)
            text = "Face : " + str(count) #format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (117, 255, 51), 1)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 153, 51), 1)
        else:
            if C_frame >= 5:
                C_frame = 0

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
vs.stop()











#print(all_face_encodings)
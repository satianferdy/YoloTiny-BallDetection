import cv2
import numpy as np
import glob
import random
import time

net = cv2.dnn.readNet("yolov3-tiny-custom.weights", "yolov3-tiny.cfg")

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()


images_path = glob.glob(r"ball.jpg") #ganti dengan direktori gambar anda
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
random.shuffle(images_path)
for img_path in images_path:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (480,360))
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    start = time.time()
    outs = net.forward(output_layers)
    end = time.time()
    print("[INFO] Waktu deteksi yolo {:.6f} detik".format(end - start))

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                print(class_id)
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    font = cv2.FONT_HERSHEY_PLAIN
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    unique, counts = np.unique(class_ids, return_counts=True)
    tambah=0
    cv2.rectangle(img, (3, 3), (165, 80), (0,0,255), 1)
    for i in range (len(counts)):
                    cv2.putText(img,str(classes[i])+" = "+str(counts[i]), (5,15+tambah),font,1, (0,0,255), 1)
                    tambah=tambah+15
    print(indexes)
    daftar=[]
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            daftar.append(label)
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            text = "{}: {:.2f}".format(label, confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

    print(daftar)
    cv2.imshow("Electronic Component Recognition", img)
    key = cv2.waitKey(0)
    
cv2.destroyAllWindows()

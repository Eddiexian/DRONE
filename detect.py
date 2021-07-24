import cv2
import numpy as np

COLORS = []

for i in range(0, 1000):
    COLORS.append(list(np.random.random(3) * 256))


def count_dis(x1, y1, x2, y2):
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))


CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4


video_path = ("DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4")

cap = cv2.VideoCapture(video_path)

net = cv2.dnn.readNet("yolov4-tiny-custom_best.weights",
                      "yolov4-tiny-custom.cfg")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (960,  540))


car_ID = []
frame_num = -1
while(True):

    ret, frame = cap.read()
    frame_num += 1
    if(frame is None):
        break
    frame = cv2.resize(frame, (960, 540))
    img = frame[100:, :]
    i = 0
    classes, scores, boxes = model.detect(
        img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % ("CAR", score)

        centerX = int(box[0]+box[2]/2)
        centerY = int(box[1]+box[3]/2+100)

        center = []
        center = [centerX, centerY]

        if(frame_num == 0):
            car_ID.append(center)
            color = COLORS[len(car_ID)]
            print(color)
            cv2.rectangle(frame, (box[0], box[1]+100),
                          (box[0]+box[2], box[1]+box[3]+100), color, 2)
            cv2.putText(frame, str(i), (box[0]+5, box[1] + box[3] + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            i += 1

        else:
            print(len(car_ID))
            shortest = 999999
            id_record = (len(car_ID)+1)
            for i in range(len(car_ID)):
                print("Count Dis"+str(i))
                if(shortest > count_dis(car_ID[i][0], car_ID[i][1], center[0], center[1])):
                    shortest = count_dis(
                        car_ID[i][0], car_ID[i][1], center[0], center[1])
                    id_record = i

            if(shortest < 50):
                color = COLORS[id_record]
                car_ID[id_record] = [center[0], center[1]]
                cv2.rectangle(frame, (box[0], box[1]+100),
                              (box[0]+box[2], box[1]+box[3]+100), color, 2)
                cv2.putText(frame, str(id_record), (box[0]+5, box[1]+box[3] + + 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            else:
                car_ID.append(center)
                color = COLORS[len(car_ID)]
                cv2.rectangle(frame, (box[0], box[1]+100),
                              (box[0]+box[2], box[1]+box[3]+100), color, 2)
                cv2.putText(frame, str(len(car_ID)+1), (box[0]+5, box[1]+box[3] + 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, "Yaoxian Ma", (430, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, str(len(car_ID)), (850, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    print("End Frame")
    out.write(frame)
    cv2.imshow("V", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()

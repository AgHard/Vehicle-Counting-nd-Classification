import torch
import torchvision
import wget
from torchvision import transforms as T

from PIL import Image
import cv2
#from google.colab.patches import cv2_imshow

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

transform = T.ToTensor()

coco_names = ["bicycle", "car", "bus", "train", "truck", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse"]

vehicle_num = 0
line_y = 300 # y-coordinate of the line
line_thickness = 5 # thickness of the line

video_capture = cv2.VideoCapture('traffic.mp4')
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    line_y = frame.shape[0] // 2

    img = transform(frame)

    with torch.no_grad():
        pred = model([img])

    bboxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
    num = torch.argwhere(scores > 0.6).shape[0]

    for i in range(num):
        x1, y1, x2, y2 = bboxes[i].numpy().astype("int")
        class_name = coco_names[labels.numpy()[i]-1]
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Check if vehicle passes the line
        if y1 <= line_y and y2 >= line_y:
            vehicle_num += 1
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), line_thickness)

        frame = cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, "Vehicles: " + str(vehicle_num), (20, 80), 0, 5, (100, 200, 0), 5)
    resized_frame = cv2.resize(frame, (640, 480))  # Resize the frame to 640x480
    cv2.imshow("frame", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Total vehicle count:", vehicle_num)

video_capture.release()
cv2.destroyAllWindows()

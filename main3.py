import torch
import torchvision
import cv2

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

coco_names = ["bicycle", "car", "bus", "train", "truck", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse"]

vehicle_num = 0
line_y = 300 # y-coordinate of the line
line_thickness = 5 # thickness of the line

video_capture = cv2.VideoCapture('My Video7.mp4')
fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video codec and filename
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('vwndeo.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    line_y = frame.shape[0] // 2

    # Detect objects in the input frame
    with torch.no_grad():
        img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        pred = model([img])

    # Filter detections with low confidence scores
    bboxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]
    num = torch.argwhere(scores > 0.6).shape[0]
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), line_thickness)
    for i in range(num):
        x1, y1, x2, y2 = bboxes[i].numpy().astype("int")
        class_name = coco_names[labels.numpy()[i]-1]
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Check if vehicle passes the line
    if y1 <= line_y+6 and y2 >= line_y-6:
        vehicle_num += 1

        frame = cv2.putText(frame, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame, "Vehicles: " + str(vehicle_num), (20, 80), 0, 5, (100, 200, 0), 5)

    out.write(frame)

video_capture.release()
out.release()
cv2.destroyAllWindows()

import numpy as np
import cv2
import torch
import torchvision

cap = cv2.VideoCapture("6.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video codec and filename
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

min_width_react = 80
min_hight_react = 80

count_line_position = 550
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx , cy

class_labels = ['__background__', 'car', 'truck', 'bus', 'motorcycle']
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def detect_objects(frame):
    # Convert the frame to a PyTorch tensor and normalize it
    image = frame.transpose(2, 0, 1)
    image = torch.from_numpy(image).float() / 255.0
    image = image.unsqueeze(0)

    # Move the tensor to the appropriate device (CPU or GPU)
    image = image.to(device)

    # Pass the tensor through the ResNet model and extract the class labels and bounding boxes
    with torch.no_grad():
        outputs = model(image)
        boxes = outputs['boxes'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()

    # Convert the bounding boxes to (x, y, w, h) format and filter out low-confidence detections
    detections = []
    for i in range(len(labels)):
        if labels[i] == 0:
            continue
        x1, y1, x2, y2 = boxes[i]
        w = x2 - x1
        h = y2 - y1
        if w < min_width_react or h < min_hight_react:
            continue
        detections.append((x1, y1, w, h, class_labels[labels[i]]))

    return detections

detect = []
offset = 6
counter = 0

while True:
    ret , frame = cap.read()
    grey = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3,3) ,5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub , np.ones((5,5)))
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (5,5))
    dilatada = cv2.morphologyEx(dilat , cv2.MORPH_CLOSE, kernal)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernal)
    countershape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 0, 0), 3)

    for (i, c) in enumerate(countershape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_hight_react)
        if not validate_counter:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if (y < (count_line_position + offset)) and (y > (count_line_position - offset)):
                counter += 1
            cv2.line(frame, (25, count_line_position), (1500, count_line_position), (255, 0, 0), 3)
            detect.remove((x, y))
            print("Vehicle Count" + str(counter))

    cv2.putText(frame, "Vehicle Counter : " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("ss", frame)
    out.frame()

    if cv2.waitKey(1) == 13:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
import torch
import torchvision
import wget
from torchvision import transforms as T

from PIL import Image
import cv2
#from google.colab.patches import cv2_imshow

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)

model.eval()
ig = Image.open("I.jpg")
transform = T.ToTensor()
img = transform(ig)

with torch.no_grad():
    pred = model([img])
pred[0].keys()

bboxes , labels , scores = pred[0]["boxes"] , pred[0]["labels"] , pred[0]["scores"]
num = torch.argwhere(scores>0.5).shape[0]

coco_names = ["bicycle", "car",  "bus", "train", "truck", "traffic light", "fire hydrant", "street sign",
                  "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                  "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie",
                  "suitcase",
                  "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat" ]
igg = cv2.imread('11.jpg')
vehicle_num = 0
for i in range(num):
    x1,y1,x2,y2 = bboxes[i].numpy().astype("int")
    class_name = coco_names[labels.numpy()[i]-1]
    igg = cv2.rectangle(igg , (x1+300,y1+300) , (x2+500,y2+250) , (0,0,255) , 3)
    igg = cv2.putText(igg , class_name , (x1 , y1-10) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255,0,0) , 1 , cv2.LINE_AA)
    vehicle_num+=1
    cv2.putText(igg, "Vehicles: " + str(vehicle_num), (20, 50), 0, 2, (100, 200, 0), 3)

    cv2.imshow("img",igg)
    cv2.waitKey(0)
print("Total current count", vehicle_num)
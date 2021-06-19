import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision import transforms
from torch import optim
from PIL import Image
from models import *
from self_curve import *
import os
import sys
import cv2
import time




def predict(img,face_detector,transform,net):
    start = time.time()
    gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_detector.detectMultiScale(gray_img, 1.32, 5)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi=img[y:y+w,x:x+h,:]#cropping region of interest i.e. face area from  image
        X = transform(roi).unsqueeze(0)
        X = X.to('cpu')
        emotion_dict = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        with torch.no_grad():
            net.eval()
            _,log_ps = net.cpu()(X)
            _, val_preds = torch.max(log_ps,1)
            pred = emotion_dict[val_preds.item()]
        cv2.putText(img, pred, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    end = time.time()
    seconds = end - start
    if seconds==0: fps=0
    else: fps = 1 / seconds
    cv2.putText(img, "FPS: " + str('%.0f' % fps), (5, 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255),1)
    resized_img = cv2.resize(img, (img.shape[1]*700//img.shape[0], 700))
    cv2.imshow('Facial emotion analysis ',resized_img)

def main():
    net = Shuffle_self_curve(num_classes=7)
    checkpoint = torch.load(os.path.join('weight', 'shuffle_self_curve_no_clhe.pt'), map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint,strict=False)

    transform = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])


    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = None
    flag = sys.argv[1]
    if flag == 'video':
        cap=cv2.VideoCapture(f'.\\Video\\{sys.argv[2]}.mp4')
    elif flag =='cam':
        cap = cv2.VideoCapture(0)
    else:
        img = cv2.imread(f'.\\Image\\{sys.argv[2]}.jpg')
    if cap != None and flag == 'video':
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        print(f"num frame: {num_frame}, width = {width} , height = {height}")#.format(num_frame,width,height))
        #fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
        #out = cv2.VideoWriter('result.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 24, (int(width),int(height)))
        for i in range(num_frame):
            ret,test_img=cap.read()# captures frame and returns boolean value and captured image
            predict(test_img,face_detector,transform,net)
            #time.sleep(0.02)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows
    elif cap != None and flag == 'cam':
        while cap.isOpened():
        #for i in range(num_frame):
            ret,test_img=cap.read()# captures frame and returns boolean value and captured image
            predict(test_img,face_detector,transform,net)
            #time.sleep(0.02)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows

    else:
        X = transform(img).unsqueeze(0)
        emotion_dict = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        with torch.no_grad():
            net.eval()
            _,log_ps = net.cpu()(X)
            #ps = torch.exp(log_ps)
            score = F.softmax(log_ps,dim=1)
            max_score,val_preds = torch.max(score,1)
            print()
            #score, val_preds = torch.max(log_ps,1)
            pred = emotion_dict[val_preds.item()]
            print(f'Model predict: {pred} with {max_score.item()*100} %')

if __name__ == "__main__":
    main()
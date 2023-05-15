import torch
import torch.nn as nn
import cv2
import json
import torchvision.models as models
import numpy as np
import pandas as pd
import statistics
import threading
import math
import subprocess
from tqdm import tqdm
from torchvision import transforms
from object_tracking import *
import time
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.eval()

class Image_Classifier(nn.Module):
    def init(self):
        super().init()
        self.model = nn.Sequential(
             Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1)),
             ReLU(),
             Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),
             ReLU(),
             MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
             Flatten(start_dim=1, end_dim=-1),
             Dropout(p=0.25, inplace=False),
             Linear(in_features=6272, out_features=132, bias=True),
             ReLU(),
             Dropout(p=0.5, inplace=False),
             Linear(in_features=132, out_features=11, bias=True),
        )

    def forward(self, x):
        return self.model(x)

color_classifier = torch.load("colour classifier.pt")
color_classifier = color_classifier.cuda()
color_classifier.eval()

body_classifier = torch.load('body classifier.pt')
body_classifier.eval()


class Detector:
    mot_tracker = Sort(max_age=30, min_hits=60) 

    object_dict = {}
    frame_count = 0

    clf_state = False
     
    def process_video(self, video_path, output_path,json_path ):
    
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ffmpeg_cmd = f"ffmpeg -y -f rawvideo -pix_fmt bgr24 -s {frame_width}x{frame_height} -r {fps} -i - -c:v libx264 -preset fast -crf 30 -pix_fmt nv12 -an -vcodec libx264 {output_path}"

        output_file = subprocess.Popen(ffmpeg_cmd.split(' '), stdin=subprocess.PIPE)

        self.mot_tracker = Sort(max_age=30, min_hits=60) 

        frame_cut = 0
        self.frame_count = 0
        self.clf_state = False

        start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Video processing completed')
                break

            self.process_frame(frame)


            output_file.stdin.write(frame.tobytes())


            self.frame_count += 1

        cap.release()
        output_file.stdin.close()
        output_file.wait()
        end = time.time()

        print(end - start)

        with open(json_path, "w") as f:
            json.dump(self.object_dict, f, indent=4)
            
            
    def process_frame(self,frame):
        frame_model = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame_model)

        track_bbs_ids = self.mot_tracker.update(results.xyxy[0][:,:4].cpu())

        for x1, y1, x2, y2, obj_id in track_bbs_ids:
            cx1 = int((x1 + x2) / 2)
            cy1 = int((y1 + y2) / 2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            diagonal = math.sqrt(width**2 + height**2)


            if diagonal > 100 :
                try: #TODO optimize
                    car_image = frame_model[int(y1):int(y2), int(x1):int(x2)]
                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Resize((32,32)),])
                    car_image = transform(car_image)
                    car_image = car_image.cuda()

                    body_car_image = frame_model[int(y1):int(y2), int(x1):int(x2)]
                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Resize((256,256)),])
                    body_car_image = transform(body_car_image)
                    body_car_image = body_car_image.cuda()
                except:
                    print('no object detected')
                    continue
                with torch.no_grad():
                    color_output = color_classifier(car_image.unsqueeze(0))
                    color_prediction = torch.argmax(color_output).item()
                    color_name = ['black','blue','brown','green','grey','orange','pink','purple','red','white','yellow']
                    color_class_name = color_name[color_prediction]

                    body_output = body_classifier(body_car_image.unsqueeze(0))
                    body_prediction = torch.argmax(body_output).item()
                    body_name = ['Heavy-Duty', 'Lorry', 'Luxury', 'Pickup', 'SUV', 'Sedan', 'Van']
                    body_class_name = body_name[body_prediction]
                    self.clf_state = True


                if obj_id not in self.object_dict:
                    self.object_dict[obj_id] = {
                        'bboxes': [(x1, y1, x2, y2)],
                        'frames':[self.frame_count],
                        'color_classifier_preds': [color_prediction],
                        'body_classifier_preds': [body_prediction]
                    }
                else:
                    self.object_dict[obj_id]['bboxes'].append(( x1, y1, x2, y2))
                    self.object_dict[obj_id]['frames'].append((self.frame_count))
                    self.object_dict[obj_id]['color_classifier_preds'].append(color_prediction)
                    self.object_dict[obj_id]['body_classifier_preds'].append(body_prediction)



                # Calculate the mode prediction of the classifier for the tracked object
                color_mode_pred = statistics.mode(self.object_dict[obj_id]['color_classifier_preds'])
                self.object_dict[obj_id]['color_mode_pred'] = str(color_name[color_mode_pred])

                body_mode_pred = statistics.mode(self.object_dict[obj_id]['body_classifier_preds'])
                self.object_dict[obj_id]['body_mode_pred'] = str(body_name[body_mode_pred])


            cv2.putText(frame, str(obj_id), (cx1, cy1), 0, 0.5, (255, 255, 255), 2)
            if self.clf_state == True:
                cv2.putText(frame, color_name[color_mode_pred], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (203, 192, 255), 2)
                cv2.putText(frame,  body_name[body_mode_pred], (int(cx1), int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (203, 192, 255), 2)
                self.clf_state = False

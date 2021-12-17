import os
import cv2
import json
import random
import datetime
import logging

import pandas as pd

class ImgProcessor:
    def __init__(self, mosaic="resize", draw_boxes=False, put_texts=False, todo_list = ['face','plate']):
        if mosaic=='color':
            self.mosaic = self.mosaic_color
        elif mosaic == "blur":
            self.mosaic = self.mosaic_blur
        elif mosaic == "resize":
            self.mosaic = self.mosaic_resize
        else:
            self.mosaic = lambda x,y:x
        
        self.draw_boxes = draw_boxes
        self.put_text = put_texts
        self.todo_list = todo_list
        self.color_dict = {name:(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for name in todo_list}
        
    def process(self, img, detections):
        classes, scores, bboxes = detections
        for c, s, bbox in zip(classes,scores,bboxes):
            x,y,w,h = bbox
            """
            filter 추가시 추가할 자리
            """
            if c in self.mosaic_list:
                img = self.mosaic(img, bbox)
                if self.draw_boxes:
                    cv2.rectangle(img, (x+2,y+2),(x+w-2,y+h-2), self.color_dict[c])
                if self.put_text:
                    cv2.putText(img, f"{c} [{float(s):.2f}]",(x,y-5),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,self.color_dict[c], 2)

        return img

    """
    mosaic functions
    """
    def mosaic_color(self, image, bbox, color=[0,0,0]):
        x,y,w,h = bbox
        image[y:y+h, x:x+w] = color
        return image

    def mosaic_resize(self, image, bbox, ratio=0.25):
        x,y,w,h = bbox
        plate = image[y:y+h, x:x+w]
        ph,pw,_ = plate.shape
        small = cv2.resize(plate, (max(int(pw*ratio),1),max(int(ph*ratio),1)))
        image[y:y+h, x:x+w] = cv2.resize(small, (pw,ph))
        return image

    def mosaic_blur(self, image, bbox, ratio=0.6):
        x,y,w,h = bbox
        m = int(max(((h*ratio)//2)*2+1,7))
        image[y:y+h, x:x+w] = cv2.GaussianBlur(image[y:y+h, x:x+w],(m, m),0)
        return image


class Annotator:
    def __init__(self,file_path):
        self.anno_dict = {}

        file_name,file_ext = os.path.splitext(os.path.basename(file_path))

        create_time_stamp = os.path.getmtime(file_path)
        create_time_local = datetime.datetime.fromtimestamp(create_time_stamp).isoformat(' ', 'milliseconds')

        file_size = os.path.getsize(file_path)
        
        self.anno_dict['FileInfo']['Name'] = file_name
        self.anno_dict['FileInfo']['Extension'] = file_ext
        self.anno_dict['FileInfo']['Created'] = create_time_local
        self.anno_dict['FileInfo']['FileSize'] = file_size
        self.anno_dict['Annotation'] = {}

        self.file_path = file_path


    def add_info(self, width, height, frame_count):
        self.anno_dict['FileInfo']['Frame-width'] = width
        self.anno_dict['FileInfo']['Frame-height'] = height
        self.anno_dict['FileInfo']['Frame-count'] = frame_count

    def write(self,detections,frame_cnt):
        classes, scores, bboxes = detections
        box_id = 0
        now_annos = {'frameNo':frame_cnt,'labels':[]}

        for c,s,bbox in zip(classes,scores,bboxes):
            obj_dict = {'class':c, 'boxcorners':bbox, 'id':box_id}
            now_annos['labels'].append(obj_dict)

        self.anno_dict['Annotation'][frame_cnt] = now_annos


    def save(self, save_path):
        with open(save_path,"w") as f:
            json.dump(self.annotate, f, indent=4, ensure_ascii=False)


    

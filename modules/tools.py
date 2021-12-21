import os
import cv2
import json
import random
import datetime
import logging

import pandas as pd

vid_ext_list = []
img_ext_list = []

class RW:
    def __init__(self, input_path, output_path):
        self.ext = os.path.splitext(input_path)[-1].lower()
        self.input_path = input_path
        self.output_path = output_path
        self.current_fcnt = 0

        if self.ext in img_ext_list:
            self.read = self.read_img
            self.write = self.write_img
            self.out = None
            self.total_fcnt = 1
        elif self.ext in vid_ext_list:
            self.cap = cv2.VideoCapture(self.input_path)
            self.read = self.read_vid
            self.out = set_saved_video(self.cap, self.output_path)
            self.write = self.write_vid
            self.total_fcnt = int(self.cap.get(cv2.CAP_RPOP_FRAME_COUNT))
        else:
            pass

    def read_vid(self):
        self.current_fcnt += 1
        ret,img = self.cap.read()
        if ret is False:
            return -1
        return img

    def read_img(self):
        self.current_fcnt += 1
        img = cv2.imread(self.input_path)
        if img is None:
            return -1
        self.input_path = ""
        return img

    def write_img(self,img):
        cv2.imwrite(self.output_path, img)

    def write_vid(self,img):
        self.out.write(img)

    def close(self):
        if self.out:
            self.out.release()
            self.cap.release()

    def __del__(self):
        if self.out:
            self.out.release()
            self.cap.release()



class ImgWorker:
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
        
    def do(self, img, detections):
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
            box_id += 1

        self.anno_dict['Annotation'][frame_cnt] = now_annos


    def save(self, save_path):
        with open(save_path,"w") as f:
            json.dump(self.annotate, f, indent=4, ensure_ascii=False)


    
def set_saved_video(cap, output_path):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'),fps,(width,height))
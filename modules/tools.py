import os
import cv2
import json
import random
import datetime
import shutil
import logging

import pandas as pd
from nltk.metrics.distance import edit_distance

vid_ext_list = ['.gif','.mov','.mp4','.avi']
img_ext_list = ['.jpg','.png','.bmp','.jpeg']
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
            filter ????????? ????????? ??????
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


def frame_select(input_path, anno_path, target):
    def select(selected_frame, candidate_list):
        # select function
        if len(selected_frame)==2:
            if candidate_list:
                n1 = int(selected_frame[0][0])
                n2 = int(selected_frame[1][0])
                new = max(candidate_list, key=lambda x: abs(n1-int(x[0]))+abs(n2-int(x[0])))
                selected_frame.append(new)
        elif len(selected_frame)==1:
            if candidate_list:
                n1 = int(selected_frame[0][0])
                new = max(candidate_list, key=lambda x:abs(n1-int(x[0])))
                selected_frame.append(new)
                candidate_list.remove(new)
                n2 = int(new[0])
                if candidate_list:
                    new = max(candidate_list, key=lambda x:abs(n1-int(x[0]))+abs(n2-int(x[0])))
                    selected_frame.append(new)
        elif len(selected_frame)==0:
            if len(candidate_list)>=3:
                selected_frame.append(candidate_list[0])
                selected_frame.append(candidate_list[len(candidate_list)//2])
                selected_frame.append(candidate_list[-1])
            else:
                selected_frame += candidate_list

        return selected_frame

    name = input_path.split("/")[-1]
    gif_flag = 0
    if name.lower().endswith("gif"):
        input_path = gif2mp4(input_path)
        gif_flag = 1
    
    cap = cv2.VideoCapture(input_path)
    
    with open(anno_path, "r") as f:
        anno_dict = json.load(f)
    
    matched_list = []
    similar_list = []
    not_matched_list = []
    for fcnt, frame_anno in anno_dict['Annotation'].items():
        fcnt_cache = []
        matched = 0
        similar = 0
        for label in frame_anno['labels']:
            if 'plate_num' in label:
                fcnt_cache.append(label['plate_num'])
                if label['plate_num']==target:
                    matched = 1
                elif label['plate_num'][-4:]==target[-4:]:
                    similar = 1
                elif ned(label['plate_num'],target)>0.7:
                    similar = 1
        if matched:
            matched_list.append([fcnt,fcnt_cache])
        elif similar:
            similar_list.append([fcnt,fcnt_cache])
        elif fcnt_cache:
            not_matched_list.append([fcnt,fcnt_cache])

    um_cand = matched_list + similar_list

    selected_frame = []
    # step 1 select from matched
    if len(matched_list)>=3:
        selected_frame.append(matched_list[0])
        selected_frame.append(matched_list[len(matched_list)//2])
        selected_frame.append(matched_list[-1])
    else:
        selected_frame += matched_list

    # step 2 select from similar
    selected_frame = select(selected_frame, similar_list)

    # step 3 select from other frame
    selected_frame = select(selected_frame, not_matched_list)

    # step 4 select from not detected frame
    total_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_cnt > 0:
        last_list = []
        for i in range(total_cnt):
            if i not in [int(j[0]) for j in selected_frame]:
                last_list.append([i,[]])
        selected_frame = select(selected_frame, last_list)

    selected_frame = sorted(selected_frame, key=lambda x:int(x[0]))

    for i,selection in enumerate(selected_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(selection[0]))
        ret,img = cap.read()
        if ret is False:
            print("cap error")
            return False
        else:
            selected_frame[i][0] = img

    if gif_flag:
        os.remove(input_path)
    return selected_frame






    

def gif2mp4(path,out_path='./temp_gif.mp4'):
    cap = cv2.VideoCapture(path)
    vid = set_saved_video(cap, out_path)
    while 1:
        ret, frame = cap.read()
        if ret is False:
            break
        vid.write(frame)
    cap.release()
    vid.release()

    return out_path


def ned(pred,gt):
    return 1 - edit_distance(pred,gt) / max(len(gt), len(pred))

def todo_add(todo, num, type='days'):
    # ex) todo = 2021122500
    todo_ts = datetime.datetime.strptime(todo, "%Y%m%d%H")
    if type=='months':
        cal = datetime.timedelta(months=num)
    elif type=='days':
        cal = datetime.timedelta(days=num)
    elif type=='hours':
        cal = datetime.timedelta(hours=num)
    elif type=='minutes':
        cal = datetime.timedelta(minutes=num)
    elif type=='seconds':
        cal = datetime.timedelta(seconds=num)
    else:
        return

    new_todo_ts = todo_ts + cal
    new_todo = datetime.datetime.strftime(new_todo_ts, "%Y%m%d%H")
    return new_todo

def remove_todo(home_path, todo):
    # remove download
    download_path = f"{home_path}/download/{todo}/"
    if os.path.isdir(download_path):
        original_list = os.listdir(download_path)
        shutil.rmtree(download_path)
    else:
        print(f"{download_path} not exist")
    
    # remove hadoop
    hadoop_path = f"{home_path}/hadoop/{todo}/"
    if os.path.isdir(hadoop_path):
        shutil.rmtree(hadoop_path)
    else:
        print(f"{hadoop_path} not exist")

    # remove nas/cvlcpt
    for file_name in original_list:
        name, ext = os.path.splitext(file_name)
        if ext in vid_ext_list:
            path = f"{home_path}/nas/cvlcpt/{todo}/{name}_unmosaic.mp4"
            if os.path.isfile(path):
                os.remove(path)
        elif ext in img_ext_list:
            path = f"{home_path}/nas/cvlcpt/{todo}/{name}_unmosaic.jpg"
            if os.path.isfile(path):
                os.remove(path)
        else:
            pass
    
    # remove nas/recg
    for file_name in original_list:
        name, ext = os.path.splitext(file_name)
        for i in range(3):
            path = f"{home_path}/nas/recg/{name}_{i}.jpg"
            if os.path.isfile(path):
                os.remove(path)

    print(f"remove done {todo}")

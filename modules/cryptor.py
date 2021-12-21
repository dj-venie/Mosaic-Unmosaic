import cv2
import json
import base64

import numpy as np
from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Util import Padding
from nltk.metrics.distance import edit_distance
class Cipher:
    def __init__(self, key, mode=AES.MODE_GCM):
        self.key = Padding.pad(str(key).encode(), 16)
        self.mode = mode

    def encrypt(self, crop_image):
        _, png_image = cv2.imencode(".png", crop_image)
        byte_image = png_image.tobytes()
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, self.mode, iv)

        return iv + cipher.encrypt(byte_image)

    def decrypt(self, enc_data):
        iv = enc_data[:16]
        cipher = AES.new(self.key, self.mode, iv)
        dec_result = cipher.decrypt(enc_data[16:])
        box_array = np.frombuffer(dec_result, dtype='uint8')
        box_image = cv2.imdecode(box_array, cv2.IMREAD_COLOR)
        return box_image
        

class EncWorker:
    def __init__(self, key):
        self.cipher = Cipher(key)
        self.enc_classes = ['plate','face']
        self.enc_dict = {}

    
    def do(self, img, detections, frame_cnt):
        classes, _, bboxes = detections
        enc_id = 0
        now_enc_dict = {}
        for cls, bbox in zip(classes, bboxes):
            x,y,w,h = bbox
            if cls in self.enc_classes:
                original = img[y:y+h, x:x+w]
                enc_data = self.cipher.encrypt(original)
                str_enc_data = base64.b64encode(enc_data).decode()
                now_enc_dict[enc_id] = {'bbox':[x,y,w,h],'encrypt':str_enc_data,'classes':cls}
            enc_id += 1
        self.enc_dict[frame_cnt] = now_enc_dict


    def save(self, save_path):
        with open(save_path,"w") as f:
            json.dump(self.enc_dict, f, indent=4, ensure_ascii=False)


class DecWorker:
    def __init__(self, key, target=""):
        self.cipher = Cipher(key)
        self.target = target
        if self.target in ['face', 'plate']:
            self.select = self.select_class
        elif self.target:
            self.select = self.select_nums
        else:
            self.select = self.select_all

    def do(self, img, fcnt, enc_dict):
        for lid in self.unmosaic_dict.get(str(fcnt)):
            plate = enc_dict[str(fcnt)][lid]
            x,y,w,h = plate['bbox']
            enc_data = plate['encrypt']
            benc_data = base64.b64decode(enc_data)
            real_plate = self.cipher.decrypt(benc_data)
            img[y:y+h,x:x+w] = real_plate
        return img

    def select_class(self, anno_dict):
        self.unmosaic_dict = {}
        for frame_num, frame_info in anno_dict['Annotation'].items():
            self.unmosaic_dict[frame_num] = []
            for label in frame_info['labels']:
                if label['class'] == self.target:
                    self.unmosaic_dict[frame_num].append(str(label['id']))
            if len(self.unmosaic_dict[frame_num])==0:
                self.unmosaic_dict.pop(frame_num)


    def select_nums(self, anno_dict):
        def ned(pred, gt):
            return 1 - edit_distance(pred,gt) / max(len(gt),len(pred))
        self.unmosaic_dict = {}
        for frame_num, frame_info in anno_dict['Annotation'].items():
            self.unmosaic_dict[frame_num] = []
            for label in frame_info['labels']:
                plate_num = label['plate_num']
                if plate_num[-4:] == self.target[-4:]:
                    self.unmosaic_dict[frame_num].append(str(label['id']))
                elif ned(plate_num, self.target) > 0.7:
                    self.unmosaic_dict[frame_num].append(str(label['id']))
            if len(self.unmosaic_dict[frame_num])==0:
                self.unmosaic_dict.pop(frame_num)


    def select_all(self, anno_dict):
        self.unmosaic_dict = {}
        for frame_num, frame_info in anno_dict['Annotation'].items():
            self.unmosaic_dict[frame_num] = []
            for label in frame_info['labels']:
                self.unmosaic_dict[frame_num].append(str(label['id']))

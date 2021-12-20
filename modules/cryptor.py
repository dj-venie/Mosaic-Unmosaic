import cv2
import json
import base64

import numpy as np
from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Util import Padding

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
import cv2
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
        
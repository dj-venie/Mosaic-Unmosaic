import cv2
import os
import json
import base64
import shutil
import datetime
import time
import traceback

from absl import flags, app
from absl.flags import FLAGS
from modules.cryptor import DecWorker
from modules.tools import RW

flags.DEFINE_string("input","./data/media/um_input.txt","path to input, anno, enc files and target")
flags.DEFINE_string("output","./um_output","path to output")
flags.DEFINE_string("key","venie","key to encrypt video")

def main(_argv):
    print(datetime.datetime().now())
    start_time = time.time()
    # check flags
    if os.path.isfile(FLAGS.input):
        input_txt = FLAGS.input
    else:
        print(f"[error] input txt not exist {FLAGS.input}")
        return

    output_dir = FLAGS.output
    os.makedirs(output_dir,exist_ok=True)

    # load input data
    with open(input_txt, "r") as f:
        input_infos = f.read().strip().split("\n")
    
    for input_info in input_infos:
        try:

            if input_info=="":
                continue
            vid_path, anno_path, enc_path, target = input_info.split()
            # read json
            if os.path.isfile(vid_path) and os.path.isfile(anno_path) and os.path.isfile(enc_path):
                with open(anno_path, "r") as af, open(enc_path, "r") as ef:
                    anno_dict = json.load(af)
                    enc_dict = json.load(ef)

            # setting tools
            vid_ext = os.path.splitext(os.path.split(vid_path)[-1])[-1]
            vid_name = os.path.splitext(anno_dict['FileInfo']['Name'])[0]
            io = RW(vid_path, f"{output_dir}/{vid_name}_unmosaic{vid_ext}")
            dec = DecWorker(FLAGS.key)
            dec.select(anno_dict)
            fcnt = 0
            while 1:
                img = io.read()
                if img == -1:
                    break
                img = dec.do(img,fcnt,enc_dict)
                io.write(img)
                fcnt += 1
            io.close()

        except:
            pass
    print(f'done {time.time() - start_time}s')

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass

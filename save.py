import glob
import os
import cv2
import time
import traceback

import pandas as pd

from absl import flags, app
from absl.flags import FLAGS
from modules.tools import frame_select, RW, img_ext_list, vid_ext_list

flags.DEFINE_string("HOME_PATH","/data/smart","home path")
flags.DEFINE_string("todo","","todo directory")

NOW_DIR = os.path.abspath("./")

def main(_argv):
    start_time = time.time()
    home_path = FLAGS.HOME_PATH
    todo = FLAGS.todo

    input_dir_path = f"{home_path}/download/{todo}"

    if os.path.exists(input_dir_path):
        print("[start] select {todo}")
    else:
        print(f"path error {input_dir_path}")
        return

    # split data
    video_list = []
    image_list = []
    for i in glob.glob(input_dir_path+"/*"):
        ext = os.path.splitext(i)[-1].lower()
        if ext in vid_ext_list:
            video_list.append(i)
        elif ext in img_ext_list:
            image_list.append(i)
        else:
            print(f"{ext} is not supported")
    # load meta df
    df = pd.read_csv(f"{home_path}/cubrid/{todo}/meta_tbl.csv")

    # dir setting
    os.makedirs(f"{home_path}/nas/recg",exist_ok=True)
    # video start
    for input_path in video_list:
        try:
            fname = input_path.split("/")[-1]
            name = os.path.splitext(fname)[0]
            sn = name.split("_")[0]
            data_sn = name.split("_")[1]

            anno_path = f"{home_path}/hadoop/{todo}/{name}_mosaic_anno.json"
            target = df.loc[(df.sn==sn) & (df.data_sn==data_sn),'vcl_no'].values[0]

            selected_frame = frame_select(input_path, anno_path, target)
            for i, selected in enumerate(selected_frame):
                img, plate_nums = selected
                img_path = f"{home_path}/nas/recg/{name}_{i}.jpg"
                selected_path = img_path
                ret = cv2.imwrite(img_path, img)
                if ret:
                    img_name = img_path.split("/")[-1]
                
        except:
            print(traceback.format_exc())

    
    # image start
    for input_path in image_list:
        try:
            fname = input_path.split("/")[-1]
            name = os.path.splitext(fname)[0]

            img_path = f"{home_path}/nas/recg/{name}_0.jpg"
            io = RW(input_path, img_path)
            img = io.read()
            io.write(img)

        except:
            print(traceback.format_exc())

    print("select done")

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass

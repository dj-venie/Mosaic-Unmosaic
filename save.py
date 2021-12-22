import glob
import os
import cv2
import json
import time
import datetime
import traceback

import pandas as pd
import numpy as np

from absl import flags, app
from absl.flags import FLAGS
from modules.tools import frame_select

flags.DEFINE_string("HOME_PATH","/data/smart","home path")
flags.DEFINE_string("todo","","todo directory")

NOW_DIR = os.path.abspath("./")

def main(_argv):
    start_time = time.time()
    home_path = FLAGS.HOME_PATH

if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass

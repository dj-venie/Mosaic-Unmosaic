import datetime
import time
import shutil
import subprocess
import os
import cv2
import glob
import time
import pandas as pd

from multiprocessing import Process
from absl import flags, app
from absl.flags import FLAGS

from modules.connector import hdfsConnection
from modules.tools import remove_todo,todo_add,vid_ext_list,img_ext_list

NOW_DIR = os.path.abspath("./")
python_path = "/home/venie/.conda/envs/py36/bin/python"

########################################################
flags.DEFINE_string("HOME_PATH","/data/","local directory path to process")
flags.DEFINE_string("key","venie","key to encrypt video")
flags.DEFINE_string("start_todo","","start todd dir")
flags.DEFINE_boolean("end_todo","inf","end todo dir")
########################################################

class SaveNode:
    def __init__(self,home_path):
        self.todo_list = []
        self.process = None
        self.next_todo_list = []
        self.home_path = home_path

    def check_status(self):
        if self.process:
            if self.process.is_alive():
                return "working"
            else:
                self.next_todo_list.append(self.todo)
                self.process = None
                return f"finish {self.todo}"

        elif self.todo_list:
            self.todo = self.todo_list.pop(0)
            pid = self.run()
            return f"start {self.todo}({pid})"
        else:
            return "waiting"

    def run(self):
        def save_process(home_path,todo):
            out = subprocess.run([python_path, f"{NOW_DIR}/save.py","--HOME_PATH",home_path,"--todo",todo],stdout=subprocess.PIPE)
        self.process = Process(target=save_process, args=(self.home_path,self.todo))
        self.process.start()
        return self.process.pid

        

class UMNode:
    def __init__(self,home_path,key, workers=4):
        self.todo_list = []
        self.process_list = []
        self.next_todo_list = []
        self.home_path = home_path
        self.workers = workers
        self.key = key

    def check_status(self):
        if self.process_list:
            flag = 0
            for process in self.process_list:
                flag |= process.is_alive()

            if flag:
                return "working"
            else:
                self.next_todo_list.append(self.todo)
                self.process_list = []
                return f"finish {self.todo}"

        elif self.todo_list:
            self.todo = self.todo_list.pop(0)
            pid = self.run()
            return f"start {self.todo}({pid})"
        else:
            return "waiting"

    def run(self):
        def um_process(input_path,output,key):
            out = subprocess.run([python_path, f"{NOW_DIR}/unmosaic.py","--input",input_path,"--output",output, "--key",key],stdout=subprocess.PIPE)

        # setting txt file
        home_path = self.home_path
        todo = self.todo
        output_dir = f"{home_path}/nas/cvlcpt/{todo}/"
        vid_list = []
        img_list = []
        for path in glob.glob(f"{home_path}/hadoop/{todo}/*"):
            ext = os.path.splitext(path)[-1].lower()
            if ext in vid_ext_list:
                vid_list.append(path)
            elif ext in img_ext_list:
                img_list.append(path)
            else:
                pass
        df = pd.read_csv(f"{home_path}/cubrid/{todo}/meta_tbl.csv")
        pids = []
        os.makedirs("temp",exist_ok=True)
        vbatch = len(vid_list)//self.workers + bool(len(vid_list)%self.workers)
        ibatch = len(img_list)//self.workers + bool(len(img_list)%self.workers)

        for worker_id in range(self.workers):
            input_path = f"temp/um_{worker_id}.txt"
            with open(input_path, "w") as f:
                input_list = vid_list[worker_id*vbatch:(worker_id+1)*vbatch] + img_list[worker_id*ibatch:(worker_id+1)*ibatch]
                for vpath in input_list:
                    vname = os.path.splitext(vpath)[0]
                    anno_path = f"{vname}_anno.json"
                    enc_path = f"{vname}_enc.json"

                    sn = vname.split("/")[-1].split("_")[0]
                    data_sn = vname.split("/")[-1].split("_")[1]
                    target = df.loc[(df.sn==int(sn))&(df.data_sn==int(data_sn)),'vcl_no'].values[0].replace(" ","")
                    if os.path.isfile(anno_path) & os.path.isfile(enc_path) & len(target):
                        f.write(f"{vpath} {anno_path} {enc_path} {target}\n")

            self.process = Process(target=um_process, args=(input_path,output_dir,self.key))
            self.process.start()
            pids.append(self.process.pid)
        return pids

class MosaicNode:
    def __init__(self,home_path,key, workers=2):
        self.todo_list = []
        self.process_list = []
        self.next_todo_list = []
        self.home_path = home_path
        GPU_NUM = int(cv2.cuda.getCudaEnabledDeviceCount())
        if GPU_NUM==0:
            self.workers = 1
        elif workers > GPU_NUM:
            self.workers = GPU_NUM
            print(f"mosaic worker must smaller than total gpunum change worker : {self.workers}")
        else:
            self.workers = workers
        self.key = key

    def check_status(self):
        if self.process_list:
            flag = 0
            for process in self.process_list:
                flag |= process.is_alive()

            if flag:
                return "working"
            else:
                self.next_todo_list.append(self.todo)
                self.process_list = []
                return f"finish {self.todo}"

        elif self.todo_list:
            self.todo = self.todo_list.pop(0)
            pid = self.run()
            return f"start {self.todo}({pid})"
        else:
            return "waiting"

    def run(self):
        def mc_process(input_path,output,key,gpu):
            out = subprocess.run([python_path, f"{NOW_DIR}/mosaic.py","--input",input_path,"--output",output, "--key",key,"--gpu",gpu],stdout=subprocess.PIPE)

        # setting txt file
        home_path = self.home_path
        todo = self.todo
        output_dir = f"{home_path}/hadoop/{todo}/"
        vid_list = []
        img_list = []
        for path in glob.glob(f"{home_path}/download/{todo}/*"):
            ext = os.path.splitext(path)[-1].lower()
            if ext in vid_ext_list:
                vid_list.append(path)
            elif ext in img_ext_list:
                img_list.append(path)
            else:
                pass

        pids = []
        os.makedirs("temp",exist_ok=True)
        vbatch = len(vid_list)//self.workers + bool(len(vid_list)%self.workers)
        ibatch = len(img_list)//self.workers + bool(len(img_list)%self.workers)

        for worker_id in range(self.workers):
            input_path = f"temp/mc_{worker_id}.txt"
            with open(input_path, "w") as f:
                input_list = vid_list[worker_id*vbatch:(worker_id+1)*vbatch] + img_list[worker_id*ibatch:(worker_id+1)*ibatch]
                for vpath in input_list:
                    f.write(f"{vpath}\n")

            self.process = Process(target=mc_process, args=(input_path,output_dir,self.key))
            self.process.start()
            pids.append(self.process.pid)
        return pids

class DownNode:
    def __init__(self,home_path):
        self.todo_list = []
        self.process = None
        self.next_todo_list = []
        self.home_path = home_path

    def check_status(self):
        if self.process:
            if self.process.is_alive():
                return "working"
            else:
                self.next_todo_list.append(self.todo)
                self.process = None
                return f"finish {self.todo}"

        elif self.todo_list:
            self.todo = self.todo_list.pop(0)
            pid = self.run()
            return f"start {self.todo}({pid})"
        else:
            return f"waiting {self.todo}"

    def run(self):
        def down_process(home_path,todo):
            out = subprocess.run([python_path, f"{NOW_DIR}/download.py","--HOME_PATH",home_path,"--todo",todo],stdout=subprocess.PIPE)
        self.process = Process(target=down_process, args=(self.home_path,self.todo))
        self.process.start()
        return self.process.pid

def main(_argv):
    home_path = FLAGS.HOME_PATH
    start_todo = FLAGS.start_todo
    end_todo = FLAGS.end_todo
    key = FLAGS.key

    d_node = DownNode(home_path)
    m_node = MosaicNode(home_path,key)
    u_node = UMNode(home_path,key)
    s_node = SaveNode(home_path)

    d_node.next_todo_list = m_node.todo_list
    m_node.next_todo_list = u_node.todo_list
    u_node.next_todo_list = s_node.todo_list

    d_node.todo_list.append(start_todo)

    while 1:
        d_status = d_node.check_status()
        m_status = m_node.check_status()
        u_status = u_node.check_status()
        s_status = s_node.check_status()

        # add down dir
        if d_status.startswith("finish") or d_status.startswith("waiting"):
            done_todo = d_status.split()[-1]
            next_todo = todo_add(done_todo, 1, 'hours')
            if next_todo >= end_todo:
                d_node.check_status = lambda x:"end of process"
            if datetime.datetime.now() > datetime.strptime(next_todo,"%Y%m%d%H")+datetime.timedelta(minutes=5):
                d_node.todo_list.append(next_todo)

        # check end point
        if s_status.startswith("finish"):
            done_todo = s_status.split()[-1]
            if done_todo==end_todo:
                print("total process done")
                break
        
        # show status 


        
                
        



if __name__=='__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
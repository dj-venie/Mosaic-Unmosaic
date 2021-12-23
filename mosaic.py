import cv2
import os
import glob
import threading
import traceback
import time
import datetime

from queue import Queue, Empty
from absl import flags, app
from absl.flags import FLAGS
from imutils.video import fps

from modules.detector import Yolo
from modules.tools import ImgWorker, Annotator, RW, img_ext_list, vid_ext_list
from modules.cryptor import EncWorker

flags.DEFINE_string("input", "./data/media/mc_input.txt", "path to input")
flags.DEFINE_string("output", "./mc_output/", "path to output")
flags.DEFINE_string("key", "venie", "key to encrypt video")
flags.DEFINE_integer("gpu",0,"gpu to use")
flags.DEFINE_boolean("ocr", False, "do ocr")

TOTAL_GPU_NUM = int(cv2.cuda.getCudaEnabledDeviceCount())

def pre_process(io, frame_q, plate_q, face_q):
    try:
        while 1:
            img  = io.read()
            if img == -1:
                break
            frame_q.put(img, timeout=2)
            plate_q.put(img, timeout=1)
            face_q.put(img, timeout=1)
    except Empty:
        pass
    except:
        print(traceback.format_exc())
        return

def detection(detector, img_q, result_q):
    try:
        while 1:
            img = img_q.get(timeout=2)

            detections = detector.detect(img)
            result_q.put(detections, timeout=1)
    except Empty:
        pass
    except:
        print(traceback.format_exc())
        return

def post_process(obj_dict, frame_q, plate_result_q, face_result_q):
    frame_timer = fps.FPS()

    io = obj_dict['Fileio']
    imwork = obj_dict['Imgprocess']
    enc = obj_dict['Encryption']
    anno = obj_dict['Annotation']
    reader = obj_dict['OCR']

    plates_dict = []
    frame_cnt = 0
    frame_timer.start()
    while 1:
        frame = frame_q.get(timeout=2)
        p_classes, p_scores, p_bboxes = plate_result_q.get(timeout=1)
        f_classes, f_scores, f_bboxes = face_result_q.get(timeout=1)

        classes = p_classes + f_classes
        scores = p_scores + f_scores
        bboxes = p_bboxes + f_bboxes

        results = [classes,scores,bboxes]
        enc.do(frame, results, frame_cnt)
        anno.write(results, frame_cnt)
        if reader:
            for label in anno.anno_dict['Annotation'][frame_cnt]['labels']:
                if label['class'] == 'plate':
                    bid = label['id']
                    x,y,w,h = label['boxcorners']
                    plates_dict[f"{frame_cnt}_{bid}"] = frame[y:y+h,x:x+w].copy()

        result_frame = imwork.do(frame,results)        
        io.write(result_frame)

        frame_cnt += 1
        frame_timer.update()
    frame_timer.stop()
    print(f"total frame : {frame_cnt}\nfps : {frame_timer.fps():.2f}({frame_timer.elapsed():.2f}s)")
    
    if reader and plates_dict:
        plates = list(plates_dict.values())
        plate_info = list(plates_dict.keys())

        result = {}
        for i in range((len(plates)-1)//1920+1):
            r = reader.read(plates[i*1920:(i+1)*1920],plate_info[i*1920:(i+1)*1920])
            result.update(r)

        for info,plate_num in result.items():
            frame_cnt, box_id = info.split("_")
            anno.anno_dict['Annotation'][int(frame_cnt)]['labels'][int(box_id)]['plate_num'] = plate_num

    
    
def main(_argv):
    print(datetime.datetime().now())
    start_time = time.time()
    # check flags
    if os.path.isfile(FLAGS.input_path):
        input_path = FLAGS.input_path
    else:
        print(f"[error] input path not exist ({FLAGS.input_path})")
        return 

    output_dir = FLAGS.output
    os.makedirs(output_dir, exist_ok=True)
    
    if TOTAL_GPU_NUM>1:
        main_gpu = FLAGS.gpu
        sub_gpu = (FLAGS.gpu + 1) % TOTAL_GPU_NUM
    elif TOTAL_GPU_NUM==1:
        main_gpu = FLAGS.gpu
        sub_gpu = FLAGS.gpu
    else:
        main_gpu = -1
        sub_gpu = -1

    # load detectors
    try: 
        plate_weights = "../data/weights/yolov4_2_best.weights"
        plate_cfg = "../data/cfg/yolov4_2.cfg"
        plate_names = "../data/cfg/0831_2.names"

        plate_yolo = Yolo(plate_cfg, plate_weights, plate_names, gpu=main_gpu)

        face_weights = "../data/weights/yolov4_face.weights"
        face_cfg = "../data/cfg/yolov4_1.cfg"
        face_names = "../data/cfg/face.names"

        face_yolo = Yolo(face_cfg, face_weights, face_names, gpu=sub_gpu)

        print(f"load yolo models done")
    except:
        print(traceback.format_exc())
        return

    if FLAGS.ocr:
        # load ocr model
        try:
            from modules.reader import Reader
            ocr_weights = "../data/weights/ocr_0908.pth"
            ocr_cfg = "../data/cfg/0908.txt"

            reader = Reader(ocr_weights, ocr_cfg, gpu=main_gpu)
        except:
            print(traceback.format_exc())
            return
    else:
        reader = None
    
    try:
        with open(input_path,"r") as f:
            file_list = f.read().strip().split("\n")
        print(f"start {len(file_list)} files")
    except:
        print(traceback.format_exc())
        return 
    
    for file_index, file_path in enumerate(file_list):
        try:
            file_name = os.path.split(file_path)[-1]
            real_name, ext = os.path.splitext(file_name)
            ext = ext.lower()
            if ext in img_ext_list:
                output_path = os.path.join(output_dir, real_name+"_mosaic.jpg")
            elif ext in vid_ext_list:
                output_path = os.path.join(output_dir, real_name+"_mosaic.mp4")
            else:
                print(f"unspported extension ({file_name})")
                continue
            # setting objects
            obj_dict = {}

            obj_dict['Fileio'] = RW(file_path, output_path)
            obj_dict['Imgprocess'] = ImgWorker(draw_boxes=True, put_texts=True)
            obj_dict['Encryption'] = EncWorker(FLAGS.key)
            obj_dict['Annotation'] = Annotator(file_path)
            obj_dict['OCR'] = reader

            # create queue
            frame_queue = Queue(maxsize=1)
            plate_image_queue = Queue(maxsize=1)
            plate_dets_queue = Queue(maxsize=1)
            face_image_queue = Queue(maxsize=1)
            face_dets_queue = Queue(maxsize=1)

            # thread start
            pre_thread = threading.Thread(target=pre_process, args=(obj_dict['Fileio'],frame_queue,plate_image_queue,face_image_queue),name='Preprocess')
            plate_thread = threading.Thread(target=detection, args=(plate_yolo,plate_image_queue,plate_dets_queue),name='Plate detect')
            face_thread = threading.Thread(target=detection, args=(face_yolo,face_image_queue,face_dets_queue),name='Face detect')
            post_thread = threading.Thread(target=post_process, args=(obj_dict,frame_queue,plate_dets_queue,face_dets_queue),name='Postprocess')

            pre_thread.start()
            plate_thread.start()
            face_thread.start()
            post_thread.start()

            # join
            main_thread = threading.currentThread()
            for thread in threading.enumerate():
                if thread in main_thread:
                    continue
                thread.join()

            obj_dict['Annotation'].save(output_path[:-4]+"_anno.json")
            obj_dict['Encryption'].save(output_path[:-4]+"_enc.json")
            obj_dict['Fileio'].close()


        except:
            print(traceback.format_exc())

    print(f"total takes {time.time()-start_time}s")


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
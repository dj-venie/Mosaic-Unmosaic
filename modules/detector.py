import cv2
import time
import numpy as np

class Detector:
    def __init__(self, cfg, weights, names, gpu=-1):
        start_time = time.time()
        self.gpu = gpu
        if gpu != -1:
            cv2.cuda.setDevice(self.gpu)
            self.network, self.class_names = self.load_opencv(cfg, weights, names)

        self.detect(np.zeros((self.height,self.width,3)).astype('uint8'))
        print(f"load detector network {time.time()-start_time:.2f}s")

    def load_opencv(self, cfg, weights, names):
        net = cv2.dnn_DetectionModel(cfg, weights)
        with open(cfg, "r") as f:
            for i in f.read().split("\n"):
                if i.startswith("width"):
                    self.width = int(i.split("=")[-1].strip())
                elif i.startswith("height"):
                    self.height = int(i.split("=")[-1].strip())
                    break
                else:
                    pass
        net.setInputScale(1/255)
        net.setInputSize((self.width, self.height))
        net.setInputSwapRB(True)
        if self.gpu!=-1:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            """
            cuda arch 7.0 이상 fp16 조건 추가
            """
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        with open(names,"r") as f:
            class_names = f.read().split("\n")[:-1]
        return net, class_names

    def detect(self, frame, th=0.7, nms=0.45):
        cap_height, cap_width, _ = frame.shape

        if self.gpu!=-1:
            cv2.cuda.setDevice(self.gpu)
        classes, scores, bboxes = self.network.detect(frame, confThreshold=th, nmsThreshold=nms)
        if len(classes):
            classes = [self.classes_names[i[0]] for i in classes]
            scores = list(scores.reshape(-1))
            bboxes = bboxes.tolist()
            
        return list(map(list,classes,scores,bboxes))
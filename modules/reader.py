import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

import torchvision.transforms as transforms

from .clova.utils import CTCLabelConverter, AttnLabelConverter
from .clova.model import Model

class Reader:
    def __init__(self, weights, option="", gpu=-1):
        if torch.cuda.is_available() and gpu!=-1:
            self.device = torch.device(f'cuda:{gpu}')
            cudnn.benchmark = True #cudnn 자동 튜너를 활성화 하여 하드웨어에 맞게 연산, input 크기 변환이 많은경우 성능저하
            cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.opt = Option(option)
        self.gpu = gpu

        self.opt.saved_model = weights
        self.opt.num_gpu = torch.cuda.device_count()
        if self.opt.Prediction == 'Attn':
            self.converter = AttnLabelConverter(self.opt.character)
        elif self.opt.Prediction == 'CTC':
            self.converter = CTCLabelConverter(self.opt.character)
        self.opt.num_class = len(self.converter.character)

        self.model = Model(self.opt)
        weights_dict = torch.load(self.opt.saved_model, map_location=self.device)
        key_list = list(weights_dict.keys())
        if key_list[0].startswith('module'):
            for key in key_list:
                data = weights_dict.pop(key)
                weights_dict[key[7:]] = data

        self.model.load_state_dict(weights_dict)
        self.model = self.model.to(self.device)

        self.model.eval()

    def read(self,img_list, file_name_list=None,th=0.25):
        if self.gpu!=-1:
            torch.cuda.set_device(self.gpu)

        if file_name_list is None:
            file_name_list = list(range(len(img_list)))
        image_tensors = self.img2tensor(img_list)
        result_dict = {}
        with torch.no_grad():
            batch_size = image_tensors.size(0)
            image = image_tensors.to(self.device)

            length_for_pred = torch.IntTensor([self.opt.batrch_max_lenght]*batch_size).to(self.device)
            text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).tod(self.device)
            preds = self.model(image, text_for_pred, is_train=False)

            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for file_name, pred, pred_max_prob in zip(file_name_list, preds_str, preds_max_prob):
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]
                pred_max_prob = pred_max_prob[:pred_EOS]

                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                if confidence_score > th:
                    result_dict[file_name] = pred

            return result_dict



    def img2tensor(self, img_list, padding=True):
        img_list2 = []
        for img in img_list:
            if padding:
                h,w,c = img.shape
                pad_num = min(h,w)
                background = np.zeros((h+pad_num, w+pad_num,c)).astype("uint8")
                background[pad_num//2:pad_num//2+h,pad_num//2:pad_num//2+w] = img
                img = background
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.reszie(gray,(self.imgW,self.imgH),cv2.INTER_CUBIC)
            timg = transforms.ToTensor()(gray)
            timg = timg.sub_(0.5).div_(0.5)
            img_list2.append(timg)
        
        image_tensors = torch.cat([img.unsqeeze(0) for img in img_list2],0)
        return image_tensors


        


class Option:
    def __init__(self,option=""):
        # default
        self.workers = 4
        self.batch_size = 192
        self.saved_model = ""
        self.batch_max_length = 25
        self.imgH = 32
        self.imgW = 100
        self.rgb = True
        self.character = '0123456789가나다라'
        self.sensitive = True
        self.PAD = True
        self.Transformation = 'TPS'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256

        self.read_option(option)
    
    def read_option(self, option):
        with open(option,"r") as f:
            option_file = f.read().strip().split("\n")

        opt_dict = {}
        flag = 1
        for line in option_file:
            if flag:
                if line.startswith("total_data"):
                    flag = 0
            else:
                if line[0].isalpha():
                    o, v= line.split(":")
                    opt_dict[o.strip()] = v.strip()

        self.batch_max_length = int(opt_dict.get('batch_max_length',self.batch_max_length))
        self.imgH = int(opt_dict.get('imgH',self.imgH))
        self.imgW = int(opt_dict.get('imgW',self.imgW))
        self.character = opt_dict.get('character',self.character)
        self.Transformation = opt_dict.get('Transformation',self.Transformation)
        self.FeatureExtraction = opt_dict.get('FeatureExtraction',self.FeatureExtraction)
        self.Prediction = opt_dict.get('Prediction',self.Prediction)
        self.num_fiducial = opt_dict.get('num_fiducial',self.num_fiducial)
        self.input_channel = opt_dict.get('input_channel',self.input_channel)
        self.output_channel = opt_dict.get('output_channel',self.output_channel)
        self.hidden_size = opt_dict.get('output_channel',self.output_channel)





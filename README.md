# Mosaic-Unmosaic

## 0. Summary
- Program for mosaic and unmosaic
- Result contains Korean license Plate recognition and object bounding-box location
- Yolov4 model used for object detection (face, korea license plate)
- Clova TRBA model used for plate ocr
- Pipeline
  - Step1 : Download from hadoop(hdfs) server, get metadata from db(cubrid)
  - Step2 : Do inference on downloaded data and mosaic about personal data 
  - Step3 : Unmosaic target data (actually except target data)
  - Step4 : Update meta tables with above results
  - Step5 : Upload to hadoop(hdfs) server, update and insert data to db(cubrid)
-----------
## 1. Fast start
- python mosaic.py --input input_paths.txt --output output_dir --key any_key_you_want
- python unmosaic.py --input vid_anno_enc_target.txt --output output_dir --key same_key_you_above
>Actually some codes are requires servers. Just use it as reference
-----------
## 2. Details
### 2.1 mosaic.py
- It contains object detection, ocr, encryption and img process.
- In order to use it, we need to create a text file with the path.
-  mosaic_input.txt :
   ```
   path/to/input_video1.mp4
   path/to/input_video2.mp4
   path/to/input_image.jpg
   ...
   ```
- 암호화 키를 설정해주어야 하고 기본값은 'venie' 입니다.
- 객체탐지의 경우 OpenCV의 dnn모듈을 사용하고 ocr은 pytorch를 사용합니다.
- 영상의 경우 FullHD 기준 40fps(v100 2장)의 속도로 진행됩니다.

### 2.2 unmosaic.py
- mosaic.py 에서 나온 결과물과 복호화 키를 기반으로 복호화 하여 영상을 저장합니다.
- 해당 프로세스에서 진행되는 파일들은 같은 키로 암호화 한 파일이어야 하고 암호가 틀릴경우 복호화 되지않습니다.
- unmosiac_input.txt :
    ```
    path/to/video1_mosaic.mp4 path/video1_anno.json path/video1_enc.json target
    path/to/image2_mosaic.jpg path/image2_anno.json path/image2_enc.json -1
    path/to/image1_mosaic.jpg path/image1_anno.json path/image1_enc.json 34나1234
    ...
    * target과 유사한 번호판에 대해서 재식별화, -1의경우 모든 번호판 재식별화
    ```
- 복호화 키를 설정해주어야 하고 기본값은 'venie' 입니다.

### 2.3 Codes for project
#### 2.3.1 main.py
- 
#### 2.3.2 download.py
- 
#### 2.3.3 save.py
- 
------------
## 3. Requirements

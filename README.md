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
- 각 시간대 별로 저장되는 디렉토리 ex)2021122500 들에 대해 자동 스케쥴링 해주는 프로그램입니다.
- 큐를 활용하여 테스크를 전달하고 네개의 프로세스를 병렬로 실행하여 전체 프로세스 속도를 높였습니다.
- 정해진 포멧의 디렉토리에 대해서 작동하며 7일 간격으로 저장된 데이터를 삭제합니다. 
#### 2.3.2 download.py
- 각 시간대 별로 저장되는 디렉토리들을 다운로드합니다.
- FROM에 해당하는 하둡 서버의 디렉토리가 설정되어야 하고 db(cubrid 서버)에 형식에 맞는 테이블도 존재해야합니다.
- modules/connector의 하둡 서버와 db 서버에 맞는 정보를 적습니다. 
#### 2.3.3 save.py
- 비정형 데이터 에서 분석결과를 기반으로 찾고자하는 차량번호가 존재하는지 확인하고 해당 프레임들(3개) 혹은 이미지를 저장합니다.
- 일치여부, 유사여부, 번호판 검출여부(일치여부x)를 순서로 프레임을 추출하고 세가지 이미지 모두 번호판이 검출되지 않을수도 있습니다.
------------
## 3. Requirements
### 3.1 Test Environment
- Linux (Ubuntu, RedHat)
- Cuda(10.1), Cudnn(7.6.5)
- OpenCV > 4.0
### 3.2 Python Packages
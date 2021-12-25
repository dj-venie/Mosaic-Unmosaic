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
- 
### 2.2 unmosaic.py
- 

### 2.3 Codes for project
#### 2.3.1 main.py
- 
#### 2.3.2 download.py
- 
#### 2.3.3 save.py
- 
------------
## 3. Requirements

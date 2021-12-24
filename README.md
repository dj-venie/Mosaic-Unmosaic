# Mosaic-Unmosaic

## 0. Summary
- Program for mosaic and unmosaic
- Result contains Korean license Plate recognition and object bounding-box location
- Yolov4 model used for object detection (face, korea license plate)
- Clova TRBA model used for plate ocr
- pipeline
  - step1 : download from hadoop(hdfs) server, get metadata from db(cubrid)
  - step2 : Inference for data and do process for personal data 
  - step3 : Unmosaic target data (actually except target data)
  - step4 : make meta tables with above results
  - step5 : upload to hadoop(hdfs) server, update and insert data to db(cubrid)

## 1. Fast start
- python mosaic.py --input input_paths.txt --output output_dir --key any_key_you_want
- python unmosaic.py --input vid_anno_enc_target.txt --output output_dir --key same_key_you_above
>Actually other codes are need to setting for servers. Just reference them

## 2. Details
### 2.1 mosaic.py
- It contains object detection, ocr, encrypt and img process
- If you use

'''
gt. txt 파일 구성 
- 메모장 text 형식
- 경로/파일명.png   라벨값(손글씨)
'''
import json
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

data_root_path = 'D:\study\deep-text-recognition-benchmark-master/test_image/'
save_root_path = 'D:/study/deep-text-recognition-benchmark-master/data/'

train_annotations = json.load(open('D:/study/_OCR project/crnn/train_annotation.json'))
gt_file = open(save_root_path + 'gt_train.txt', 'w')
for file_name in tqdm(train_annotations):
    annotations = train_annotations[file_name]
    image = cv2.imread(data_root_path+file_name)    #cv2.imread : 이미지 파일이 넘파이 배열 값들로 넘어오고 이 숫자가 해당 위치에서의 색을 의미
    text = annotations[0]['text']
    #   (key : images) file_name.png 
    #   (value : annotations) [ id, image_id, text, attributes ]
    #  {"00000022.png": [{"id": "00000022", "image_id": "00000022", "text": "\uac00\ub839", "attributes": {"type": "\ub2e8\uc5b4(\uc5b4\uc808)", "gender": "\uc5ec", "age": "29", "job": "\ud68c\uc0ac\uc6d0"}}],
    cv2.imwrite(save_root_path + 'train/' + file_name, image)
    gt_file.write('train/{}\t{}\n'.format(file_name, text))

validation_annotations = json.load(open('D:\study\_OCR project\crnn\\validation_annotation.json'))
gt_file = open(save_root_path + 'gt_validation.txt', 'w')
for file_name in tqdm(validation_annotations):
    annotations = validation_annotations[file_name]
    image = cv2.imread(data_root_path+file_name)  
    text = annotations[0]['text']
    cv2.imwrite(save_root_path + 'validation/' + file_name, image)
    gt_file.write('validation/{}\t{}\n'.format(file_name, text))    

test_annotations = json.load(open('D:/study/_OCR project/crnn/test_annotation.json'))
gt_file = open(save_root_path + 'gt_test.txt', 'w')
for file_name in tqdm(test_annotations):
    annotations = test_annotations[file_name]
    image = cv2.imread(data_root_path+file_name)  
    text = annotations[0]['text']
    cv2.imwrite(save_root_path + 'test/' + file_name, image)
    gt_file.write('test/{}\t{}\n'.format(file_name, text))        
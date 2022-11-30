'''
gt. txt 파일 구성 
- 메모장 text 형식
- 경로/파일명.png   라벨값
'''
import json
import random
import os  
import tqdm as tqdm    

file = json.load(open('D:\study\deep-text-recognition-benchmark-master\\test_image\dataset_info.json','rt', encoding='UTF8'))

# print(file.keys())      # dict_keys(['info', 'images', 'annotations', 'licenses'])
# print(file['info'])      # {'name': '한글 손글씨 데이터', 'date_created': '2019-08-24 20:06:53'}
# print(type(file['images'])) #<class 'list'>
#  print(file['images'][:3]) 

'''
[{'id': '00000001', 'width': 332, 'height': 227, 'file_name': '00000001.png'}, 
{'id': '00000002', 'width': 333, 'height': 227, 'file_name': '00000002.png'}, 
{'id': '00000003', 'width': 333, 'height': 227, 'file_name': '00000003.png'}]
'''

# "단어(어절)" 클래스 데이터 추출
annotation = [a for a in file['annotations'] if a ['attributes']['type']=='단어(어절)']
# print(annotation[:1])

ocr_files = os.listdir('./deep-text-recognition-benchmark-master/test_image/')
# print(len(ocr_files)) # 31

train = int(len(ocr_files) * 0.7)
validation = int(len(ocr_files) * 0.15)
test = int(len(ocr_files) * 0.15)

print(train, validation, test)  # 21 4 4

train_files = ocr_files[ : train ]
validation_files = ocr_files[ train : train + validation ]
test_files = ocr_files[ -test: ]

#  train/validation/test  이미지들에 해당하는  id 값을 저장
# separate image id - train, validation, test 
train_img_ids = {}
validation_img_ids = {}
test_img_ids = {}

for image in file['images']:
    if image['file_name'] in train_files:
        train_img_ids[image['file_name']] =image['id']
    elif image['file_name'] in validation_files:
        validation_img_ids[image['file_name']] = image['id'] 
    elif image['file_name'] in test_files:
        test_img_ids[image['file_name']] = image['id']   

# train/validation/test 이미지들에 해당하는 annotaion들 저장
train_annotations = { f : [ ] for f in train_img_ids.keys()}    # {image id} : [ ]
validation_annotations = { f : [ ] for f in validation_img_ids.keys()}
test_annotations = { f : [ ] for f in test_img_ids.keys()}

train_ids_img = {train_img_ids[id_]: id_ for id_ in train_img_ids}
validation_ids_img = {validation_img_ids[id_]: id_ for id_ in validation_img_ids}
test_ids_img = {test_img_ids[id_]: id_ for id_ in test_img_ids}

# tqdm 작업진행율 
# for idx, annotaion in enumerat(x) => enumerate 사용 시 하나하나 세어가며 열거, idx(index) 함께 출력 가능
for idx, annotation in tqdm.tqdm(enumerate(file['annotations'])) : 
    if idx % 5000 == 0 :
        print(idx, '/', len(file['annotations']), 'processed')
    if annotation['attributes']['type']  != '단어(어절)' :
        continue
    if annotation['image_id'] in train_ids_img:
        train_annotations[train_ids_img[annotation['image_id']]].append(annotation)
    elif annotation['image_id'] in validation_ids_img:
        validation_annotations[validation_ids_img[annotation['image_id']]].append(annotation)
    elif annotation['image_id'] in test_ids_img:
        test_annotations[test_ids_img[annotation['image_id']]].append(annotation)    
#   {"00000022.png": [{"id": "00000022", "image_id": "00000022", "text": "\uac00\ub839", "attributes": {"type": "\ub2e8\uc5b4(\uc5b4\uc808)", "gender": "\uc5ec", "age": "29", "job": "\ud68c\uc0ac\uc6d0"}}],
#   (key : images) file_name.png : (value : annotations) id, image_id, text, attributes

with open('train_annotation.json', 'w') as file:
    json.dump(train_annotations, file)
with open('validation_annotation.json', 'w') as file:
    json.dump(validation_annotations, file) 
with open('test_annotation.json', 'w') as file  :
    json.dump(test_annotations, file)          


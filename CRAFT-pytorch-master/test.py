"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

# cpu로 돌아갈 수 있게 작업
from collections import OrderedDict #입력된 아이템들(items)의 순서를 기억하는 Dictionary 클래스
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

# cuda인자에 아래 5개 중 하나 기재 시 True 적용
def str2bool(v): 
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')   # 이미지 확대율
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args() # argument command 창에 기재하지 않으면 defalut 값으로 실행

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder) #img_files, mask_files, gt_files


result_folder = './result/'
if not os.path.isdir(result_folder): #result 폴더 없으면 만들어줌
    os.mkdir(result_folder) #make directory

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize (img_resized, target_ratio = 확대된 이미지 사이즈, 이미지 비율)
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio #세로, 가로 동일한 비율로 resize

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized) # 이미지 평균과 표준편차 이용하여 정규화
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w] : 채널 변환(파이토치 형태)
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w] : 차원 증가
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():   # gradient를 계산하지 않음, test 단계이기 때문에 gradient 계산할 필요 없음
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy() # 1장:모든 행:모든 열:region score 
    score_link = y[0,:,:,1].cpu().data.numpy() # 1장:모든 행:모든 열:affinity score

    # refine link (특수한 데이터에서 쓰는 모델 : 이미지-좌표값으로만 이루어진 데이터 훈련시킴)
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment 좌표 조정 
    # img_resized를 기준으로 word box를 생성 후, 원본사이즈 기준으로 워드박스 조정
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h) 
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k] #poly가 None(false)이면 boxes를 넣어준다.

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link)) # np.hstack : 배열 옆으로 붙이기 (가로 결합)
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


# 터미널창에서 실행되는 부분
if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net) # 데이터 병렬처리
        cudnn.benchmark = False

    net.eval() #craft 모델 평가모드로 전환
    '''
    훈련단계에서 적용된 normalization, reguralization, dropout 등의 하이퍼파라미터
    테스트 단계에서는 자동 미적용
    >> model.eval 로 평가모드 전환 
    '''
    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        # return boxes, polys, ret_score_text

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path)) #os.path.splitext(filename) --> 확장자만 따로 분류
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text) # mask_file 경로에 score_text 저장 (ret_score_text)
                  
        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder) 
        # image[:, :, ::-1] : image를 뒤집는다. opencv bgr로 읽기 때문에 rgb로 바꾸기 위해 뒤집음
        # 박스 생성된 이미지, 좌표값 적힌 text 파일 생성
        # res_file = dirname + "res_" + filename + '.txt' #좌표
        # res_img_file = dirname + "res_" + filename + '.jpg' #박스 생성된 이미지 파일
        
    print("elapsed time : {}s".format(time.time() - t))


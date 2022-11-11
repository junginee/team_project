"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
'''
Word-Level QuadBox Inference
Region Score와 Affinity Score를 통해 각 문자를 하나의 단어로 그룹화하는 방법
1. 원본 이미지와 동일한 크기의 이진 맵 M을 모든 값이 0이 되도록 초기화
2. 픽셀 p에 대해 Region Score(p) 혹은 Affinity Score(p)가 각각 역치값 Tr(p), Ta(p)보다 클 경우 해당 픽셀을 1로 설정
3. 이진 맵 M에 대해 Connected component Labeling을 수행
4. 각 Label을 둘러싸는 최소 영역의 회전된 직사각형 찾기(OpenCV에서 제공하는 connectedComponents()와 minAreaRect()를 활용)

Polygon Inference

'''
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math

""" auxilary functions (보조 기능) """
# unwarp corodinates (비뚤어진 좌표)
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])
""" end of auxilary functions """


def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    # prepare data (데이터 준비)
    linkmap = linkmap.copy()    # affinity score 
    textmap = textmap.copy()    # region score
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)  # 0.4 이하면 1, 이상이면 0 parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0) # 0.7 이하면 1, 이상이면 0 parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')    
    
    # cv2.threshold(img, threshold_value, value, flag) : 이미지 이진화
    # img: 변환할 이미지
    # threshold: 스레시홀딩 임계값
    
    text_score_comb = np.clip(text_score + link_score, 0, 1)    # np.clip(array, 하한값, 상한값)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    # connectedComponentsWithStats : 이진화된 이미지에서 연결되어 있는 픽셀들 grouping
    # connectedComponentsWithStats는 8비트 이하의 정보만 사용할 수 있어서 unit8로 변환 
    # connectivity 4 or 8의 값을 줌, 4 = 동, 서, 남, 북
    
    det = []
    mapper = []
    for k in range(1,nLabels):
        # size filtering (크기 필터링)
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding 
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area (affinity 영역 삭제)
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        # getStructuringElement : 이미지에 가해지는 변형을 결정하는 구조화 요소 (커널과 같은 역할)
        # cv2.MORPH_RECT : 직사각형 (구조화 요소 커널 모양)
        # (1 + niter, 1 + niter) : 구조화 요소 커널의 크기
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        #cv2.dilate 굵게 만듬

        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2) #넘파이 배열을 굴리는 (roll) 함수
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        
        '''
        cv2.minAreaRect
        입력받은 2차원 포인터 집합을 바탕으로 입력받은 포인터들을 모두 포함하면서 최소한의 영역을 차지하는 직사각형을 찾음
        직사각형은 boundingRect 함수처럼 곧게 서있을 수도, 회전할 수도 있음
        좌상단의 x좌표, y좌표, 가로와 세로의 폭, 기울어진 각도 순으로 반환
        
        cv2.boxPoints
        회전된 직사각형의 4개의 꼭짓점을 찾는 함수
        float형으로 2차원 포인터들의 집합으로 반환
        '''
        
        # align diamond-shape (마름모꼴로 맞춤)
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]) # np.linalg.norm : 넘파이 벡터 정규화 
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order (시계 방향 정렬)
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    # configs (구성)
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []  
    for k, box in enumerate(boxes):
        # size filter for small instance (작은 객체용 크기 필터)
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 10 or h < 10:
            polys.append(None); continue

        # warp image
        tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, tar)   
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        '''
        원근 변환 적용 핵심코드
        getPerspectiveTransform(원본 이미지 좌표, 변경할 이미지 좌표) : 원근변환을 원하는 4개의 전/후 좌표를 통해 변환행렬 계산
        cv2.warpPerspective : 원근 변환 적용
        '''
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None); continue

        # binarization for selected label (선택한 레이블에 대한 이진화)
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        """ Polygon generation """
        # find top/bottom contours (상단/하단 등고선 찾기) : 파란색 선
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:,i] != 0)[0]
            if len(region) < 2 : continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length

        # pass if max_len is similar to h (max_len이 h와 유사할 경우 통과)
        if h * max_len_ratio < max_len:
            polys.append(None); continue

        # get pivot points with fixed length (고정된 길이로 피벗 포인트 얻기) : 노란색 선
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     # segment width
        pp = [None] * num_cp    # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0,len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables (변수 재설정)
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1)/2)] = (x, cy)
                seg_height[int((seg_num - 1)/2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh is smaller than character height 
        # (피벗 수가 충분하지 않거나 세그먼트 너비가 문자 높이보다 작은 경우 통과)
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None); continue

        # calc median maximum of pivot points (계산 피벗 점의 최대 중위수)
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots (수평 피벗을 만드는 데 적용) : 파란색과 노란색을 구하고 그것에 코사인을 적용해서 빨간색을 찾음
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:     # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps (character 히트맵 덮을 가장자리 점 가져오기)
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None); continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)

    if poly:
        polys = getPoly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes) # dummy (poly가 false인 경우 에러 나지 않도록 dummy 형태 만들어줌)

    return boxes, polys

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys

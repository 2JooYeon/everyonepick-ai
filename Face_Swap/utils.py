import requests
from PIL import Image
import io
import cv2
import imutils
import numpy as np
from numpy.linalg import norm as l2norm
from face_data import get_bbox
from face_segmentation import get_seg_bbox


# 가장 많은 선택을 받은 사진의 인덱스를 찾는 함수
def find_base_photo_id(user_choices, photo_id_count):
    # 아무도 선택 하지 않은 경우
    if len(user_choices) == 0:
        return -1

    for choices in user_choices.values():
        for choice in choices:
            photo_id_count[choice] += 1

    # 사진 별 선택 수 (내림차순 정렬)
    sorted_photo_id_count = sorted(photo_id_count.items(), reverse=True, key=lambda item: item[1])
    # base 사진, 받은 선택 수
    base_photo_id = sorted_photo_id_count[0][0]

    return base_photo_id


# face swap이 필요한 user_id와 source 사진을 찾는 함수
def list_of_face_swap(user_choices, base_photo_id):
    user_id_choice = []

    for user_id, choices in user_choices.items():
        if base_photo_id not in choices:
            # 첫 번째 선택 사진을 source 사진으로 결정
            user_id_choice.append((user_id, choices[0]))

    return user_id_choice


# S3 url로부터 이미지를 다운받는 함수
def download_s3_url(url):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


# target_embedding과 그룹 사진 얼굴들과의 유사도 측정하는 함수
def compute_face_similarity(group_embeddings, target_embedding):
    # 임베딩 정규화
    normed_target_embedding = target_embedding / l2norm(target_embedding)
    normed_group_embeddings = []
    for embedding in group_embeddings:
        normed_group_embeddings.append(embedding / l2norm(embedding))

    normed_group_embeddings = np.array(normed_group_embeddings, dtype=np.float32)
    # 코사인 유사도 계산
    sims = np.dot(normed_target_embedding, normed_group_embeddings.T)

    return sims


def resize_img(source_face, target_face, img):
    source_x1, source_y1, source_x2, source_y2 = get_bbox(source_face)
    source_bbox_w = source_x2 - source_x1
    source_bbox_h = source_y2 - source_y1

    target_x1, target_y1, target_x2, target_y2 = get_bbox(target_face)
    target_bbox_w = target_x2 - target_x1
    target_bbox_h = target_y2 - target_y1

    height_scale = target_bbox_h / source_bbox_h
    img_resized = imutils.resize(img, height=int(img.shape[0] * height_scale))

    if (source_bbox_w * height_scale) < target_bbox_w:
        width_scale = target_bbox_w / (source_bbox_w * height_scale)
        img_resized = imutils.resize(img_resized, width=int(img_resized.shape[1] * width_scale))

    return img_resized


def adjust_swap_position(seg_label, source_face, target_face, target_img):
    seg_x1, seg_y1, seg_x2, seg_y2 = get_seg_bbox(seg_label)

    source_seg_h = seg_y2 - seg_y1
    source_seg_w = seg_x2 - seg_x1

    source_x1, source_y1, source_x2, source_y2 = get_bbox(source_face)
    source_cx, source_cy = int((source_x1 + source_x2) // 2), int((source_y1 + source_y2) // 2)

    target_x1, target_y1, target_x2, target_y2 = get_bbox(target_face)
    target_cx, target_cy = int((target_x1 + target_x2) // 2), int((target_y1 + target_y2) // 2)

    start_y = target_cy - (source_cy - seg_y1)
    start_x = target_cx - (source_cx - seg_x1)
    end_y = start_y + source_seg_h
    end_x = start_x + source_seg_w

    '''얼굴이 잘리는 경우 예외처리'''
    t_start_x = 0
    t_end_x = 0
    t_start_y = 0
    t_end_y = 0

    s_start_x = 0
    s_end_x = 0
    s_start_y = 0
    s_end_y = 0

    if start_y <= 0:
        t_start_y = 0
        s_start_y = seg_y1 - start_y
    if start_y > 0:
        t_start_y = start_y
        s_start_y = seg_y1
    if start_x <= 0:
        t_start_x = 0
        s_start_x = seg_x1 - start_x
    if start_x > 0:
        t_start_x = start_x
        s_start_x = seg_x1
    if end_y > target_img.shape[0]:
        t_end_y = target_img.shape[0]
        s_end_y = seg_y2 - (end_y - target_img.shape[0])
    if end_y <= target_img.shape[0]:
        t_end_y = end_y
        s_end_y = seg_y2
    if end_x > target_img.shape[1]:
        t_end_x = target_img.shape[1]
        s_end_x = seg_x2 - (end_x - target_img.shape[1])
    if end_x <= target_img.shape[1]:
        t_end_x = end_x
        s_end_x = seg_x2

    return [(s_start_x, s_start_y, s_end_x, s_end_y), (t_start_x, t_start_y, t_end_x, t_end_y)]
import imutils
import numpy as np
import cv2
from utils import adjust_swap_position, resize_img
from face_data import get_faceobj, find_user_face, get_angle
from face_segmentation import get_seg_label


def swap_face(user_face_embedding, source_img, target_img, target_faces):

    source_faces = get_faceobj(source_img)

    '''이미지에서 사용자 얼굴 찾기'''
    source_face = find_user_face(source_faces, user_face_embedding)
    target_face = find_user_face(target_faces, user_face_embedding)

    '''target 얼굴 각도에 맞춰 source 얼굴 회전시키기'''
    source_angle = get_angle(source_face)
    target_angle = get_angle(target_face)
    rotate_angle = source_angle - target_angle
    source_img_rotated = imutils.rotate_bound(source_img, rotate_angle)

    source_faces = get_faceobj(source_img_rotated)
    '''target 얼굴 크기에 맞춰 source 얼굴 크기 조정하기'''
    source_face = find_user_face(source_faces, user_face_embedding)
    source_img_resized = resize_img(source_face, target_face, source_img_rotated)

    source_faces = get_faceobj(source_img_resized)
    source_face = find_user_face(source_faces, user_face_embedding)

    '''segmentation 진행'''
    source_seg_label = get_seg_label(source_img_resized, source_face)
    target_seg_label = get_seg_label(target_img, target_face)

    source_position, target_position = adjust_swap_position(source_seg_label, source_face, target_face, target_img)
    s_start_x, s_start_y, s_end_x, s_end_y = source_position
    t_start_x, t_start_y, t_end_x, t_end_y = target_position

    source_face_image = cv2.bitwise_and(source_img_resized, source_img_resized, mask=source_seg_label)

    '''target 이미지 크기와 동일한 마스크 생성(3차원)'''
    source_face_mask = np.zeros(target_img.shape, np.uint8)
    source_face_mask[t_start_y:t_end_y, t_start_x:t_end_x] = source_face_image[s_start_y:s_end_y, s_start_x:s_end_x]

    '''target 이미지 크기와 동일한 마스크 생성(1차원)'''
    target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    source_seg_mask = np.zeros_like(target_img_gray)

    source_seg_mask[t_start_y:t_end_y, t_start_x:t_end_x] = source_seg_label[s_start_y:s_end_y,s_start_x:s_end_x]

    '''얼굴에 해당하는 픽셀만 255, 나머지 0'''
    source_face_seg = np.where((0 < source_seg_mask) & (source_seg_mask < 10), 255, 0)
    source_face_seg = source_face_seg.astype(np.uint8)

    source_face_seg_box = np.where(source_face_seg > 0)
    source_face_seg_y2 = max(source_face_seg_box[0])

    '''hair label에 해당하는 픽셀만 255, 나머지 0'''
    source_hair_seg = np.where(source_seg_mask == 10, 255, 0)

    '''face segmentation 보다 아래에 있는 머리는 지우도록'''
    source_hair_seg[source_face_seg_y2:, :] = 0
    source_hair_seg = source_hair_seg.astype(np.uint8)

    '''target (hair+얼굴) 픽셀 범위에 존재하는 source hair 픽셀만 남기도록'''
    hair_mask = cv2.bitwise_and(source_hair_seg, source_hair_seg, mask=target_seg_label)
    hair_face_mask = cv2.add(hair_mask, source_face_seg)

    source_fg = cv2.bitwise_and(source_face_mask, source_face_mask, mask=hair_face_mask)
    target_bg = cv2.bitwise_and(target_img, target_img, mask=cv2.bitwise_not(hair_face_mask))
    result = cv2.add(target_bg, source_fg)

    return result

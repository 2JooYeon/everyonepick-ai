import torch
import numpy as np
from insightface.app import FaceAnalysis
import json
from collections import defaultdict
import boto3
import io
import cv2
from numpy.linalg import norm as l2norm
import requests
from PIL import Image
import math
import imutils
import facer


with open('input.json') as f:
    json_object = json.load(f)

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

photo_id_url = {}
photo_id_count = {}
user_choices = defaultdict(list)
user_embedding = {}
s3 = boto3.client('s3')

'''전체 이미지 리스트에 대한 정보'''
for photo in json_object["pick"]["photos"]:
    photo_id_url[photo["id"]] = photo["photo_url"]
    photo_id_count[photo["id"]] = 0

'''사용자 별 사진 선택에 대한 정보'''
for pick in json_object["pick_info_photos"]:
    user_id = pick["user_id"]
    for photo_id in pick["photo_ids"]:
        user_choices[user_id].append(photo_id["id"])

    '''사용자 별 임베딩 정보'''
    f = io.BytesIO()
    s3.download_fileobj("everyonepick-ai-face-embedding-bucket", f"face{user_id}_embedding", f)
    f.seek(0)
    embedding = np.frombuffer(f.read(), dtype=np.float32)
    user_embedding[user_id] = embedding


# 가장 많은 선택을 받은 사진의 인덱스를 찾는 함수
def find_base_photo_id(user_choices):
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


# insightface 모델로 얼굴 정보 분석하는 함수
# bbox, kps, det_score, landmark_3d_68, pose, landmark_2d_106, gender, embedding 포함
# def face_analysis(img_path):
#     img = cv2.imread(img_path)
#     faces = app.get(img)
#     return faces


# 얼굴 임베딩 리스트 반환하는 함수
def get_embeddings(faces):
    embeddings = []
    for face in faces:
        embeddings.append(face['embedding'])

    return embeddings


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


def get_angle(face):
  landmark = face['landmark_3d_68']
  tan_theta = (landmark[34][0] - landmark[28][0]) / (landmark[34][1] - landmark[28][1])
  theta = np.arctan(tan_theta)
  rotate_angle = theta * 180 / math.pi
  return rotate_angle


def get_bbox(face):
  bbox = face['bbox']
  return (bbox[0], bbox[1], bbox[2], bbox[3])


def get_kps(face):
  kps = face['kps']
  kps = kps.tolist()
  return kps


''' face swap 함수 테스트 (사용자 선택 고려 O)'''
if __name__ == "__main__":
    target_img_id = find_base_photo_id(user_choices)
    target_img_url = photo_id_url[target_img_id]
    response = requests.get(target_img_url)
    target_img = Image.open(io.BytesIO(response.content))
    target_img = np.array(target_img)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

    target_faces = app.get(target_img)
    target_embeddings = get_embeddings(target_faces)

    face_swap_list = list_of_face_swap(user_choices, target_img_id)

    for user_id, choice in face_swap_list:
        face_embedding = user_embedding[user_id]
        # user_img_path = user_profiles[user]

        source_img_url = photo_id_url[choice]
        response = requests.get(source_img_url)
        source_img = Image.open(io.BytesIO(response.content))
        source_img = np.array(source_img)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)


        # user_face = face_analysis(user_img_path)
        source_faces = app.get(source_img)

        # user_embedding = user_face[0]['embedding']
        source_embeddings = get_embeddings(source_faces)

        target_user_face_index = compute_face_similarity(target_embeddings, face_embedding).argmax()
        source_user_face_index = compute_face_similarity(source_embeddings, face_embedding).argmax()

        source_face = source_faces[source_user_face_index]
        target_face = target_faces[target_user_face_index]

        '''여기부터 얼굴 segmentation 합성 코드 추가'''
        '''이미지에서 얼굴 회전 각도 찾기'''
        source_angle = get_angle(source_face)
        target_angle = get_angle(target_face)
        rotate_angle = source_angle - target_angle

        '''target 얼굴 각도에 맞춰 source 얼굴 회전시키기'''
        source_img_rotated = imutils.rotate_bound(source_img, rotate_angle)

        '''이미지에서 얼굴 바운딩 박스 크기 검출'''
        source_face_rotated = app.get(source_img_rotated)
        source_embeddings = get_embeddings(source_face_rotated)
        source_user_face_index = compute_face_similarity(source_embeddings, face_embedding).argmax()
        source_face = source_face_rotated[source_user_face_index]

        source_x1, source_y1, source_x2, source_y2 = get_bbox(source_face)
        source_cx, source_cy = int((source_x1 + source_x2) // 2), int((source_y1 + source_y2) // 2)
        source_bbox_w = source_x2 - source_x1
        source_bbox_h = source_y2 - source_y1

        target_x1, target_y1, target_x2, target_y2 = get_bbox(target_face)
        target_cx, target_cy = int((target_x1 + target_x2) // 2), int((target_y1 + target_y2) // 2)
        target_bbox_w = target_x2 - target_x1
        target_bbox_h = target_y2 - target_y1

        '''target 얼굴 크기에 맞춰 source 얼굴 크기 조정하기'''
        height_scale = target_bbox_h / source_bbox_h
        source_img_resized = imutils.resize(source_img_rotated, height=int(source_img_rotated.shape[0] * height_scale))

        if (source_bbox_w * height_scale) < target_bbox_w:
            width_scale = target_bbox_w / (source_bbox_w * height_scale)
            source_img_resized = imutils.resize(source_img_resized,
                                                width=int(source_img_resized.shape[1] * width_scale))

        '''source 얼굴 segementation 진행'''
        source_tensor = torch.from_numpy(source_img_resized)
        target_tensor = torch.from_numpy(target_img)

        source_tensor = source_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        target_tensor = target_tensor.unsqueeze(0).permute(0, 3, 1, 2)

        '''points:[]에 segmentation 진행하고자 하는 얼굴의 kps 담기'''
        '''image_ids:[]에 segmentation 진행하고자 하는 얼굴의 수 만큼 0 삽입'''
        source_face_resized = app.get(source_img_resized)
        source_embeddings = get_embeddings(source_face_resized)
        source_user_face_index = compute_face_similarity(source_embeddings, face_embedding).argmax()
        source_face = source_face_resized[source_user_face_index]

        source_img_points = torch.Tensor(np.array([get_kps(source_face)]))
        source_img_ids = torch.tensor(np.array([0]), dtype=torch.int64)
        source_faces = {'points': source_img_points, 'image_ids': source_img_ids}

        target_img_points = torch.Tensor(np.array([get_kps(target_face)]))
        target_img_ids = torch.tensor(np.array([0]), dtype=torch.int64)
        target_faces = {'points': target_img_points, 'image_ids': target_img_ids}

        face_parser = facer.face_parser('farl/lapa/448', device=device)
        with torch.inference_mode():
            source_faces = face_parser(source_tensor, source_faces)
            target_faces = face_parser(target_tensor, target_faces)

        '''label names order -> ['background', 'face', 'rb', 'lb', 're', 'le', 'nose', 'ulip', 'imouth', 'llip', 'hair']'''
        source_seg_logits = source_faces['seg']['logits'][0]
        source_seg_probs = source_seg_logits.softmax(dim=0)
        source_seg_labels = source_seg_probs.argmax(dim=0)
        source_seg_labels = source_seg_labels.to(torch.uint8)
        source_seg_labels = source_seg_labels.numpy()

        target_seg_logits = target_faces['seg']['logits'][0]
        target_seg_probs = target_seg_logits.softmax(dim=0)
        target_seg_labels = target_seg_probs.argmax(dim=0)
        target_seg_labels = target_seg_labels.to(torch.uint8)
        target_seg_labels = target_seg_labels.numpy()

        '''source segmentation 경계 추출'''
        source_seg_box = np.where(source_seg_labels > 0)

        '''x1, y1, x2, y2'''
        seg_x1, seg_y1, seg_x2, seg_y2 = [min(source_seg_box[1]), min(source_seg_box[0]), max(source_seg_box[1]),
                                          max(source_seg_box[0])]
        source_seg_h = seg_y2 - seg_y1
        source_seg_w = seg_x2 - seg_x1

        source_x1, source_y1, source_x2, source_y2 = get_bbox(source_face)
        source_cx, source_cy = int((source_x1 + source_x2) // 2), int((source_y1 + source_y2) // 2)

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

        seg_start_x = 0
        seg_start_y = 0
        seg_end_x = 0
        seg_end_y = 0

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

        source_face_image = cv2.bitwise_and(source_img_resized, source_img_resized, mask=source_seg_labels)

        '''target 이미지 크기와 동일한 마스크 생성(3차원)'''
        source_face_mask = np.zeros(target_img.shape, np.uint8)
        source_face_mask[t_start_y:t_end_y + 1, t_start_x:t_end_x + 1] = source_face_image[s_start_y:s_end_y + 1,
                                                                         s_start_x:s_end_x + 1]

        '''target 이미지 크기와 동일한 마스크 생성(1차원)'''
        target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        source_seg_mask = np.zeros_like(target_img_gray)

        source_seg_mask[t_start_y:t_end_y + 1, t_start_x:t_end_x + 1] = source_seg_labels[s_start_y:s_end_y + 1,
                                                                        s_start_x:s_end_x + 1]

        '''얼굴에 해당하는 픽셀만 255, 나머지 0'''
        source_face_seg = np.where((0 < source_seg_mask) & (source_seg_mask < 10), 255, 0)
        source_face_seg = source_face_seg.astype(np.uint8)

        source_face_seg_box = np.where(source_face_seg > 0)
        face_seg_y2 = max(source_face_seg_box[0])

        '''hair label에 해당하는 픽셀만 255, 나머지 0'''
        source_hair_seg = np.where(source_seg_mask == 10, 255, 0)
        '''face segmentation 보다 아래에 있는 머리는 지우도록'''
        source_hair_seg[face_seg_y2:, :] = 0
        source_hair_seg = source_hair_seg.astype(np.uint8)

        '''target (hair+얼굴) 픽셀 범위에 존재하는 source hair 픽셀만 남기도록'''
        hair_mask = cv2.bitwise_and(source_hair_seg, source_hair_seg, mask=target_seg_labels)
        hair_face_mask = cv2.add(hair_mask, source_face_seg)

        source_fg = cv2.bitwise_and(source_face_mask, source_face_mask, mask=hair_face_mask)
        target_bg = cv2.bitwise_and(target_img, target_img, mask=cv2.bitwise_not(hair_face_mask))
        result = cv2.add(target_bg, source_fg)
        cv2.imshow("result", result)
        cv2.waitKey()
        cv2.destroyAllWindows()
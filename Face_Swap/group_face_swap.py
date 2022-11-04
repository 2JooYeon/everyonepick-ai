import torch
import numpy as np
from insightface.app import FaceAnalysis
import json
from collections import defaultdict
import boto3
import io

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
        print(user_id, choices)
        if base_photo_id not in choices:
            # 첫 번째 선택 사진을 source 사진으로 결정
            user_id_choice.append((user_id, choices[0]))

    return user_id_choice

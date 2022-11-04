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
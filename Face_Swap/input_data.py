from collections import defaultdict
import boto3
import io
import numpy as np


def refine_input(record):
    pick_id = record["pick"]["id"]
    photo_id_url = {}
    photo_id_count = {}
    user_choices = defaultdict(list)
    user_embedding = {}
    s3 = boto3.client('s3')

    '''전체 이미지 리스트에 대한 정보'''
    for photo in record["pick"]["photos"]:
        photo_id_url[photo["id"]] = photo["photoUrl"]
        photo_id_count[photo["id"]] = 0

    '''사용자 별 사진 선택에 대한 정보'''
    for pick in record["pickInfoPhotos"]:
        user_id = pick["userId"]
        user_choices[user_id] = pick["photoIds"]

        '''사용자 별 임베딩 정보'''
        f = io.BytesIO()
        s3.download_fileobj("everyonepick-ai-face-embedding-bucket", f"face{user_id}_embedding", f)
        f.seek(0)
        embedding = np.frombuffer(f.read(), dtype=np.float32)
        user_embedding[user_id] = embedding

    return pick_id, photo_id_url, photo_id_count, user_choices, user_embedding

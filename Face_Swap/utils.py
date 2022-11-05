import requests
from PIL import Image
import io
import numpy as np
import cv2

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
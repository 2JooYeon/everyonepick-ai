from insightface.app import FaceAnalysis
import numpy as np
import math

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))


# Face 객체를 반환하는 함수
# bbox, kps, det_score, landmark_3d_68, pose, landmark_2d_106, gender, embedding 포함
def get_faceobj(img):
    return app.get(img)


# 얼굴 임베딩 리스트 반환하는 함수
def get_embeddings(faces):
    embeddings = []
    for face in faces:
        embeddings.append(face['embedding'])

    return embeddings


def get_bbox(face):
  bbox = face['bbox']

  return (bbox[0], bbox[1], bbox[2], bbox[3])


def get_kps(face):
  kps = face['kps']
  kps = kps.tolist()

  return kps


def find_user_face(faces, embedding):
    from utils import compute_face_similarity
    embeddings = get_embeddings(faces)
    user_face_id = compute_face_similarity(embeddings, embedding).argmax()

    return faces[user_face_id]


def get_angle(face):
  landmark = face['landmark_3d_68']
  tan_theta = (landmark[34][0] - landmark[28][0]) / (landmark[34][1] - landmark[28][1])
  theta = np.arctan(tan_theta)
  rotate_angle = theta * 180 / math.pi

  return rotate_angle
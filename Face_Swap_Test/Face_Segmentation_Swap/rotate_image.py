import math
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))


def get_angle(cv_img):
  face = app.get(cv_img)
  landmark = face[0].landmark_3d_68
  tan_theta = (landmark[34][0] - landmark[28][0]) / (landmark[34][1] - landmark[28][1])
  theta = np.arctan(tan_theta)
  rotate_angle = theta * 180 / math.pi
  return rotate_angle


def get_bbox(img):
  face = app.get(img)
  return (face[0]['bbox'][0], face[0]['bbox'][1], face[0]['bbox'][2], face[0]['bbox'][3])
import torch
import facer
import numpy as np
from face_data import *


def get_seg_label(img, face):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tensor = torch.from_numpy(img)
    tensor = tensor.unsqueeze(0).permute(0, 3, 1, 2)

    face_kps = torch.Tensor(np.array([get_kps(face)]))
    img_ids = torch.tensor(np.array([0]), dtype=torch.int64)
    face_info = {'points': face_kps, 'image_ids': img_ids}

    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        face_seg = face_parser(tensor, face_info)
    seg_logits = face_seg['seg']['logits'][0]
    seg_probs = seg_logits.softmax(dim=0)
    seg_labels = seg_probs.argmax(dim=0)
    seg_labels = seg_labels.to(torch.uint8)
    seg_labels = seg_labels.numpy()

    return seg_labels


def get_seg_bbox(seg_label):
    seg_bbox = np.where(seg_label>0)
    seg_x1, seg_y1, seg_x2, seg_y2 = [min(seg_bbox[1]), min(seg_bbox[0]),
                                      max(seg_bbox[1]), max(seg_bbox[0])]

    return seg_x1, seg_y1, seg_x2, seg_y2


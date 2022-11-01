import facer
import torch
import cv2
import rotate_image
from rotate_image import *
import imutils
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''이미지 불러오기'''
source_img = cv2.imread('source_img_file_path')
target_img = cv2.imread('target_img_file_path')

'''이미지에서 얼굴 회전 각도 찾기'''
source_angle = get_angle(source_img)
target_angle = get_angle(target_img)
rotate_angle = source_angle - target_angle

'''target 얼굴 각도에 맞춰 source 얼굴 회전시키기'''
source_img_rotated = imutils.rotate_bound(source_img, rotate_angle)

'''이미지에서 얼굴 바운딩 박스 크기 검출'''
source_x1, source_y1, source_x2, source_y2 = get_bbox(source_img_rotated)
source_cx, source_cy = int((source_x1+source_x2)//2), int((source_y1+source_y2)//2)
source_bbox_w = source_x2 - source_x1
source_bbox_h = source_y2 - source_y1

target_x1, target_y1, target_x2, target_y2 = get_bbox(target_img)
target_cx, target_cy = int((target_x1+target_x2)//2), int((target_y1+target_y2)//2)
target_bbox_w = target_x2 - target_x1
target_bbox_h = target_y2 - target_y1


'''target 얼굴 크기에 맞춰 source 얼굴 크기 조정하기'''
height_scale = target_bbox_h/source_bbox_h
source_img_resized = imutils.resize(source_img_rotated, height=int(source_img_rotated.shape[0]*height_scale))

if (source_bbox_w*height_scale) < target_bbox_w:
    width_scale = target_bbox_w/(source_bbox_w*height_scale)
    source_img_resized = imutils.resize(source_img_resized, width=int(source_img_resized.shape[1]*width_scale))


'''source 얼굴 segementation 진행'''
source_tensor = torch.from_numpy(source_img_resized)
target_tensor = torch.from_numpy(target_img)

source_tensor = source_tensor.unsqueeze(0).permute(0, 3, 1, 2)
target_tensor = target_tensor.unsqueeze(0).permute(0, 3, 1, 2)

face_detector = facer.face_detector('retinaface/mobilenet', device=device)
with torch.inference_mode():
    source_faces = face_detector(source_tensor)
    target_faces = face_detector(target_tensor)

face_parser = facer.face_parser('farl/lapa/448', device=device)
with torch.inference_mode():
    source_faces = face_parser(source_tensor, source_faces)
    target_faces = face_parser(target_tensor, target_faces)


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
white_source_seg = np.where(source_seg_labels>0, 255, 0)
white_source_seg=white_source_seg.astype(np.uint8)


# x1, y1, x2, y2
seg_x1, seg_y1, seg_x2, seg_y2 = [min(source_seg_box[1]), min(source_seg_box[0]), max(source_seg_box[1]), max(source_seg_box[0])]
source_seg_h = seg_y2-seg_y1
source_seg_w = seg_x2-seg_x1
print(source_seg_h, source_seg_w)

source_face_image = cv2.bitwise_and(source_img_resized, source_img_resized, mask=source_seg_labels)


'''target 이미지 크기와 동일한 마스크 생성'''
d3_mask=np.zeros(target_img.shape, np.uint8)

source_x1, source_y1, source_x2, source_y2 = get_bbox(source_img_resized)
source_cx, source_cy = int((source_x1+source_x2)//2), int((source_y1+source_y2)//2)

start_y = target_cy-(source_cy-seg_y1)
start_x = target_cx-(source_cx-seg_x1)
end_y = start_y+source_seg_h
end_x = start_x+source_seg_w

#얼굴이 잘리는 경우 예외처리
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
if start_y > 0 :
    t_start_y = start_y
    s_start_y = seg_y1
if start_x <=0:
    t_start_x = 0
    s_start_x = seg_x1 - start_x
if start_x > 0:
    t_start_x = start_x
    s_start_x = seg_x1
if end_y > target_img.shape[0]:
    t_end_y = target_img.shape[0]
    s_end_y = seg_y2 - (end_y-target_img.shape[0])
if end_y <= target_img.shape[0]:
    t_end_y = end_y
    s_end_y = seg_y2
if end_x > target_img.shape[1]:
    t_end_x = target_img.shape[1]
    s_end_x = seg_x2 - (end_x-target_img.shape[1])
if end_x <= target_img.shape[1]:
    t_end_x = end_x
    s_end_x = seg_x2


'''ValueError: could not broadcast input array from shape (4910,1987,3) into shape (2690,1987,3)'''
d3_mask[t_start_y:t_end_y, t_start_x:t_end_x] = source_face_image[s_start_y:s_end_y, s_start_x:s_end_x]

img_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
d1_mask = np.zeros_like(img_gray)
d1_mask[t_start_y:t_end_y, t_start_x:t_end_x] = white_source_seg[s_start_y:s_end_y, s_start_x:s_end_x]
mask = cv2.bitwise_and(d1_mask, d1_mask, mask=target_seg_labels)

source_fg = cv2.bitwise_and(d3_mask, d3_mask, mask=mask)
target_bg = cv2.bitwise_and(target_img, target_img, mask=cv2.bitwise_not(mask))
result = cv2.add(target_bg, source_fg)
cv2.imshow("result", result)
cv2.waitKey()
cv2.destroyAllWindows()

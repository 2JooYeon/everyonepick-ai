from kafka import KafkaProducer
from json import dumps
import traceback
import base64
import cv2

producer=KafkaProducer(
    bootstrap_servers=["b-1.everyonepickkafka.v7ao1r.c3.kafka.ap-northeast-2.amazonaws.com:9092",
                       "b-3.everyonepickkafka.v7ao1r.c3.kafka.ap-northeast-2.amazonaws.com:9092",
                       "b-2.everyonepickkafka.v7ao1r.c3.kafka.ap-northeast-2.amazonaws.com:9092"],
    value_serializer=lambda x: dumps(x).encode('utf-8')
)

def send_data(pick_id, np_img):
    record = {}
    record["pick_id"] = pick_id

    # numpy 형식의 이미지를 base64 인코딩하기
    base64_img = base64.b64encode(cv2.imencode('.jpg', np_img)[1]).decode()
    record["face_swap_result"] = base64_img
    try:
        response = producer.send(topic='everyonepick.faceswap.result', value=record).get()
        producer.flush()
    except:
        traceback.print_exc()


from kafka import KafkaConsumer
from json import loads
from input_data import refine_input
from utils import find_base_photo_id, list_of_face_swap, download_s3_url
from face_data import get_faceobj, get_embeddings
from face_swap import swap_face
from producer import send_data


consumer = KafkaConsumer("everyonepick.faceswap.request",
                         bootstrap_servers=["b-1.everyonepickkafka.v7ao1r.c3.kafka.ap-northeast-2.amazonaws.com:9092",
                                            "b-3.everyonepickkafka.v7ao1r.c3.kafka.ap-northeast-2.amazonaws.com:9092",
                                            "b-2.everyonepickkafka.v7ao1r.c3.kafka.ap-northeast-2.amazonaws.com:9092"],
                         value_deserializer=lambda x: loads(x.decode('utf-8')),
                         auto_offset_reset="earliest",
                         enable_auto_commit=False,
                         max_poll_interval_ms=600000,
                         max_poll_records=1
                         )

while True:
    # 30초 지나고 return 하라는 의미
    for _, records in consumer.poll(30000).items():
        for record in records:
            try:
                pick_id, photo_id_url, photo_id_count, user_choices, user_embedding = refine_input(record.value)
                target_img_id = find_base_photo_id(user_choices, photo_id_count)
                face_swap_list = list_of_face_swap(user_choices, target_img_id)
                target_img = download_s3_url(photo_id_url[target_img_id])
                # 합성이 필요 없는 경우
                if len(face_swap_list) == 0:
                    send_data(pick_id, target_img)
                # 합성이 필요한 경우
                else:
                    target_faces = get_faceobj(target_img)
                    target_embeddings = get_embeddings(target_faces)
                    for user_id, choice_id in face_swap_list:
                        user_face_embedding = user_embedding[user_id]
                        source_img = download_s3_url(photo_id_url[choice_id])
                        swap_result = swap_face(user_face_embedding, source_img, target_img, target_faces)
                        target_img = swap_result
                    send_data(pick_id, swap_result)
            except:
                import logging
                import traceback
                logging.error(traceback.format_exc())

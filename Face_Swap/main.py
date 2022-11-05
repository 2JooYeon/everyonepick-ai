from kafka import KafkaConsumer
from json import loads
import logging
import traceback
from input_data import refine_input
from utils import *

consumer = KafkaConsumer("everyonepick.faceswap.request",
                         bootstrap_servers=["b-1.everyonepickkafka.v7ao1r.c3.kafka.ap-northeast-2.amazonaws.com:9092", "b-3.everyonepickkafka.v7ao1r.c3.kafka.ap-northeast-2.amazonaws.com:9092", "b-2.everyonepickkafka.v7ao1r.c3.kafka.ap-northeast-2.amazonaws.com:9092"],
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
                pick_id, photo_id_url, photo_id_count, user_choices, user_embedding = refine_input(record)
                target_img_id = find_base_photo_id(user_choices, photo_id_count)

            except:
                logging.error(traceback.format_exc())

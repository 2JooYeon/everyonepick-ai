from kafka import KafkaProducer
from json import dumps
import traceback

producer=KafkaProducer(
    bootstrap_servers=["b-1.everyonepickkafka.v7ao1r.c3.kafka.ap-northeast-2.amazonaws.com:9092",
                       "b-3.everyonepickkafka.v7ao1r.c3.kafka.ap-northeast-2.amazonaws.com:9092",
                       "b-2.everyonepickkafka.v7ao1r.c3.kafka.ap-northeast-2.amazonaws.com:9092"],
    value_serializer=lambda x: dumps(x).encode('utf-8')
)

record = [{"pick_id":1, "image":"byte code"}]
try:
    response = producer.send(topic='everyonepick.faceswap.result', value=record).get()
    print(response)
    producer.flush()
except:
    traceback.print_exc()


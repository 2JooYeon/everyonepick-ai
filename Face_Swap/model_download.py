import boto3
import os


def download_face_model():
    home_path = os.getenv("HOME")
    model_dir = os.path.join(home_path, '.insightface', 'models')
    model_name = 'antelopev2'
    bucket_name = 'everyonepick-ai-model-bucket'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(os.path.join(model_dir, model_name)):
        os.makedirs(os.path.join(model_dir, model_name))

        resource = boto3.resource('s3')
        bucket = resource.Bucket(bucket_name)
        for object in bucket.objects.filter(Prefix=model_name):
            bucket.download_file(object.key, f"{model_dir}/{object.key}")


def download_seg_model():
    home_path = os.getenv("HOME")
    hub_dir = os.path.join(home_path, '.cache', 'torch', 'hub')
    model_dir = os.path.join(hub_dir, 'checkpoints')

    segmentation_model_file = 'face_parsing.farl.lapa.main_ema_136500_jit191.pt'
    bucket_name = 'everyonepick-ai-model-bucket'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(os.path.join(model_dir, segmentation_model_file)):
        client = boto3.client('s3')
        cached_file = os.path.join(model_dir, segmentation_model_file)
        with open(cached_file, 'wb') as file_name:
            client.download_fileobj(bucket_name, segmentation_model_file, file_name)

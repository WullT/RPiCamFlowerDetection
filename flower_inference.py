import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
from yolo_model import YoloModel
import argparse
import time
from PIL import Image, ImageOps
import numpy as np
from utils import *
import os
import datetime
import gc
import yaml
import argparse
import socket

parser = argparse.ArgumentParser(description="YOLO Object Detection")
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="configuration.yaml",
    help="path to configuration file",
)
args = parser.parse_args()


with open(args.config, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit(1)


cfg = yaml.load(open(args.config))
model_config = cfg.get("model")
weights = model_config.get("weights_path")
classes = model_config.get("classes")
image_size = model_config.get("image_size")
confidence_threshold = model_config.get("confidence_threshold")
iou_threshold = model_config.get("iou_threshold")
margin = model_config.get("margin")

image_source = cfg.get("image_source")
image_source_url = image_source.get("url")
image_source_username = image_source.get("username")
image_source_password = image_source.get("password")

dest_endpoint = cfg.get("dest_endpoint")
dest_endpoint_url = dest_endpoint.get("url")
dest_endpoint_username = dest_endpoint.get("username")
dest_endpoint_password = dest_endpoint.get("password")

capture_interval = cfg.get("capture_interval")


HOSTNAME = socket.gethostname()
if "cam-" in HOSTNAME:
    HOSTNAME = HOSTNAME.replace("cam-", "")



model = YoloModel(
    weights, image_size, confidence_threshold, iou_threshold, classes=classes, margin=margin
)


i = 0
while True:

    logging.info("downloading image {}".format(i))
    t0 = time.time()
    download_time = datetime.datetime.utcnow()
    image = download_image(
        image_source_url, username=image_source_username, password=image_source_password
    )
    t1 = time.time()
    logging.info("downloading image {} took {}".format( i, t1 - t0))

    crops, result_class_names, result_scores = model.get_crops(image)
    t2 = time.time()
    logging.info("processing image {} took {}".format( i, t2 - t1))


    meta = {
        "time_download": (t1 - t0),
        "time_process": (t2 - t1),
        "capture_size": image.size,
        "conf_thres": confidence_threshold,
        "iou_thres": iou_threshold,
        "node_id": HOSTNAME,
        "capture_time": download_time.isoformat(),
        "margin": model.margin,
        "model_name": weights.split("/")[-1]
    }
    upload_json(
        crops,
        result_class_names,
        result_scores,
        dest_endpoint_url,
        username=dest_endpoint_username,
        password=dest_endpoint_password,
        record_date=download_time,
        metadata=meta,
    )
    t3 = time.time()
    logging.info("uploading image {} took {}".format( i,  t3 - t2))
    logging.info("TOTAL TIME: {}".format(t3 - t0))
    logging.info("Collecting")
    gc.collect()
    i += 1
    while time.time() - t0 < (capture_interval - 0.1):
        time.sleep(0.05)

from yolo_model import YoloModel
import argparse
import time
from PIL import Image, ImageOps
import numpy as np
from utils import *
import os
import datetime
import gc
import logging
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
if not os.path.exists(args.config):
    raise Exception("Configuration file not found")


cfg = yaml.load(open(args.config))
model_config = cfg.get("model")
weights = model_config.get("weights_path")
classes = model_config.get("classes")
image_size = model_config.get("image_size")
confidence_threshold = model_config.get("confidence_threshold")
iou_threshold = model_config.get("iou_threshold")
image_source = cfg.get("image_source")
image_source_url = image_source.get("url")
image_source_username = image_source.get("username")
image_source_password = image_source.get("password")

dest_endpoint = cfg.get("dest_endpoint")
dest_endpoint_url = dest_endpoint.get("url")
dest_endpoint_username = dest_endpoint.get("username")
dest_endpoint_password = dest_endpoint.get("password")

HOSTNAME = socket.gethostname()
if "cam-" in HOSTNAME:
    HOSTNAME = HOSTNAME.replace("cam-", "")

capture_interval = cfg.get("capture_interval")


model = YoloModel(
    weights, image_size, confidence_threshold, iou_threshold, classes=classes
)


i = 0
while True:

    print("downloading image", i)
    t0 = time.time()
    download_time = datetime.datetime.utcnow()
    image = download_image(
        image_source_url, username=image_source_username, password=image_source_password
    )
    t1 = time.time()
    print("downloading image", i, "took", t1 - t0)
    crops, result_class_names, result_scores = model.get_crops(image)
    t2 = time.time()
    print("processing image", i, "took", t2 - t1)
    meta = {
        "time_download": (t1 - t0),
        "time_process": (t2 - t1),
        "capture_size": image.size,
        "conf_thres": confidence_threshold,
        "iou_thres": iou_threshold,
        "node_id": HOSTNAME,
        "capture_time": download_time.isoformat(),
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
    print("uploading image", i, "took", t3 - t2)
    print("TOTAL TIME:", t3 - t0)
    print("Collecting")
    gc.collect()
    i += 1
    while time.time() - t0 < (capture_interval - 0.1):
        time.sleep(0.05)

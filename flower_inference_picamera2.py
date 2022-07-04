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
from picamera2 import Picamera2



camera = Picamera2()

def capture_image():
    np_array = camera.capture_array()
    image = Image.fromarray(np_array)
    return image

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
margin = model_config.get("margin")

picamera_config = cfg.get("picamera_config")
picamera_width = picamera_config.get("width",4656)
picamera_height = picamera_config.get("height",3496)

dest_endpoint = cfg.get("dest_endpoint")
dest_endpoint_url = dest_endpoint.get("url")
dest_endpoint_username = dest_endpoint.get("username")
dest_endpoint_password = dest_endpoint.get("password")

capture_interval = cfg.get("capture_interval")



cam_config = camera.still_configuration()
cam_config["main"]["size"] = (picamera_width, picamera_height)
camera.configure(cam_config)
camera.start()

HOSTNAME = socket.gethostname()
if "cam-" in HOSTNAME:
    HOSTNAME = HOSTNAME.replace("cam-", "")


model = YoloModel(
    weights, image_size, confidence_threshold, iou_threshold, classes=classes, margin=margin
)


i = 0
while True:

    logging.info("getting image {}".format( i))
    t0 = time.time()
    download_time = datetime.datetime.utcnow()
    image = capture_image()
    t1 = time.time()
    logging.info("capture image {} took {}".format( i, t1 - t0))
    crops, result_class_names, result_scores = model.get_crops(image)
    t2 = time.time()
    logging.info("get crops {} took {}".format( i, t2 - t1))
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
    logging.info("uploading image {} took {}".format(i,  t3 - t2))


    logging.info("TOTAL TIME: {}".format( t3 - t0))
    logging.info("Collecting")
    gc.collect()
    i += 1
    while time.time() - t0 < (capture_interval - 0.1):
        time.sleep(0.05)

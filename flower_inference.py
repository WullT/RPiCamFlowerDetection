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
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit(1)


model_config = cfg.get("model")
MODEL_WEIGHTS = model_config.get("weights_path")
MODEL_CLASSES = model_config.get("classes")
MODEL_IMG_SIZE = model_config.get("image_size")
MODEL_CONFIDENCE_THRESHOLD = model_config.get("confidence_threshold")
MODEL_IOU_THRESHOLD = model_config.get("iou_threshold")
MODEL_MARGIN = model_config.get("margin")

input_config = cfg.get("input")
INPUT_TYPE = input_config.get("type", "url")
INPUT_URL = None
INPUT_USERNAME = None
INPUT_PASSWORD = None

camera = None

if INPUT_TYPE == "url":
    input_config_server = input_config.get("server")
    if input_config_server is None:
        logging.error("No server configuration found")
        exit(1)
    INPUT_URL = input_config_server.get("url")
    if INPUT_URL is None:
        logging.error("No input url found")
        exit(1)
    INPUT_USERNAME = input_config_server.get("username")
    INPUT_PASSWORD = input_config_server.get("password")

elif INPUT_TYPE == "camera":
    from picamera2 import Picamera2
    camera = Picamera2()
    input_config_camera = input_config.get("camera")
    INPUT_CAMERA_WIDTH = input_config_camera.get("width", 4656)
    INPUT_CAMERA_HEIGHT = input_config_camera.get("height",3496)
    cam_config = camera.still_configuration()
    cam_config["main"]["size"] = (INPUT_CAMERA_WIDTH, INPUT_CAMERA_HEIGHT)
    camera.configure(cam_config)
    camera.start()



output_config = cfg.get("output")
OUTPUT_URL = output_config.get("url")
OUTPUT_USERNAME = output_config.get("username")
OUTPUT_PASSWORD = output_config.get("password")

CAPTURE_INTERVAL = cfg.get("capture_interval")


HOSTNAME = socket.gethostname()
if "cam-" in HOSTNAME:
    HOSTNAME = HOSTNAME.replace("cam-", "")

def capture_image():
    if INPUT_TYPE == "url":
        logging.info("Capturing image from {}".format(INPUT_URL))
        image = download_image(
        INPUT_URL, username=INPUT_USERNAME, password=INPUT_PASSWORD
        )
    elif INPUT_TYPE == "camera":
        logging.info("Capturing image from camera")
        np_array = camera.capture_array()
        image = Image.fromarray(np_array)
    return image



model = YoloModel(
    MODEL_WEIGHTS,
    MODEL_IMG_SIZE,
    MODEL_CONFIDENCE_THRESHOLD,
    MODEL_IOU_THRESHOLD,
    classes=MODEL_CLASSES,
    margin=MODEL_MARGIN,
)


i = 0
while True:

    logging.info("downloading image {}".format(i))
    t0 = time.time()
    download_time = datetime.datetime.utcnow()
    image = capture_image()
    t1 = time.time()
    logging.info("downloading image {} took {}".format(i, t1 - t0))

    crops, result_class_names, result_scores = model.get_crops(image)
    t2 = time.time()
    logging.info("processing image {} took {}".format(i, t2 - t1))

    meta = {
        "time_download": (t1 - t0),
        "time_process": (t2 - t1),
        "capture_size": image.size,
        "conf_thres": MODEL_CONFIDENCE_THRESHOLD,
        "iou_thres": MODEL_IOU_THRESHOLD,
        "node_id": HOSTNAME,
        "capture_time": download_time.isoformat(),
        "margin": model.margin,
        "model_name": MODEL_WEIGHTS.split("/")[-1],
    }
    upload_json(
        crops,
        result_class_names,
        result_scores,
        OUTPUT_URL,
        username=OUTPUT_USERNAME,
        password=OUTPUT_PASSWORD,
        record_date=download_time,
        metadata=meta,
    )
    t3 = time.time()
    logging.info("uploading image {} took {}".format(i, t3 - t2))
    logging.info("TOTAL TIME: {}".format(t3 - t0))
    logging.info("Collecting")
    gc.collect()
    i += 1
    while time.time() - t0 < (CAPTURE_INTERVAL - 0.1):
        time.sleep(0.05)

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import cv2
import logging
import sys
import warnings
import pickle as pk
import time
import argparse
import imutils
from tqdm import trange
from imantics import Polygons, Mask
import json 
import os
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from PIL import Image

# PyTorch and Detectron2 Libraries
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Boxes

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.ImageFilter as ImageFilter

from enumeration_by_counting import Sort
from tensorRT_model import TensorRTInfer

def preprocess_image(image, shape):

    def resize_pad(image, shape, pad_color=(0, 0, 0)):
        """
        A subroutine to implement padding and resizing. This will resize the image to fit fully within the input
        size, and pads the remaining bottom-right portions with the value provided.
        :param image: The PIL image object
        :pad_color: The RGB values to use for the padded area. Default: Black/Zeros.
        :return: Two values: The PIL image object already padded and cropped, and the resize scale used.
        """
        min_size_test = 800
        max_size_test = 1333
        if shape[1] == 3:
            FORMAT = "NCHW"
            HEIGHT = shape[2]
            WIDTH = shape[3]
        elif shape[3] == 3:
            FORMAT = "NHWC"
            HEIGHT = shape[1]
            WIDTH = shape[2]

        # Get characteristics.
        width, height = image.size
        
        # Replicates behavior of ResizeShortestEdge augmentation.
        size = min_size_test * 1.0
        pre_scale = size / min(height, width)
        if height < width:
            newh, neww = size, pre_scale * width
        else:
            newh, neww = pre_scale * height, size

        # If delta between min and max dimensions is so that max sized dimension reaches self.max_size_test
        # before min dimension reaches self.min_size_test, keeping the same aspect ratio. We still need to
        # maintain the same aspect ratio and keep max dimension at self.max_size_test.
        if max(newh, neww) > max_size_test:
            pre_scale = max_size_test * 1.0 / max(newh, neww)
            newh = newh * pre_scale
            neww = neww * pre_scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        # Scaling factor for normalized box coordinates scaling in post-processing. 
        scaling = max(newh/height, neww/width)

        # Padding.
        image = image.resize((neww, newh), resample=Image.BILINEAR)
        pad = Image.new("RGB", (WIDTH, HEIGHT))
        pad.paste(pad_color, [0, 0, WIDTH, HEIGHT])
        pad.paste(image)
        return pad, scaling

    scale = None
    # Pad with mean values of COCO dataset, since padding is applied before actual model's
    # preprocessor steps (Sub, Div ops), we need to pad with mean values in order to reverse 
    # the effects of Sub and Div, so that padding after model's preprocessor will be with actual 0s.
    image, scale = resize_pad(image, shape, (124, 116, 104))
    image = np.asarray(image, dtype=np.float32)
    # Change HWC -> CHW.
    image = np.transpose(image, (2, 0, 1))
    # Change RGB -> BGR.
    return image[[2,1,0]], scale

#Overlay mask with transparency on top of the image.
def overlay(image, mask, color, alpha_transparency=0.4):
    for channel in range(3):
        image[:, :, channel] = np.where(mask == 1,
                              image[:, :, channel] *
                              (1 - alpha_transparency) + alpha_transparency * color[channel] * 255,
                              image[:, :, channel])
    return image

def visualize_masks(image, masks, bboxes, color, iou_threshold=0.5):
    # Get image dimensions.
    im_width, im_height = image.size
    line_width = 2
    font = ImageFont.load_default()
    index = 0
    for detectedMask in masks:
        # Dynamically convert PIL color into RGB numpy array.
        pixel_color = Image.new("RGB",(1, 1), color)
        # Normalize.
        np_color = (np.asarray(pixel_color)[0][0])/255
        # TRT instance segmentation masks.
        if isinstance(detectedMask, np.ndarray) and detectedMask.shape == (28, 28):
            # Get detection bbox resolution.
            det_width = round(bboxes[index][2] - bboxes[index][0])
            det_height = round(bboxes[index][3] - bboxes[index][1])
            # Create an image out of predicted mask array.
            small_mask = Image.fromarray(detectedMask)
            # Upsample mask to detection bbox's size.
            mask = small_mask.resize((det_width, det_height), resample=Image.BILINEAR)
            # Create an original image sized template for correct mask placement.
            pad = Image.new("L", (im_width, im_height))
            # Place your mask according to detection bbox placement.
            pad.paste(mask, (round(bboxes[index][0]), (round(bboxes[index][1]))))
            # Reconvert mask into numpy array for evaluation.
            padded_mask = np.array(pad)
            #Creat np.array from original image, copy in order to modify.
            image_copy = np.asarray(image).copy()
            # Image with overlaid mask.
            masked_image = overlay(image_copy, padded_mask, np_color)
            # Reconvert back to PIL.
            image = Image.fromarray(masked_image)
        index += 1

    return image

def main():

    CLASSES = {0 : 'OpenBoll', 1 : 'ClosedBoll', 2 : 'Flower', 3 : 'Square'}
    COLORS = {0 : (255, 0, 255), 1 : (255, 0, 0), 2 : (0, 0, 255), 3 : (255, 255, 0)}

    folder = os.path.split(args["input"])[0]
    finalFolder = os.path.join(folder, 'Bag Results With TensorRT')
    os.makedirs(finalFolder, exist_ok=True)

    # Log to stdout
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s"
    )

    objectCounts = np.zeros(4, dtype=np.uint32)
    totalObjectCounts = np.zeros(4, dtype=np.uint32)

    jsonOutput = {'frames': []}
    # GPU is available
    gpu = torch.cuda.is_available()
    logging.info(f"GPU available - { gpu }")

    # setup Realsense Camera
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, args['input'])
    profile = pipeline.start(config)

    playback=profile.get_device().as_playback() # get playback device
    playback.set_real_time(False) # disable real-time playback

    logging.info(f"Video Stream has started!")

    trt_infer = TensorRTInfer(args['model'])

    # initialize the video writer (we'll instantiate later if need be)
    writer = None
    writer2 = None
    #create instance of the SORT tracker
    tracker = Sort(max_age=args['max_age'], 
                   min_hits=args['min_hits'],
                   min_count=args['min_count'],
                   iou_threshold=args['iou_threshold'])
   
    # start the frames per second throughput estimator
    fps = FPS().start()
    frameCount = 0
    prevPosition = -1
    # cv2.namedWindow('Detections')
    try:
        logging.info(f"Tracking has started...")

        # outputsFile = []
        while True:
            # Capture frame-by-frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            newPosition = playback.get_position()
            if prevPosition > newPosition:
                break
            else:
                prevPosition = newPosition

            color_image = np.asanyarray(color_frame.get_data())

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            if finalFolder is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(os.path.join(finalFolder, 'RGB_Video.avi'), fourcc, args['fps'],
                    (args['width'], args['height']), True)

            if writer is not None:
                writer.write(color_image)

            if (finalFolder is not None) and (writer2 is None) and args['visualization']:
                fourcc2 = cv2.VideoWriter_fourcc(*"MJPG")
                writer2 = cv2.VideoWriter(os.path.join(finalFolder, 'RGB_Video_Visualization.avi'), fourcc2, args['fps'],
                    (args['width'], args['height']), True)

                    # f'RGB_Video_MinHits{args["min_hits"]}_MaxAge{args["max_age"]}_MinCount{args["min_count"]}.avi'), fourcc, args['fps'],
                    # (args['width'], args['height']), True)


            shape, _ = trt_infer.input_spec()
            preprocessedImage, scale = preprocess_image(Image.fromarray(color_image), shape)
            detections = trt_infer.infer_frame(preprocessedImage, scale, args['nms_threshold'])

            # Filter based on confidence
            detections = [detection for detection in detections if detection['score'] >= args['confidence']]

            dets = np.zeros((len(detections), 5))
            classes = [0]*len(detections)
            masks = [0]*len(detections)
            index = 0
            for det in detections:
                dets[index]= np.array([det['ymin'], det['xmin'], det['ymax'], det['xmax'], det['score']])
                classes[index]= int(det['class'])
                # Slight scaling, to get binary masks after float32 -> uint8
                # conversion, if not scaled all pixels are zero. 
                mask = det['mask'] > 0.5
                masks[index] = mask.tolist()
                index += 1
            
            if args['visualization']:
                # Get Mask Indices
                OpenBoll_indices = [i for i, value in enumerate(classes) if value == 0]
                ClosedBoll_indices = [i for i, value in enumerate(classes) if value == 1]
                Flower_indices = [i for i, value in enumerate(classes) if value == 2]
                Square_indices = [i for i, value in enumerate(classes) if value == 3]
                # Get Masks
                OpenBoll_masks = list(masks[i] for i in OpenBoll_indices)
                ClosedBoll_masks = list(masks[i] for i in ClosedBoll_indices)
                Flower_masks = list(masks[i] for i in Flower_indices)
                Square_masks = list(masks[i] for i in Square_indices)
                # get bboxes
                OpenBoll_bboxes = list(dets[i,:4] for i in OpenBoll_indices)
                ClosedBoll_bboxes = list(dets[i,:4] for i in ClosedBoll_indices)
                Flower_bboxes = list(dets[i,:4] for i in Flower_indices)
                Square_bboxes = list(dets[i,:4] for i in Square_indices)
                # Overlay Masks
                frame = Image.fromarray(color_image)
                if len(OpenBoll_masks) != 0:
                    frame = visualize_masks(frame, np.array(OpenBoll_masks, dtype= np.uint8), OpenBoll_bboxes, COLORS[0])
                if len(ClosedBoll_masks) != 0:
                    frame = visualize_masks(frame, np.array(ClosedBoll_masks, dtype= np.uint8), ClosedBoll_bboxes, COLORS[1])
                if len(Flower_masks) != 0:
                    frame = visualize_masks(frame, np.array(Flower_masks, dtype= np.uint8), Flower_bboxes, COLORS[2])
                if len(Square_masks) != 0:
                    frame = visualize_masks(frame, np.array(Square_masks, dtype= np.uint8), Square_bboxes, COLORS[3])
                frame = np.asarray(frame)

            trackers, trackerLabels, objectCounts = tracker.update(dets, classes, objectCounts)

            if args['visualization']:
                for d, label in zip(trackers, trackerLabels):
                    d = d.astype(np.int32)
                    frame = cv2.rectangle(frame, (d[0],d[1]), (d[2],d[3]), COLORS[label], 1)
                    cv2.putText(frame, f"{CLASSES[label]}", (d[0],d[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[label], 1)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("OpenBolls", objectCounts[0]),
                ("ClosedBolls", objectCounts[1]),
                ("Flowers", objectCounts[2]),
                ("Squares", objectCounts[3])
            ]
            # loop over the info tuples and draw them on our frame
            if args['visualization']:
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, args['height'] - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
            # # check to see if we should write the frame to disk
            if (writer2 is not None) and args['visualization']:
                writer2.write(frame)
            
            totalObjectCounts[0] += classes.count(0)
            totalObjectCounts[1] += classes.count(1)
            totalObjectCounts[2] += classes.count(2)
            totalObjectCounts[3] += classes.count(3)

            jsonOutput['frames'].append({'frame':frameCount, 'classes': classes,'bboxes':dets[:,:4].tolist(),
                                         'segmentations':masks, 'trackedClasses_wSort' : trackerLabels,
                                         'trackedBBoxes_wSort': trackers[:,0:4].tolist(), 'trackedIDs_wSort': trackers[:,4].tolist(), 
                                         'numOfObjectsInFrame': {'OpenBolls': classes.count(0), 
                                         'ClosedBolls': classes.count(1), 'Flowers': classes.count(2),
                                         'Squares': classes.count(3)}, 'objectCount_wSort': {'OpenBolls':int(objectCounts[0]), 
                                         'ClosedBolls':int(objectCounts[1]), 'Flowers':int(objectCounts[2]), 
                                         'Squares':int(objectCounts[3])}})

            if frameCount % 1800 == 0 and frameCount != 0:
                print(f'Finished minute {frameCount / 1800} ...')
            
            # then update the FPS counter
            fps.update()
            frameCount += 1
    except RuntimeError:
        pass

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    with open(os.path.join(finalFolder, "ProcessingSpeed.txt"), "w") as f:
        f.write(f"Elapsed Time: {fps.elapsed():.2f}\nFPS: {fps.fps():.2f}")
            
    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()
    if (writer2 is not None) and args['visualization']:
        writer2.release()
    
    with open(os.path.join(finalFolder, "RGB_Processing.json"), "w") as outfile:
        json.dump(jsonOutput, outfile)
    
    results = {'numOfObjectsDetected': {'OpenBolls':int(totalObjectCounts[0]), 'ClosedBolls':int(totalObjectCounts[1]), 
               'Flowers':int(totalObjectCounts[2]), 'Squares':int(totalObjectCounts[3])}, 
               'objectCount_wSort': {'OpenBolls':int(objectCounts[0]), 'ClosedBolls':int(objectCounts[1]), 
               'Flowers':int(objectCounts[2]), 'Squares':int(objectCounts[3])}}

    with open(os.path.join(finalFolder, "RGB_Processing_Results.json"), 'w') as outfiletxt:
        json.dump(results, outfiletxt)
    
    # Save Dist Figure for Organ Count
    Classes=['OpenBolls', 'ClosedBolls', 'Flowers', 'Squares']
    Dist=[objectCounts[0], objectCounts[1], objectCounts[2], objectCounts[3]]
    fig = plt.figure()
    plt.bar(Classes, Dist)
    for index, value in enumerate(Dist):
        plt.text(index, value, str(value))
    plt.title('Organ Count')
    plt.savefig(os.path.join(finalFolder, "RGB_Video") + '_OrganCount.png')
    plt.close(fig)

    # Save Dist Figure for Organ Count
    Classes=['OpenBolls', 'ClosedBolls', 'Flowers', 'Squares']
    Dist=[totalObjectCounts[0], totalObjectCounts[1], totalObjectCounts[2], totalObjectCounts[3]]
    fig = plt.figure()
    plt.bar(Classes, Dist)
    for index, value in enumerate(Dist):
        plt.text(index, value, str(value))
    plt.title('Total Number Of Organs Detected')
    plt.savefig(os.path.join(finalFolder, "RGB_Video") + '_TotalNumOfOrgans.png')
    plt.close(fig)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to model")
    ap.add_argument("-i", "--input", type=str, required=True,
        help="Input Video")
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
        help="minimum probability to filter weak detections")
    ap.add_argument("-w", "--width", type=int, default=1280,
        help="image width")
    ap.add_argument("-t", "--height", type=int, default=720,
        help="image height")
    ap.add_argument("-f", "--fps", type=int, default=30,
        help="Camera FPS")
    ap.add_argument("--min_hits", 
        help="Minimum number of associated detections before track is initialised.", 
        type=int, default=3)
    ap.add_argument("--max_age", 
        help="Maximum number of frames to keep alive a track without associated detections.", 
        type=int, default=3)
    ap.add_argument("--min_count", 
        help="Minimum number of frames to count an object!", 
        type=int, default=6)
    ap.add_argument("--nms_threshold", 
        help="NMS Threshold", 
        type=float, default=0.6)
    ap.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    ap.add_argument("--visualization", help="Should we visualize segmentation and tracking.", type=bool, default=False)
    args = vars(ap.parse_args())
    main()
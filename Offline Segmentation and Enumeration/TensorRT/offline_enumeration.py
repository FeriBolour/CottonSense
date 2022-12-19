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

def main():

    NAME = input('Please enter the name of the experiment:\n')

    os.makedirs(os.path.join(args['outputFolder'], NAME), exist_ok=True)

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

    vs = cv2.VideoCapture(args["input"])
    totalFrames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"Video Stream has started!")

    trt_infer = TensorRTInfer(args['model'])

    # initialize the video writer (we'll instantiate later if need be)
    writer = None
    #create instance of the SORT tracker
    tracker = Sort(max_age=args['max_age'], 
                   min_hits=args['min_hits'],
                   min_count=args['min_count'],
                   iou_threshold=args['iou_threshold'])
   
    # start the frames per second throughput estimator
    fps = FPS().start()

    # cv2.namedWindow('Detections')
    try:
        logging.info(f"Tracking has started...")

        # outputsFile = []
        for count in trange(totalFrames):
            # Capture frame-by-frame
            color_image = vs.read()
            color_image = color_image[1] if args.get("input", False) else color_image
            # if we are viewing a video and we did not grab a frame then we
            # have reached the end of the video
            if args["input"] is not None and color_image is None:
                break

            if (args['width'] == None) or (args['height'] == None):
                args['height'], args['width'], _ = color_image.shape

            # if args["outputFolder"] is not None and writer is None:
            #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            #     writer = cv2.VideoWriter(os.path.join(args['outputFolder'], NAME, NAME + '.avi'), fourcc, args['fps'],
            #         (args['width'], args['height']), True)


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

            trackers, trackerLabels, objectCounts = tracker.update(dets, classes, objectCounts)

            # # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(color_image)
            
            totalObjectCounts[0] += classes.count(0)
            totalObjectCounts[1] += classes.count(1)
            totalObjectCounts[2] += classes.count(2)
            totalObjectCounts[3] += classes.count(3)

            jsonOutput['frames'].append({'frame':count, 'classes': classes,'bboxes':dets[:,:4].tolist(),
                                         'segmentations':masks, 'trackedClasses_wSort' : trackerLabels,
                                         'trackedBBoxes_wSort': trackers[:,0:4].tolist(), 'trackedIDs_wSort': trackers[:,4].tolist(), 
                                         'numOfObjectsInFrame': {'OpenBolls': classes.count(0), 
                                         'ClosedBolls': classes.count(1), 'Flowers': classes.count(2),
                                         'Squares': classes.count(3)}, 'objectCount_wSort': {'OpenBolls':int(objectCounts[0]), 
                                         'ClosedBolls':int(objectCounts[1]), 'Flowers':int(objectCounts[2]), 
                                         'Squares':int(objectCounts[3])}})

            # then update the FPS counter
            fps.update()
    except KeyboardInterrupt:
        pass

    # release the video file pointer
    vs.release()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()
    
    with open(os.path.join(args['outputFolder'], NAME, NAME + ".json"), "w") as outfile:
        json.dump(jsonOutput, outfile)
    
    results = {'numOfObjectsDetected': {'OpenBolls':int(totalObjectCounts[0]), 'ClosedBolls':int(totalObjectCounts[1]), 
               'Flowers':int(totalObjectCounts[2]), 'Squares':int(totalObjectCounts[3])}, 
               'objectCount_wSort': {'OpenBolls':int(objectCounts[0]), 'ClosedBolls':int(objectCounts[1]), 
               'Flowers':int(objectCounts[2]), 'Squares':int(objectCounts[3])}}

    with open(os.path.join(args['outputFolder'], NAME, NAME + "_results.json"), 'w') as outfiletxt:
        json.dump(results, outfiletxt)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to model")
    ap.add_argument("-i", "--input", type=str, required=True,
        help="Input Video")
    ap.add_argument("-o", "--outputFolder", type=str,
        help="path to optional output video folder")
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
    args = vars(ap.parse_args())
    main()
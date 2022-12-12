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

# PyTorch and Detectron2 Libraries
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.structures import Boxes

import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from torchvision.ops import nms


from enumeration_by_counting import Sort

def initialize_predictor(modelPath, confidence):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 
    #----------------------------------------------------------------------- FPN Options
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    #-----------------------------------------------------------------------
    cfg.MODEL.WEIGHTS = modelPath
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    predictor = DefaultPredictor(cfg)

    return predictor

def main():

    # NAME = input('Please enter the name of the experiment:\n')
    NAME = os.path.basename(args["input"])
    NAME = NAME[:-4]

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

    predictor = initialize_predictor(args['model'], args['confidence'])

    # initialize the video writer (we'll instantiate later if need be)
    # writer = None
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


            outputs = predictor(color_image)

            bboxes = outputs['instances'].pred_boxes.tensor.cpu().detach().numpy()
            classes = outputs["instances"].pred_classes.cpu().detach().numpy().tolist()
            polygons = []
            for mask in outputs['instances'].pred_masks.cpu().detach().numpy():
                polygons.append(Mask(mask).polygons().segmentation[0])


            dets = np.column_stack((bboxes, outputs['instances'].scores.cpu().detach().numpy()))

            trackers, trackerLabels, objectCounts = tracker.update(dets, classes, objectCounts)

            # # check to see if we should write the frame to disk
            # if writer is not None:
            #     writer.write(color_image)
            
            totalObjectCounts[0] += classes.count(0)
            totalObjectCounts[1] += classes.count(1)
            totalObjectCounts[2] += classes.count(2)
            totalObjectCounts[3] += classes.count(3)

            jsonOutput['frames'].append({'frame':count, 'classes': classes,'bboxes':bboxes.tolist(),
                                         'segmentations':polygons, 'trackedClasses_wSort' : trackerLabels,
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
    # if writer is not None:
    #     writer.release()
    
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
    ap.add_argument("-w", "--width", type=int, default=None,
        help="image width")
    ap.add_argument("-t", "--height", type=int, default=None,
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
    ap.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    ap.add_argument("--nms_threshold", help="Minimum NMS for removing boxes.", type=float, default=0.7)
    args = vars(ap.parse_args())
    main()
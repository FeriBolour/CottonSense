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
from imantics import Polygons, Mask
import json 
import os
import matplotlib.pyplot as plt
import pyrealsense2 as rs

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

    CLASSES = {0 : 'OpenBoll', 1 : 'ClosedBoll', 2 : 'Flower', 3 : 'Square'}
    COLORS = {0 : (255, 0, 255), 1 : (255, 0, 0), 2 : (0, 0, 255), 3 : (255, 255, 0)}

    folder = os.path.split(args["input"])[0]
    finalFolder = os.path.join(folder, 'Bag Results With Detectron2')
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

    predictor = initialize_predictor(args['model'], args['confidence'])

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


            outputs = predictor(color_image)

            bboxes = outputs['instances'].pred_boxes.tensor.cpu().detach().numpy()
            classes = outputs["instances"].pred_classes.cpu().detach().numpy().tolist()
            polygons = []
            for mask in outputs['instances'].pred_masks.cpu().detach().numpy():
                polygons.append(Mask(mask).polygons().segmentation[0])

            if args['visualization']:
                # Get Polygon Indices
                OpenBoll_indices = [i for i, value in enumerate(classes) if value == 0]
                ClosedBoll_indices = [i for i, value in enumerate(classes) if value == 1]
                Flower_indices = [i for i, value in enumerate(classes) if value == 2]
                Squares_indices = [i for i, value in enumerate(classes) if value == 3]
                # Create Polygons
                OpenBoll_polygons = Polygons(polygons[i] for i in OpenBoll_indices)
                ClosedBoll_polygons = Polygons(polygons[i] for i in ClosedBoll_indices)
                Flower_polygons = Polygons(polygons[i] for i in Flower_indices)
                Squares_polygons = Polygons(polygons[i] for i in Squares_indices)
                # Draw Polygons
                frame = OpenBoll_polygons.draw(color_image, COLORS[0])
                frame = ClosedBoll_polygons.draw(frame, COLORS[1])
                frame = Flower_polygons.draw(frame, COLORS[2])
                frame = Squares_polygons.draw(frame, COLORS[3])

            dets = np.column_stack((bboxes, outputs['instances'].scores.cpu().detach().numpy()))

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

            jsonOutput['frames'].append({'frame':frameCount, 'classes': classes,'bboxes':bboxes.tolist(),
                                         'segmentations':polygons, 'trackedClasses_wSort' : trackerLabels,
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
    except KeyboardInterrupt:
        pass

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
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
    ap.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    ap.add_argument("--visualization", help="Should we visualize segmentation and tracking.", type=bool, default=False)
    args = vars(ap.parse_args())
    main()
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
import matplotlib.pyplot as plt
import os

def main():

    CLASSES = {0 : 'OpenBoll', 1 : 'ClosedBoll', 2 : 'Flower', 3 : 'Square'}
    COLORS = {0 : (255, 0, 255), 1 : (255, 0, 0), 2 : (0, 0, 255), 3 : (255, 255, 0)}
    # Log to stdout
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s"
    )

    vs = cv2.VideoCapture(os.path.join(args["inputFolder"], os.path.basename(args["inputFolder"]) + '.avi'))
    totalFrames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"Video Stream has started!")

    # load json files
    F = open(os.path.join(args["inputFolder"], os.path.basename(args["inputFolder"]) + '.json'))
    data = json.load(F)
    F2 = open(os.path.join(args["inputFolder"], os.path.basename(args["inputFolder"]) + '_results.json'))
    results = json.load(F2)

    logging.info(f"JSON Files Loaded!")

    # initialize the video writer (we'll instantiate later if need be)
    writer = None
   
    # start the frames per second throughput estimator
    fps = FPS().start()

    # cv2.namedWindow('Detections')
    try:

        # outputsFile = []
        for count in trange(totalFrames):
            # Capture frame-by-frame
            color_image = vs.read()
            color_image = color_image[1]
            h, w, c = color_image.shape
            # if we are viewing a video and we did not grab a frame then we
            # have reached the end of the video
            if args["inputFolder"] is not None and color_image is None:
                break

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            if args["inputFolder"] is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(os.path.join(args["inputFolder"], os.path.basename(args["inputFolder"]) + '_visualizer.avi'), fourcc, args['fps'], (w, h), True)


            # Get Polygon Indices
            OpenBoll_indices = [i for i, value in enumerate(data['frames'][count]['classes']) if value == 0]
            ClosedBoll_indices = [i for i, value in enumerate(data['frames'][count]['classes']) if value == 1]
            Flower_indices = [i for i, value in enumerate(data['frames'][count]['classes']) if value == 2]
            Squares_indices = [i for i, value in enumerate(data['frames'][count]['classes']) if value == 3]
            # Create Polygons
            OpenBoll_polygons = Polygons(list(data['frames'][count]['segmentations'][i] for i in OpenBoll_indices))
            ClosedBoll_polygons = Polygons(list(data['frames'][count]['segmentations'][i] for i in ClosedBoll_indices))
            Flower_polygons = Polygons(list(data['frames'][count]['segmentations'][i] for i in Flower_indices))
            Squares_polygons = Polygons(list(data['frames'][count]['segmentations'][i] for i in Squares_indices))
            # Draw Polygons
            frame = OpenBoll_polygons.draw(color_image, COLORS[0])
            frame = ClosedBoll_polygons.draw(frame, COLORS[1])
            frame = Flower_polygons.draw(frame, COLORS[2])
            frame = Squares_polygons.draw(frame, COLORS[3])

            # Get trackedBBoxes Indices
            OpenBoll_indices = [i for i, value in enumerate(data['frames'][count]['trackedClasses_wSort']) if value == 0]
            ClosedBoll_indices = [i for i, value in enumerate(data['frames'][count]['trackedClasses_wSort']) if value == 1]
            Flower_indices = [i for i, value in enumerate(data['frames'][count]['trackedClasses_wSort']) if value == 2]
            Squares_indices = [i for i, value in enumerate(data['frames'][count]['trackedClasses_wSort']) if value == 3]
            # Get trackedBBoxes
            OpenBoll_bboxes = list(data['frames'][count]['trackedBBoxes_wSort'][i] for i in OpenBoll_indices)
            ClosedBoll_bboxes = list(data['frames'][count]['trackedBBoxes_wSort'][i] for i in ClosedBoll_indices)
            Flower_bboxes = list(data['frames'][count]['trackedBBoxes_wSort'][i] for i in Flower_indices)
            Squares_bboxes = list(data['frames'][count]['trackedBBoxes_wSort'][i] for i in Squares_indices)
            # Draw trackedBBoxes
            index = 0
            for bboxes in [OpenBoll_bboxes, ClosedBoll_bboxes, Flower_bboxes, Squares_bboxes]:
                for bbox in bboxes:
                    frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), COLORS[index], 1)
                    cv2.putText(frame, f"{CLASSES[index]}", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[index], 1)               
                index += 1

            
            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("OpenBolls", data['frames'][count]['objectCount_wSort']['OpenBolls']),
                ("ClosedBolls", data['frames'][count]['objectCount_wSort']['ClosedBolls']),
                ("Flowers", data['frames'][count]['objectCount_wSort']['Flowers']),
                ("Squares", data['frames'][count]['objectCount_wSort']['Squares'])
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, h - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)
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
    
    # Save Dist Figure for Organ Count
    Classes=list(results['objectCount_wSort'].keys())
    Dist=list(results['objectCount_wSort'].values())
    fig = plt.figure()
    plt.bar(Classes, Dist)
    for index, value in enumerate(Dist):
        plt.text(index, value, str(value))
    plt.title('Organ Count')
    plt.savefig(os.path.join(args["inputFolder"], os.path.basename(args["inputFolder"]) + '_OrganCount.png'))
    plt.close(fig)

    # Save Dist Figure for Total Number of Organs Detected
    fig2 = plt.figure()
    Classes=list(results['numOfObjectsDetected'].keys())
    Dist=list(results['numOfObjectsDetected'].values())
    plt.bar(Classes, Dist)
    for index, value in enumerate(Dist):
        plt.text(index, value, str(value))
    plt.title('Total Number Of Organs Detected')
    plt.savefig(os.path.join(args["inputFolder"], os.path.basename(args["inputFolder"]) + '_TotalNumOfOrgans.png'))
    plt.close(fig2)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--fps", type=int, default=1,
        help="Video FPS")
    ap.add_argument("-i", "--inputFolder", type=str, required=True,
        help="Input Folder")
    args = vars(ap.parse_args())
    main()
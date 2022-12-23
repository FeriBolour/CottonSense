import numpy as np
import cv2
import sys
import warnings
import argparse
import pyrealsense2 as rs
import os
from imutils.video import FPS
import logging

def main():

    NAME = input('Please enter the name of the experiment:\n')

    os.makedirs(os.path.join(args['outputFolder'], NAME), exist_ok=True)

    # Log to stdout
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s"
    )

    # setup Realsense Camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args['width'], args['height'], rs.format.bgr8, args['fps'])
    config.enable_stream(rs.stream.depth, args['width'], args['height'], rs.format.z16, args['fps'])
    config.enable_record_to_file(os.path.join(args['outputFolder'], NAME, NAME + '_RealSenseData.bag'))
    pipeline.start(config)

    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]

    if args['exposure'] != 0:
        if args['exposure'] > 1 and args['exposure'] < 10000:
            sensor.set_option(rs.option.enable_auto_exposure, False)
            sensor.set_option(rs.option.exposure, args['exposure'])
            logging.info(f"Exposure level set to {args['exposure']}.")
        else:
            logging.info("Exposure level not in between 1 and 10000. Will continue using Auto-Exposure")

    if args['white_balance'] != 0:
        if args['white_balance'] > 2800 and args['white_balance'] < 6500:
            sensor.set_option(rs.option.enable_auto_white_balance, False)
            sensor.set_option(rs.option.white_balance, args['white_balance'])
            logging.info(f"White Balance level set to {args['white_balance']}.")
        else:
            logging.info("White Balance level not in between 2800 and 6500. Will continue using Auto-White-Balance")

    logging.info(f"Camera Stream has started!")

    # start the frames per second throughput estimator
    fps = FPS().start()

    cv2.namedWindow("RGB Stream", cv2.WINDOW_AUTOSIZE)

    frameCount = 0
    while True:
        # Capture frame-by-frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        color_image = np.asanyarray(color_frame.get_data())

        cv2.imshow("RGB Stream", color_image)

        # then update the FPS counter
        fps.update()
        frameCount += 1

        if cv2.waitKey(1) == ord("q"):
            break


    # When everything done, release the capture
    pipeline.stop()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outputFolder", type=str,
        help="path to optional output video folder")
    ap.add_argument("-w", "--width", type=int, default=1280,
        help="image width")
    ap.add_argument("-t", "--height", type=int, default=720,
        help="image height")
    ap.add_argument("-f", "--fps", type=int, default=15,
        help="Camera FPS")
    ap.add_argument("-e", "--exposure", 
        help="Exposure Level", 
        type=int, default=0)
    ap.add_argument("-b", "--white_balance", 
        help="White Balance Level", 
        type=int, default=0)
    args = vars(ap.parse_args())
    main()
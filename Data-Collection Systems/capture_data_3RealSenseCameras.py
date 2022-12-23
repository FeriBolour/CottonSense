import numpy as np
import cv2
import sys
import warnings
import time
import argparse
import pyrealsense2 as rs
import os
from imutils.video import FPS
import logging

def main():

    NAME = 'test3cameras'
    
    os.makedirs(os.path.join(args['outputFolder'], NAME), exist_ok=True)

    # Log to stdout
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(message)s"
    )

    # setup Realsense Cameras
    # ...from Camera 1
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()
    config_1.enable_device('044322070446')
    config_1.enable_stream(rs.stream.color, args['width'], args['height'], rs.format.bgr8, args['fps'])
    config_1.enable_stream(rs.stream.depth, args['width'], args['height'], rs.format.z16, args['fps'])
    config_1.enable_record_to_file(os.path.join(args['outputFolder'], NAME, NAME + '_Camera1_RealSenseData.bag'))
    # ...from Camera 2
    pipeline_2 = rs.pipeline()
    config_2 = rs.config()
    config_2.enable_device('034422073886')
    config_2.enable_stream(rs.stream.color, args['width'], args['height'], rs.format.bgr8, args['fps'])
    config_2.enable_stream(rs.stream.depth, args['width'], args['height'], rs.format.z16, args['fps'])
    config_2.enable_record_to_file(os.path.join(args['outputFolder'], NAME, NAME + '_Camera2_RealSenseData.bag'))
    # ...from Camera 3
    pipeline_3 = rs.pipeline()
    config_3 = rs.config()
    config_3.enable_device('044122070445')
    config_3.enable_stream(rs.stream.color, args['width'], args['height'], rs.format.bgr8, args['fps'])
    config_3.enable_stream(rs.stream.depth, args['width'], args['height'], rs.format.z16, args['fps'])
    config_3.enable_record_to_file(os.path.join(args['outputFolder'], NAME, NAME + '_Camera3_RealSenseData.bag'))

    pipeline_1.start(config_1)
    pipeline_2.start(config_2)
    pipeline_3.start(config_3)

    sensor1 = pipeline_1.get_active_profile().get_device().query_sensors()[1]
    sensor2 = pipeline_2.get_active_profile().get_device().query_sensors()[1]
    sensor3 = pipeline_3.get_active_profile().get_device().query_sensors()[1]

    if args['exposure'] != 0:
        if args['exposure'] > 1 and args['exposure'] < 10000:
            sensor1.set_option(rs.option.enable_auto_exposure, False)
            sensor1.set_option(rs.option.exposure, args['exposure'])
            sensor2.set_option(rs.option.enable_auto_exposure, False)
            sensor2.set_option(rs.option.exposure, args['exposure'])
            sensor3.set_option(rs.option.enable_auto_exposure, False)
            sensor3.set_option(rs.option.exposure, args['exposure'])
            logging.info(f"Exposure level set to {args['exposure']}.")
        else:
            logging.info("Exposure level not in between 1 and 10000. Will continue using Auto-Exposure")

    if args['white_balance'] != 0:
        if args['white_balance'] > 2800 and args['white_balance'] < 6500:
            sensor1.set_option(rs.option.enable_auto_white_balance, False)
            sensor1.set_option(rs.option.white_balance, args['white_balance'])
            sensor2.set_option(rs.option.enable_auto_white_balance, False)
            sensor2.set_option(rs.option.white_balance, args['white_balance'])
            sensor3.set_option(rs.option.enable_auto_white_balance, False)
            sensor3.set_option(rs.option.white_balance, args['white_balance'])
            logging.info(f"White Balance level set to {args['white_balance']}.")
        else:
            logging.info("White Balance level not in between 2800 and 6500. Will continue using Auto-White-Balance")

    logging.info(f"Camera Stream has started!")

    # start the frames per second throughput estimator
    fps = FPS().start()
    
    window4 = np.zeros((args['height'], args['width'], 3), dtype=np.uint8)
    cv2.namedWindow("RGB Stream", cv2.WINDOW_NORMAL)

    frameCount = 0
    try:
        while True:
            # Capture frame-by-frame
            frames1 = pipeline_1.wait_for_frames()
            color_frame1 = frames1.get_color_frame()

            frames2 = pipeline_2.wait_for_frames()
            color_frame2 = frames2.get_color_frame()

            frames3 = pipeline_3.wait_for_frames()
            color_frame3 = frames3.get_color_frame()

            # if (not color_frame1) or (not color_frame2) or (not color_frame3):
            #     continue
            
            color_image1 = np.asanyarray(color_frame1.get_data())
            color_image2 = np.asanyarray(color_frame2.get_data())
            color_image3 = np.asanyarray(color_frame3.get_data())

            images = np.vstack((np.hstack((color_image1, color_image2)),
                                np.hstack((color_image3, window4))))
            cv2.imshow("RGB Stream", images)

            # then update the FPS counter
            fps.update()
            frameCount += 1

            if cv2.waitKey(1) == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    # When everything done, release the capture
    pipeline_1.stop()
    pipeline_2.stop()
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
import libtmux
import os
import keyboard
import time
from Open_terminals import open_terminals
from Editlaunchfile_function import editlaunchfile
from Defaultlaunchfile_function import defaultlaunchfile

# OPEN TERMINALS (opens 2 terminals that can be controlled with keyboard commands)
Panes = open_terminals()

# Default count value (required to choose autoexposure only once at the beginning of scanning session)
count = 1

# Edits launchfile name (user given name)
#Filenames = editlaunchfile()

#Given_name = Filenames[3] # User given name

database_path = 'somepath' # This needs to be defaulted to a path to avoid retyping every time!!! FOR SAVING + FOR LAUNCH FILES!!!

# THIS IS A GOOD IDEA TO KEEP TRACK OF ALL THE FILES BEING SCANNED!!! OR IT COULD BE POSTPROCESSED (IF TRUSTED TO BE NOT FORGOTTEN LATER)
# Create New Main Folder
# Create 2 sub folders inside Main Folder (for .db and .bag)

# Find and Set White Balance Level
WB_input = str(input("Would You like to change white balance levels? 'y' or 'n': "))
if WB_input == 'y' or WB_input == 'Y':
    while True:
        Panes[0].send_keys("roslaunch realsense2_camera opensource_tracking.launch")
        time.sleep(5)
        Panes[1].send_keys("rosrun dynamic_reconfigure dynparam set /camera/rgb_camera enable_auto_white_balance False")
        WB_input = str(input("Enter your white balance level: "))
        Panes[1].send_keys(f"rosrun dynamic_reconfigure dynparam set /camera/rgb_camera white_balance {WB_input}")
        WB = str(input("Is the white balance good? 'y' or 'n': "))
        if WB == 'y' or WB == 'Y':
            break
        else:
            continue
# Stop Process in Terminal # 1
Panes[1].send_keys('C-c', enter=False, suppress_history=False)
Panes[0].send_keys('C-c', enter=False, suppress_history=False)

# Camera Parameter change Commands
change_param = 'rosrun dynamic_reconfigure dynparam set /camera/rgb_camera '
enable_auto_exposure = 'enable_auto_exposure '
auto_exposure_argument = input('Run using Auto Exposure ("y" or "n")?: ')
while True:
    if auto_exposure_argument == 'y':
        auto_exposure_input = 'True'
        break
    elif auto_exposure_argument == 'n':
        auto_exposure_input = 'False'
        exposure_input= str(input('Enter the desired camera exposure level: '))
        exposure_level = 'exposure ' + exposure_input 
        print(exposure_level)
        break
    else:
        print('Error: Need a valid input --> "y" or "n" ')
        auto_exposure_argument = input('Run using Auto Exposure ("y" or "n")?: ')

#print('Outside while loop')    
#auto_exposure_input = 'False' # default (can be changed to True if desired) # Need adjustments!!! (give the ability to use autoexposure --> True or False)
#exposure_input= str(input('Enter the desired camera exposure level: ')) # Need adjustments!!! If 'True' don't let the user give input
#exposure_level = 'exposure ' + exposure_input # Need adjustments!!! skip this if 'True'
#print(exposure_level)

# rosbag record -O Row1.bag /camera/aligned_depth_to_color/camera_info  camera/aligned_depth_to_color/image_raw

# Commands to save Rosbag file
rosbag_record = 'rosbag record -o /home/avl/Documents/BagFiles/'
rosbag_data = '.bag /camera/aligned_depth_to_color/camera_info  camera/aligned_depth_to_color/image_raw /camera/color/camera_info /camera/color/image_raw /camera/imu /camera/imu_info /tf_static'

# Commands to start roslaunch
launch_ros = 'roslaunch'

# Commands to stop the process
    # ctrl + C

while True:

    # NEED TO ASK THE NAME OF THE .db file at every restart (space button push)!!!

    # count = count + 1 # NOT REQUIRED!
    # newfilename = myfilename + '_' + str(count) # NOT REQUIRED! ASK FOR USER INPUT EVERYTIME!!!
    # print(newfilename)


    #print('Inside the main while loop')
    print("Please click ctrl to continue Scanning or esc to exit session!")
    # WAITS FOR AN INPUT FROM KEYBOARD EVERYTIME!!!
    input_3 = keyboard.read_key()
    
    # If ESC BUTTON IS PUSHED EXIT THE CODE
    if input_3 == 'esc':
        #Panes[1].send_keys('C-b',enter=False, suppress_history=False)
        #Panes[1].send_keys('x', enter=True)
        #Panes[1].send_keys('y', enter=True)
        #Panes[0].send_keys('C-b',enter=False, suppress_history=False)
        #Panes[0].send_keys('x', enter=True)
        # Panes[0].send_keys('y', enter=True)
        os.system("tmux kill-server")
        print('Exiting Code!')
        # SAVE METADATAFILE BEFORE EXITING THE CODE!!!
        break
    # IF SPACE BUTTON IS PUSHED START THE SCAN
    elif input_3 == 'ctrl':
        # Edits launchfile name (user given name)
        Filenames = editlaunchfile()
        print(Filenames[0])
        #print(Filenames[4])
        Given_name = Filenames[3] # User given name
        print(Given_name)

        # Run ROS LAUNCH (Terminal # 1)
        roslaunch_command = launch_ros + ' ' + Filenames[0]
        print(roslaunch_command)
        Panes[1].send_keys(roslaunch_command, enter=True) # COMMAND TO START THE SCANNING SESSION # roslaunch /home/avl/Desktop/LaunchFiles/Camera1_with_IMU.launch

        time.sleep(7)  # Wait 7 seconds to give ROS Launch enough time to start

        # Turn off Auto Exposure and set exposure level (Terminal # 2)
        if count == 1:  # If you want to execute it everytime, remove this if line (maybe keep this for setting autoexpose only once? still does not require count update but, needs a default count value == 1)
            if auto_exposure_argument == 'n':
                set_auto_exposure = change_param + enable_auto_exposure + auto_exposure_input  # Turn off auto exposure (give more options with if and elif for autoexposure enabled vs disabled)
                set_exposure_level = change_param + exposure_level  # Set exposure level
                Panes[0].send_keys(set_auto_exposure, enter=True)
                time.sleep(7)
                Panes[0].send_keys(set_exposure_level, enter=True)
            # elif auto_exposure_argument == 'y':
            #     set_auto_exposure = change_param + enable_auto_exposure + auto_exposure_input 
            #     Panes[0].send_keys(set_auto_exposure, enter=True)
            #     time.sleep(7)


        # Start Recording Rosbag file
        record_rosbagfile = rosbag_record + Given_name + rosbag_data # Command to start recording data into a rosbag file
        Panes[0].send_keys(record_rosbagfile, enter=True) # START SAVING THE ROSBAG FILE!!!

        # WAIT FOR A KEYBOARD PUSH (SPACE)
        keyboard.wait('ctrl')

        # GET THE KEYBOARD VARIABLE NAME TO GIVE COMMANDS => if input_2 = 'space' STOP BOTH TERMINAL-1 AND TERMINAL-2 PROCESSES
        input_2 = keyboard.get_hotkey_name()
        
        if input_2 == 'ctrl':
            
            # Stop Process in Terminal # 1
            Panes[1].send_keys('C-c', enter=False, suppress_history=False)
            time.sleep(3)
            
            # Stop Process in Terminal # 2
            Panes[0].send_keys('C-c', enter=False, suppress_history=False)
            time.sleep(1)
            
            # Brings the edited launcfilename back to its default settings ('Choosename.db')
            defaultlaunchfile(Filenames[0], Filenames[1], Filenames[2])
            time.sleep(5)
            for root, dirs, files in os.walk('/home/avl/Documents/BagFiles'):
                for file in files:
                    if file.startswith(Given_name):
                        os.system(f"chmod 777 /home/avl/Documents/BagFiles/{file}")
                        os.rename(f"/home/avl/Documents/BagFiles/{file}", f"/home/avl/Documents/BagFiles/{Given_name}.bag")
            for root, dirs, files in os.walk('/home/avl/Documents/RTAB-Map'):
                for file in files:
                    if file.startswith(Given_name):
                        os.system(f"chmod 777 /home/avl/Documents/RTAB-Map/{file}")

    # IF ANY BUTTON OTHER THAN 'SPACE' OR 'ESC' IS PUSHED, PRINT A WARNING!!! IF BUTTONS ARE PUSHED BY MISTAKE DURING THE SCANNING SESSION!!!
    else:
        print('Wrong Key! Try again!')
        time.sleep(1)

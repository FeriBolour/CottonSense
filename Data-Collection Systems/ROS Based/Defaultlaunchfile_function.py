def defaultlaunchfile(Full_path_launch_file, Full_path_launch_file_new_format, Full_Given_name):
    
    # ATTENTION! WARNING! Global variables are used from editlaunchfile() function! (defaultlaunchfile should be entered after editlaunchfile becaus they are dependent on each other)

    # Change the name of the database (.db) back to default "Choosename.db" when exiting the scannning session from the previously "user given name"

    # Change file extension from .launch to .txt to edit the file
    import os 

    space = ' '
    quote = "'"
    new_terminal_command = "gnome-terminal -- /bin/sh -c "
    os_command = 'mv '
    full_command = new_terminal_command + quote + os_command + Full_path_launch_file + space + Full_path_launch_file_new_format + quote
    #os.system("gnome-terminal -- /bin/sh -c 'mv /home/avl/Desktop/LaunchFiles/Camera1_with_IMU.launch /home/avl/Desktop/LaunchFiles/Camera1_with_IMU.txt' ") --> Full command example
    os.system(full_command)

    # Edit the .txt file as desired (main goal: change the database name in the launch file automatically)
    with open(Full_path_launch_file_new_format,'r') as f:
        contents = f.read()

    contents = contents.replace(Full_Given_name,'Choosename.db') # Default back to Choosename.db

    with open(Full_path_launch_file_new_format,'w') as f:
        f.write(contents)

    # Change the extension back to .launch again (from .txt)
    full_command = new_terminal_command + quote + os_command + Full_path_launch_file_new_format + space + Full_path_launch_file + quote
    #os.system("gnome-terminal -- /bin/sh -c 'mv /home/avl/Desktop/LaunchFiles/Camera1_with_IMU.launch /home/avl/Desktop/LaunchFiles/Camera1_with_IMU.txt' ") --> Full command example
    os.system(full_command)
def editlaunchfile():
    # This function will help change the launch file 'database name' automatically within the specified launch file while scanning... 
    # ...The user has to input the desired 'database name' for each scanning session and has to specify the 'launch file name'

    # ATTENTION! WARNING! The launch file 'database name' has to be 'Choosename.db' for this code to work properly. If that's not the case, change the 'database name' to 'Choosename.db' and save the launch file before running this script
    # ATTENTION! WARNING! The path to launch files is assumed to be the path to 'Original_path' variable and it has to be the same for all launch files. If that's not the case, please create a default path with...
    # ...with all launch files inside and change the 'Original_path' variable to the new default path
    import os

    Original_path = '/home/avl/Desktop/LaunchFiles/' # This path file is always assumed to be true (to avoid entering the path --> otherwise no reason to automate)

    Launch_file_name = str(input('Launch file name:'))
    Launch_file_format = '.launch'
    New_format = '.txt'
    space = ' '
    quote = "'"
    Full_path_launch_file = Original_path + Launch_file_name + Launch_file_format
    Full_path_launch_file_new_format = Original_path + Launch_file_name + New_format

    # Change file extension from .launch to .txt to edit the file
    new_terminal_command = "gnome-terminal -- /bin/sh -c "
    os_command = 'mv '
    full_command = new_terminal_command + quote + os_command + Full_path_launch_file + space + Full_path_launch_file_new_format + quote
    #os.system("gnome-terminal -- /bin/sh -c 'mv /home/avl/Desktop/LaunchFiles/Camera1_with_IMU.launch /home/avl/Desktop/LaunchFiles/Camera1_with_IMU.txt' ") --> Full command example
    os.system(full_command)

    # Edit the .txt file as desired (main goal: change the database name in the launch file automatically)
    with open(Full_path_launch_file_new_format,'r') as f:
        contents = f.read()

    Given_name = str(input('Enter database file name (.db): '))
    Given_name_format = '.db'
    Full_Given_name = Given_name + Given_name_format

    contents = contents.replace('Choosename.db', Full_Given_name) # Choosename.db is always assumed to be true within the specified launch file (if not edit launch file to Choosename.db)

    with open(Full_path_launch_file_new_format,'w') as f:
        f.write(contents) 

    full_command = new_terminal_command + quote + os_command + Full_path_launch_file_new_format + space + Full_path_launch_file + quote
    #os.system("gnome-terminal -- /bin/sh -c 'mv /home/avl/Desktop/LaunchFiles/Camera1_with_IMU.launch /home/avl/Desktop/LaunchFiles/Camera1_with_IMU.txt' ") --> Full command example
    os.system(full_command)

    return [Full_path_launch_file, Full_path_launch_file_new_format, Full_Given_name,Given_name]
    
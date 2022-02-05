import tesserocr
from tesserocr import PyTessBaseAPI, get_languages, PSM, OEM, RIL
tesserocr.PyTessBaseAPI(path='C:\\Program Files\\Tesseract-OCR\\tessdata\\')
import cv2
import ctypes
import numpy as np
import time
from minecraft import *


def move_mouse(x, y):
    ctypes.windll.user32.mouse_event(0x01, x, y, 0, 0)


def scan_63(player: MinecraftPlayer):
    angles = [
        # Row 1
        Vector2(0, 60),
        Vector2(47, 50),
        Vector2(65, 37),
        Vector2(70, 22),
        Vector2(70, 3),

        # Row 2
        Vector2(90, 60),
        Vector2(90, 40),
        Vector2(90, 22),
        Vector2(90, 3),

        # Row 3
        Vector2(-179, 60),
        Vector2(135, 50),
        Vector2(117, 37),
        Vector2(112, 22),
        Vector2(112, 3),
    ]

    for angle in angles:
        player.add_rotation_to_queue(angle)


def scan_333(player: MinecraftPlayer):
    """
    From the current coordinate, look a the 3x3 cube surrounding the player.
    """
    y_levels = [90, 60, 35, 0, -30, -55, -90]
    x_levels = [0, 30, 50, 90, 120, 144, 180, -140, -188, -90, -50, -30, 0]


    ey_levels = [50, -45]
    ex_levels = [45, 135, -135, -44]

    # Handle front back left right
    for x in x_levels:
        for y in y_levels:
            player.add_rotation_to_queue(Vector2(x, y))

    for x in ex_levels:
        for y in ey_levels:
            player.add_rotation_to_queue(Vector2(x, y))


def walk_fill_square(player: MinecraftPlayer, n):
    """
    Walk over the filled square from current pos to n
    """
    origin = Vector3(0.5, -60, 0.5)

    player.add_coordinates_to_queue(origin)

    # Look down
    player.add_rotation_to_queue(Vector2(0, 0))

    
    for i in range(n):
        for j in range(n):
            player.add_coordinates_to_queue(Vector3(origin.x + i, origin.y, origin.z + j))
            player.add_rotation_to_queue(Vector2(0, 90))
            player.add_rotation_to_queue(Vector2(0, 0))

    player.add_rotation_to_queue(Vector2(0, 0))

def walk_square(player: MinecraftPlayer, n):
    origin = Vector3(1.5, -60, -27.5)

    player.add_coordinates_to_queue(origin)

    player.add_coordinates_to_queue(Vector3(origin.x, origin.y, origin.z + n))
    player.add_coordinates_to_queue(Vector3(origin.x - n, origin.y, origin.z + n))
    player.add_coordinates_to_queue(Vector3(origin.x - n, origin.y, origin.z))
    player.add_coordinates_to_queue(Vector3(origin.x, origin.y, origin.z))


def make_square(player: MinecraftPlayer, n):
    """
    Constructs an nxn square.
    """
    assert n > 2
    player.add_rotation_to_queue(Vector2(0, 0))
    player.add_coordinates_to_queue(Vector3(0.5,-60,0.5))

    for i in range(0, n+1):
        player.add_coordinates_to_queue(Vector3(i, 0, 0))
        player.add_rotation_to_queue(Vector2(0, 50))
        player.add_click_to_queue("right")

    player.add_coordinates_to_queue(Vector3(n+1, 0, 0))
    player.add_coordinates_to_queue(Vector3(n+1, 0, 2))

    for i in range(2, n+1):
        if i == 2:
            i += 1
        player.add_coordinates_to_queue(Vector3(n+1, 0, i))
        player.add_rotation_to_queue(Vector2(90, 50))
        player.add_click_to_queue("right")
    
    player.add_coordinates_to_queue(Vector3(n+1, 0, n+1))
    player.add_coordinates_to_queue(Vector3(n-1, 0, n+1))

    for i in range(2, n+1):
        player.add_coordinates_to_queue(Vector3(n-i, 0, n+1))
        player.add_rotation_to_queue(Vector2(180, 50))
        player.add_click_to_queue("right")

    player.add_coordinates_to_queue(Vector3(-1, 0, n+1))
    player.add_coordinates_to_queue(Vector3(-1, 0, n-1))

    for i in range(2, n):
        if i == 2:
            i = 1.5
        player.add_coordinates_to_queue(Vector3(-1, 0, n-i))
        player.add_rotation_to_queue(Vector2(-90, 50))
        player.add_click_to_queue("right")
    


def start(shared_map, shared_lock, running):
    bb_coords = (45, 210, 400, 230)
    bb_rotation = (75, 265, 550, 285)
    bb_block_coords = (2560-400, 370, 2560, 391)
    bb_block_type = (2560-500, 393, 2560, 411)


    #bb_coords = (45, 190, 400, 210)
    #bb_rotation = (75, 245, 550, 265)
    #bb_block_coords = (2560-400, 370, 2560, 391)
    #bb_block_type = (2560-500, 393, 2560, 411)

    #bb_block_type = (2560-250, 393, 2560, 412)


    player = MinecraftPlayer(bb_coords, bb_rotation, bb_block_coords, bb_block_type, shared_map, shared_lock)
    print(f"Loaded Languages:\n", get_languages('C:\\Program Files\\Tesseract-OCR\\tessdata\\'))

    with PyTessBaseAPI(lang='mc', psm=13, oem=3) as api:
        api.SetVariable("load_freq_dawg", "false")
        api.SetVariable("load_system_dawg", "false")

        time.sleep(4)
        print("Starting")

        player.add_coordinate_to_queue(Vector3(30, -60, 30))
        player.add_coordinate_to_queue(Vector3(40, -60, 30))
        player.add_coordinate_to_queue(Vector3(40, -60, 40))
        player.add_coordinate_to_queue(Vector3(30, -60, 40))
        player.add_coordinate_to_queue(Vector3(30, -60, 30))
        
    
        # used to record the time when we processed last frame
        prev_frame_time = 0
        
        # used to record the time at which we processed current frame
        new_frame_time = 0

        # Run forever unless you press Esc
        while running.is_set():

            success = player.update(api)

            height, width = player.current_position_image.shape
            resized_rotation_image = cv2.resize(player.current_rotation_image, (width, height))
            resized_block_position_image = cv2.resize(player.current_block_position_image, (width, height))
            resized_block_type_image = cv2.resize(player.current_block_type_image, (width, height))
            
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
        
            # converting the fps into integer
            fps = int(fps)
        
            # converting the fps to string so that we can display it on frame
            # by using putText function
            fps = "fps: "+ str(fps)
            empty = np.zeros((height, width))
            cv2.putText(empty, fps, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            images = (empty, player.current_position_image, resized_rotation_image, resized_block_position_image, resized_block_type_image)

            frame = np.concatenate(images, axis=0)

            #cv2.imshow("player", frame)

            # This line will break the while loop when you press Esc
            #if cv2.waitKey(1) == 27:
                #break

    # This will make sure all windows created from cv2 is destroyed
    #cv2.destroyAllWindows()
from math import ceil, cos, sin, radians
import tesserocr
from tesserocr import PyTessBaseAPI, get_languages, PSM, OEM, RIL
tesserocr.PyTessBaseAPI(path='C:\\Program Files\\Tesseract-OCR\\tessdata\\')
# OCR Screen Scanner
# By Dornu Inene
# Libraries that you show have all installed
import cv2
import ctypes
import numpy as np
import re
import time
import string
import pydirectinput

# We only need the ImageGrab class from PIL
from PIL import ImageGrab, Image



"""Transforms the value x from the input range to the output range."""
def linmap(x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def process_image(im):
    """
    Converts the image to a numpy array, then applies preprocessing.
    """
    im_arr = np.array(im)
        
    height, width, depth = im_arr.shape

    for i in range(height):
        for j in range(width):
            r, g, b = im_arr[i][j]

            r = (r + 150) / 2
            g = (g + 150) / 2
            b = (b + 150) / 2

            mean = (r + g + b) / 3
            diffr = abs(mean - r)
            diffg = abs(mean - g)
            diffb = abs(mean - b)

            maxdev = 2

            if (diffr + diffg + diffb) > maxdev:
                im_arr[i][j][0] = 0
                im_arr[i][j][1] = 0
                im_arr[i][j][2] = 0
            


    im_arr = cv2.cvtColor(im_arr, cv2.COLOR_BGR2GRAY)

        #cap_arr = cv2.threshold(cap_arr,127,255,cv2.THRESH_BINARY)
    
    # Otsu's thresholding after Gaussian filtering
    #blur = cv2.GaussianBlur(cap_arr,(3,3),0)
    ret3, im_arr = cv2.threshold(im_arr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return im_arr


def image_to_text(api, image):
    """
    Returns the text and the processed image as a numpy array.
    """
    image_array = process_image(image)
    image = Image.fromarray(np.uint8(image_array))
    api.SetImage(image)
    return api.GetUTF8Text(), image_array


def press_key_for_t(key, t):
    start_time = time.time()
    pydirectinput.keyDown(key)
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= t:
            pydirectinput.keyUp(key)
            break

class Vector3:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

    def reassign(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def magnitude(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5

    def subtract(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    
    def rotate_about_origin_xy(self, origin, angle):
        angle = radians(angle)
        x = self.x - origin.x
        z = self.z - origin.z
        tx = x * cos(angle) - z * sin(angle)
        tz = x * sin(angle) + z * cos(angle)
        tx += origin.x
        tz += origin.z

        return Vector3(tx, self.y, tz)


    def __repr__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"

class Vector2:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def reassign(self, x, y):
        self.x = x
        self.y = y

    def magnitude(self):
        return (self.x**2 + self.y**2)**0.5

    def subtract(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


class Block:
    def __init__(self, position: Vector3, block_type) -> None:
        self.position = position
        self.type = block_type

    def __hash__(self) -> int:
        return hash((self.position.x, self.position.y, self.position.z))


class Map:
    def __init__(self, init_with_blocks=None) -> None:
        self.current_map = {}
        if init_with_blocks != None:
            self.current_map.update({x for x in init_with_blocks})
    
    def get_all_blocks_filtered(self, predicate):
        """
        predicate is a function that takes in a string, if the function returns True, the block is kept.
        If False, the block is removed. Does not alter map, just alters the returned blocks from this call.
        """
        return {x for x in self.current_map if predicate(x)}


class MinecraftPlayer:
    def __init__(self, bb_coords, bb_rotation, bb_block_coords, bb_block_type) -> None:
        self.bb_coords = bb_coords
        self.bb_rotation = bb_rotation
        self.bb_block_coords = bb_block_coords
        self.bb_block_type = bb_block_type

        self.coordinate_regex = re.compile(r'([+-]?\d+\.\d+)/([+-]?\d+\.\d+)/([+-]?\d+\.\d+)')
        self.rotation_regex = re.compile(r'[a-zA-Z]+\([a-zA-Z]+\)\(([+-]?\d+\.\d+)/([+-]?\d+\.\d+)\)')
        self.target_block_position_regex = re.compile(r'(\d), (\d), (\d)')
        self.target_block_type_regex = re.compile(r'minecraft\:(.*?)')
        self.position = Vector3(0, 0, 0)
        self.prev_position = None
        self.rotation = Vector2(0, 0)
        self.prev_rotation = None
        self.time = 0
        self.prev_time = None

        self.max_speed_tolerance = 10
        self.max_rotation_tolerance = 720

        self.current_position_image = None
        self.current_rotation_image = None
        self.current_block_position_image = None
        self.current_block_type_image = None

        self.map = Map()

        self.action_queue = []

        self.rotation_action_queue = []

        self.movement_action_queue = []

    
    def add_rotation_to_queue(self, position: Vector2):
        self.action_queue.append({"type": "rotation", "value": position})

    def add_movement_to_queue(self, units_forward: float):
        self.action_queue.append({"type": "movement", "value": {"units": units_forward, "original": None, "keydown": "None", "slow": False}})

    def add_coordinates_to_queue(self, position: Vector3):
        self.action_queue.append({"type": "coordinate", "value": {"coord": position, "axis": "forward", "keydown": "None", "slow": False, "original": None} })

    def add_click_to_queue(self, msb):
        self.action_queue.append({ "type": "click", "value": msb})

    def serve_action(self):
        if len(self.action_queue) <= 0:
            return
        
        action = self.action_queue[0]

        if action["type"] == "rotation":
            result = self.serve_rotation()
            if result:
                self.action_queue.pop(0)
        elif action["type"] == "movement":
            result = self.serve_movement()
            if result:
                self.action_queue.pop(0)
        elif action["type"] == "coordinate":
            result = self.serve_coordinates()
            if result:
                self.action_queue.pop(0)
        elif action["type"] == "click":
            result = self.serve_click()
            if result:
                self.action_queue.pop(0)
        else:
            raise Exception(f"Unkown action {action}")
        


    def serve_click(self):
        click = self.action_queue[0]["value"]
        if click == "right":
            pydirectinput.rightClick(duration=0.1)
            print("righclicked")
        elif click == "left":
            pydirectinput.leftClick(duration=0.1)
            print("righclicked")
        
        return True

    def serve_coordinates(self):
        """
        Need to define a control scheme:
        w, a, s, d
        ctrl + (w, a ,s, d)
        
        Given this control scheme, a players current coords, current orientation, go to desired coordinates (expected that no blocks block the way)

        Move forward (w) until the left and right (a,d) perpendicular axis lines up with the point then move along that axis.

        """
        desired_position = self.action_queue[0]["value"]["coord"]

        if self.action_queue[0]["value"]["original"] is None:
            self.action_queue[0]["value"]["original"] = Vector3(self.position.x, self.position.y, self.position.z)

        curr_position = self.position

        curr_rotation = self.rotation

        if curr_rotation.x < 0:
            curr_rotation.x = 180 + (180 + curr_rotation.x)

        desired_position_rotated = desired_position.rotate_about_origin_xy(self.action_queue[0]["value"]["original"], -curr_rotation.x)
        curr_position_rotated = curr_position.rotate_about_origin_xy(self.action_queue[0]["value"]["original"], -curr_rotation.x)

        print(f"Moving {curr_position_rotated} -> {desired_position_rotated} Difference {desired_position_rotated.subtract(curr_position_rotated)}")

        forwards = "s"
        backwards = "d"
        left = "a"
        right = "d"


        error_tolerance = 1

        # Step 1:
        # Move forward or backwards until x axis aligns with desired_position_rotated.x
        if self.action_queue[0]["value"]["axis"] == "forward":
            difference = desired_position_rotated.z - curr_position_rotated.z

            if abs(difference) < 2:
                if not self.action_queue[0]["value"]["slow"]:
                    self.action_queue[0]["value"]["slow"] = True
                    pydirectinput.keyDown("ctrl")
            else:
                if self.action_queue[0]["value"]["slow"]:
                    pydirectinput.keyUp("ctrl")
                    self.action_queue[0]["value"]["slow"] = False

            

            if abs(difference) > error_tolerance:
                if difference < 0:
                    # Move foward
                    if self.action_queue[0]["value"]["keydown"] != forwards:
                        if self.action_queue[0]["value"]["keydown"] != "None":
                            pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                    pydirectinput.keyDown(forwards)
                    self.action_queue[0]["value"]["keydown"] = forwards

                elif difference > 0:
                    # move backwards
                    if self.action_queue[0]["value"]["keydown"] != backwards:
                        if self.action_queue[0]["value"]["keydown"] != "None":
                            pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                    pydirectinput.keyDown(backwards)
                    self.action_queue[0]["value"]["keydown"] = backwards
            
            else:
                if self.action_queue[0]["value"]["keydown"] != "None":
                    pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                    self.action_queue[0]["value"]["keydown"] = "None"

                if self.action_queue[0]["value"]["slow"]:
                    self.action_queue[0]["value"]["slow"] = False
                    pydirectinput.keyUp("ctrl")
                self.action_queue[0]["value"]["axis"] = "right"
                print(f"Aligned forward axis")

        # Step 2:
        # Move left or right until y axis aligns with desired_position_rotated.y
        else:
            difference = desired_position_rotated.x - curr_position_rotated.x

            if abs(difference) < 2:
                if not self.action_queue[0]["value"]["slow"]:
                    self.action_queue[0]["value"]["slow"] = True
                    pydirectinput.keyDown("ctrl")
            else:
                if self.action_queue[0]["value"]["slow"]:
                    pydirectinput.keyUp("ctrl")
                    self.action_queue[0]["value"]["slow"] = False

            if abs(difference) > error_tolerance:
                if difference < 0:
                    # Move left
                    if self.action_queue[0]["value"]["keydown"] != right:
                        if self.action_queue[0]["value"]["keydown"] != "None":
                            pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                    pydirectinput.keyDown(right)
                    self.action_queue[0]["value"]["keydown"] = right

                elif difference > 0:
                    # move right
                    if self.action_queue[0]["value"]["keydown"] != left:
                        if self.action_queue[0]["value"]["keydown"] != "None":
                            pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                    pydirectinput.keyDown(left)
                    self.action_queue[0]["value"]["keydown"] = left
            
            else:
                if self.action_queue[0]["value"]["keydown"] != "None":
                    pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                    self.action_queue[0]["value"]["keydown"] = "None"

                if self.action_queue[0]["value"]["slow"]:
                    self.action_queue[0]["value"]["slow"] = False
                    pydirectinput.keyUp("ctrl")

                print(f"Aligned left axis")
                print(f"Done moving to coordinates {desired_position}, difference {difference}")
                return True

        return False


    def serve_movement(self):
        if self.action_queue[0]["value"]["original"] is None:
            self.action_queue[0]["value"]["original"] = Vector3(self.position.x, self.position.y, self.position.z)

        orig_position = self.action_queue[0]["value"]["original"]
        curr_position = self.position

        units_desired = self.action_queue[0]["value"]["units"]

        units_moved = curr_position.subtract(orig_position).magnitude()

        difference = units_desired - units_moved

        if abs(difference) < 2:
            if not self.action_queue[0]["value"]["slow"]:
                self.action_queue[0]["value"]["slow"] = True
                pydirectinput.keyDown("ctrl")
        else:
            if self.action_queue[0]["value"]["slow"]:
                pydirectinput.keyUp("ctrl")
                self.action_queue[0]["value"]["slow"] = False

        if abs(difference) <= 0.1:
            if self.action_queue[0]["value"]["keydown"] != "None":
                print("keyup")
                pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])

            if self.action_queue[0]["value"]["slow"]:
                pydirectinput.keyUp("ctrl")
                self.action_queue[0]["value"]["slow"] = False

            print(f"moved {units_moved} / {units_desired} from {orig_position} to {self.position}")
            print(f"Done moving {units_desired} units")
            return True
        else:
            print(f"moved {units_moved} / {units_desired} from {orig_position} to {self.position}")
            if difference > 0:
                if self.action_queue[0]["value"]["keydown"] != "w":
                    if self.action_queue[0]["value"]["keydown"] != "None":
                        pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                pydirectinput.keyDown("w")
                self.action_queue[0]["value"]["keydown"] = "w"
            else:
                if self.action_queue[0]["value"]["keydown"] != "s":
                    if self.action_queue[0]["value"]["keydown"] != "None":
                        pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                pydirectinput.keyDown("s")
                self.action_queue[0]["value"]["keydown"] = "s"

        return False

        
        
    def serve_rotation(self):
        desired_rotation = self.action_queue[0]["value"]
        curr_rotation = self.rotation

        # Convert to 360 degree version
        if curr_rotation.x < 0:
            curr_rotation.x = 180 + (180 + curr_rotation.x)
        

        #mouse_x = (desired_rotation.x - curr_rotation.x) % 360

        diff = ( desired_rotation.x - curr_rotation.x + 180 ) % 360 - 180
        mouse_x = diff + 360 if diff < -180 else diff
        
        mouse_y = curr_rotation.y - desired_rotation.y
        
        
        

        
        error_tol = 0.2

        if abs(mouse_x) < error_tol:
            mouse_x = 0
        if abs(mouse_y) < error_tol:
            mouse_y = 0

        mx = linmap(mouse_x, -360, 360, -1500, 1500)
        my = linmap(mouse_y, -90, 90, -600, 600)

        if abs(mouse_x) < error_tol and abs(mouse_y) < error_tol:
            print(f"Done rotating to {desired_rotation}")
            return True
        else:
            mx = ceil(mx)
            my = ceil(my)
            print(f"moving mouse: {(int(mx), int(my))}")
            ctypes.windll.user32.mouse_event(0x01, int(mx), int(-my), 0, 0)
        return False

    
    def update(self, api):
        """
        Grab all new text, update time, then update pos, rot, block, etc...
        """
        # Position
        api.SetVariable("tessedit_char_whitelist", "./-0123456789")
        im_position = ImageGrab.grab(bbox=self.bb_coords)
        pos_text, self.current_position_image = image_to_text(api, im_position)

        # Rotation
        api.SetVariable("tessedit_char_whitelist", "./-()" + string.digits + string.ascii_letters.replace('S', ""))
        im_rotation = ImageGrab.grab(bbox=self.bb_rotation)
        rot_text, self.current_rotation_image = image_to_text(api, im_rotation)

        self.time = time.time()
        if self.prev_time is None:
                self.prev_time = self.time
        coord = self.update_coords(pos_text)
        rot = self.update_rotation(rot_text)
        self.prev_time = self.time

        if coord and rot:
            # Go through actions
            self.serve_action()

        return coord and rot

    
    def update_coords(self, pos_text):
        
        match = self.coordinate_regex.match(pos_text)

        if match:
            x, y, z = match.groups()
            self.position.reassign(float(x), float(y), float(z))

            if self.prev_position is None:
                self.prev_position = Vector3(self.position.x, self.position.y, self.position.z)

            dt = self.time - self.prev_time
            if dt == 0:
                dt = 0.00001
            dx = (self.position.x - self.prev_position.x) / dt
            dy = (self.position.y - self.prev_position.y) / dt
            dz = (self.position.z - self.prev_position.z) / dt

            speed = Vector3(dx, dy, dz)
            
            self.prev_position.reassign(self.position.x, self.position.y, self.position.z)

            if speed.magnitude() > self.max_speed_tolerance:
                print("Coord error, invalid speed")
            #print(f"\tCoords - X: {self.position.x}, Y: {self.position.y}, Z: {self.position.z}", f"\t\tSpeed - dx: {dx}, dy: {dy}, dz: {dz}, dt: {dt}")
        else:
            #print(f"[Error] Could not parse coordinates this frame")
            return False
        return True


    def update_rotation(self, rot_text):
        
        #print(f'Rotation: {rot_text}')

        match = self.rotation_regex.match(rot_text)

        if match:
            x, y = match.groups()
            self.rotation.reassign(float(x), float(y))

            if self.prev_rotation is None:
                self.prev_rotation = Vector2(self.rotation.x, self.rotation.y)

            dt = self.time - self.prev_time
            if dt == 0:
                dt = 0.00001
            dx = (self.rotation.x - self.prev_rotation.x) / dt
            dy = (self.rotation.y - self.prev_rotation.y) / dt

            speed = Vector2(dx, dy)
            
            self.prev_rotation.reassign(self.rotation.x, self.rotation.y)

            if speed.magnitude() > self.max_rotation_tolerance:
                print("Coord error, invalid rotation")
            #print(f"\tRotation - X: {self.rotation.x}, Y: {self.rotation.y}", f"\t\tSpeed - dx: {dx}, dy: {dy}, dt: {dt}")
        else:
            print(f"[Error] Could not parse rotation this frame {rot_text}")
            return False
        return True


def move_mouse(x, y):
    ctypes.windll.user32.mouse_event(0x01, x, y, 0, 0)




bb_coords = (45, 210, 400, 230)
bb_rotation = (75, 265, 550, 285)
bb_block_coords = (0, 0, 0, 0)
bb_block_type = (0, 0, 0, 0)


player = MinecraftPlayer(bb_coords, bb_rotation, bb_block_coords, bb_block_type)
print(f"Loaded Languages:\n", get_languages('C:\\Program Files\\Tesseract-OCR\\tessdata\\'))


with PyTessBaseAPI(lang='mc', psm=13, oem=3) as api:

    time.sleep(4)
    print("Starting")
    #player.add_rotation_to_queue(Vector2(-90, 0))
    #
    player.add_rotation_to_queue(Vector2(0, 0))
    player.add_movement_to_queue(20)
    player.add_rotation_to_queue(Vector2(-180, 0))
    player.add_movement_to_queue(20)


    # Run forever unless you press Esc
    while True:

        success = player.update(api)

        # Lets try moving the mouse to a desired direction (0, 0)
        




        #frame = np.vstack((player.current_position_image, player.current_rotation_image))
        height, width = player.current_position_image.shape
        resized_rotation_image = cv2.resize(player.current_rotation_image, (width, height))
        frame = np.concatenate((player.current_position_image, resized_rotation_image), axis=0)


        # cv2.imshow() shows a window display and it is using the image that we got
        # use array as input to image
        cv2.imshow("player", frame)
        

        # This line will break the while loop when you press Esc
        if cv2.waitKey(1) == 27:
            break

# This will make sure all windows created from cv2 is destroyed
cv2.destroyAllWindows()
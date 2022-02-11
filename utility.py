"""
Module handles all the utility functions and classes for minecraft.py
"""

from cmath import isclose
from math import ceil, cos, sin, radians
import cv2
import ctypes
import numpy as np
import re
import time
import string
import pydirectinput
pydirectinput.PAUSE = 0
import cython

from PIL import ImageGrab, Image


"""Transforms the value x from the input range to the output range."""
def linmap(x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def get_neighboring_blocks(block_position):
    """
    Returns an array of BlockRotation elements.
    """
    return [
                # Forward
                BlockRotation(Vector2(0,0), block_position.add(Vector3(0, 1, 1))),
                BlockRotation(Vector2(0,60), block_position.add(Vector3(0, 0, 1))),
                BlockRotation(Vector2(0,60), block_position.add(Vector3(0, -1, 1))),
                BlockRotation(Vector2(0,90), block_position.add(Vector3(0, -1, 0))),
                # Left
                BlockRotation(Vector2(-90, 0), block_position.add(Vector3(1, 1, 0))),
                BlockRotation(Vector2(-90, 60), block_position.add(Vector3(1, 0, 0))),
                BlockRotation(Vector2(-90, 60), block_position.add(Vector3(1, -1, 0))),
                # Backwards
                BlockRotation(Vector2(180, 0), block_position.add(Vector3(0, 1, -1))),
                BlockRotation(Vector2(180, 60), block_position.add(Vector3(0, 0, -1))),
                BlockRotation(Vector2(180, 60), block_position.add(Vector3(0, -1, -1))),
                # Right
                BlockRotation(Vector2(90, 0), block_position.add(Vector3(-1, 1, 0))),
                BlockRotation(Vector2(90, 60), block_position.add(Vector3(-1, 0, 0))),
                BlockRotation(Vector2(90, 60), block_position.add(Vector3(-1, -1, 0))),
            ]


def get_neighboring_blocks_dict(block_position):
    return {
        "forward": {
            "start": BlockRotation(Vector2(0,60), block_position.add(Vector3(0, -1, 1))),
            "rest": [
                BlockRotation(Vector2(0,0), block_position.add(Vector3(0, 1, 1))),
                BlockRotation(Vector2(0,60), block_position.add(Vector3(0, 0, 1))),
            ],
        },
        "left": {
            "start": BlockRotation(Vector2(-90, 60), block_position.add(Vector3(1, -1, 0))),
            "rest": [
                BlockRotation(Vector2(-90, 0), block_position.add(Vector3(1, 1, 0))),
                BlockRotation(Vector2(-90, 60), block_position.add(Vector3(1, 0, 0))),
            ],
        },
            
        "backwards": {
            "start": BlockRotation(Vector2(180, 60), block_position.add(Vector3(0, -1, -1))),
            "rest": [
                BlockRotation(Vector2(180, 0), block_position.add(Vector3(0, 1, -1))),
                BlockRotation(Vector2(180, 60), block_position.add(Vector3(0, 0, -1))),
            ],
        }, 
        "right": {
            "start": BlockRotation(Vector2(90, 60), block_position.add(Vector3(-1, -1, 0))),
            "rest": [
                BlockRotation(Vector2(90, 0), block_position.add(Vector3(-1, 1, 0))),
                BlockRotation(Vector2(90, 60), block_position.add(Vector3(-1, 0, 0))),
            ],
        },
    }


def v_filter_gray(rgb):
    # Expect len 3
    r, g, b = rgb

    r = (r + 150) / 2
    g = (g + 150) / 2
    b = (b + 150) / 2

    mean = (r + g + b) / 3
    diffr = abs(mean - r)
    diffg = abs(mean - g)
    diffb = abs(mean - b)

    maxdev = 2

    if (diffr + diffg + diffb) > maxdev:
        return 0
    return rgb



def filter_gray(na):
    mean = np.mean(na, axis=2, keepdims=True)

    res = (na[..., 2] / na[..., 0]) * 100/l
    return res.astype(np.uint8)


def process_image(im, crop_to_activity=False, crop_extra=0):
    """
    Converts the image to a numpy array, then applies preprocessing.
    """
    im_arr = np.array(im)
        
    height, width, depth = im_arr.shape

    hsv = cv2.cvtColor(im_arr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([179,2,255]))

    im_arr = cv2.bitwise_and(im_arr, im_arr, mask = mask) 

    #im_arr = np.apply_along_axis(v_filter_gray, 2, im_arr)


    """for i in range(height):
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
                im_arr[i][j][2] = 0"""
            
    #im_arr = cv2.cvtColor(im_arr, cv2.COLOR_HLS2BGR)

    im_arr = cv2.cvtColor(im_arr, cv2.COLOR_BGR2GRAY)

        #cap_arr = cv2.threshold(cap_arr,127,255,cv2.THRESH_BINARY)
    
    # Otsu's thresholding after Gaussian filtering
    #blur = cv2.GaussianBlur(cap_arr,(3,3),0)
    ret3, im_arr = cv2.threshold(im_arr,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    if crop_to_activity:
        positions = np.nonzero(im_arr)
        if len(positions) > 0:
            left = np.min(positions[1], initial=width-1)
            last_column = max(0, left+crop_extra)
            im_arr = im_arr[:, last_column:]
        """last_column = -1
        for j in range(width):
            for i in range(height):
                v = im_arr[i][j]

                if v != 0:
                    last_column = j
                    break
            if last_column != -1:
                break
        
        last_column = max(0, last_column+crop_extra)
        im_arr = im_arr[:, last_column:]"""
    
    im_arr = cv2.bitwise_not(im_arr)

    return im_arr


def image_to_text(api, image, crop_to_activity=False, crop_extra=0):
    """
    Returns the text and the processed image as a numpy array.
    """
    image_array = process_image(image, crop_to_activity, crop_extra)
    try:
        image = Image.fromarray(np.uint8(image_array))
    except:
        image_array = process_image(image, crop_to_activity)
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

class BlockRotation:
    def __init__(self, rotation, position) -> None:
        self.rotation = rotation
        self.position = position

    def copy(self):
        return BlockRotation(self.rotation.copy(), self.position.copy())

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

    def add(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def rotate_about_origin_xy(self, origin, angle):
        angle = radians(angle)
        x = self.x - origin.x
        z = self.z - origin.z
        tx = x * cos(angle) - z * sin(angle)
        tz = x * sin(angle) + z * cos(angle)
        tx += origin.x
        tz += origin.z

        return Vector3(tx, self.y, tz)
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Vector3):
            return isclose(self.x, other.x, rel_tol=1e-09, abs_tol=0.0) \
                and isclose(self.y, other.y, rel_tol=1e-09, abs_tol=0.0) \
                and isclose(self.z, other.z, rel_tol=1e-09, abs_tol=0.0)
            #return self.x == other.x and self.y == other.y and self.z == other.z
        return False
    
    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def copy(self):
        return Vector3(self.x, self.y, self.z)

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

    def copy(self):
        return Vector2(self.x, self.y)

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
        self.instantiated = False

    def __hash__(self) -> int:
        return hash((self.position.x, self.position.y, self.position.z))

    def __str__(self) -> str:
        return f"{self.type} - {self.position}"

    def __repr__(self) -> str:
        return f"{self.type} - {self.position}"
    
    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.position.x == other.position.x and self.position.y == other.position.y and self.position.z == other.position.z


class Map:
    def __init__(self, init_with_blocks=None, shared_map=None, shared_lock=None) -> None:
        self.current_map = {}
        self.shared_map = shared_map
        self.shared_lock = shared_lock
        if init_with_blocks != None:
            self.current_map.update({x for x in init_with_blocks})
    
    def get_all_blocks_filtered(self, predicate):
        """
        predicate is a function that takes in a string, if the function returns True, the block is kept.
        If False, the block is removed. Does not alter map, just alters the returned blocks from this call.
        """
        return {x for x in self.current_map if predicate(x)}

    
    def add_block(self, name, position: Vector3):
        self.current_map[position] = Block(position, name)
        with self.shared_lock:
            if self.shared_map is not None:
                block = Block(position, name)
                self.shared_map.add(block)



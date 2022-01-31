from ursina import *

from threading import Thread, Lock, Event
import datacollector
from minecraft import *


app = Ursina()

# Define a Voxel class.
# By setting the parent to scene and the model to 'cube' it becomes a 3d button.

class Voxel(Button):
    def __init__(self, position=(0,0,0), colorrgb=(255, 255, 255)):
        super().__init__(
            parent = scene,
            position = position,
            model = 'cube',
            origin_y = .5,
            texture = 'white_cube',
            color = color.rgb(colorrgb[0], colorrgb[1], colorrgb[1]),
            highlight_color = color.lime,
        )


    def input(self, key):
        if self.hovered:
            if key == 'left mouse down':
                voxel = Voxel(position=self.position + mouse.normal)

            if key == 'right mouse down':
                destroy(self)


def update():
    global free_roam, pressed_last_frame
    camera.x += held_keys["d"] * 0.1
    camera.x -= held_keys["a"] * 0.1
    camera.z += held_keys["w"] * 0.1
    camera.z -= held_keys["s"] * 0.1
    camera.y += held_keys["space"] * 0.1
    camera.y -= held_keys["shift"] * 0.1

    camera.rotation_x += held_keys["down arrow"] * 0.8
    camera.rotation_x -= held_keys["up arrow"] * 0.8

    camera.rotation_y += held_keys["right arrow"] * 0.8
    camera.rotation_y -= held_keys["left arrow"] * 0.8

    if held_keys["f"] == 1 and not pressed_last_frame:
        pressed_last_frame = True
        free_roam = not free_roam
    else:
        pressed_last_frame = False


    sx = 0
    sy = 0
    sz = 0
    with shared_lock:
        for block in shared_map:
            sx += block.position.x
            sy += block.position.y
            sz += block.position.z
            if not block.instantiated:
                print(f"Instantiating block {block}")
                if len(block.type) < 3:
                    block.type += "extra"
                voxel = Voxel(position=(block.position.x, block.position.y, block.position.z), colorrgb=[(ord(c.lower())-97)*8 for c in block.type[:3]])
                block.instantiated = True
        sl = len(shared_map)
        #print("Num Blocks ", sl)
        if sl != 0 and not free_roam:
            sx /= sl
            sy /= sl
            sz /= sl
            camera.x = sx - 10
            camera.y = sy + 10
            camera.z = sz - 10



shared_lock = Lock()
shared_map = set()

existing_blocks = set()

running = Event()
running.set()



voxel = Voxel(position=(0,-60,0))

free_roam = False
pressed_last_frame = False

camera.y = -50
camera.rotation_x = 30
camera.rotation_y = 50

window.borderless = False

window.windowed_size = 0.3
window.update_aspect_ratio()
window.late_init()

window.position = Vec2(3500, 1920/3)
Sky()

data = Thread(target=datacollector.start, args=(shared_map, shared_lock, running))

data.start()
try:
    app.run()
except:
    print("[Error]")
    running.clear()
    data.join()

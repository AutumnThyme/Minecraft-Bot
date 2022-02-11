"""
Module offers api for controlling the player through computer vision and simulating keypresses.
"""
from math import floor

from utility import *
import mss

class MinecraftPlayer:
    def __init__(self, bb_coords, bb_rotation, bb_block_coords, bb_block_type, shared_map, shared_lock) -> None:
        self.bb_coords = bb_coords
        self.bb_rotation = bb_rotation
        self.bb_block_coords = bb_block_coords
        self.bb_block_type = bb_block_type

        self.coordinate_regex = re.compile(r'([+-]?\d+\.\d+)/([+-]?\d+\.\d+)/([+-]?\d+\.\d+)')
        self.rotation_regex = re.compile(r'[a-zA-Z]+\([a-zA-Z]+\)\(([+-]?\d+\.\d+)/([+-]?\d+\.\d+)\)')
        self.target_block_position_regex = re.compile(r'([+-]?\d+),([+-]?\d+),([+-]?\d+)')
        self.target_block_type_regex = re.compile(r'(.+)')
        self.position = Vector3(0, 0, 0)
        self.prev_position = None
        self.rotation = Vector2(0, 0)
        self.prev_rotation = None
        self.target_position = Vector3(0,0,0)
        self.prev_target_position = None
        self.target_type = ""
        self.prev_target_type = None

        self.time = 0
        self.prev_time = None

        self.max_speed_tolerance = 10
        self.max_rotation_tolerance = 720
        self.max_target_tolerance = 100

        self.current_position_image = None
        self.current_rotation_image = None
        self.current_block_position_image = None
        self.current_block_type_image = None

        self.map = Map(shared_map=shared_map, shared_lock=shared_lock)

        self.action_queue = []

        self.rotation_action_queue = []

        self.movement_action_queue = []

        self.screen = mss.mss()

    
    def add_rotation_to_queue(self, position: Vector2):
        self.action_queue.append({"type": "rotation", "value": position})

    def add_movement_to_queue(self, units_forward: float):
        self.action_queue.append({"type": "movement", "value": {"units": units_forward, "original": None, "keydown": "None", "slow": False}})

    def add_coordinates_long_to_queue(self, position: Vector3):
        self.action_queue.append({"type": "coordinates_long", "value": {"coord": position, "axis": "forward", "keydown": "None", "slow": False, "original": None} })

    def add_coordinate_to_queue(self, position: Vector3):
        self.action_queue.append({"type": "coordinate", "value": {"coord": position, "keydown_z": None, "keydown_x": None, "crouch": False, "original": None} })

    def add_pathfind_coordinate_to_queue(self, position: Vector3):
        self.action_queue.append({"type": "pathfind", "value": {"coord": position, "state": "start", "original": None, "visited": []} })

    def _insert_events(self, events, position):
        self.action_queue = self.action_queue[:position] + events + self.action_queue[position:]

    def _get_event(self, event_name, value):
        if event_name == "rotation":
            return {"type": "rotation", "value": value}
        elif event_name == "coordinate":
            return {"type": "coordinate", "value": {"coord": value, "keydown_z": None, "keydown_x": None, "crouch": False, "original": None} }
        else:
            raise Exception(f"[ERROR] Unknown Event {event_name}")

    def add_click_to_queue(self, msb):
        self.action_queue.append({ "type": "click", "value": msb})

    def add_scan_to_queue(self):
        """
        Add an action to scan the 3x3 cube surrounding the player.
        """
        self.action_queue.append({"type": "scan", "Origin": None, })


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
            result = self.serve_coordinate()
            if result:
                self.action_queue.pop(0)
        elif action["type"] == "click":
            result = self.serve_click()
            if result:
                self.action_queue.pop(0)
        elif action["type"] == "pathfind":
            result = self.serve_pathfind()
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


    def serve_pathfind(self):
        """
        Uses simple pathfinding to move to a coordinate (currently only x and z pathfinding)
        avoids simple blocked path (blocked = blocks in way, holes)
        If all paths are blocked, backtrack and set current node to blocked. (Binary d*?)
        We use event loops to break away to other actions while pathfinding.
        Event loops work by inserting events after this one, then after the inserted events
        we insert another pathfinding event with some managed state.
        """
        desired_position = self.action_queue[0]["value"]["coord"]
        state = self.action_queue[0]["value"]["state"]
        original = self.action_queue[0]["value"]["original"]

        current_position = self.position

        block_position = self.position.copy()
        block_position.x = min(floor(block_position.x), ceil(block_position.x))
        block_position.y = min(floor(block_position.y), ceil(block_position.y))
        block_position.z = min(floor(block_position.z), ceil(block_position.z))
        
        # For now lets just do 2d pathfinding with 3d obstacles.
        delta_position = desired_position.subtract(current_position)

        # Should be larger than coordinate event tolerance
        coordinate_tolerance = 0.5

        if abs(delta_position.x) < coordinate_tolerance and abs(delta_position.y) < coordinate_tolerance:
            print(f"[INFO] Pathfinding complete, arrived at {desired_position} with error {delta_position}")
            return True
        
        if state == "start":
            print("Scanning")
            # Search area for coordinates and obstacles (defined by 3x3 ground + walls on each side of 3x3 ground)
            # We can use the insertion of rotation events + reinsertion of pathfind event at index 1 afterwards to interrupt the current process
            # and start the rotation events, after those complete, return to pathfinding.
            
            # Recall that player position = block_on.y + 1
            rotations = get_neighboring_blocks(block_position)

            # Remove unneeded block scans
            events = [self._get_event("rotation", x.rotation) for x in rotations if x.position not in self.map.current_map]
            self.action_queue[0]["value"]["state"] = "move"
            events.append(self.action_queue[0])
            
            # Insert loop into event queue
            self._insert_events(events, 1)
            return True
        elif state == "move":
            print("moving")
            # Given the map and current position, determine the best position to pick.
            # If current position is completely blocked, set current position to blocked and backtrack.
            # We can block the position by inserting a block into the map with id "blocked coordinate" and the given position.y+2.
            neighbors = get_neighboring_blocks_dict(block_position)

            choices = []

            # Remove blocked neighbors
            for ne in neighbors:
                # This is the direction
                n = neighbors[ne]
                # Check that node does not have blocks blocking it.
                blocked = False
                hole = False
                for nw in n["rest"]:
                    blocked = blocked or nw.position in self.map.current_map
                    if nw.position in self.map.current_map:
                        print(f"Block {nw.position} blocks {n['start'].position}.")
                
                # Check that node is not a hole
                hole = hole or n["start"].position not in self.map.current_map
                if n["start"].position not in self.map.current_map:
                    print(f"Block {n['start'].position} does not exist to stand on.")
                if not (blocked or hole):
                    choices.append(n["start"])
            
            choices = [x for x in choices if x.position not in self.action_queue[0]["value"]["visited"]]

            # Exit if we run out of possible paths
            if len(choices) <= 0:
                print(f"[ERROR] Pathfinding failed to find path to {desired_position}, all neighboring blocks are blocked at coordinate {current_position}")
                print(f"Map: {self.map.current_map}")
                return True
            
            if len(choices) == 1 and not hole:
                # Current block should become blocked - so we wall the area off in the map.
                self.map.add_block("PATHFINDING_BLOCKED", block_position)
                self.map.add_block("PATHFINDING_BLOCKED", block_position.add(Vector3(0,1,0)))

            # Pick the choice that minimizes the vector to the desired position (this is 100% a shitty heuristic)
            choice = choices[0]
            rest = choices[1:]
            for curr in rest:
                delta_choice = desired_position.subtract(choice.position)
                delta_current = desired_position.subtract(curr.position)
                if delta_choice.magnitude() > delta_current.magnitude():
                    choice = curr.copy()


            self.action_queue[0]["value"]["visited"].append(choice.position.copy())

            #choice.position.x -= 0.5 if choice.position.x < 0 else -0.5
            #choice.position.y -= 0.5 if choice.position.y < 0 else -0.5
            #hoice.position.z -= 0.5 if choice.position.z < 0 else -0.5
            choice_dir = choice.position.subtract(block_position)
            choice_position = current_position.add(choice_dir)

            print(f"Adding pathfind node {choice_position}")

            events = [self._get_event("coordinate", choice_position.copy())]
            self.action_queue[0]["value"]["state"] = "start"
            events.append(self.action_queue[0])
            
            # Insert loop into event queue
            self._insert_events(events, 1)
            return True
        else:
            raise Exception(f"[ERROR] Unknown Pathfinding State: {state}")


    def serve_coordinate(self):
        """
        Walk to the given coordinate with no pathfinding.
        """
        desired_position = self.action_queue[0]["value"]["coord"]
        if self.action_queue[0]["value"]["original"] is None:
                self.action_queue[0]["value"]["original"] = self.position.copy()

        curr_rotation = self.rotation
        if curr_rotation.x < 0:
            curr_rotation.x = 180 + (180 + curr_rotation.x)

        desired_position_rotated: Vector3 = desired_position.rotate_about_origin_xy(self.action_queue[0]["value"]["original"], -curr_rotation.x)
        curr_position_rotated: Vector3 = self.position.rotate_about_origin_xy(self.action_queue[0]["value"]["original"], -curr_rotation.x)

        delta_position = desired_position_rotated.subtract(curr_position_rotated)

        forwards = "w"
        backwards = "s"
        left = "a"
        right = "d"
        crouch = "ctrl"

        # Set tolerances
        tap_duration = 0.03
        tap_tolerance = 0.3
        crouch_tolerance = 2
        error_tolerance = 0.05

        if abs(delta_position.x) < error_tolerance and abs(delta_position.z) < error_tolerance:
            # Clean up keys pressed and return True to indicate task completion.
            if self.action_queue[0]["value"]["crouch"]:
                pydirectinput.keyUp(crouch)
            pydirectinput.keyUp(self.action_queue[0]["value"]["keydown_z"])
            pydirectinput.keyUp(self.action_queue[0]["value"]["keydown_x"])
            print(f"[TASK] Done moving to coordinates {desired_position}")
            return True

        if delta_position.magnitude() < crouch_tolerance:
                # Crouch if needed
                self.action_queue[0]["value"]["crouch"] = True
                pydirectinput.keyDown(crouch)

        # Handle forward backwards.
        if abs(delta_position.z) > error_tolerance:
            # Need to move forward or backwards
            if delta_position.z >= 0:
                # Behind point, move forwards
                if abs(delta_position.z) < tap_tolerance:
                    # Tap direction if needed.
                    pydirectinput.keyUp(self.action_queue[0]["value"]["keydown_z"])
                    self.action_queue[0]["value"]["keydown_z"] = None
                    pydirectinput.keyDown(forwards)
                    time.sleep(tap_duration)
                    pydirectinput.keyUp(forwards)
                else:
                    if self.action_queue[0]["value"]["keydown_z"] != forwards:
                        pydirectinput.keyUp(self.action_queue[0]["value"]["keydown_z"])
                        pydirectinput.keyDown(forwards)
                        self.action_queue[0]["value"]["keydown_z"] = forwards
            else:
                # In front of point, move backwards
                if abs(delta_position.z) < tap_tolerance:
                    # Tap direction if needed.
                    pydirectinput.keyUp(self.action_queue[0]["value"]["keydown_z"])
                    self.action_queue[0]["value"]["keydown_z"] = None
                    pydirectinput.keyDown(backwards)
                    time.sleep(tap_duration)
                    pydirectinput.keyUp(backwards)
                else:
                    if self.action_queue[0]["value"]["keydown_z"] != backwards:
                        pydirectinput.keyUp(self.action_queue[0]["value"]["keydown_z"])
                        pydirectinput.keyDown(backwards)
                        self.action_queue[0]["value"]["keydown_z"] = backwards

        # Handle left right.
        if abs(delta_position.x) > error_tolerance:
            # Need to move forward or backwards
            if delta_position.x >= 0:
                # Right of point, move left.
                if abs(delta_position.x) < tap_tolerance:
                    # Tap direction if needed.
                    pydirectinput.keyUp(self.action_queue[0]["value"]["keydown_x"])
                    self.action_queue[0]["value"]["keydown_x"] = None
                    pydirectinput.keyDown(left)
                    time.sleep(tap_duration)
                    pydirectinput.keyUp(left)
                else:
                    if self.action_queue[0]["value"]["keydown_x"] != left:
                        pydirectinput.keyUp(self.action_queue[0]["value"]["keydown_x"])
                        pydirectinput.keyDown(left)
                        self.action_queue[0]["value"]["keydown_x"] = left
            else:
                # Left of point, move right.
                if abs(delta_position.x) < tap_tolerance:
                    # Tap direction if needed.
                    pydirectinput.keyUp(self.action_queue[0]["value"]["keydown_x"])
                    self.action_queue[0]["value"]["keydown_x"] = None
                    pydirectinput.keyDown(right)
                    time.sleep(tap_duration)
                    pydirectinput.keyUp(right)
                else:
                    if self.action_queue[0]["value"]["keydown_x"] != right:
                        pydirectinput.keyUp(self.action_queue[0]["value"]["keydown_x"])
                        pydirectinput.keyDown(right)
                        self.action_queue[0]["value"]["keydown_x"] = right
                
        return False


    def serve_coordinates_long(self):
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

        #print(f"Moving {curr_position} -> {desired_position} Difference {desired_position_rotated.subtract(curr_position_rotated)}")

        forwards = "s"
        backwards = "w"
        left = "a"
        right = "d"

        tap = False

        tap_duration = 0.04
        tap_tolerance = 0.6

        error_tolerance = 0.1

        differencez = desired_position_rotated.z - curr_position_rotated.z
        differencex = desired_position_rotated.x - curr_position_rotated.x

        if differencez > error_tolerance and self.action_queue[0]["value"]["axis"] != "forward":
            self.action_queue[0]["value"]["axis"] = "forward"
        elif differencex > error_tolerance and self.action_queue[0]["value"]["axis"] != "right":
            self.action_queue[0]["value"]["axis"] = "right"

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

            if abs(difference) <= tap_tolerance:
                tap = True

            if abs(difference) > error_tolerance:
                if difference < 0:
                    if tap:
                        if self.action_queue[0]["value"]["keydown"] != "None":
                            pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                            self.action_queue[0]["value"]["keydown"] = "None"
                        pydirectinput.keyDown(forwards)
                        time.sleep(tap_duration)
                        pydirectinput.keyUp(forwards)
                    else:
                        # Move foward
                        if self.action_queue[0]["value"]["keydown"] != forwards:
                            if self.action_queue[0]["value"]["keydown"] != "None":
                                pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                        pydirectinput.keyDown(forwards)
                        self.action_queue[0]["value"]["keydown"] = forwards

                elif difference > 0:
                    # move backwards
                    if tap:
                        if self.action_queue[0]["value"]["keydown"] != "None":
                            pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                            self.action_queue[0]["value"]["keydown"] = "None"
                        pydirectinput.keyDown(backwards)
                        time.sleep(tap_duration)
                        pydirectinput.keyUp(backwards)
                    else:
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
                #print(f"Aligned forward axis")

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

            if abs(difference) <= tap_tolerance:
                tap = True

            if abs(difference) > error_tolerance:
                if difference < 0:
                    # Move left
                    if tap:
                        if self.action_queue[0]["value"]["keydown"] != "None":
                            pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                            self.action_queue[0]["value"]["keydown"] = "None"
                        pydirectinput.keyDown(right)
                        time.sleep(tap_duration)
                        pydirectinput.keyUp(right)
                    else:
                        if self.action_queue[0]["value"]["keydown"] != right:
                            if self.action_queue[0]["value"]["keydown"] != "None":
                                pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                        pydirectinput.keyDown(right)
                        self.action_queue[0]["value"]["keydown"] = right

                elif difference > 0:
                    # move right
                    if tap:
                        if self.action_queue[0]["value"]["keydown"] != "None":
                            pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])
                            self.action_queue[0]["value"]["keydown"] = "None"
                        pydirectinput.keyDown(left)
                        time.sleep(tap_duration)
                        pydirectinput.keyUp(left)
                    else:
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

                #print(f"Aligned left axis")
                print(f"Done moving to coordinates {desired_position}, real {curr_position}")
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
                pydirectinput.keyUp(self.action_queue[0]["value"]["keydown"])

            if self.action_queue[0]["value"]["slow"]:
                pydirectinput.keyUp("ctrl")
                self.action_queue[0]["value"]["slow"] = False

            #print(f"moved {units_moved} / {units_desired} from {orig_position} to {self.position}")
            print(f"Done moving {units_desired} units")
            return True
        else:
            #print(f"moved {units_moved} / {units_desired} from {orig_position} to {self.position}")
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
            #print(f"Done rotating to {desired_rotation}")
            return True
        else:
            mx = ceil(mx)
            my = ceil(my)
            #print(f"moving mouse: {(int(mx), int(my))}")
            ctypes.windll.user32.mouse_event(0x01, int(mx), int(-my), 0, 0)
        return False

    
    def update(self, api):
        """
        Grab all new text, update time, then update pos, rot, block, etc...
        """
        # Position
        api.SetVariable("tessedit_char_whitelist", "./-0123456789")
        #im_position = ImageGrab.grab(bbox=self.bb_coords)
        mon = {"top": self.bb_coords[1], "left": self.bb_coords[0], "width": self.bb_coords[2] - self.bb_coords[0], "height": self.bb_coords[3] - self.bb_coords[1]}
        im_position = self.screen.grab(mon)
        pos_text, self.current_position_image = image_to_text(api, im_position)

        # Rotation
        api.SetVariable("tessedit_char_whitelist", "./-()" + string.digits + string.ascii_letters.replace('S', ""))
        #im_rotation = ImageGrab.grab(bbox=self.bb_rotation)

        mon = {"top": self.bb_rotation[1], "left": self.bb_rotation[0], "width": self.bb_rotation[2] - self.bb_rotation[0], "height": self.bb_rotation[3] - self.bb_rotation[1]}
        im_rotation = self.screen.grab(mon)
        rot_text, self.current_rotation_image = image_to_text(api, im_rotation)


        # Block Look Position
        api.SetVariable("tessedit_char_whitelist", "-," + string.digits + string.ascii_letters.replace('S', ""))
        #im_block_position = ImageGrab.grab(bbox=self.bb_block_coords)

        mon = {"top": self.bb_block_coords[1], "left": self.bb_block_coords[0], "width": self.bb_block_coords[2] - self.bb_block_coords[0], "height": self.bb_block_coords[3] - self.bb_block_coords[1]}
        im_block_position = self.screen.grab(mon)
        block_position_text, self.current_block_position_image = image_to_text(api, im_block_position, crop_to_activity=True, crop_extra=160)
        block_position_text = block_position_text.replace(' ',  '')

        # Block Look Type
        api.SetVariable("tessedit_char_whitelist", "_" + string.ascii_lowercase)
        #im_block_type = ImageGrab.grab(bbox=self.bb_block_type)

        mon = {"top": self.bb_block_type[1], "left": self.bb_block_type[0], "width": self.bb_block_type[2] - self.bb_block_type[0], "height": self.bb_block_type[3] - self.bb_block_type[1]}
        im_block_type = self.screen.grab(mon)
        block_type_text, self.current_block_type_image = image_to_text(api, im_block_type, crop_to_activity=True, crop_extra=97)

        #block_type_text = block_type_text.replace(' ', '')


        self.time = time.time()
        if self.prev_time is None:
                self.prev_time = self.time
        coord = self.update_coords(pos_text)
        rot = self.update_rotation(rot_text)
        block_pos = self.update_block_position(block_position_text)
        block_type = self.update_block_type(block_type_text)

        self.prev_time = self.time

        if block_pos and block_type:
            #print(f"Adding block to map {self.target_type}: {self.target_position}")
            self.map.add_block(self.target_type, self.target_position.copy())
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
            print(f"[Error] Could not parse coordinates this frame: {pos_text}")
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

            #if speed.magnitude() > self.max_rotation_tolerance:
                #print("Coord error, invalid rotation")
            #print(f"\tRotation - X: {self.rotation.x}, Y: {self.rotation.y}", f"\t\tSpeed - dx: {dx}, dy: {dy}, dt: {dt}")
        else:
            #print(f"[Error] Could not parse rotation this frame: {rot_text}")
            return False
        return True


    def update_block_position(self, block_position_text):
        match = self.target_block_position_regex.match(block_position_text)

        if match:
            x, y, z = match.groups()
            self.target_position.reassign(float(x), float(y), float(z))

            if self.prev_target_position is None:
                self.prev_target_position = Vector3(self.target_position.x, self.target_position.y, self.target_position.z)

            dt = self.time - self.prev_time
            if dt == 0:
                dt = 0.00001
            dx = (self.target_position.x - self.prev_target_position.x) / dt
            dy = (self.target_position.y - self.prev_target_position.y) / dt
            dz = (self.target_position.z - self.prev_target_position.z) / dt

            speed = Vector3(dx, dy, dz)
            
            self.prev_target_position.reassign(self.target_position.x, self.target_position.y, self.target_position.z)

            #if speed.magnitude() > self.max_target_tolerance:
                #print("Coord error, invalid speed")
            #print(f"\tTargetCoords - X: {self.target_position.x}, Y: {self.target_position.y}, Z: {self.target_position.z}", f"\t\tSpeed - dx: {dx}, dy: {dy}, dz: {dz}, dt: {dt}")
        else:
            #print(f"[Error] Could not parse target coordinates this frame: {block_position_text}")
            return False
        return True


    def update_block_type(self, block_type_text):
        match = self.target_block_type_regex.match(block_type_text)

        if match:
            self.prev_target_type = self.target_type
            self.target_type = match.group(0)
            #print(f"\tTargetType - {self.target_type}")
        else:
            #print(f"[Error] Could not parse target type this frame: {block_type_text}")
            return False
        return True
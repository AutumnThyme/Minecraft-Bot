from warnings import catch_warnings
from utility import Vector3
from math import floor, ceil


def stateNameToCoords(name):
    try:
        return [int(name.split('x')[1].split('y')[0]), int(name.split('x')[1].split('y')[1])]
    except Exception as e:
        print(f"Error parsing value: {name}\n", e)

def coordsToStateName(pos: Vector3):
    return f"x{int(pos.x)}y{int(pos.z)}"

def world2Grid(position: Vector3, start: Vector3, padding):
    block_position = Vector3(0,0,0)
    block_position.x = min(floor(start.x), ceil(start.x))
    block_position.y = min(floor(start.y), ceil(start.y))
    block_position.z = min(floor(start.z), ceil(start.z))
    m = position.subtract(block_position)
    #m.x = abs(m.x)
    #m.z = abs(m.z)
    return m.add(Vector3(padding, 0, padding))

def grid2World(x, y, start: Vector3, padding):
    return start.add(Vector3(x, 0, y).subtract(Vector3(padding, 0, padding)))

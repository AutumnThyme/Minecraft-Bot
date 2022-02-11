from utility import Vector3


def stateNameToCoords(name):
    return [int(name.split('x')[1].split('y')[0]), int(name.split('x')[1].split('y')[1])]

def coordsToStateName(pos: Vector3):
    return f"x{int(pos.x)}y{int(pos.z)}"

def world2Grid(position: Vector3, start: Vector3, padding):
    return position.subtract(start).add(Vector3(padding, 0, padding))

def grid2World(x, y, start: Vector3, padding):
    return start.subtract(Vector3(padding, 0, padding)).add(Vector3(x, 0, y))

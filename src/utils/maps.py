import enum
from sketchgraphs.data import sketch as datalib

"""
Define the maps between identifiers and types.
"""

# the nodes and edges that are considered
keep_edge = [datalib.ConstraintType.Coincident, datalib.ConstraintType.Distance, datalib.ConstraintType.Horizontal,
             datalib.ConstraintType.Parallel, datalib.ConstraintType.Vertical, datalib.ConstraintType.Tangent,
             datalib.ConstraintType.Length, datalib.ConstraintType.Perpendicular, datalib.ConstraintType.Midpoint,
             datalib.ConstraintType.Equal, datalib.ConstraintType.Diameter, datalib.ConstraintType.Radius,
             datalib.ConstraintType.Concentric, datalib.ConstraintType.Angle, datalib.ConstraintType.Subnode]
keep_node = [datalib.EntityType.Point, datalib.EntityType.Line,
             datalib.EntityType.Circle, datalib.EntityType.Arc,
             datalib.SubnodeType.SN_Start, datalib.SubnodeType.SN_End, datalib.SubnodeType.SN_Center,
             datalib.EntityType.External, datalib.EntityType.Stop]

EDGE_IDX_MAP = {}
i = 0
for t in datalib.ConstraintType:
    if t in keep_edge:
        EDGE_IDX_MAP[t] = i
        i += 1
EDGE_IDX_MAP_REVERSE = {i: t for t, i in EDGE_IDX_MAP.items()}

NODE_IDX_MAP = {}
i = 0
for t in list(datalib.EntityType) + list(datalib.SubnodeType):
    if t in keep_node:
        NODE_IDX_MAP[t] = i
        i += 1
PADDING_IDX = len(NODE_IDX_MAP)
NODE_IDX_MAP['void'] = PADDING_IDX
NODE_IDX_MAP_REVERSE = {i: t for t, i in NODE_IDX_MAP.items()}


NODES_PARAMETRIZED = {
    datalib.EntityType.Point: ['isConstruction', 'x', 'y'],
    datalib.EntityType.Line: ['isConstruction', 'dirX', 'dirY', 'pntX', 'pntY', 'startParam', 'endParam'],
    datalib.EntityType.Circle: ['isConstruction', 'xCenter', 'yCenter', 'xDir', 'yDir', 'radius', 'clockwise'],
    datalib.EntityType.Arc: ['isConstruction', 'xCenter', 'yCenter', 'xDir', 'yDir', 'radius', 'startParam', 'endParam', 'clockwise']
}

EDGES_PARAMETRIZED = {
    datalib.ConstraintType.Angle: ['aligned', 'clockwise', 'angle'],
    datalib.ConstraintType.Length: ['direction', 'length'],
    datalib.ConstraintType.Distance: ['direction', 'halfSpace0', 'halfSpace1', 'length'],
    datalib.ConstraintType.Diameter: ['length'],
    datalib.ConstraintType.Radius: ['length']
}

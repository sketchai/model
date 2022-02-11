import enum


class MockConstraintType(enum.IntEnum):
    """This enumeration represents the type of the constraint."""
    Coincident = 0
    Diameter = 12
    Angle = 17
    Unknown = 29

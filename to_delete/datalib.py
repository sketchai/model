"""This module implements parsing and representation of sketch constraints.
"""

import abc
import enum
import typing
import sys

# pylint: disable=invalid-name, too-many-arguments, too-many-return-statements, too-many-instance-attributes


class ConstraintType(enum.IntEnum):
    """This enumeration represents the type of the constraint."""
    Coincident = 0
    Projected = 1
    Mirror = 2
    Distance = 3
    Horizontal = 4
    Parallel = 5
    Vertical = 6
    Tangent = 7
    Length = 8
    Perpendicular = 9
    Midpoint = 10
    Equal = 11
    Diameter = 12
    Offset = 13
    Radius = 14
    Concentric = 15
    Fix = 16
    Angle = 17
    Circular_Pattern = 18
    Pierce = 19
    Linear_Pattern = 20
    Centerline_Dimension = 21
    Intersected = 22
    Silhoutted = 23
    Quadrant = 24
    Normal = 25
    Minor_Diameter = 26
    Major_Diameter = 27
    Rho = 28
    Unknown = 29
    Subnode = 101

    @property
    def _numeric_schema(self):
        """Supported numerical constraint schemas; will be added to over time.

        For Offset we do 2nd most popular since we can only handle 2 entity references atm.
        """
        return {
            ConstraintType.Angle: ('aligned', 'angle', 'clockwise', 'local0',
                                   'local1'),
            ConstraintType.Distance: ('direction', 'halfSpace0', 'halfSpace1',
                                      'length', 'local0', 'local1'),
            ConstraintType.Length: ('direction', 'length', 'local0'),
            ConstraintType.Offset: ('local0', 'local1'),
            ConstraintType.Diameter: ('length', 'local0'),
            ConstraintType.Radius: ('length', 'local0')
        }.get(self)

    @property
    def has_parameters(self) -> bool:
        """A boolean value indicating whether the given constraint has parameters."""
        return self._numeric_schema is not None

    def normalize(self, schema):
        """Normalizes supported schemas for numerical constraintType.

        We will consider identical any schemas that only differ by local# vs. localWord.

        Parameters
        ----------
        schema: an interable containing parameter ID strings
        constraitType: an instance of ConstraintType

        Returns
        -------
        If schema is supported, returns ref_schema, the set of relevant params; otherwise returns False.
        """
        ignored_params = ('labelAngle', 'labelDistance', 'labelRatio', 'radiusDisplay')
        ref_schema = self._numeric_schema
        if ref_schema is None:
            raise ValueError("Only call schema_comp on numerical constraints.")
        norm_schema = []
        for param_id in schema:
            if param_id in ignored_params:
                continue
            # TODO: address these magic strings
            if param_id == 'localFirst':
                param_id = 'local0'
            elif param_id == 'localSecond':
                param_id = 'local1'
            norm_schema.append(param_id)
        if set(norm_schema) == set(ref_schema):
            return set(ref_schema)
        else:
            return False


string_to_constraint_type = {e.name.upper(): e for e in ConstraintType}

print(f'test  : {ConstraintType.Angle.name}')
#print(f'string = {string_to_constraint_type}')

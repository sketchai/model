from typing import Dict
from sam.catalog_primitive import Point, Line, Circle, Arc
from sam.primitive import Primitive
from sam.constraint import Constraint
import logging
logger = logging.getLogger(__name__)

class SubnodeConstraint(Constraint):
    """Subnode Constraint, temporary class for encoding"""

    def __init__(self, references = []):
        elt_type = lambda _: None
        elt_type.name = 'Subnode'
        super().__init__(elt_type=elt_type, references=references)

        
def format_for_encoding(ops):
    """Adds Subnodes and reference indexes to the sequence"""
    new_sequence = []
    primitive_ops = [op for op in ops if isinstance(op, Primitive)]
    # current_subnode_index = len(primitive_ops)
    current_node_index = 0
    for op in ops:
        new_sequence.append(op)
        if isinstance(op,Primitive):
            op.node_index = current_node_index
            current_node_index+=1
            if isinstance(op, Line) or isinstance(op, Arc):
                op.pnt1.subnode_type = 'SN_pnt1'
                op.pnt1.node_index = current_node_index
                constraint1 = SubnodeConstraint(references=[op,op.pnt1])
                op.pnt2.subnode_type = 'SN_pnt2'
                op.pnt2.node_index = current_node_index + 1
                constraint2 = SubnodeConstraint(references=[op,op.pnt2])
                new_sequence.extend([op.pnt1, constraint1, op.pnt2, constraint2])
                current_node_index += 2

            if isinstance(op, Circle) or isinstance(op, Arc):
                op.center.subnode_type = 'SN_center'
                op.center.node_index = current_node_index
                current_node_index += 1
                constraint = SubnodeConstraint(references=[op,op.center])
                new_sequence.extend([op.center,constraint])
    return new_sequence
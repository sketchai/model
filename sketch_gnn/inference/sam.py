from unicodedata import category
from sam.catalog_constraint import Horizontal, Vertical, Parallel, Length, Coincident, Perpendicular, Distance, Radius, Tangent, Midpoint, Equal, Angle, HorizontalDistance, VerticalDistance
from sam.constraint import Constraint
from sam.sketch import Sketch
from sam.primitive import Primitive
from sketch_gnn.utils.format_encoding import format_for_encoding
from sketch_gnn.inference.eval import EvalPrediction
import logging
logger = logging.getLogger(__name__)


NAME_TO_CONSTRAINT_MAP = {
    'HORIZONTAL': Horizontal,
    'VERTICAL': Vertical,
    'PARALLEL': Parallel,
    'LENGTH': Length,
    'COINCIDENT': Coincident,
    'PERPENDICULAR': Perpendicular,
    'DISTANCE': Distance,
    'RADIUS': Radius,
    'TANGENT': Tangent,
    'MIDPOINT': Midpoint,
    'EQUAL': Equal,
    'ANGLE': Angle,
    'HORIZONTAL_DISTANCE': HorizontalDistance,
    'VERTICAL_DISTANCE': VerticalDistance,
}

class PredictionToSam:

    def get_constraints(sketch: Sketch, pred: EvalPrediction)->dict:
        """
        returns sam constraints (except true negatives)

        note that for '_wrong_type' categories, the constraint type
        is the one predicted, not the ground truth
        """        
        encoded_sequence = format_for_encoding(sketch.sequence)
        prim_ops = [op for op in encoded_sequence if isinstance(op, Primitive)]

        l_constraints = []
        for edge_info in pred:
            if edge_info['category'] == 'true_negatives':
                l_constraints.append(None)
                continue

            refs = edge_info['references']
            label = edge_info.get('true_label')
            prim_refs = [prim_ops[ref] for ref in refs]
            
            if label == 0 or label == 1:
                type_name = edge_info['predicted_type_name']
            elif label is None:
                type_name = edge_info['true_type_name']

            if type_name == 'Subnode':
                l_constraints.append('Subnode')
                continue

            new_constraint = NAME_TO_CONSTRAINT_MAP[type_name](references=prim_refs)
            l_constraints.append(new_constraint)
        return l_constraints
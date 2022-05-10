from sketch_data.catalog_constraint import Horizontal, Vertical, Parallel, Length, Coincident, Perpendicular, Distance, Radius, Tangent, Midpoint, Equal, Angle, HorizontalLength, VerticalLength
from sketch_data.constraint import Constraint
from sketch_data.sketch import Sketch
from sketch_data.primitive import Primitive
from src.utils.format_encoding import format_for_encoding
from src.utils.logger import logger
from src.inference.eval import EvalPrediction


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
    'HORIZONTAL_LENGTH': HorizontalLength,
    'VERTICAL_LENGTH': VerticalLength,
}

class PredictionToCastor:

    def get_constraints(sketch: Sketch, pred: EvalPrediction):
        edge_idx_map_reverse = list(pred.edge_idx_map.keys())
        predicted_type_name = [edge_idx_map_reverse[idx] for idx in pred.predicted_type]
        true_type_name = [edge_idx_map_reverse[idx] for idx in pred.true_type]

        encoded_sequence = format_for_encoding(sketch.sequence)
        prim_ops = [op for op in encoded_sequence if isinstance(op, Primitive)]

        predicted_constraints = []
        for i in range(len(pred.true_label)):
            refs = tuple(pred.all_references[i].tolist())
            prim_refs = [prim_ops[ref] for ref in refs]
            new_constraint = NAME_TO_CONSTRAINT_MAP[predicted_type_name[i]](references=prim_refs)
            predicted_constraints.append(new_constraint)

        true_constraints = []
        for i in range(len(pred.true_type)):
            refs = tuple(pred.all_references[i].tolist())
            prim_refs = [prim_ops[ref] for ref in refs]
            new_constraint = NAME_TO_CONSTRAINT_MAP[true_type_name[i]](references=prim_refs)
            true_constraints.append(new_constraint)

        return predicted_constraints, true_constraints

    def get_sorted_constraints(sketch: Sketch, pred: EvalPrediction):
        d_edges_idx = pred.sort_edges()
        predicted_constraints, true_constraints = PredictionToCastor.get_constraints(sketch, pred)

        d_constr_pred = {}
        for key in ['true_positives', 'true_positives_wrong_type', 'false_positives',
                        'false_negatives', 'false_negatives_wrong_type']:
            idxes = d_edges_idx[key]
            predicted = [predicted_constraints[idx] for idx in idxes]
            d_constr_pred[key] = predicted

        d_constr_gt = {}
        for key in ['true_positives_wrong_type', 'false_negatives_wrong_type']:
            idxes = d_edges_idx[key]
            ground_truth = [true_constraints[idx] for idx in idxes]
            d_constr_gt[key] = ground_truth
            
        return d_constr_pred, d_constr_gt
import matplotlib.patches
import matplotlib.pyplot as plt
from src.inference.castor import PredictionToCastor
from src.inference.eval import EvalPrediction
from sketch_data.sketch import Sketch
from sketch_data.primitive import Primitive
from sketch_data.constraint import Constraint
from sketch_data.catalog_primitive import Line, Point, Circle, Arc


def render_sketch(sketch: Sketch, ax: plt.Axes) -> None:
    """Renders a given sketch"""
    fig, ax = sketch._prepare_draw(ax=ax, show_axes=False)
    plt.axis('equal')
    for s in sketch.sequence:
        if isinstance(s, Primitive):
            render_prim(s, ax)
    # Rescale axis limits to handle text outside [0,1]
    ax.set_xlim([-0.2, 1.2])
    ax.set_ylim([-0.2, 1.2])


def render_prim(prim: Primitive, ax: plt.Axes, color='black', linewidth=1) -> None:
    """
    Renders a given primitive
    This function wraps the plot method of the sketch_data module
    it adds linewidth support and does not plot pnt1 and pnt2
    """
    if isinstance(prim, Primitive):
        if prim.get_name == 'LINE':
            ax.plot(
                [prim.pnt1.x, prim.pnt2.x],
                [prim.pnt1.y, prim.pnt2.y],
                color=color,
                linestyle=prim._get_linestyle(),
                linewidth=linewidth)
        elif prim.get_name == 'ARC':
            prim.radian = True
            prim.plot(ax, color=color, linewidth=linewidth)
        elif prim.get_name == 'POINT':
            prim.plot(ax, color=color)
        else:
            prim.plot(ax, color=color, linewidth=linewidth)


def get_prim_coords(prim):
    if isinstance(prim, Point):
        x, y = prim.x, prim.y
    if isinstance(prim, Circle) or isinstance(prim, Arc):
        x, y = prim.center.x, prim.center.y
    if isinstance(prim, Line):
        x, y = (prim.pnt1.x + prim.pnt2.x) / 2, (prim.pnt1.y + prim.pnt2.y) / 2
    return x, y


def highlight_constraint(constraint, color='red', ax=None, color_prim=True, y_offset=0.1):
    """
    Adds a patch highlighting a constraint to a plt.Axes.
    - adds an arrow pointing from ref0 to ref1
    - adds a text label
    - re render the references in a highlight color
    """
    primitive_a, primitive_b = constraint.references[0], constraint.references[-1]
    if ax is None:
        ax = plt.gca()

    if color_prim:
        render_prim(primitive_a, ax=ax, color=color)
    xa, ya = get_prim_coords(primitive_a)

    if primitive_b != primitive_a:
        if color_prim:
            render_prim(primitive_b, ax=ax, color=color)
        xb, yb = get_prim_coords(primitive_b)
        arrow = matplotlib.patches.FancyArrowPatch(posA=(xa, ya),
                                                   posB=(xb, yb),
                                                   path=None,
                                                   arrowstyle='<->',
                                                   connectionstyle='arc',
                                                   mutation_scale=10,
                                                   color=color
                                                   )
        ax.add_patch(arrow)
        xtext, ytext = (xa + xb) / 2, (ya + yb) / 2 + y_offset
    else:
        xtext, ytext = xa, ya + y_offset

    ax.text(xtext, ytext, constraint.type.name.lower(), ha="center", va="center", rotation=0,
            size=15, color=color)


def display_constraint(sketch: Sketch, n=-1, size=10, color=None):
    """
    Creates a plt.figure representing the sketch with constraint highlighted depending on its type

    n= -1 to highlight all constraints
    n= i  to highlight only i_th constraint
    """
    fig, ax = plt.subplots(figsize=(size, size))
    render_sketch(sketch, ax=ax)
    constr_ops = [op for op in sketch.sequence if isinstance(op, Constraint)]
    # cmap = np.array(plt.get_cmap('Set3').colors).reshape(-1,3)
    cmap = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'] * 2
    if n == -1:
        for i, constr in enumerate(constr_ops):
            color = cmap[constr.type]
            highlight_constraint(constr, color=color, color_prim=False)
    else:
        constr = constr_ops[n]
        if color is None:
            color = cmap[constr.type]
        highlight_constraint(constr, color=color, color_prim=True)


COLOR_MAP = {
    'green': 'true_positives',
    'red': 'false_positives',
    'blue': 'true_positives_wrong_type',
    'orange': 'false_negatives',
    'purple': 'false_negatives_wrong_type',
}

OFFSET_MAP = {
    'green': 0.02,
    'red': 0.04,
    'blue': 0.06,
    'orange': 0.08,
    'purple': 0.1,
}

def display_inference(sketch, pred: EvalPrediction, legend=True):
    """
    Creates a plt.figure representing the sketch with constraint highlighted depending on model output

    Category                            color       gt      pred    gt_type     pred_type

    True positives:                     green       1       1       A           A
    True positives Wrong Type:          blue        1       1       A           B
    False positives:                    red         0       1       /           A
    False negatives:                    orange      1       0       A           A
    False negatives Wrong type:         purple      1       0       A           B
    True negatives:                     /           0       0       /           /
    
    TODO: add given constraints in grey/black?
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    render_sketch(sketch, ax=ax)

    d_constr_pred, d_constr_gt = PredictionToCastor.get_sorted_constraints(sketch, pred)

    colored_constraints = {k:d_constr_pred[v] for k,v in COLOR_MAP.items()}

    for color, l_constraints in colored_constraints.items():
        offset = OFFSET_MAP[color]
        for constr in l_constraints:
            highlight_constraint(constr, color=color, color_prim=False, y_offset=offset)

    if legend:
        add_legend(colored_constraints, ax)
        

def add_legend(colored_constraints, ax):
    handles = []
    legend_names = {
            'green':    f'True Positive {len(colored_constraints["green"]):<3}',
            'blue':     f'True Positive wrong type {len(colored_constraints["blue"]):<3}',
            'red':      f'False Positive {len(colored_constraints["red"]):<3}',
            'orange':   f'False Negative {len(colored_constraints["orange"]):<3}',
            'purple':   f'False Negative wrong type {len(colored_constraints["purple"]):<3}',
        }
    for color, name in legend_names.items():
        h = matplotlib.patches.Patch(color=color, label=name)
        handles.append(h)
    legend = ax.legend(handles=handles, loc='upper left')

def display_specific_constraint(sketch, pred: EvalPrediction, request, legend=True, hide_tp=True):
    """
    Same as display_inference but only a specific constraint is highlighted
    Extra information is shown
    """
    if isinstance(request,int):
        l_idx_constraint = [request]

    fig, ax = plt.subplots(figsize=(10, 10))
    render_sketch(sketch, ax)
    d_constr_pred, d_constr_gt = PredictionToCastor.get_sorted_constraints(sketch, pred)

    colored_constraints = {k:d_constr_pred[v] for k,v in COLOR_MAP.items()}
    if legend:
        add_legend(colored_constraints,ax)

    ### Highlights
    if hide_tp:
        d_constr_pred['true_positives'] = []

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin,xmax+0.3)
    xtext = xmax
    ytext = ax.get_ylim()[1] - 0.04
    idx = 0
    for color, key in COLOR_MAP.items():
        l_constr_pred = d_constr_pred.get(key)
        l_constr_gt = d_constr_gt.get(key)
        for j, constraint in enumerate(l_constr_pred):
            # highlight
            if idx in l_idx_constraint:
                text = constraint.type.name
                font = 'bold'
                if l_constr_gt is not None:
                    gt_constraint = l_constr_gt[j]
                    text += f' (gt = {gt_constraint.type.name})'
                y_offset = OFFSET_MAP[color]
                highlight_constraint(constraint, color=color, ax=ax, color_prim=True, y_offset=y_offset)
            else:
                text = constraint.type.name.lower()
                font = 'regular'

            #generate text list
            ax.text(xtext, ytext, text, ha="center", va="center", rotation=0,
                size=15, color=color, fontweight=font)
            ytext -= 0.04
            idx+=1


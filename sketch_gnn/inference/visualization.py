from unicodedata import category
import matplotlib.patches
import matplotlib.pyplot as plt
from sketch_gnn.inference.sam import PredictionToSam
from sketch_gnn.inference.eval import EvalPrediction
from sam.sketch import Sketch
from sam.primitive import Primitive
from sam.constraint import Constraint
from sam.catalog_primitive import Line, Point, Circle, Arc

PARENT_COLOR = {
    'green': 'lightgreen',
    'red': 'pink',
    'blue': 'lightblue',
    'darkorange': 'yellow',
    'purple': 'violet',
    'grey': 'lightgrey',
}

NAME_TO_SYMBOL_MAP = {
    'COINCIDENT':       'C     ',
    'HORIZONTAL':       ' H    ',
    'VERTICAL':         ' V    ',
    'PARALLEL':         '  //  ',
    'LENGTH':           '    L ',
    'PERPENDICULAR':    '     ⊥',
    'DISTANCE':     '\n\nD     ',
    'RADIUS':       '\n\n R    ',
    'TANGENT':      '\n\n  T   ',
    'MIDPOINT':     '\n\n   M  ',
    'EQUAL':        '\n\n    = ',
    'ANGLE':        '\n\n     ∡',
    'HORIZONTAL_DISTANCE':'hL',
    'VERTICAL_DISTANCE':  'vL',
}

COLOR_MAP = {
    'true_positives': 'green',
    'false_positives': 'red',
    'true_positives_wrong_type': 'blue',
    'false_negatives': 'darkorange',
    'false_negatives_wrong_type': 'purple',
    'given': 'grey',
}

OFFSET_MAP = {
    'true_positives': 0.02,
    'false_positives': 0.04,
    'true_positives_wrong_type': 0.06,
    'false_negatives': 0.08,
    'false_negatives_wrong_type': 0.1,
    'given': -0.04,
}


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
    This function wraps the plot method of the sam module
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
            prim.plot(ax, color=color, linewidth=linewidth)
        elif prim.get_name == 'POINT':
            prim.plot(ax, color=color, s=10*(2*linewidth)**2)
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


def highlight_constraint(constraint, color='red', ax=None, color_prim=True, y_offset=0.1, arrow=True):
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
        for primitive in (primitive_a, primitive_b):
            render_prim(primitive, ax=ax, color=color, linewidth=3)
            parent = primitive.__dict__.get('parent')
            if parent:
                parent_color = PARENT_COLOR.get(color) or color
                render_prim(parent, ax=ax, color=parent_color, linewidth=2)
    xa, ya = get_prim_coords(primitive_a)
    xb, yb = get_prim_coords(primitive_b)

    if primitive_b != primitive_a:
        if arrow:
            arrow_patch = matplotlib.patches.FancyArrowPatch(posA=(xa, ya),
                                                    posB=(xb, yb),
                                                    path=None,
                                                    arrowstyle='<->',
                                                    connectionstyle='arc',
                                                    mutation_scale=10,
                                                    color=color
                                                    )
            ax.add_patch(arrow_patch)
        xtext, ytext = (xa + xb) / 2, (ya + yb) / 2 + y_offset
    else:
        xtext, ytext = xa, ya + y_offset

    text = NAME_TO_SYMBOL_MAP[constraint.type.name]
    ax.text(xtext, ytext, text, ha="center", va="center", rotation=0,
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


def display_inference(sketch: Sketch, pred: EvalPrediction, legend=True, categories=None):
    """
    Creates a plt.figure representing the sketch with constraint highlighted depending on model output

    Category                            color       gt      pred    gt_type     pred_type

    True positives:                     green       1       1       A           A
    False positives:                    red         0       1       /           A
    True positives Wrong Type:          blue        1       1       A           B
    False negatives:                    orange      1       0       A           A
    False negatives Wrong type:         purple      1       0       A           B
    Given:                              grey        1       /       A           /
    True negatives:                     /           0       0       /           /
    
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    render_sketch(sketch, ax=ax)
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    l_constraints = PredictionToSam.get_constraints(sketch, pred)

    indexes_to_show = filter_edges_for_visu(pred, hide_tp=False, hide_given=False, categories=categories)
    
    for idx in indexes_to_show:
        constraint = l_constraints[idx]
        info= pred[idx]
        highlight_constraint_with_info(constraint, info, color_prim=False)

    if legend:
        add_legend(pred, ax)

    return fig, ax
        

def add_legend(pred, ax):
    handles = []
    d_categories = pred.d_categories
    legend_labels = {
            'true_positives': 'True Positives {} /{}',
            'false_positives': 'False Positives {} /{}',
            'true_positives_wrong_type': 'True Positives wrong type {} /{}',
            'false_negatives': 'False Negatives {} /{}',
            'false_negatives_wrong_type': 'False Negatives wrong type {} /{}',
    }
    n_total_edges = sum((len(d_categories[category]) for category in legend_labels))
    for category, label in legend_labels.items():
        legend_labels[category] = label.format(len(d_categories[category]),n_total_edges)
    # exclude subnodes
    n_constr_given = len([idx for idx in d_categories['given'] if pred[idx]['true_type_name'] != 'Subnode'])
    legend_labels['given'] = f'Given {n_constr_given}'
    for category, color in COLOR_MAP.items():
        legend_label = legend_labels[category]
        h = matplotlib.patches.Patch(color=color, label=legend_label)
        handles.append(h)
    legend = ax.legend(handles=handles, loc='upper left',fontsize="x-large")
    return legend

def display_specific_constraint(sketch: Sketch, pred: EvalPrediction, request=-1, legend=True, hide_tp=False, hide_given=False, categories=None):
    """
    Same as display_inference but only a specific constraint is highlighted
    and extra information is shown
    """

    fig, ax = plt.subplots(figsize=(15, 15))
    render_sketch(sketch, ax)
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    l_constraints = PredictionToSam.get_constraints(sketch, pred)

    if legend:
        add_legend(pred,ax)

    indexes_to_show = filter_edges_for_visu(pred, hide_tp=hide_tp, hide_given=hide_given, categories=categories)
    
    list_idx = 0
    for idx in indexes_to_show:
        constraint = l_constraints[idx]
        info = pred[idx]

        bold=False
        highlight=False
        if list_idx == request:
            color_prim=True
            arrow=False
            bold=True
            highlight=True
        elif request == -1:
            color_prim=False
            arrow=True
            highlight=True

        if highlight:
            highlight_constraint_with_info(constraint, info, color_prim=color_prim, arrow=arrow)

        add_text_item(constraint, info, list_idx, bold)

        list_idx+=1
    return fig, ax

def filter_edges_for_visu(pred,hide_tp=True, hide_given=True, categories=None, sort=True):
    """
    filter indexes by category
    removes tn and subnode edges
    """
    l_indexes = []
    d_categories = pred.d_categories
    if categories is None:
        categories = list(d_categories.keys())
        categories.remove('true_negatives')
        if hide_tp:
            categories.remove('true_positives')
        if hide_given:
            categories.remove('given')

    for category in categories:
        idxes = d_categories.get(category)
        if category == 'given':
            idxes = [i for i in idxes if pred[i].get('true_type_name')!='Subnode']
        l_indexes.extend(idxes)

    if sort:
        l_indexes = sorted(l_indexes, key= lambda i: pred[i].get('predicted_sigmoid') or -1, reverse=True)

    return l_indexes


def add_text_item(constraint, info, list_idx, bold:bool, ax=None):
    if ax is None:
        ax = plt.gca()
    
    category = info['category']
    color = COLOR_MAP[category]

    # generate text list
    if bold:
        text = constraint.type.name
        score = info.get("predicted_sigmoid")
        if score:
            text += f' confidence={score:.2f}'
        font = 'bold'
        true_name = info.get('true_type_name')
        if true_name is not None and true_name != constraint.type.name:
            text += f' (gt = {true_name})'
    else:
        text = constraint.type.name.lower()
        font = 'regular'

    xtext = ax.get_xlim()[1]
    ytext = ax.get_ylim()[1] - 0.04*(list_idx+1)
    ax.text(xtext, ytext, text, ha="center", va="center", rotation=0,
            size=15, color=color, fontweight=font)

def highlight_constraint_with_info(constraint, info, color_prim=True, arrow=False, ax=None):
    if ax is None:
        ax = plt.gca()
        
    category = info['category']
    color = COLOR_MAP[category]
    y_offset = OFFSET_MAP[category]
    highlight_constraint(constraint, color=color, ax=ax, color_prim=color_prim, y_offset=y_offset, arrow=arrow)
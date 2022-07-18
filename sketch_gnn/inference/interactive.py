from sam.sketch import Sketch
from sketch_gnn.inference.visualization import display_constraint, display_inference, display_specific_constraint, filter_edges_for_visu
from ipywidgets import interact, widgets
import warnings
from sam.constraint import Constraint

def interactive_sketch(sketch):
    def f(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            display_constraint(sketch,n)
    n_constraints = len([op for op in sketch.sequence if isinstance(op,Constraint)])-1
    interact(f,n=widgets.IntSlider(min=-1,max=n_constraints,step=1,value=-1))

def interactive_inference(sketch,pred,categories=None):
    def f(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            display_specific_constraint(sketch,pred,request=n,hide_tp=False,hide_given=False,categories=categories)
    # count elements to display
    n_constraints = len(filter_edges_for_visu(pred,hide_tp=False,hide_given=False,categories=categories))-1
    interact(f,n=widgets.IntSlider(min=-1,max=n_constraints,step=1,value=-1))
import os
from pinocchio.visualize import MeshcatVisualizer

def maybe_create_viz(rmodel, collision_model, visual_model):
    viz = MeshcatVisualizer(rmodel, collision_model, visual_model)
    viz.initViewer(loadModel=True, open=True)
    viz.loadViewerModel()
    return viz
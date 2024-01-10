from .fedrolex_submodel import extract_submodel_densenet_structure as fedrolex_extract_submodel_densenet_structure
from .fedrolex_submodel import extract_submodel_resnet_structure as fedrolex_extract_submodel_resnet_structure


def extract_submodel_resnet_structure(ratio, global_model, round_n=None):
    # HeteroFL is just FedRolex with never rolling over the paramters, hence round_n=0
    return fedrolex_extract_submodel_resnet_structure(ratio, global_model, round_n=0)


def extract_submodel_densenet_structure(ratio, global_model, round_n=None):
    # HeteroFL is just FedRolex with never rolling over the paramters, hence round_n=0
    return fedrolex_extract_submodel_densenet_structure(ratio, global_model, round_n=0)
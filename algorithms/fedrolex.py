
from .fedavg import FedAvgEvaluation

from .fd import SubsetDevice, FederatedDropoutServer


class FedRolexServer(FederatedDropoutServer):
    _device_class = SubsetDevice
    _device_evaluation_class = FedAvgEvaluation

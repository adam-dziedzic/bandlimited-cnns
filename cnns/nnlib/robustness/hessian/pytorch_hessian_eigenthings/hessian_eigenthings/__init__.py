""" Top-level module for hessian eigenvec computation """
from hessian_eigenthings.power_iter import power_iteration,\
    deflated_power_iteration
from hessian_eigenthings.lanczos import lanczos
from hessian_eigenthings.hvp_operator import HVPOperatorParams,\
    compute_hessian_eigenthings
from hessian_eigenthings.hvp_operator import HVPOperatorInputs

__all__ = [
    'power_iteration',
    'deflated_power_iteration',
    'lanczos',
    'HVPOperatorParams',
    'HVPOperatorInputs',
    'compute_hessian_eigenthings'
]

name = 'hessian_eigenthings'

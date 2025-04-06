from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry

WEIGHT_INIT_REGISTRY = Registry("WEIGHT_INIT")  # noqa F401 isort:skip
WEIGHT_INIT_REGISTRY.__doc__ = """
Registry for weight-initializations, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def weight_init(cfg):
    """
    Initialize the layers, defined by ``cfg.MODEL.WEIGHT_INIT``.
    Note that it does not load any weights from ``cfg``.
    """
    weight_init = cfg.MODEL.WEIGHT_INIT
    model = WEIGHT_INIT_REGISTRY.get(weight_init)(cfg)
    _log_api_usage("modeling.weight_init." + weight_init)
    return model

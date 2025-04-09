import os
import pkgutil

# Dynamically import all submodules and add them to __all__
__all__ = []

current_dir = os.path.dirname(__file__)
for _, module_name, is_pkg in pkgutil.iter_modules([current_dir]):
    __all__.append(module_name)
    if is_pkg:
        __import__(f"{__name__}.{module_name}")
from typing import Dict, Type, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.input_adapters.base_input_adapter import BaseInputAdapter

_ADAPTER_REGISTRY: Dict[str, Type['BaseInputAdapter']] = {}


def register_adapter(cls):
    """Class decorator to register an input adapter by class name."""
    _ADAPTER_REGISTRY[cls.__name__] = cls
    return cls


def get_adapter(name: str) -> 'BaseInputAdapter':
    """Instantiate a registered adapter by name."""
    if name not in _ADAPTER_REGISTRY:
        raise ValueError(
            f'Unknown adapter: {name!r}. '
            f'Available: {list(_ADAPTER_REGISTRY.keys())}'
        )
    return _ADAPTER_REGISTRY[name]()


def list_adapters() -> List[str]:
    return list(_ADAPTER_REGISTRY.keys())
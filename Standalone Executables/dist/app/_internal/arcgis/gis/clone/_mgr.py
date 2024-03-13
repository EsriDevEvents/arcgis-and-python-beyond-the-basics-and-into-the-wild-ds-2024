from typing import Any, Dict, Union
from ._base import (
    BaseCloneDefinition,
    BaseCloneItemDefinition,
    BaseCloneTextItemDefinition,
)

_CLONE_REGISTRY = {}


###########################################################################
def register(
    item_type: str,
    cls: Union[
        BaseCloneDefinition, BaseCloneItemDefinition, BaseCloneTextItemDefinition
    ],
) -> bool:
    """
    Loads a custom cloner class into the clone registry

    ================    ===============================================================
    **Parameter**        **Description**
    ----------------    ---------------------------------------------------------------
    item_type           Required String. The name of the item type to clone.
    ----------------    ---------------------------------------------------------------
    cls                 Required Class. The class to perform the cloning operation.
    ================    ===============================================================

    :return: bool
    """
    global _CLONE_REGISTRY
    types = (
        BaseCloneDefinition,
        BaseCloneItemDefinition,
        BaseCloneTextItemDefinition,
    )
    if issubclass(cls, types) or isinstance(cls, types):
        _CLONE_REGISTRY[item_type] = cls
        return True
    return False


###########################################################################
def unregister(item_type: str) -> bool:
    """
    Loads a custom cloner class into the clone registry

    ================    ===============================================================
    **Parameter**        **Description**
    ----------------    ---------------------------------------------------------------
    item_type           Required String. The name of the item type to delete from the clone registry
    ================    ===============================================================

    :return: bool
    """
    global _CLONE_REGISTRY
    lll = [k.lower() for k in _CLONE_REGISTRY.keys()]
    if item_type.lower() in lll:
        idx = lll.index(item_type.lower())
        key = list(_CLONE_REGISTRY.keys())[idx]
        del _CLONE_REGISTRY[key]
        return True
    return False


###########################################################################
def clone_registry() -> Dict[str, Any]:
    """
    Returns the Clone Registry

    :return: Dict[str, Any]
    """
    global _CLONE_REGISTRY
    return _CLONE_REGISTRY

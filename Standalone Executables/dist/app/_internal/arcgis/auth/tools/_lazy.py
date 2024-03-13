import types
import importlib


###########################################################################
class LazyLoader(types.ModuleType):
    """
    Lazy load modules

    np = LazyLoader("numpy")
    pd = LazyLoader("pandas")
    arcpy = LazyLoader("arcpy", strict=True)
    cf = LazyLoader(module_name="concurrent.futures")

    """

    def __init__(self, module_name: str, submod_name=None, strict=False):
        """lazying loading of libraries

        Set Strict == True if the module is required
        """
        if strict:
            if LazyLoader.check_module_exists(module_name) == False:
                raise ModuleNotFoundError(f"Required {module_name} not found.")

        self._module_name = "{}{}".format(
            module_name, submod_name and ".{}".format(submod_name) or ""
        )
        self._mod = None
        super(LazyLoader, self).__init__(self._module_name)

    @staticmethod
    def check_module_exists(name: str) -> bool:
        """Checks if a module exists"""
        try:
            res = importlib.util.find_spec(name)
            if res is None:
                return False
            return True
        except:
            return False

    def _load(self):
        if self._mod is None:
            self._mod = importlib.import_module(self._module_name)
        return self._mod

    def __getattr__(self, attrb):
        return getattr(self._load(), attrb)

    def __dir__(self):
        return dir(self._load())

try:
    from traitlets import Dict
except ImportError:

    class Dict:
        pass


class observablelist(list):
    """Acts just like a `list` primitive, but calls `on_change()`
    every time the list mutates."""

    def __setitem__(self, *args, **kwargs):
        old = list(self)
        super().__setitem__(*args, **kwargs)
        new = list(self)

        self._on_change(old, new)

    def __delitem__(self, *args, **kwargs):
        old = list(self)
        super().__delitem__(*args, **kwargs)
        new = list(self)

        self._on_change(old, new)

    def append(self, *args, **kwargs):
        old = list(self)
        super().append(*args, **kwargs)
        new = list(self)

        self._on_change(old, new)

    def extend(self, *args, **kwargs):
        old = list(self)
        super().extend(*args, **kwargs)
        new = list(self)

        self._on_change(old, new)

    def insert(self, *args, **kwargs):
        old = list(self)
        super().insert(*args, **kwargs)
        new = list(self)

        self._on_change(old, new)

    def remove(self, *args, **kwargs):
        old = list(self)
        super().remove(*args, **kwargs)
        new = list(self)

        self._on_change(old, new)

    def pop(self, *args, **kwargs):
        old = list(self)
        super().pop(*args, **kwargs)
        new = list(self)

        self._on_change(old, new)

    def clear(self, *args, **kwargs):
        old = list(self)
        super().clear(*args, **kwargs)
        new = list(self)

        self._on_change(old, new)

    def _on_change(self, old, new):
        """Called everytime the list is mutated (i.e. my_list.append('foo'))
        Overwrite this function with what you want your callback to be

        Args
        ----
        old: a copy of the list object before the mutation
        new: a copy of the list object after the mutation
        """
        pass


class observabledict(dict):
    """Acts just like a `dict` primitive, but calls `on_change()`
    every time the dict mutates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replace_children_with_observables(self)

    def _replace_children_with_observables(self, dict_):
        """For each nested value in a multi-layer dict object,
        replace all `dict` subchildren with `observabledict`,
        replace all `list` subchildren with `observablelist`
        """
        if isinstance(dict_, dict):
            for key in dict_:
                value = dict_[key]
                if isinstance(value, dict):
                    dict_[key] = observabledict(value)
                    dict_[key]._on_change = self._child_on_change
                if isinstance(value, list):
                    dict_[key] = observablelist(value)
                    dict_[key]._on_change = self._child_on_change

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = observabledict(value)
            value._on_change = self._child_on_change
        elif isinstance(value, list):
            value = observablelist(value)
            value._on_change = self._child_on_change

        old = dict(self)
        super().__setitem__(key, value)
        new = dict(self)

        self._on_change(old, new)

    def __delitem__(self, *args, **kwargs):
        old = dict(self)
        super().__delitem__(*args, **kwargs)
        new = dict(self)

        self._on_change(old, new)

    def update(self, *args, **kwargs):
        old = dict(self)
        super().update(*args, **kwargs)
        new = dict(self)

        self._on_change(old, new)

    def _child_on_change(self, old_child, new_child):
        """Make any child `on_change` call bubble up the parent `on_change`,
        but with the correct old/new values. `self` has already been updated,
        so find what has changed and replace it with the old value
        """
        new = dict(self)
        old = dict(self)
        self._recurs_replace_child_with(old, new_child, old_child)

        self._on_change(old, new)

    def _recurs_replace_child_with(self, dict_, child_a, child_b):
        if isinstance(dict_, dict):
            for key in dict_:
                value = dict_[key]
                if value == child_a:
                    dict_[key] = child_b
                elif isinstance(value, dict):
                    self._recurs_replace_child_with(value, child_a, child_b)

    def _on_change(self, old, new):
        """Called everytime the dict is mutated (i.e. my_dict['foo'] = 'bar')
        Overwrite this function with what you want your callback to be

        Args
        ----
        old: a copy of the dictionary object before the mutation
        new: a copy of the dictionary object after the mutation
        """
        pass


class ObservableDict(Dict):
    """Mimics the traitlets `Dict` class, but will fire a `change` event when
    a dictionary object is updated (new value added, etc.). Can get hooked up
    with traitlets/ipywidget `observe()` func and some other thigns"""

    _parent = None

    def get(self, obj, cls):
        self._parent = obj
        orig_out = super().get(obj, cls)
        new_out = observabledict(orig_out)
        new_out._on_change = self._on_change
        self._parent.set_trait(self.name, new_out)
        return new_out

    def _on_change(self, old, new):
        self._parent.notify_change(
            {
                "name": self.name,
                "old": old,
                "new": new,
                "owner": self._parent,
                "type": "change",
            }
        )

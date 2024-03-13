""" Defines the Model abstract base class.
"""


class Model:
    """The abstract base class for all workforce model objects."""

    @property
    def id(self):
        raise NotImplementedError()

    def _validate(self, **kwargs):
        """
        Checks the validity of a model as it exists locally.
            :returns: A list of ValidationErrors
        """
        return []

    def _validate_for_add(self, **kwargs):
        """Checks the validity of a model, and ensures that the model is ready to be added to the
        backend.  This may involve executing queries against the backend to ensure that the
        model is valid in the context of the project.
        :returns: A list of ValidationErrors
        """
        return self._validate(**kwargs)

    def _validate_for_update(self, **kwargs):
        """Checks the validity of a model, and ensures that the model is ready to be updated on the
        backend.  This may involve executing queries against the backend to ensure that the
        model is valid in the context of the project.
        :returns: A list of ValidationErrors
        """

        return self._validate(**kwargs)

    def _validate_for_remove(self, **kwargs):
        """Checks to ensure that the model can be removed from the backend.  This may involve
        executing queries against the backend to ensure that the project integrity will be
        maintained when the model is removed.
        :returns: A list of ValidationErrors
        """
        return []

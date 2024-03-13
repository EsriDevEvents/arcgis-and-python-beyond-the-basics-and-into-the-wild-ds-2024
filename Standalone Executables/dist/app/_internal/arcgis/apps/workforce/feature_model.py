""" Defines the FeatureModel object.
"""

from arcgis.features import Feature
from .exceptions import ValidationError
from .model import Model
from .utils import to_arcgis_date, from_arcgis_date


class FeatureModel(Model):
    """An abstract base class for all Workforce model objects that are stored as Features."""

    def __init__(self, project=None, feature_layer=None, feature=None):
        """Creates a new instance that is owned by the project.
        :param project: The project that owns this model.
        :type project: Project
        """
        self._schema = None  # Implement in subclasses
        self.project = project
        self.feature_layer = feature_layer
        self._feature = feature
        if self._feature is None:
            self._feature = Feature(attributes={})

    @property
    def id(self):
        """The object (version 1) or global id (version 2) of the feature"""
        # Returns global ID in upper case form if it exists for v2 project, otherwise returns nothing
        if self.project._is_v2_project and self.global_id is not None:
            return self.global_id.upper()
        elif not self.project._is_v2_project:
            return self.object_id
        else:
            return None

    @property
    def feature(self):
        """The :class:`~arcgis.features.Feature`"""
        return self._feature

    @property
    def object_id(self):
        """The object id of the feature"""
        return self._feature.attributes.get(self._schema.object_id)

    @property
    def global_id(self):
        """The global id of the feature"""
        return self._feature.attributes.get(self._schema.global_id)

    @property
    def creator(self):
        """The named user that created the Feature."""
        return self._feature.attributes.get(self._schema.creator)

    @property
    def creation_date(self):
        """The :class:`datetime` at which the Feature was created."""
        return self._get_datetime_attr(self._schema.creation_date)

    @property
    def editor(self):
        """The named user that last edited the Feature."""
        return self._feature.attributes.get(self._schema.editor)

    @property
    def edit_date(self):
        """The :class:`datetime` at which the Feature was last edited."""
        return self._get_datetime_attr(self._schema.edit_date)

    @property
    def geometry(self):
        """Gets/Sets the geometry for the Feature."""
        return self._feature.geometry

    def _validate_for_update(self, **kwargs):
        """Checks the validity of a model, and ensures that the model is ready to be updated on the
        backend.
        :returns: A list of ValidationError
        """
        return super()._validate_for_update(**kwargs) + self._validate_object_id()

    def _validate_for_remove(self, **kwargs):
        """Checks to ensure that the model can be removed from the backend.
        :returns: A list of ValidationError
        """
        return super()._validate_for_remove(**kwargs) + self._validate_object_id()

    def _validate_object_id(self):
        errors = []
        if self.project._is_v2_project and self.object_id is None:
            errors.append(ValidationError("Model requires an object_id", self))
        return errors

    def _get_datetime_attr(self, name):
        """Gets an attribute and converts it to a datetime.
        :param name: The attribute name
        :returns: The value of the attribute converted to a datetime, or None if the attribute
        is null or not present.
        :rtype: datetime.datetime or None
        """
        timestamp = self._feature.attributes.get(name)
        return from_arcgis_date(timestamp) if timestamp is not None else None

    def _set_datetime_attr(self, name, value):
        """Sets an attribute to a datetime value.
        :param name: The attribute name
        :param value: The datetime value
        :type value: datetime.datetime
        """
        timestamp = to_arcgis_date(value) if value is not None else None
        self._feature.attributes[name] = timestamp

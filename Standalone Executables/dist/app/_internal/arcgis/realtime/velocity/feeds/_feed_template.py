from collections import abc
from dataclasses import dataclass, field
from typing import ClassVar

from arcgis.realtime.velocity._reserved_fields import _ReservedFields
from .time import _START_TIME_TAG, _END_TIME_TAG

_TRACK_ID_TAG: str = "TRACK_ID"


@dataclass
class _FeedTemplate:
    """
    Base Class for Feeds
    """

    # Non default fields or fields that will not be initialized by constructor
    label: str
    description: str

    # abstract properties - to be assigned by all derived classes
    _name: ClassVar[str]

    _fields: dict = field(default_factory=dict, init=False)

    # ---> Inheriting classes MUST declare the following Optional fields. <---
    # These are commented variables because of a limitation in dataclass where base class properties get ordered before
    # the derived class properties. Since these are optional properties, the init will order them before the non-optional
    # derived class properties leading to errors like - TypeError: non-default argument 'rss_url' follows default argument
    #
    # track_id_field: Optional[str] = None

    # abstract methods to be implemented by all Feeds
    def _build(self) -> dict:
        """
        Build the feed configuration object from the current state of this object. This feed configuration can be posted to the API endpoint.
        To be implemented by concrete classes.
        For example:

        :return: Feed configuration
        """
        raise NotImplemented()

    def _generate_feed_properties(self) -> dict:
        """
        Builds part of the dictionary object that is pertinent to feed and format properties from the current state of this object.
        To be implemented by concrete classes.

        :return: Dictionary object that contains feed and format properties which can be used to POST request to velocity.
        """
        raise NotImplemented

    def _generate_schema_transformation(self) -> dict:
        """
        Builds the final Schema-transformation dictionary object from the current state of _fields object in the format that can be used
        to create POST request to ArcGIS Velocity.

        :return: Dictionary object that contains the schema-transformation properties.
        """
        # validate feature_schema
        if self._fields is None or not self._fields["attributes"]:
            raise ValueError("Invalid feed schema. Cannot proceed")

        input_schema = {}
        field_mappings = []

        input_schema = self._fields.copy()
        for field in input_schema["attributes"]:
            if field["toField"]:
                if _ReservedFields.is_reserved(field["toField"]):
                    # A toField cannot be one of the reserved names.
                    raise ValueError(
                        f"'{field['toField']}' is a reserved field name. It must be renamed or dropped"
                    )

                field_mappings.append(
                    {
                        "fromField": field["name"],
                        "toField": field["toField"],
                        "tags": field["tags"],
                    }
                )

            # input_schema's attributes should not have "toField" property
            del field["toField"]

        return {
            "schemaTransformation": {
                "inputSchema": input_schema,
                "fieldMappings": field_mappings,
            }
        }

    # Feature schema manipulation methods
    def _set_fields(self, feature_schema):
        """
        Reads the feature schema config from the sample_messages response json/dict into self._fields object property.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        feature_schema         dict. The sample_messages_response["featureSchema"] json/dict object.
        ==================     ====================================================================
        """
        self._fields = feature_schema

        # validate feature_schema
        if self._fields is None or not self._fields["attributes"]:
            raise ValueError("Invalid feed schema. Cannot proceed")

        for field in self._fields["attributes"]:
            # add a property 'toField' to each attribute. This property will be used to generate the fieldMappings dict.
            field["toField"] = field["name"]

    def rename_field(self, current_name: str, new_name: str) -> bool:
        """
        Rename a field.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        current_name           String. Current field name.
        ------------------     --------------------------------------------------------------------
        new_name               String. New field name.
        ==================     ====================================================================

        :return: Field collection after transformation
        """
        if not new_name.strip():
            raise ValueError("new_name cannot be empty")
        elif _ReservedFields.is_reserved(new_name):
            raise ValueError(
                f"'{new_name}' is a reserved field name and cannot be used."
            )

        if self._fields is not None and self._fields["attributes"]:
            attributes = self._fields["attributes"]
            is_success = False
            for attr in attributes:
                if attr["name"] == current_name:
                    attr["toField"] = new_name
                    is_success = True
            if is_success:
                return True
            else:
                raise ValueError(f"could not find field `{current_name}` in the schema")
        else:
            raise Exception("Invalid feed schema. Cannot proceed")

    def change_field_data_type(self, name, new_data_type) -> bool:
        """
        Used to specify the expected data type of a field.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   String. Field name.
        ------------------     --------------------------------------------------------------------
        new_data_type          String. New data type for the field.
        ==================     ====================================================================

        :return: Boolean - True if data type change was successful
        """
        is_success = False
        if self._fields is not None and self._fields["attributes"]:
            fields = self._fields["attributes"]
            for field in fields:
                if field["name"] == name or field["toField"] == name:
                    field["dataType"] = new_data_type
                    is_success = True

            if is_success:
                return True
            else:
                raise ValueError(f"could not find field `{name}` in the schema")
        else:
            raise Exception("Invalid feed schema. Cannot proceed")

    def remove_field(self, name: str) -> bool:
        """
        Remove a field from the schema.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        name                   String. Field to be removed from the schema.
        ==================     ====================================================================

        :return: Boolean - True if a field is removed, False otherwise
        """
        is_success = False
        if self._fields is not None and self._fields["attributes"]:
            fields = self._fields["attributes"]
            for field in fields:
                if field["name"] == name:
                    # flag the field for removal from schema by clearing the `toField` value
                    field["toField"] = None
                    if _TRACK_ID_TAG in field["tags"]:
                        field["tags"].clear()
                    elif any(
                        elem in [_START_TIME_TAG, _END_TIME_TAG]
                        for elem in field["tags"]
                    ):
                        self.reset_time_config()

                    is_success = True

        if is_success:
            return True
        else:
            raise ValueError(f"could not find field `{name}` in the schema")

    def set_track_id(self, field_name: str):
        """
         Set the track ID field for the feed.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        field_name             String. Name of the track ID field. Either the original name or renamed
                               field name can be used to specify the track ID.
        ==================     ====================================================================
        """
        is_success = False
        for field in self._fields["attributes"]:
            if field["name"] == field_name or field["toField"] == field_name:
                field["tags"] = [_TRACK_ID_TAG]
                is_success = True
            elif _TRACK_ID_TAG in field["tags"]:
                field["tags"].clear()

        if is_success:
            if self.track_id_field is None:
                self.track_id_field = field_name
            return True
        else:
            raise ValueError(f"invalid field_name: '{field_name}'")

    def _dict_deep_merge(self, dct, merge_dct):
        """Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
        updating only top-level keys, dict_merge recurses down into dicts nested
        to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
        ``dct``.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        dct                    String. Dict onto which the merge is executed.
        ------------------     --------------------------------------------------------------------
        merge_dct              String. This dict will be merged into dct.
        ==================     ====================================================================
        """
        # Future enhancement - Move this to a util?
        # Or use https://anaconda.org/conda-forge/deepmerge package instead
        for k, v in merge_dct.items():
            if (
                k in dct
                and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], abc.Mapping)
            ):
                self._dict_deep_merge(dct[k], merge_dct[k])
            elif k in dct:
                raise ValueError(
                    f"property name collision found for `{k}` key. Found in dct and merge_dct dictionaries"
                )
            else:
                dct[k] = merge_dct[k]

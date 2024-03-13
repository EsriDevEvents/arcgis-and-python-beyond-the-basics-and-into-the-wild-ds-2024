"""
A module for managing forms in the ArcGIS platform
"""

import json
from typing import Optional, Union
from arcgis import mapping
from arcgis._impl.common._mixins import PropertyMap
from arcgis.gis import Item
from arcgis.features import FeatureLayer
import copy


class FormCollection:
    """
    Represents a collection of forms in a webmap or item. A form is the editable counterpart to a popup
    -- it controls the appearance and behavior of your data collection experience in ArcGIS Field Maps
    and Map Viewer Beta.  These forms can then be used in the ArcGIS Field Maps mobile app and other applications.
    A form is stored as "formInfo" in the layer JSON on the webmap.
    This class will create a :class:`~arcgis.mapping.forms.FormInfo` object for each layer or table in the webmap/item data and return it
    as a list. You can then modify the :class:`~arcgis.mapping.forms.FormInfo` object to add and edit your form.

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    parent                 Required :class:`~arcgis.mapping.WebMap` or :class:`~arcgis.gis.Item`.
                           This is the object which contains the layer, either an item of type
                           :class:`Feature Layer Collection <arcgis.features.FeatureLayerCollection>`
                           or a :class:`Web Map <arcgis.mapping.WebMap>`, where the forms are located.
                           This is needed to save your form changes to the backend.
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Get Form Collection from WebMap
        wm = arcgis.mapping.WebMap(item)
        wm.add_layer(manhole_inspection)

        # get forms from webmap, get individual form from FormCollection, modify form
        form_collection = wm.forms
        form_info_2 = form_collection[0]
        form_info = form_collection.get(title="Manhole Inspection")
        form_info.clear()
        form_info.add_field(field_name="inspector", label="Inspector", description="This is the inspector")
        form_info.add_group(label="Group 1",initial_state="collapsed")
        form_info.update()

        # USAGE EXAMPLE 2: Create Form Collection

        from arcgis.mapping.forms import FormCollection
        wm = arcgis.mapping.WebMap(item)
        form_collection = FormCollection(wm)
        form_info_1 = form_collection.get(item_id="232323232323232323")
        form_info_1.title = "New Form"

    """

    def __init__(self, parent):
        self._parent = parent
        self._index = 0
        self._refresh_forms(parent)

    def __getitem__(self, key):
        return self.forms[key]

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self.forms):
            raise StopIteration
        else:
            self._index += 1
            return self.forms[self._index - 1]

    def _refresh_forms(self, parent):
        if isinstance(parent, mapping.WebMap):
            self.forms = self._get_forms_from_webmap(parent)
        elif isinstance(parent, Item):
            if parent.type != "Feature Layer Collection":
                raise ValueError("Item must be feature layer collection to have forms")
            self.forms = self._get_forms_from_item(parent)
        else:
            raise ValueError("Parent item must be webmap or feature layer collection")

    def get(
        self,
        item_id: Optional[str] = None,
        title: Optional[str] = None,
        layer_id: Optional[str] = None,
    ):
        """
        Returns the form for the first layer with a matching item_id, title, or layer_id in the webmap's
        operational layers. Pass one of the three parameters into the method to return the form.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        item_id                Optional :class:`str`. Pass the item_id for the layer or table whose
                               form you'd like to return.
        ------------------     --------------------------------------------------------------------
        title                  Optional :class:`str`. Pass the title for the layer or table whose
                               form you'd like to return.
        ------------------     --------------------------------------------------------------------
        layer_id               Optional :class:`str`. Pass the layer_id for the layer whose form
                               you'd like to return
        ==================     ====================================================================

        :return: :class:`~arcgis.mapping.forms.FormInfo` or `None`
        """
        if item_id is None and title is None and layer_id is None:
            raise ValueError("Please pass at least one parameter into the function")
        self._refresh_forms(self._parent)
        if self.forms:
            for form in self.forms:
                # item id is optional in the webmap spec, so we need to try/except
                try:
                    if (
                        (title == form._layer_data["title"])
                        or (layer_id == form._layer_data["id"])
                        or (item_id == form._layer_data["itemId"])
                    ):
                        return form
                except Exception:
                    pass
        return None

    @staticmethod
    def _construct_forms_array(forms, layers, parent):
        """This is a shared method which creates an array of forms if given an array of layers."""
        # we save parent here into the form in order to distinguish between whether from_item or from_webmap was used and to get a usable GIS
        for layer in layers:
            FormCollection._get_forms_within_group_layer(forms, layer, parent)

    @staticmethod
    def _get_forms_within_group_layer(forms, layer, parent, subtype_gl_layer=None):
        if layer.layerType == "GroupLayer":
            for sub_layer in reversed(layer.layers):
                FormCollection._get_forms_within_group_layer(forms, sub_layer, parent)
        elif layer.layerType == "SubtypeGroupLayer":
            for sub_layer in reversed(layer.layers):
                FormCollection._get_forms_within_group_layer(
                    forms, sub_layer, parent, subtype_gl_layer=layer
                )
        elif layer.layerType == "ArcGISFeatureLayer" and not hasattr(
            layer, "featureCollection"
        ):
            forms.append(FormInfo(layer, parent, subtype_gl_layer))

    def _get_forms_from_item(self, item):
        """Populates self.forms given an item."""
        # we only support from_item for the instance where the form is saved into the feature layer item
        forms = []
        item_data = item.get_data()
        if "layers" in item_data:
            self._construct_forms_array(forms, item_data["layers"], parent=item)
        return forms

    def _get_forms_from_webmap(self, webmap):
        """Populates self.forms given a webmap."""
        forms = []
        self._construct_forms_array(forms, webmap.layers, parent=webmap)
        self._construct_forms_array(forms, webmap.tables, parent=webmap)
        return forms


class FormInfo:
    """
    Represents a form in ArcGIS Field Maps and other applications. This matches with
    the formInfo property in a webmap's operational layer.

    For more please see: https://developers.arcgis.com/web-map-specification/objects/formInfo/

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    layer_data             Required :class:`PropertyMap` or :class:`dict`. This is the
                           operational layer which contains the formInfo dict. It can be
                           retrieved from a webmap using `arcgis.mapping.WebMap(item).layers[0]`
    ------------------     --------------------------------------------------------------------
    parent                 Required :class:`~arcgis.mapping.WebMap` or :class:`~arcgis.gis.Item`.
                           This is the object which contains the layer, either an item of type
                           `Feature Layer Collection` or a webmap. This is needed to save your
                           form changes to the backend.
    ------------------     --------------------------------------------------------------------
    subtype_gl_data        Optional :class:`PropertyMap` or :class:`dict`. This is the
                           operational layer representing a subtype group layer which contains
                           the layer containing the form. It can be retrieved from a webmap
                           using `arcgis.mapping.WebMap(item).layers[0]`
    ==================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Modify FormInfo
            from arcgis.mapping.forms import FormFieldElement
            wm = arcgis.mapping.WebMap(item)
            wm.add_layer(manhole_inspection)
            form_collection = wm.forms
            form_info = form_collection.get_form(title="Manhole Inspection")

            # edit form properties
            form_info.title = "Manhole Inspection Form"
            form_info.description = "The editable experience in ArcGIS Field Maps for data collection"

            # get element, add element
            new_element = FormFieldElement(label="Inspector Name", field_name="inspectornm",
                                          editable=True)
            form_info.add(element=new_element)
            same_element = form_info.get(label="Inspector Name")

            # add group, add second group, add to group
            form_info.add_group(label="Inspector Group",description="This is a group for inspectors")
            new_group = FormGroupElement(label="Group 1", description="New Group")
            group = form_info.add(element=new_group)
            group.add_field(field_name="inspection_date", label="Inspection Date")

            # move element, delete element
            form_info.move(element=new_group, index=0)
            form_info.delete(element=new_group)

            # save form into backend
            form_info.update()
    """

    def __init__(self, layer_data, parent, subtype_gl_data=None, **kwargs):
        if not isinstance(layer_data, (dict, PropertyMap)):
            raise ValueError(
                "Incorrect layer type passed to FormInfo class. Please pass in a property map"
            )
        if subtype_gl_data and not isinstance(subtype_gl_data, (dict, PropertyMap)):
            raise ValueError(
                "Incorrect subtype group layer type passed to FormInfo class. Please pass in a property map"
            )
        self._kwargs = kwargs
        self._subtype_group_layer_data = subtype_gl_data
        self._original_layer = layer_data
        self._layer_data = copy.deepcopy(layer_data)
        self._form = self._layer_data.get("formInfo", {})
        self._title = self._form.get("title", self._layer_data.get("title", None))
        self._description = self._form.get("description")
        self._expression_infos = []
        for exp in self._form.get("expressionInfos", []):
            expression = FormExpressionInfo(
                expression=exp.get("expression"),
                name=exp.get("name"),
                title=exp.get("title"),
                return_type=exp.get("returnType"),
            )
            self._expression_infos.append(expression)
        self._form_elements = self._get_form_element_objects(
            self._form.get("formElements", []), self
        )
        self._parent = parent
        try:
            if self._subtype_group_layer_data:
                url = self._subtype_group_layer_data["url"]
            else:
                url = self._original_layer["url"]
            self.feature_layer = FeatureLayer(url=url, gis=self._parent._gis)
            self._fields = self._get_fields()
            self._edit_fields = self._get_edit_fields()
            self._id_fields = self._get_id_fields()
            self._required_fields = self._get_required_fields()
        except Exception:
            raise ValueError(
                "A layer url which can be used to generate a feature layer is required to use this module"
            )

    def __repr__(self):
        if self.title:
            return "Form: " + self.title
        else:
            return "Form"

    def __str__(self):
        if self.exists():
            return json.dumps(self.to_dict(), indent=2)
        return "None"

    def exists(self):
        """Returns whether the form exists for that particular layer or not."""
        return len(self._form_elements) > 0

    def clear(self):
        """Clears the form to an empty state. Deletes all form elements currently in the form."""
        self._expression_infos = []
        self._form_elements = []
        self._title = None
        self._description = None

    def get(self, label: Optional[str] = None):
        """
        Returns a matching FormElement given a label

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        label                  Optional :class:`str`
                               The label of the form element you want to return from the list of
                               form elements
        ==================     ====================================================================

        :return: :class:`~arcgis.mapping.forms.FormFieldElement` or :class:`~arcgis.mapping.forms.FormGroupElement` or `None`
        """
        try:
            for el in self._form_elements:
                if el.label.lower() == label.lower():
                    return el
                if el.element_type == "group":
                    for grouped_field in el._form_elements:
                        if grouped_field.label.lower() == label.lower():
                            return grouped_field
        except Exception:
            return None

    def feature_layer(self):
        """The feature layer associated with the form"""
        return self.feature_layer

    @property
    def title(self):
        """Gets/sets the title of the form"""
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    @property
    def description(self):
        """Gets/sets the description of the form"""
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def elements(self):
        """Returns elements in the form to the user - a list of :class:`~arcgis.mapping.forms.FormElement`"""
        return self._form_elements

    @property
    def expressions(self):
        """
        Returns Arcade expressions used in the form to the user - a list of :class:`~arcgis.mapping.forms.FormExpressionInfo`
        """
        return self._expression_infos

    def update(self):
        """
        Saves the form to the backend. If the form was derived from an :class:`~arcgis.gis.Item`, calling this function is required
        to save the form into the item. If the form was derived from a :class:`~arcgis.mapping.WebMap`, you can either call this
        function or :attr:`~arcgis.mapping.WebMap.update()`. If form has been cleared, removes formInfo from webmap
        """
        if self.exists():
            self._validate_all_required_fields_in_form()
        if isinstance(self._parent, Item):
            item_data = self._parent.get_data()
            if self.exists():
                item_data["layers"][self._layer_data["id"]]["formInfo"] = self.to_dict()
            else:
                item_data["layers"][self._layer_data["id"]].pop("formInfo", None)
            self._parent.update(data=item_data)
        if isinstance(self._parent, mapping.WebMap):
            if self.exists():
                self._original_layer["formInfo"] = self.to_dict()
            else:
                self._original_layer.pop("formInfo", None)
            try:
                self._parent.update()
            except RuntimeError:
                raise ValueError(
                    "WebMap item does not exist yet. Form is now on your webmap - please use WebMap.save() to persist these changes"
                )
        else:
            pass

    def to_dict(self):
        self._hydrate_expression_infos()
        data = {
            "expressionInfos": [exp.to_dict() for exp in self._expression_infos],
            "formElements": [element.to_dict() for element in self._form_elements],
        }
        if self._description:
            data["description"] = self._description
        if self._title:
            data["title"] = self._title
        for key, value in self._kwargs.items():
            data[key] = value
        return data

    def add_all_attributes(self):
        """Adds all fields which can be valid form elements (string, date, int, double, small) to the form."""
        fields = self._fields
        for field in fields:
            try:
                element = FormFieldElement(
                    label=field.get("alias"),
                    editable=field.get("editable"),
                    field_name=field.get("name"),
                )
                self.add(element=element)
            except Exception:
                continue

    def add(
        self,
        element=None,
        index: Optional[int] = None,
    ):
        """
        Adds a single :class:`~arcgis.mapping.forms.FormElement` to the form. You can add to the form either by instantiating your
        own :class:`~arcgis.mapping.forms.FormFieldElement` or :class:`~arcgis.mapping.forms.FormGroupElement` and passing it into the element parameter here,
        or you can type in a valid field_name in the form's layer and this function will attempt
        to add it to the form.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        element                Optional :class:`~arcgis.mapping.forms.FormFieldElement` or
                               :class:`~arcgis.mapping.forms.FormGroupElement`.
        ------------------     --------------------------------------------------------------------
        index                  Optional :class:`int`.
                               The index where you'd like the element in the form. If not provided,
                               this function will add the new element to the end of the form.
        ==================     ====================================================================

        :return: The element that was added - :class:`~arcgis.mapping.forms.FormFieldElement` or :class:`~arcgis.mapping.forms.FormGroupElement`
        """
        self._validate_input(element=element)
        self._validate_element(element)
        if index is None:
            index = len(self._form_elements)
        self._form_elements.insert(index, element)
        self._hydrate_expression_infos()
        return element

    def add_field(
        self,
        field_name: str,
        label: str,
        description: Optional[str] = None,
        visibility_expression=None,
        domain: Optional[dict] = None,
        editable: Optional[bool] = None,
        hint: Optional[str] = None,
        input_type: Optional[Union[str, dict]] = None,
        required_expression=None,
        index: Optional[int] = None,
        editable_expression=None,
        value_expression=None,
        **kwargs,
    ):
        """
        Adds a single field :class:`~arcgis.mapping.forms.FormElement` element to the end of the form.

        For more please see: https://developers.arcgis.com/web-map-specification/objects/formFieldElement/

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        field_name                 Required :class:`str`.
                                   The field name the form element corresponds to (where the data
                                   will be collected)
        ----------------------     --------------------------------------------------------------------
        label                      Required :class:`str`.
                                   The label of the form element
        ----------------------     --------------------------------------------------------------------
        description                Optional :class:`str`.
                                   The description of the form element
        ----------------------     --------------------------------------------------------------------
        visibility_expression      Optional :class:`~arcgis.mapping.forms.FormExpressionInfo`.
                                   The conditional visibility Arcade expression determining the
                                   visibility of the form element during data collection
        ----------------------     --------------------------------------------------------------------
        domain                     Optional :class:`dict`.
                                   The domain of the form element
        ----------------------     --------------------------------------------------------------------
        editable                   Optional :class:`bool`.
                                   Whether or not the form element is editable
        ----------------------     --------------------------------------------------------------------
        hint                       Optional :class:`str`.
                                   The hint for the user filling out the form element
        ----------------------     --------------------------------------------------------------------
        input_type                 Optional :class:`str` or :class:`dict`.
                                   The input type for the form element in ArcGIS Field Maps.
                                   Options include: "text-area", "text-box", "barcode-scanner",
                                   "combo-box", "radio-buttons", "datetime-picker"
        ----------------------     --------------------------------------------------------------------
        required_expression        Optional :class:`~arcgis.mapping.forms.FormExpressionInfo`.
                                   The conditional visibility Arcade expression determining the
                                   necessity of the form element during data collection
        ----------------------     --------------------------------------------------------------------
        index                      Optional :class:`int`.
                                   The index where you'd like the element in the form. If not provided,
                                   this function will add the new element to the end of the form.
        ----------------------     --------------------------------------------------------------------
        editable_expression        Optional :class:`~arcgis.mapping.forms.FormExpressionInfo`.
                                   The Arcade expression determining the
                                   editablity of the form element during data collection
        ----------------------     --------------------------------------------------------------------
        value_expression           Optional :class:`~arcgis.mapping.forms.FormExpressionInfo`.
                                   The Arcade expression which calculates a value for the form
                                   element during data collection
        ======================     ====================================================================

        :return: The element that was added - :class:`~arcgis.mapping.forms.FormGroupElement`
        """

        element = FormFieldElement(
            form=self,
            field_name=field_name,
            label=label,
            description=description,
            visibility_expression=visibility_expression,
            domain=domain,
            editable=editable,
            hint=hint,
            input_type=input_type,
            required_expression=required_expression,
            editable_expression=editable_expression,
            value_expression=value_expression,
            **kwargs,
        )
        return self.add(element, index=index)

    def add_group(
        self,
        label: str,
        description: Optional[str] = None,
        visibility_expression=None,
        initial_state: Optional[str] = None,
        index: Optional[int] = None,
        **kwargs,
    ):
        """
        Adds a single :class:`~arcgis.mapping.forms.GroupElement` to the form

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        label                      Required :class:`str`. The label of the group
        ----------------------     --------------------------------------------------------------------
        description                Optional :class:`str`. The description of the group
        ----------------------     --------------------------------------------------------------------
        visibility_expression      Optional :class:`~arcgis.mapping.forms.FormExpressionInfo`.
                                   The conditional visibility Arcade expression determining the
                                   visibility of the form element during data collection
        ----------------------     --------------------------------------------------------------------
        initial_state              Optional :class:`str`. The initial state of the group
        ----------------------     --------------------------------------------------------------------
        index                      Optional :class:`int`.
                                   The index where you'd like the element in the form. If not provided,
                                   this function will add the new element to the end of the form.
        ======================     ====================================================================

        :return: The element that was added - :class:`~arcgis.mapping.forms.FormGroupElement`
        """
        group_el = FormGroupElement(
            label=label,
            description=description,
            visibility_expression=visibility_expression,
            initial_state=initial_state,
            **kwargs,
        )
        return self.add(group_el, index=index)

    def delete(
        self,
        element=None,
        label: Optional[str] = None,
    ):
        """
        Deletes element from the form. You can use either the element param
        with a form element you get using :attr:`~arcgis.mapping.FormInfo.get()` or you can pass the label of the
        form element you'd like to move into the label param.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        element                Optional :class:`~arcgis.mapping.forms.FormFieldElement` or
                               :class:`~arcgis.mapping.forms.FormGroupElement`
        ------------------     --------------------------------------------------------------------
        label                  Optional :class:`str`.
                               An actual field name (not alias) corresponding to a field in the
                               form's feature layer
        ==================     ====================================================================

        :return: :class:`~arcgis.mapping.forms.FormFieldElement` or
                 :class:`~arcgis.mapping.forms.FormGroupElement`
                 or `False`
        """
        self._validate_input(element=element, field=label)
        if label:
            element = self.get(label=label)
        try:
            self._form_elements.remove(element)
            self._hydrate_expression_infos()
            return element
        except Exception:
            return False

    def move(
        self,
        element=None,
        label: Optional[str] = None,
        destination=None,
        index: Optional[int] = None,
    ):
        """
        Moves a form element in the form to a new location. You can use either the element param
        with a form element you get using :attr:`~arcgis.mapping.FormInfo.get()` or you can pass the label of the
        form element you'd like to move into the label param.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        element                Optional :class:`~arcgis.mapping.forms.FormFieldElement` or
                               :class:`~arcgis.mapping.forms.FormGroupElement`
        ------------------     --------------------------------------------------------------------
        label                  Optional :class:`str`
                               The label of the form element you'd like to move to a new location
        ------------------     ----------------------------------------------------------------------
        destination            Optional `arcgis.mapping.forms.FormGroupElement`
                               If moving an element into a group, please
                               provide this parameter. Otherwise, you will only be allowed to move
                               within the form.
        ------------------     --------------------------------------------------------------------
        index                  Optional :class:`int`.
                               The index where you'd like the element to move to in the form.
        ==================     ====================================================================

        :return: `True`
        """
        if index is None:
            raise ValueError("Please provide an index to this function")
        self._validate_input(element=element, field=label)
        if label:
            element = self.get(label=label)
        self.delete(element)
        if destination is None:
            destination = self
        return destination.add(element, index=index)

    def _get_matching_field(self, field_name):
        """Get the feature layer field given a popup field."""
        for field in self._fields:
            if field_name.lower() == field["name"].lower():
                return field
        else:
            raise ValueError("No matching field in the layer")

    def _get_fields(self):
        """Get feature layer's fields"""
        return self.feature_layer.properties.get("fields")

    @staticmethod
    def _is_valid_field_type(field_type):
        """Check the field type from the feature layer field is valid to be added to the form"""
        return field_type in [
            "esriFieldTypeDate",
            "esriFieldTypeDouble",
            "esriFieldTypeInteger",
            "esriFieldTypeSingle",
            "esriFieldTypeSmallInteger",
            "esriFieldTypeString",
            "esriFieldTypeGUID",
        ]

    @staticmethod
    def _get_default_input_type(field):
        """Gets the default input type based on the field type."""
        if field.get("domain", None):
            return "combo-box"
        if "date" in field["type"].lower():
            return "datetime-picker"
        return "text-box"

    @staticmethod
    def _validate_input(element=None, field=None):
        """Validate the inputs provided to add are valid."""
        if element and field:
            raise ValueError("Please use either element or field, not both")
        if element and not isinstance(element, FormElement):
            raise ValueError("Please pass a form element into the element parameter")
        if field and not isinstance(field, str):
            raise ValueError("Please pass a string into the element parameter")

    def _validate_element(self, element):
        """Validate the element passed to or created by add element is correct."""
        if element._form is None:
            element._form = self
        if not element.label:
            raise ValueError("Element must have label")
        if element.element_type == "field":
            matching_field = self._get_matching_field(element.field_name)
            if self._is_valid_field_type(matching_field["type"]):
                if not element.input_type:
                    element.input_type = self._get_default_input_type(matching_field)
            else:
                raise ValueError("Not a valid field type to add to the form")
            if self._is_geometry_field(element.field_name):
                raise ValueError("Cannot add a geometry field to the form")
            if not self._validate_unrestricted_field_name(element.field_name.lower()):
                raise ValueError(
                    "Cannot add a GPS metadata or editor tracking fields to the form"
                )
            for form_el in self._form_elements:
                if form_el.element_type == "group":
                    if element.field_name in [
                        el.field_name for el in form_el._form_elements
                    ]:
                        raise ValueError(
                            "Field already exists in a group, cannot add to form"
                        )
                else:
                    if element.field_name == form_el.field_name:
                        raise ValueError(
                            "Field already exists in the form, cannot add to the form"
                        )
        elif element.element_type == "group":
            for el in element.elements:
                self._validate_element(el)

    def _validate_unrestricted_field_name(self, field_name):
        """Validates the field is not a GPS metdata, edit, or id field."""
        return (
            "esrignss" not in field_name
            and "esrisnsr" not in field_name
            and field_name not in self._edit_fields
            and field_name not in self._id_fields
        )

    def _is_geometry_field(self, field_name):
        return "geometryProperties" in self.feature_layer.properties and (
            (
                "shapeAreaFieldName"
                in self.feature_layer.properties["geometryProperties"]
                and self.feature_layer.properties["geometryProperties"][
                    "shapeAreaFieldName"
                ].lower()
                == field_name.lower()
            )
            or (
                "shapeLengthFieldName"
                in self.feature_layer.properties["geometryProperties"]
                and self.feature_layer.properties["geometryProperties"][
                    "shapeLengthFieldName"
                ].lower()
                == field_name.lower()
            )
        )

    def _get_id_fields(self):
        """Returns the id fields in lower case."""
        try:
            return [
                self.feature_layer.properties.get("objectIdField").lower(),
                self.feature_layer.properties.get("globalIdField").lower(),
            ]
        except Exception:
            return []

    def _get_edit_fields(self):
        """Gets the edit fields for the feature layer in order to filter them out of the form."""
        try:
            return [
                x.lower()
                for x in list(
                    self.feature_layer.properties.get("editFieldsInfo", {}).values()
                )
            ]
        except Exception:
            return []

    @staticmethod
    def _get_expression_info(form, expression_name):
        return next((e for e in form.expressions if e.name == expression_name), None)

    @staticmethod
    def _get_form_element_objects(form_elements, form=None):
        """Shared between FormInfo and FormGroupElement to construct an array of FormElement objects from dictionaries for external usage."""
        elements = []
        for element in form_elements:
            if element["type"] == "field":
                el = FormFieldElement(
                    form=form,
                    description=element.get("description"),
                    label=element.get("label"),
                    visibility_expression=element.get("visibilityExpression"),
                    domain=element.get("domain"),
                    editable=element.get("editable"),
                    field_name=element.get("fieldName"),
                    hint=element.get("hint"),
                    input_type=element.get("inputType"),
                    required_expression=FormInfo._get_expression_info(
                        form, element.get("requiredExpression")
                    )
                    if form
                    else None,
                    editable_expression=FormInfo._get_expression_info(
                        form, element.get("editableExpression")
                    )
                    if form
                    else None,
                    value_expression=FormInfo._get_expression_info(
                        form, element.get("valueExpression")
                    )
                    if form
                    else None,
                )
            elif element["type"] == "group":
                el = FormGroupElement(
                    form=form,
                    elements=element.get("formElements"),
                    initial_state=element.get("initialState"),
                    description=element.get("description"),
                    label=element.get("label"),
                    visibility_expression=FormInfo._get_expression_info(
                        form, element.get("visibilityExpression")
                    )
                    if form
                    else None,
                )
            else:
                el = FormElement(
                    form=form,
                    element_type=element.get("type"),
                    description=element.get("description"),
                    label=element.get("label"),
                    visibility_expression=FormInfo._get_expression_info(
                        form, element.get("visibilityExpression")
                    )
                    if form
                    else None,
                )
            elements.append(el)
        return elements

    def _hydrate_expression_infos(self):
        for form_el in self._form_elements:
            self._hydrate_element_expressions(form_el)
            if form_el.element_type == "group":
                for el in form_el.elements:
                    self._hydrate_element_expressions(el)

    def _hydrate_element_expressions(self, element):
        """For each element's visibility and required expression, check if it exists in the expression info list. If it's a :class:`~arcgis.mapping.forms.FormExpressionInfo` object,
        add it to the list. If it's not, remove the expression as it does not point to anything and we can't form a :class:`~arcgis.mapping.forms.FormExpressionInfo`
        """
        if element.visibility_expression:
            if isinstance(element.visibility_expression, FormExpressionInfo):
                if (
                    self._get_expression_info(self, element.visibility_expression.name)
                    is None
                ):
                    self._expression_infos.append(element.visibility_expression)
            else:
                if (
                    self._get_expression_info(self, element.visibility_expression)
                    is None
                ):
                    element._visibility_expression = None
        if element.element_type == "field":
            if element.required_expression:
                if isinstance(element.required_expression, FormExpressionInfo):
                    if (
                        self._get_expression_info(
                            self, element.required_expression.name
                        )
                        is None
                    ):
                        self._expression_infos.append(element.required_expression)
                else:
                    if (
                        self._get_expression_info(self, element.required_expression)
                        is None
                    ):
                        element._required_expression = None
            if element.editable_expression:
                if isinstance(element.editable_expression, FormExpressionInfo):
                    if (
                        self._get_expression_info(
                            self, element.editable_expression.name
                        )
                        is None
                    ):
                        self._expression_infos.append(element.editable_expression)
                else:
                    if (
                        self._get_expression_info(self, element.editable_expression)
                        is None
                    ):
                        element._editable_expression = None
            if element.value_expression:
                if isinstance(element.value_expression, FormExpressionInfo):
                    if (
                        self._get_expression_info(self, element.value_expression.name)
                        is None
                    ):
                        self._expression_infos.append(element.value_expression)
                else:
                    if (
                        self._get_expression_info(self, element.value_expression)
                        is None
                    ):
                        element._value_expression = None

    def _get_required_fields(self):
        required_fields = []
        for field in self._fields:
            if (
                "nullable" in field
                and field["nullable"] is False
                and field["name"].lower() not in self._edit_fields
                and field["name"].lower() not in self._id_fields
            ):
                required_fields.append(field["name"])
        return required_fields

    def _validate_all_required_fields_in_form(self):
        found = False
        for field_name in self._required_fields:
            for el in self._form_elements:
                if el.element_type == "group":
                    if field_name.lower() in [
                        el.field_name.lower() for el in el._form_elements
                    ]:
                        found = True
                        break
                else:
                    if field_name.lower() == el.field_name.lower():
                        found = True
                        break
            if not found:
                raise ValueError(
                    str(field_name)
                    + " is a required field not found in the form. Please add to the form"
                )
            found = False


class FormElement:
    """
    The superclass class for :class:`~arcgis.mapping.forms.FormFieldElement` and :class:`~arcgis.mapping.forms.FormGroupElement`. Contains properties common to
    the two types of field elements. Instantiate a FormFieldElement or FormGroupElement instead of this class.
    """

    def __init__(
        self,
        form=None,
        element_type=None,
        description=None,
        label=None,
        visibility_expression=None,
        **kwargs,
    ):
        self._form = form
        self._element_type = element_type
        self._description = description
        self._label = label
        self._visibility_expression = visibility_expression
        self._kwargs = kwargs

    @property
    def description(self):
        """Gets/sets the description of the form element."""
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def label(self):
        """Gets/sets the label of the form element."""
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def element_type(self):
        """Get/Set the element type of the form element."""
        return self._element_type

    @element_type.setter
    def element_type(self, value):
        if self.element_type is None:
            self._element_type = value
        else:
            raise ValueError("Cannot change element type once it has been set")

    @property
    def visibility_expression(self):
        """Gets/sets the visibility expression of the form element."""
        return self._visibility_expression

    @visibility_expression.setter
    def visibility_expression(self, value):
        if isinstance(value, FormExpressionInfo) or value is None:
            self._visibility_expression = value
        else:
            raise ValueError("Please pass a FormExpressionInfo object")

    def to_dict(self):
        el_dict = {}
        if self._description:
            el_dict["description"] = self._description
        if self._label:
            el_dict["label"] = self._label
        if self._element_type:
            el_dict["type"] = self._element_type
        if self._visibility_expression:
            el_dict["visibilityExpression"] = self._visibility_expression.name
        for key, value in self._kwargs.items():
            el_dict[key] = value
        return el_dict


class FormFieldElement(FormElement):
    """
    Represents a single field (non-group) element in a form. This corresponds with a field
    in the feature layer where you are collecting data. This is a subclass of FormElement, so
    you can modify properties such as label, description, and visibility_expression on these
    objects as well.

    For more please see: https://developers.arcgis.com/web-map-specification/objects/formFieldElement/

    ======================     ====================================================================
    **Parameter**               **Description**
    ----------------------     --------------------------------------------------------------------
    form                       Optional :class:`~arcgis.mapping.forms.FormInfo`.
                               The form which contains this field element.
    ----------------------     --------------------------------------------------------------------

    description                Optional :class:`str`.
                               The description of the form element
    ----------------------     --------------------------------------------------------------------
    label                      Optional :class:`str`.
                               The label of the form element
    ----------------------     --------------------------------------------------------------------
    visibility_expression      Optional :class:`~arcgis.mapping.forms.FormExpressionInfo`.
                               The conditional visibility Arcade expression determining the
                               visibility of the form element during data collection
    ----------------------     --------------------------------------------------------------------
    domain                     Optional :class:`dict`.
                               The domain of the form element
    ----------------------     --------------------------------------------------------------------
    editable                   Optional :class:`bool`.
                               Whether or not the form element is editable
    ----------------------     --------------------------------------------------------------------
    field_name                 Optional :class:`str`.
                               The field name the form element corresponds to (where the data
                               will be collected)
    ----------------------     --------------------------------------------------------------------
    hint                       Optional :class:`str`.
                               The hint for the user filling out the form element
    ----------------------     --------------------------------------------------------------------
    input_type                 Optional :class:`str` or :class:`dict`.
                               The input type for the form element in ArcGIS Field Maps.

                               Options include:
                                    "text-area", "text-box", "barcode-scanner", "combo-box", "radio-buttons", "datetime-picker"
    ----------------------     --------------------------------------------------------------------
    required_expression        Optional :class:`~arcgis.mapping.forms.FormExpressionInfo`.
                               The conditional visibility Arcade expression determining the
                               necessity of the form element during data collection
    ======================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Edit properties on form element
        from arcgis.mapping.forms import FormExpressionInfo
        wm = arcgis.mapping.WebMap(item)
        wm.add_layer(manhole_inspection)
        form_collection = wm.forms
        form_info = form_collection.get_form(title="Manhole Inspection")

        # edit element properties
        form_element = form_info.get(label="Inspector Name")
        form_element.label = "Inspector Name(s)"
        form_element.description = "The inspector(s) who completed this manhole inspection")

        # set visibility expression
        el = form_info.add_field(field_name="jake_only", label="jake_only")
        expression_info = FormExpressionInfo(name="expr0",title="New Expression",expression="$feature.inspector == 'Jake'")
        el.visibility_expression = expression_info
    """

    def __init__(
        self,
        form=None,
        description=None,
        label=None,
        visibility_expression=None,
        domain=None,
        editable=None,
        field_name=None,
        hint=None,
        input_type=None,
        required_expression=None,
        editable_expression=None,
        value_expression=None,
        **kwargs,
    ):
        super().__init__(
            form=form,
            element_type="field",
            description=description,
            label=label,
            visibility_expression=visibility_expression,
            **kwargs,
        )
        self._domain = domain
        self._editable = editable
        self._field_name = field_name
        self._hint = hint
        self.input_type = input_type
        self._required_expression = required_expression
        self._editable_expression = editable_expression
        self._value_expression = value_expression

    def __repr__(self):
        if self._label:
            return "Field " + self._label
        else:
            return "Field"

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    @property
    def domain(self):
        """Gets/sets the domain of the form element."""
        return self._domain

    @domain.setter
    def domain(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError("Please pass a dict into this function")
        self._domain = value

    @property
    def editable(self):
        """Gets/sets the editability of the form element."""
        return self._editable

    @editable.setter
    def editable(self, value: bool):
        self._editable = value

    @property
    def field_name(self):
        """Gets the field name for the form element."""
        if self._field_name:
            return self._field_name.lower()
        else:
            return None

    @field_name.setter
    def field_name(self, value):
        if self._field_name is None:
            self._field_name = value
        else:
            raise ValueError("Cannot modify field name once it has been set")

    @property
    def hint(self):
        """Gets/sets the hint of the form element."""
        return self._hint

    @hint.setter
    def hint(self, value):
        self._hint = value

    @property
    def input_type(self):
        """
        Gets/sets the input type of the form element.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required string or dictionary.

                            Values:

                                "text-area" | "text-box" | "barcode-scanner" | "combo-box" | "radio-buttons" | "datetime-picker"
        ===============     ====================================================================

        :return: dictionary that represents the input type
        """
        return self._input_type

    @input_type.setter
    def input_type(self, value: Union[str, dict]):
        if value in [
            "text-area",
            "text-box",
            "barcode-scanner",
            "combo-box",
            "radio-buttons",
            "datetime-picker",
        ]:
            value = {"type": value}
        self._input_type = value

    @property
    def required_expression(self):
        """Gets/sets the required expression of the form element. Takes an object of :class:`~arcgis.mapping.forms.FormExpressionInfo` ."""
        return self._required_expression

    @required_expression.setter
    def required_expression(self, value):
        if isinstance(value, FormExpressionInfo) or value is None:
            self._required_expression = value
        else:
            raise ValueError("Please pass a FormExpressionInfo object")

    @property
    def editable_expression(self):
        """Gets/sets the editable expression of the form element. Takes an object of :class:`~arcgis.mapping.forms.FormExpressionInfo` ."""
        return self._editable_expression

    @editable_expression.setter
    def editable_expression(self, value):
        if isinstance(value, FormExpressionInfo) or value is None:
            self._editable_expression = value
        else:
            raise ValueError("Please pass a FormExpressionInfo object")

    @property
    def value_expression(self):
        """Gets/sets the value expression of the form element. Takes an object of :class:`~arcgis.mapping.forms.FormExpressionInfo`  ."""
        return self._value_expression

    @value_expression.setter
    def value_expression(self, value):
        if isinstance(value, FormExpressionInfo) or value is None:
            self._value_expression = value
        else:
            raise ValueError("Please pass a FormExpressionInfo object")

    def to_dict(self):
        el_dict = super().to_dict()
        if self._domain:
            el_dict["domain"] = self._domain
        if self._editable:
            el_dict["editable"] = self._editable
        if self._field_name:
            el_dict["fieldName"] = self._field_name
        if self._hint:
            el_dict["hint"] = self._hint
        if self._input_type:
            el_dict["inputType"] = self._input_type
        if self._required_expression:
            el_dict["requiredExpression"] = self._required_expression.name
        if self._editable_expression:
            el_dict["editableExpression"] = self._editable_expression.name
        if self._value_expression:
            el_dict["valueExpression"] = self._value_expression.name
        return el_dict


class FormGroupElement(FormElement):
    """
    Represents a single group element in a form. This is a subclass of FormElement, so
    you can modify properties such as label, description, and visibility_expression on these
    objects as well.

    For more please see: https://developers.arcgis.com/web-map-specification/objects/formGroupElement/

    ======================     ====================================================================
    **Parameter**               **Description**
    ----------------------     --------------------------------------------------------------------
    form                       Optional :class:`~arcgis.mapping.forms.FormInfo`.
                               The form which contains this group element.
    ----------------------     --------------------------------------------------------------------
    description                Optional :class:`str`.
                               The description of the group element
    ----------------------     --------------------------------------------------------------------
    label                      Optional :class:`str`.
                               The label of the group element
    ----------------------     --------------------------------------------------------------------
    visibility_expression      Optional :class:`~arcgis.mapping.forms.FormExpressionInfo`.
                               The conditional visibility Arcade expression determining the
                               visibility of the form element during data collection
    ----------------------     --------------------------------------------------------------------
    initial_state              Optional :class:`dict`.
                               Options are "collapsed" and "expanded"
    ======================     ====================================================================

    .. code-block:: python

        # USAGE EXAMPLE 1: Edit properties on group, add to group
        from arcgis.mapping.forms import FormExpressionInfo
        wm = arcgis.mapping.WebMap(item)
        wm.add_layer(manhole_inspection)
        form_collection = wm.forms
        form_info = form_collection.get_form(title="Manhole Inspection")

        # edit group properties, access elements within group
        group_element = form_info.get(label="Group 1")
        grouped_form_element = group_element.get(label="Inspector Name")
        grouped_form_element.label = "Inspector Name(s)
        group_element.label = "Inspector Information"
        group_element.initial_state = "collapsed"

        # add group, add to group, delete from group, delete group
        new_group = FormGroupElement(form_info, label="Group 2", initial_state="expanded")
        group = form_info.add(element=new_group)
        grouped_element = group.add_field(field_name="inspection_date", label="Inspection Date")
        group.add_field(field_name="inspection_city", label="Inspection City")
        grouped_element.label = "Inspection Date"
        group.move(grouped_element, index=1)
        group.delete(grouped_element)
        form_info.delete(group)

    """

    def __init__(
        self,
        form=None,
        elements=None,
        initial_state=None,
        description=None,
        label=None,
        visibility_expression=None,
        **kwargs,
    ):
        super().__init__(
            form=form,
            element_type="group",
            description=description,
            label=label,
            visibility_expression=visibility_expression,
            **kwargs,
        )
        if elements is None:
            elements = []
        self._form_elements = FormInfo._get_form_element_objects(elements, form=form)
        self._initial_state = initial_state

    def __repr__(self):
        return "Group: " + self._label

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    @property
    def elements(self):
        """Returns elements in the group to the user - a list of :class:`~arcgis.mapping.forms.FormElement`."""
        return self._form_elements

    @property
    def initial_state(self):
        """
        Gets/sets the initial state of the form element.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        value               Required string.

                            Values:

                                "collapsed" | "expanded"
        ===============     ====================================================================
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value: str):
        if value not in ["collapsed", "expanded"]:
            raise ValueError("Value can either be collapsed or expanded")
        self._initial_state = value

    def add(
        self, element: Optional[FormFieldElement] = None, index: Optional[int] = None
    ):
        """
        Adds a single form element to the group. You can add to the group either by instantiating
        a FormFieldElement and passing it into the element parameter here,
        or you can type in a valid field_name in the form's layer and this function will attempt
        to add it to the form.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        element                Optional :class:`~arcgis.mapping.forms.FormFieldElement`
        ------------------     --------------------------------------------------------------------
        index                  Optional :class:`int`.
                               The index where you'd like the element in the group. If not provided,
                               this function will add the new element to the end of the group.
        ==================     ====================================================================

        :return: The element that was added - :class:`~arcgis.mapping.forms.FormFieldElement`
        """
        self._validate_element(element)
        if index is None:
            index = len(self._form_elements)
        self._form_elements.insert(index, element)
        self._form._hydrate_expression_infos()
        return element

    def add_field(
        self,
        field_name: str,
        label: str,
        description: Optional[str] = None,
        visibility_expression=None,
        domain: Optional[dict] = None,
        editable: Optional[bool] = None,
        hint: Optional[str] = None,
        input_type: Optional[Union[dict, str]] = None,
        required_expression=None,
        index: Optional[int] = None,
        editable_expression=None,
        value_expression=None,
        **kwargs,
    ):
        """
        Adds a single field :class:`~arcgis.mapping.forms.FormElement` element to the end of the group.

        For more please see: https://developers.arcgis.com/web-map-specification/objects/formFieldElement/

        ======================     ====================================================================
        **Parameter**               **Description**
        ----------------------     --------------------------------------------------------------------
        field_name                 Required :class:`str`.
                                   The field name the form element corresponds to (where the data
                                   will be collected)
        ----------------------     --------------------------------------------------------------------
        label                      Required :class:`str`.
                                   The label of the form element
        ----------------------     --------------------------------------------------------------------
        description                Optional :class:`str`.
                                   The description of the form element
        ----------------------     --------------------------------------------------------------------
        visibility_expression      Optional :class:`~arcgis.mapping.forms.FormExpressionInfo`.
                                   The conditional visibility Arcade expression determining the
                                   visibility of the form element during data collection
        ----------------------     --------------------------------------------------------------------
        domain                     Optional :class:`dict`.
                                   The domain of the form element
        ----------------------     --------------------------------------------------------------------
        editable                   Optional :class:`bool`.
                                   Whether or not the form element is editable
        ----------------------     --------------------------------------------------------------------
        hint                       Optional :class:`str`.
                                   The hint for the user filling out the form element
        ----------------------     --------------------------------------------------------------------
        input_type                 Optional :class:`str` or :class:`dict`.
                                   The input type for the form element in ArcGIS Field Maps.
                                   Options include: "text-area", "text-box", "barcode-scanner",
                                   "combo-box", "radio-buttons", "datetime-picker"
        ----------------------     --------------------------------------------------------------------
        required_expression        Optional :class:`~arcgis.mapping.forms.FormExpressionInfo`.
                                   The conditional visibility Arcade expression determining the
                                   necessity of the form element during data collection
        ----------------------     --------------------------------------------------------------------
        index                      Optional :class:`int`.
                                   The index where you'd like the element in the form. If not provided,
                                   this function will add the new element to the end of the form.
        ----------------------     --------------------------------------------------------------------
        editable_expression        Optional :class:`~arcgis.mapping.forms.FormExpressionInfo`.
                                   The Arcade expression determining how much editable the form element is during data collection
        ----------------------     --------------------------------------------------------------------
        value_expression           Optional :class:`~arcgis.mapping.forms.FormExpressionInfo`.
                                   The Arcade expression which calculates a value for the form
                                   element during data collection
        ======================     ====================================================================

        """
        element = FormFieldElement(
            form=self,
            field_name=field_name,
            label=label,
            description=description,
            visibility_expression=visibility_expression,
            domain=domain,
            editable=editable,
            hint=hint,
            input_type=input_type,
            required_expression=required_expression,
            editable_expression=editable_expression,
            value_expression=value_expression,
            **kwargs,
        )
        return self.add(element, index=index)

    def delete(
        self, element: Optional[FormFieldElement] = None, label: Optional[str] = None
    ):
        """
        Deletes form element from the group. You can use either the element param
        with a form element you get using `FormInfo.get()` or you can pass the label of the
        form element you'd like to move into the label param.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        element                Optional :class:`~arcgis.mapping.forms.FormFieldElement`
        ------------------     --------------------------------------------------------------------
        label                  Optional :class:`str`
                               An actual field name (not alias) corresponding to a field in the
                               form's feature layer
        ==================     ====================================================================

        :return: The deleted element - :class:`~arcgis.mapping.forms.FormFieldElement` or `False`
        """
        if label:
            element = self.get(label=label)
        try:
            self._form_elements.remove(element)
            return element
        except Exception:
            return False

    def move(
        self,
        element: Optional[FormFieldElement] = None,
        label: Optional[str] = None,
        destination=None,
        index: Optional[int] = None,
    ):
        """
        Moves a form element in the group to a new location. You can use either the element param
        with a form element you get using `FormGroupElement.get() <:attr:`arcgis.mapping.forms.FormGroupElement.get`>` or you can pass the label
        of the form element you'd like to move into the label param.

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        element                Optional :class:`~arcgis.mapping.forms.FormFieldElement`
        ------------------     --------------------------------------------------------------------
        label                  Optional :class:`str`
                               The label of the form element you'd like to move to a new location
        ------------------     ----------------------------------------------------------------------
        destination            Optional :class:`~arcgis.mapping.forms.FormInfo` or
                               `arcgis.mapping.forms.FormGroupElement`
                               If moving out of the group (to the form or to new group), please
                               provide this parameter. Otherwise, you will only be allowed to move
                               within this group.
        ------------------     --------------------------------------------------------------------
        index                  Optional :class:`int`.
                               The index where you'd like the element to move to in the group.
        ==================     ====================================================================

        :return: The element that was moved - :class:`~arcgis.mapping.forms.FormFieldElement`
        """
        if index is None:
            raise ValueError("Please provide an index")
        if label:
            element = self.get(label=label)
        self.delete(element)
        if destination is None:
            destination = self
        return destination.add(element, index=index)

    def get(self, label: Optional[str] = None):
        """
        Returns a matching FormFieldElement in the group given a label

        ==================     ====================================================================
        **Parameter**           **Description**
        ------------------     --------------------------------------------------------------------
        label                  Optional :class:`str`
                               The label of the form element you want to return from the list of
                               form elements
        ==================     ====================================================================

        :return: :class:`~arcgis.mapping.forms.FormFieldElement` or `None`
        """
        try:
            return next(
                el for el in self._form_elements if el.label.lower() == label.lower()
            )
        except Exception:
            return None

    def to_dict(self):
        el_dict = super().to_dict()
        # don't use if form_elements here as the empty array is required
        el_dict["formElements"] = [element.to_dict() for element in self._form_elements]
        if self._initial_state:
            el_dict["initialState"] = self._initial_state
        return el_dict

    def _validate_element(self, element):
        """Validate the element passed to or created by add element is correct"""
        if element.element_type == "group":
            raise ValueError("You cannot add a group to another group")
        if self._form is not None:
            self._form._validate_element(element)


class FormExpressionInfo:
    """
    This class corresponds to a single expressionInfo in the expressionInfos list within a form.

    For more please see: https://developers.arcgis.com/web-map-specification/objects/formExpressionInfo/

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    expression             Optional :class:`str`
                           This is an Arcade expression which you want to evaluate in ArcGIS
                           Field Maps on whether or not this form element shows.
    ------------------     --------------------------------------------------------------------
    name                   Optional :class:`str`
                           The unique identifier for the expressionInfo
    ------------------     --------------------------------------------------------------------
    title                  Optional :class:`int`.
                           The user friendly name for the expressionInfo
    ------------------     --------------------------------------------------------------------
    return_type            Optional :class:`str`.
                           The return type of the expression in expressionInfo
    ==================     ====================================================================
    """

    def __init__(
        self, expression=None, name=None, title=None, return_type="boolean", **kwargs
    ):
        self._expression = expression
        self._name = name
        self._return_type = return_type
        self._title = title
        self._kwargs = kwargs

    def __repr__(self):
        if self._title:
            return "Expression: " + self._title
        else:
            return "Expression"

    @property
    def expression(self):
        """Gets/sets the expression for the expression info."""
        return self._expression

    @expression.setter
    def expression(self, value):
        if value:
            self._expression = value.replace('"', '"')
        else:
            raise ValueError("Expression must be set")

    @property
    def name(self):
        """Gets/sets the name for the expression info."""
        return self._name

    @name.setter
    def name(self, value):
        if value:
            self._name = value
        else:
            raise ValueError("Name must be set")

    @property
    def return_type(self):
        """Gets/sets the return type for the expression info."""
        return self._return_type

    @return_type.setter
    def return_type(self, value):
        self._return_type = value

    @property
    def title(self):
        """Gets/sets the title for the expression info."""
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    def to_dict(self):
        exp_dict = {}
        if self._name:
            exp_dict["name"] = self._name
        if self._title:
            exp_dict["title"] = self._title
        if self._expression:
            exp_dict["expression"] = self._expression
        if self._return_type:
            exp_dict["returnType"] = self._return_type
        for key, value in self._kwargs.items():
            exp_dict[key] = value
        return exp_dict

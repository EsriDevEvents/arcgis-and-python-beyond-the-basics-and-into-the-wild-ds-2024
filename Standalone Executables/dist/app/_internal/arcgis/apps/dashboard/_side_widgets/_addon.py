import uuid


class DatePicker(object):
    """
    Creates a Date Selector widget for Side Panel or Header.

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    range                       Optional boolean. True to create a range
                                selector.
    -------------------------   -------------------------------------------
    operator                    Optional String. Operator for non range
                                datepicker.

                                Options:

                                    | "is", "is not", "is before",
                                    | "is or is before", "is after",
                                    | "is or is after".
    -------------------------   -------------------------------------------
    label                       Optional String. Label for the widget.
    -------------------------   -------------------------------------------
    kwargs                      If "range" is True, provide two parameters
                                "min_value" and "max_value". If "range" is
                                False provide single parameter "value".

                                Allowed values:

                                    | None, "Today", or a fixed value in 24 hours format
                                    | (year, month, day, hour, minutes)
                                    | or
                                    | (year, month, day)
    =========================   ===========================================
    """

    def __init__(self, range=False, operator="is", label="", **kwargs):
        self._id = str(uuid.uuid4())
        self.type = "dateSelectorWidget"

        self._operator_mapping = {
            "is": "is_on",
            "is not": "is_not_on",
            "is before": "is_before",
            "is or is before": "is_on_before",
            "is after": "is_after",
            "is or is after": "is_on_after",
        }

        self._selection_type = "single"
        if range:
            self._selection_type = "range"
            self.operator = "between"
            min_value = kwargs.get("min_value", None)
            max_value = kwargs.get("max_value", None)
            if min_value:
                if isinstance(min_value, str) and min_value.lower() == "today":
                    self._min_value = {
                        "type": "date",
                        "includeTime": False,
                        "defaultToToday": True,
                    }
                else:
                    self._min_value = {
                        "type": "date",
                        "includeTime": False,
                        "defaultToToday": False,
                        "year": min_value[0],
                        "month": min_value[1] - 1,
                        "date": min_value[2],
                        "hours": min_value[3] if len(min_value) > 3 else 0,
                        "minutes": min_value[4] if len(min_value) > 3 else 0,
                        "seconds": 0,
                        "milliSeconds": 0,
                    }
            else:
                # raise Exception("Please provide min_value parameter")
                self._min_value = None

            if max_value:
                if isinstance(max_value, str) and max_value.lower() == "today":
                    self._max_value = {
                        "type": "date",
                        "includeTime": False,
                        "defaultToToday": True,
                    }
                else:
                    self._max_value = {
                        "type": "date",
                        "includeTime": False,
                        "defaultToToday": False,
                        "year": max_value[0],
                        "month": max_value[1] - 1,
                        "date": max_value[2],
                        "hours": max_value[3] if len(max_value) > 3 else 0,
                        "minutes": max_value[4] if len(max_value) > 3 else 0,
                        "seconds": 0,
                        "milliSeconds": 0,
                    }
            else:
                # raise Exception("Please provide max_value parameter")
                self._max_value = None
        else:
            self.operator = operator
            min_value = kwargs.get("value", None)
            self._max_value = None
            if min_value:
                if isinstance(min_value, str) and min_value.lower() == "today":
                    self._min_value = {
                        "type": "date",
                        "includeTime": False,
                        "defaultToToday": True,
                    }
                else:
                    self._min_value = {
                        "type": "date",
                        "includeTime": False,
                        "defaultToToday": False,
                        "year": min_value[0],
                        "month": min_value[1] - 1,
                        "date": min_value[2],
                        "hours": min_value[3] if len(min_value) > 3 else 0,
                        "minutes": min_value[4] if len(min_value) > 3 else 0,
                        "seconds": 0,
                        "milliSeconds": 0,
                    }
            else:
                self._min_value = None
                # raise Exception("Please provide value parameter")

        self.label = label

    def _convert_to_json(self):
        if self._selection_type == "range":
            self._operator_logic = "between"
        else:
            self._operator_logic = self._operator_mapping.get(self.operator, "is_on")

        data = {
            "type": "dateSelectorWidget",
            "optionType": "datePicker",
            "datePickerOption": {
                "type": "datePicker",
                "selectionType": self._selection_type,
                "operator": self._operator_logic,
            },
            "id": self._id,
            "name": "Date Selector (1)",
            "caption": self.label,
            "showLastUpdate": True,
            "noDataVerticalAlignment": "middle",
            "showCaptionWhenNoData": True,
            "showDescriptionWhenNoData": True,
        }

        if self._min_value:
            data["datePickerOption"]["minDefaultValue"] = self._min_value

        if self._max_value:
            data["datePickerOption"]["maxDefaultValue"] = self._max_value

        return data

    def _repr_html_(self):
        from arcgis.apps.dashboard import Dashboard
        from arcgis.apps.dashboard import SidePanel

        sp = SidePanel()
        sp.add_selector(self)
        url = Dashboard._publish_random(sp)
        return f"""<iframe src={url} width=300 height=300>"""


class NumberSelector(object):
    """
    Creates a Number Selector widget for Side Panel or Header.

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    range                       Optional boolean. True to create a range
                                selector.
    -------------------------   -------------------------------------------
    display_type                Optional String. Display type can be from
                                "spinner", "slider", "input".
    -------------------------   -------------------------------------------
    label                       Optional string. Label for the selector.
    =========================   ===========================================

    **Keyword Arguments**

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    operator                    Optional string for non-range input.

                                Allowed:

                                    | "equal", "not equal", "greater than",
                                    | "greater than or equal", "less than",
                                    | "less than or equal".

                                    | Default: "equal"
    -------------------------   -------------------------------------------
    increment_factor            Optional int for slider and spinner input.
    =========================   ===========================================
    """

    def __init__(
        self, range=False, display_type="spinner", label="Select a number", **kwargs
    ):
        self._json = {}
        self._display_type = display_type
        self.type = "numberSelectorWidget"
        self._id = str(uuid.uuid4())
        self._label = label if label else ""
        self._increment = kwargs.get("increment_factor", 1)
        self._lower_limit = 0
        self._upper_limit = 100
        self._lower_default = 0
        self._upper_default = 100
        self._range = range
        self._operator = kwargs.get("operator", "equal")

        self._left_placeholder_text = ""
        self._right_placeholder_text = ""

        self._dataset = None
        self._operator_mapping = {
            "equal": "equal",
            "not equal": "not_equal",
            "greater than": "greater",
            "greater than or equal": "greater_or_equal",
            "less than": "less",
            "less than or equal": "less_or_equal",
        }

        self._constraint = {
            "type": "fixed",
            "lowerLimit": self._lower_limit,
            "upperLimit": self._upper_limit,
            "firstDefault": self._lower_default,
            "secondDefault": self._upper_default,
        }

    def _repr_html_(self):
        from arcgis.apps.dashboard import Dashboard
        from arcgis.apps.dashboard import SidePanel

        sp = SidePanel()
        sp.add_selector(self)
        url = Dashboard._publish_random(sp)
        return f"""<iframe src={url} width=300 height=300>"""

    @property
    def placeholder_text(self):
        """
        :return: Text for left place holder in range type or default place holder.
        """
        return self._left_placeholder_text

    @placeholder_text.setter
    def placeholder_text(self, value):
        """
        Text for left place holder in range type or default place holder.
        """
        self._left_placeholder_text = value

    @property
    def right_placeholder_text(self):
        """
        :return: Text for right place holder in range type.
        """
        return self._right_placeholder_text

    @right_placeholder_text.setter
    def right_placeholder_text(self, value):
        """
        Text for right place holder in range type.
        """
        self._right_placeholder_text = value

    def set_statistics_limits(self, item, field, default="min", layer_id=0):
        """
        Set the item to pick values from for spinner and slider display type.

        =========================   ===========================================
        **Parameter**                **Description**
        -------------------------   -------------------------------------------
        item                        Required Portal :class:`~arcgis.gis.Item` . Item to pick values from.
        -------------------------   -------------------------------------------
        field                       Required String. Field from the Portal Item.
        -------------------------   -------------------------------------------
        default                     Optional String. Default value statistic.
                                    Options: "min", "max", "avg"
        -------------------------   -------------------------------------------
        layer_id                    Optional integer. Layer Id for the item.
        =========================   ===========================================
        """

        self._stat_dataset = {
            "type": "serviceDataset",
            "dataSource": {
                "type": "featureServiceDataSource",
                "itemId": item.itemid,
                "layerId": layer_id,
                "table": True,
            },
            "outFields": ["*"],
            "groupByFields": [],
            "orderByFields": [],
            "statisticDefinitions": [
                {
                    "onStatisticField": field,
                    "outStatisticFieldName": "lowerLimit",
                    "statisticType": "min",
                },
                {
                    "onStatisticField": field,
                    "outStatisticFieldName": "upperLimit",
                    "statisticType": "max",
                },
                {
                    "onStatisticField": field,
                    "outStatisticFieldName": "averageStatisticValue",
                    "statisticType": "avg",
                },
            ],
            "maxFeatures": 50,
            "querySpatialRelationship": "esriSpatialRelIntersects",
            "returnGeometry": False,
            "clientSideStatistics": False,
            "name": "main",
        }

        self._constraint = {"type": "statistic", "defaultStatistic": "min"}

        if default in ["min", "max", "avg"]:
            self._constraint["defaultStatistic"] = default

    def set_defined_limits(self, lower_limit=0, upper_limit=100, **kwargs):
        """
        Set the item to pick values from for spinner and slider display type.

        =========================   ===========================================
        **Parameter**                **Description**
        -------------------------   -------------------------------------------
        lower_limit                 Optional integer. Set the lower limit.
        -------------------------   -------------------------------------------
        upper_limit                 Optional integer. Set the upper limit.
        =========================   ===========================================

        **Keyword Arguments**

        =========================   ===========================================
        **Parameter**                **Description**
        -------------------------   -------------------------------------------
        default                     Optional integer. Set default value for
                                    non-range selector.
        -------------------------   -------------------------------------------
        lower_default               Optional integer. Set the lower default
                                    value for range selector.
        -------------------------   -------------------------------------------
        upper_default               Optional integer. Set the upper default
                                    value for range selector.
        =========================   ===========================================
        """

        self._lower_limit = lower_limit
        self._upper_limit = upper_limit
        self._lower_default = kwargs.get("default", kwargs.get("lower_default", 0))
        self._upper_default = kwargs.get("upper_default", 100)
        self._constraint = {
            "type": "fixed",
            "lowerLimit": self._lower_limit,
            "upperLimit": self._upper_limit,
            "firstDefault": self._lower_default,
            "secondDefault": self._upper_default,
        }

    def _convert_to_json(self):
        # self._dataset = {
        #     "type": "serviceDataset",
        #     "dataSource": {
        #         "type": "featureServiceDataSource",
        #         "itemId": self.item.itemid,
        #         "layerId": 0,
        #         "table": True
        #     },
        #     "outFields": ["*"],
        #     "groupByFields": [],
        #     "orderByFields": [],
        #     "statisticDefinitions": [
        #         {"onStatisticField": field, "outStatisticFieldName": "lowerLimit",
        #          "statisticType": "min"},
        #         {"onStatisticField": field, "outStatisticFieldName": "upperLimit",
        #          "statisticType": "max"},
        #         {"onStatisticField": field, "outStatisticFieldName": "averageStatisticValue",
        #          "statisticType": "avg"}
        #     ],
        #     "maxFeatures": 50,
        #     "querySpatialRelationship": "esriSpatialRelIntersects",
        #     "returnGeometry": False,
        #     "clientSideStatistics": False,
        #     "name": "main"
        # }

        json = {
            "type": "numericSelectorWidget",
            "displayType": self._display_type,
            "increment": self._increment,
            "valueLabelFormat": {
                "name": "value",
                "type": "decimal",
                "prefix": False,
                "pattern": "#,###",
            },
            "selection": {"type": "single" if not self._range else "range"},
            "datasets": [],
            "id": self._id,
            "name": "Number Selector (1)",
            "caption": self._label,
            "showLastUpdate": True,
            "noDataVerticalAlignment": "middle",
            "showCaptionWhenNoData": True,
            "showDescriptionWhenNoData": True,
        }

        if not self._range:
            json["selection"]["operator"] = self._operator

        if self._constraint:
            json["constraint"] = self._constraint

        if self._display_type == "input":
            json["selection"]["placeholderText"] = self._left_placeholder_text
            if self._range:
                json["selection"]["rightPlaceHolderText"] = self._right_placeholder_text

        if getattr(self, "_stat_dataset", None):
            json["datasets"].append(self._stat_dataset)
        # else:
        #     json["datasets"].append(self._dataset)

        return json


class CategorySelector(object):
    """
    Creates a Category Selector widget for Side Panel or Header.
    """

    def __init__(self):
        self._categories_from = "static"

        self._id = str(uuid.uuid4())
        self._selector = CategorySelectorProperties._create_selector_properties()

        self._dataset = None

    def _repr_html_(self):
        from arcgis.apps.dashboard import Dashboard
        from arcgis.apps.dashboard import SidePanel

        sp = SidePanel()
        sp.add_selector(self)
        url = Dashboard._publish_random(sp)
        return f"""<iframe src={url} width=300 height=300>"""

    def set_defined_values(self, key_value_pairs, value_type="string"):
        """
        Set defined values for the dropdown.

        =========================   ===========================================
        **Parameter**                **Description**
        -------------------------   -------------------------------------------
        key_value_pairs             Optional list of tuples. The tuple should
                                    contain labels and their corresponding values.
        -------------------------   -------------------------------------------
        value_type                  Optional String.
                                    The data type of the values in the tuple.
                                    "integer" or "string
        =========================   ===========================================
        """
        type_caster = str
        self._categories_from = "static"
        if value_type == "integer":
            type_caster = int

        if value_type not in ["string", "integer"]:
            value_type = "string"

        self._dataset = {
            "type": "staticDataset",
            "data": {"type": "staticValues", "dataType": value_type, "values": []},
            "name": "main",
        }

        id = 0
        for pair in key_value_pairs:
            self._dataset["data"]["values"].append(
                {
                    "type": " labelledValue",
                    "id": str(id),
                    "label": pair[0],
                    "value": type_caster(pair[1]),
                }
            )
            id = id + 1

    def set_feature_options(
        self, item, line_item_text="", field_name=None, max_features=50
    ):
        """
        Set feature values for dropdown.

        =========================   ===========================================
        **Parameter**                **Description**
        -------------------------   -------------------------------------------
        item                        Required Portal :class:`~arcgis.gis.Item` . Dropdown values will be populated from this.
        -------------------------   -------------------------------------------
        line_item_text              Optional String. This text will be displayed with options.
        -------------------------   -------------------------------------------
        field_name                  Optional String. Data from this field will be added to list.
        -------------------------   -------------------------------------------
        max_features                Optional Integer. Set max features to display.
        =========================   ===========================================
        """
        self._categories_from = "features"

        self._line_item_text = line_item_text if line_item_text else ""
        if field_name is not None:
            self._line_item_text = self._line_item_text + "{" + field_name + "}"

        self._dataset = {
            "type": "serviceDataset",
            "dataSource": {
                "type": "featureServiceDataSource",
                "itemId": item.itemid,
                "layerId": 0,
                "table": True,
            },
            "outFields": ["*"],
            "groupByFields": [],
            "orderByFields": [],
            "statisticDefinitions": [],
            "maxFeatures": max_features,
            "querySpatialRelationship": "esriSpatialRelIntersects",
            "returnGeometry": False,
            "clientSideStatistics": False,
            "name": "main",
        }

    def set_group_by_values(self, item, category_field, max_features=50):
        """
        Set group by values for dropdown.

        =========================   ===========================================
        **Parameter**                **Description**
        -------------------------   -------------------------------------------
        item                        Required Portal :class:`~arcgis.gis.Item` .
                                    Dropdown values will be populated from this.
        -------------------------   -------------------------------------------
        category_field              Optional String. This string denotes the
                                    field to pick the values from.
        -------------------------   -------------------------------------------
        max_features                Optional Integer.
                                    Set max features to display.
        =========================   ===========================================
        """
        self._categories_from = "groupByValues"

        self._dataset = {
            "type": "serviceDataset",
            "dataSource": {
                "type": "featureServiceDataSource",
                "itemId": item.itemid,
                "layerId": 0,
                "table": True,
            },
            "outFields": ["*"],
            "groupByFields": [category_field],
            "orderByFields": [category_field + " asc"],
            "statisticDefinitions": [
                {
                    "onStatisticField": category_field,
                    "outStatisticFieldName": "count_result",
                    "statisticType": "count",
                }
            ],
            "maxFeatures": max_features,
            "querySpatialRelationship": "esriSpatialRelIntersects",
            "returnGeometry": False,
            "clientSideStatistics": False,
            "name": "main",
        }

    @property
    def selector(self):
        """
        :return: Selector Properties Object, set label, preferred display, display threshold, operator etc.
        """
        return self._selector

    def _convert_to_json(self):
        json = {
            "type": "categorySelectorWidget",
            "category": {},
            "selection": {
                "type": self._selector._selection_type,
                "defaultSelection": "0",
                "operator": self._selector._operator,
            },
            "preferredDisplayType": self._selector._preferred_display,  # dropdown, button_bar, radio_buttons
            "displayThreshold": self._selector._display_threshold,
            "datasets": [],
            "id": self._id,
            "name": "Category Selector (1)",
            "caption": self._selector._label,
            "showLastUpdate": True,
            "noDataVerticalAlignment": "middle",
            "showCaptionWhenNoData": True,
            "showDescriptionWhenNoData": True,
        }

        if self._selector._none_option:
            json["noneLabelPlacement"] = self._selector._none_placement
            json["noneLabel"] = self._selector._none_label

        if self._categories_from == "static":
            json["category"] = {"type": "static"}
        elif self._categories_from == "features":
            json["category"] = {"type": "features", "itemText": self._line_item_text}
        elif self._categories_from == "groupByValues":
            json["category"] = {
                "type": "groupByValues",
                "nullLabel": "Null",
                "blankLabel": "Blank",
                "labelOverrides": [],
            }

        if self._dataset:
            json["datasets"].append(self._dataset)

        return json


class CategorySelectorProperties(object):
    @classmethod
    def _create_selector_properties(cls):
        selector = cls()
        selector._preferred_display = "dropdown"
        selector._default_selection = 0
        selector._label = ""
        selector._selection_type = "single"
        selector._operator = "equal"
        selector._display_threshold = 10

        selector._none_option = False
        selector._none_placement = "first"
        selector._none_label = "None"

        return selector

    @property
    def preferred_display(self):
        """
        :return: Preferred display for the selector.
        """
        return self._preferred_display

    @preferred_display.setter
    def preferred_display(self, value):
        """
        Set preferred display from "dropdown", "button_bar" or "radio_buttons"
        """
        if value not in ["dropdown", "button_bar", "radio_buttons"]:
            raise Exception("Invalid preferred display")

        self._preferred_display = value

    @property
    def label(self):
        """
        :return: Label Text
        """
        return self._label

    @label.setter
    def label(self, value):
        """
        Set label text.
        """
        self._label = value

    @property
    def multiple_selection(self):
        """
        :return: True or False for multiple selection
        """
        if self._selection_type == "single":
            return False

        return True

    @multiple_selection.setter
    def multiple_selection(self, value):
        """
        Set selection type to True or False.
        """
        if value:
            self._selection_type = "multiple"
            if self._operator == "equal":
                self._operator = "is_in"
            elif self._operator == "not_equal":
                self._operator = "is_not_in"
        else:
            self._selection_type = "single"
            if self._operator == "is_in":
                self._operator = "equal"
            elif self._operator == "is_not_in":
                self._operator = "not_equal"

    @property
    def include_values(self):
        """
        :return: True if values selected are to be taken.
        """
        if self._operator in ["is_in", "equal"]:
            return True

        return False

    @include_values.setter
    def include_values(self, value):
        """
        Set True to take selected values, False to take unselected values.
        """
        if value:
            if self._selection_type == "multiple":
                self._operator = "is_in"
            else:
                self._operator = "equal"
        else:
            if self._selection_type == "multiple":
                self._operator = "is_not_in"
            else:
                self._operator = "not_equal"

    @property
    def display_threshold(self):
        """
        :return: Return display threshold.
        """
        return self._display_threshold

    @display_threshold.setter
    def display_threshold(self, value):
        """
        Set Dropdown display threshold
        """
        self._display_threshold = value

    @property
    def none(self):
        """
        :return: Label for None option if set else None.
        """
        return self._none_label if self._none_option == True else None

    @none.setter
    def none(self, value):
        """
        Set Label for None option. Set None to disable
        """
        if value is None:
            self._none_option = False
            self._none_label = ""
        else:
            self._none_option = True
            self._none_label = value

    @property
    def none_placement(self):
        """
        :return: None Placement "first" or "last".
        """
        return self._none_placement

    @none_placement.setter
    def none_placement(self, value):
        """
        Set none option placement to "first" or "last".
        """

        if value not in ["first", "last"]:
            raise Exception("Invalid value")

        self._none_placement = value

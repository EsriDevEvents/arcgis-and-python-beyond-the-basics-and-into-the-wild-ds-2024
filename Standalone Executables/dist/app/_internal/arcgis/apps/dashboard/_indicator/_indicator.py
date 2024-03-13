import uuid
import arcgis
from .._utils._basewidget import _BaseWidget
from .._utils._basewidget import NoDataProperties


class Indicator(_BaseWidget):
    """
    Creates a dashboard Indicator widget.

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    item                        Required Portal :class:`~arcgis.gis.Item` object. Item object can
                                be a :class:`~arcgis.features.FeatureLayer`  or a MapWidget.
    -------------------------   -------------------------------------------
    name                        Optional string. Name of the Indicator
                                widget.
    -------------------------   -------------------------------------------
    title                       Optional string. Title or Caption for the
                                widget.
    -------------------------   -------------------------------------------
    layer                       Optional integer. Layer number when item is
                                a mapwidget.
    -------------------------   -------------------------------------------
    description                 Optional string. Description for the widget.
    =========================   ===========================================
    """

    def __init__(self, item, name="Indicator", layer=0, title="", description=""):
        super().__init__(name, title, description)

        if item.type not in ["Feature Service", "mapWidget"]:
            raise Exception("Please specify an item")

        self.item = item
        self.type = "indicatorWidget"
        self.layer = layer

        self._nodata = NoDataProperties._nodata_init()
        self._novalue = NoDataProperties._nodata_init()

        self._data = IndicatorData._create_data()
        if not self._data._value_field in [
            fld["name"] for fld in item.layers[layer].properties.fields
        ]:
            self._data._value_field = [
                i["fields"]
                for i in item.layers[layer].properties.indexes
                if i["isUnique"]
            ][0]
        self._reference = ReferenceData._create_data()

        self._max_display_features = 50
        self._show_last_update = True

    @classmethod
    def _from_json(cls, widget_json):
        gis = arcgis.env.active_gis
        itemid = widget_json["datasets"]["datasource"]["itemid"]
        name = widget_json["name"]
        item = gis.content.get(itemid)
        title = widget_json["caption"]
        description = widget_json["description"]
        indicator = Indicator(item, name, title, description)

        return indicator

    @property
    def data(self):
        """
        :return: Indicator Data object. Set data properties, categories and values.
        """
        return self._data

    @property
    def reference(self):
        """
        :return: Indicator reference object. Set data properties, categories and values.
        """
        return self._reference

    @property
    def max_features(self):
        """
        :return: Maximum features for widget.
        """
        return self._max_features

    @max_features.setter
    def max_features(self, value):
        """
        Set max features for widget.
        """
        self._max_features = int(value)

    @property
    def show_last_update(self):
        """
        :return: Show last update or not.
        """
        return self._show_last_update

    @show_last_update.setter
    def show_last_update(self, value):
        """
        Set show last update or not.
        """
        self.show_last_update = value
        if not isinstance(value, bool):
            self._show_last_update = True

    def _convert_to_json(self):
        self._reference_statistic = []
        self._statistic_definition = []

        if self._data.value_type == "statistic":
            self._statistic_definition.append(
                {
                    "onStatisticField": self.data.value_field,
                    "outStatisticFieldName": "value",
                    "statisticType": self._data.statistic,
                }
            )

        if self.item.type == "mapWidget":
            wlayer = self.item.layers[self.layer]
            widget_id = self.item._id
            layer_id = wlayer["id"]
            self._datasource = {"id": str(widget_id) + "#" + str(layer_id)}
        else:
            self._datasource = {
                "type": "featureServiceDataSource",
                "itemId": self.item.itemid,
                "layerId": self.layer,
                "table": True,
            }

        self._datasets = [
            {
                "type": "serviceDataset",
                "dataSource": self._datasource,
                "outFields": ["*"],
                "groupByFields": [],
                "orderByFields": [],
                "statisticDefinitions": self._statistic_definition,
                "querySpatialRelationship": "esriSpatialRelIntersects",
                "returnGeometry": False,
                "clientSideStatistics": False,
                "name": "main",
            }
        ]
        if self._reference.reference_type == "fixed":
            self._datasets.append(
                {
                    "type": "staticDataset",
                    "data": self._reference.fixed_value,
                    "name": "reference",
                }
            )
        elif self._reference.reference_type == "statistic":
            self._datasets.append(
                {
                    "type": "serviceDataset",
                    "dataSource": {
                        "type": "featureServiceDataSource",
                        "itemId": self.item.itemid,
                        "layerId": 0,
                        "table": True,
                    },
                    "outFields": ["*"],
                    "groupByFields": [],
                    "orderByFields": [],
                    "statisticDefinitions": [
                        {
                            "onStatisticField": self.reference.reference_field,
                            "outStatisticFieldName": "value",
                            "statisticType": self._reference.statistic,
                        }
                    ],
                    "querySpatialRelationship": "esriSpatialRelIntersects",
                    "returnGeometry": False,
                    "clientSideStatistics": False,
                    "name": "reference",
                }
            )

        json_data = {
            "type": "indicatorWidget",
            "defaultSettings": {
                "topSection": {"fontSize": 80, "textInfo": {}},
                "middleSection": {"fontSize": 160, "textInfo": {"text": "{value}"}},
                "bottomSection": {"fontSize": 80, "textInfo": {}},
            },
            "comparison": self._reference.reference_type,
            "valueField": self._data.value_field,
            "referenceField": self._reference.reference_field,
            "valueConversion": {
                "factor": self._data.factor,
                "offset": self._data.offset,
            },
            "referenceConversion": {
                "factor": self._reference.factor,
                "offset": self._reference.offset,
            },
            "valueFormat": {"name": "value", "type": "decimal", "prefix": False},
            "percentageFormat": {
                "name": "percentage",
                "type": "decimal",
                "prefix": False,
            },
            "ratioFormat": {"name": "ratio", "type": "decimal", "prefix": False},
            "valueType": self._data.value_type,
            "noValueVerticalAlignment": self._novalue.alignment,
            "showCaptionWhenNoValue": self._novalue.show_title,
            "showDescriptionWhenNoValue": self._novalue.show_description,
            "datasets": self._datasets,
            "id": self._id,
            "name": self.name,
            "caption": self.title,
            "description": self.description,
            "showLastUpdate": self.show_last_update,
            "noDataVerticalAlignment": self._nodata.alignment,
            "showCaptionWhenNoData": self._nodata.show_title,
            "showDescriptionWhenNoData": self._nodata._show_description,
        }
        if self._text_color:
            json_data["defaultSettings"]["textColor"] = self._text_color

        if self._background_color:
            json_data["defaultSettings"]["backgroundColor"] = self._background_color

        if self.title:
            json_data["defaultSettings"] = {
                "topSection": {"fontSize": 80, "textInfo": {"text": self.title}},
                "middleSection": {"fontSize": 160, "textInfo": {"text": "{value}"}},
                "bottomSection": {"fontSize": 80, "textInfo": {}},
            }

        return json_data


class IndicatorData(object):
    @classmethod
    def _create_data(cls, value_type="statistic"):
        data = IndicatorData()

        data._value_type = "statistic"
        data._statistic = "count"
        data._value_field = ""
        data._factor = 1
        data._offset = 0
        data._filters = []

        data.value_type = value_type

        return data

    @property
    def value_type(self):
        """
        :return: value type of data.
        """
        return self._value_type

    @value_type.setter
    def value_type(self, value):
        """
        Set value type of data.
        """
        if value in ["statistic", "feature"]:
            self._value_type = str(value)
        else:
            raise Exception(
                "Please set correct value type. Supported value types are 'statistic', 'feature'"
            )

    @property
    def statistic(self):
        """
        :return: statistic for data.
        """
        return self._statistic

    @statistic.setter
    def statistic(self, value):
        """
        Set statistic for data.
        """
        if self._value_type == "statistic":
            if value in ["count", "avg", "min", "max", "stddev", "sum"]:
                self._statistic = value
        else:
            raise Exception("Can set statistic only for reference type 'statistic'")

    @property
    def factor(self):
        """
        :return: value conversion factor.
        """
        return self._factor

    @factor.setter
    def factor(self, value):
        """
        Set value conversion factor.
        """
        self._factor = value

    @property
    def offset(self):
        """
        :return: value conversion offset.
        """
        return self._offset

    @offset.setter
    def offset(self, value):
        """
        Set value conversion offset.
        """
        self._offset = value

    @property
    def value_field(self):
        """
        :return: value field for data.
        """
        return self._value_field

    @value_field.setter
    def value_field(self, value):
        """
        Set value field for data.
        """
        if self._value_type in ["feature", "statistic"]:
            self._value_field = value
        else:
            raise Exception(
                "Can add value field only for value type 'feature' or 'statistic'"
            )

    @property
    def filters(self):
        """
        :return: filters associated with widget
        """
        return self._filters

    def add_filter(self, field, join, condition, **kwargs):
        """
        Add filters associated with widget. The filters are applied to the
        layer used    in the initialization of the Indicator widget. Please
        see `Filter data <https://doc.arcgis.com/en/dashboards/get-started/filter-data.htm>`_
        for detailed description of how filtering works with dashboard elements.

        ===============     ====================================================
        **Parameter**        **Description**
        ---------------     ----------------------------------------------------
        field               The layer's attribute field name that will be used
                            to limit the features visualized in the widget.
        ---------------     ----------------------------------------------------
        join                Specify `AND` or `OR` to indicate whether there
                            will be one or multiple filter conditions.
        ---------------     ----------------------------------------------------
        condition           The operator used to evaluate the attributes and
                            return the subset of results. The operators
                            available change depending upon the type of the
                            attribute field. See
                            `Filter condition components <https://doc.arcgis.com/en/dashboards/get-started/filter-data.htm>`_
                            for details on what conditions apply to an
                            attribute field based on its contents.
        ===============     ====================================================

        In addition to explicitly named parameters,
        the :meth:`~arcgis.apps.dashboard.IndicatorData.add_filter` method supports an
        optional key word argument when the condition is `equal`, `not  equal`
        `greater than`, `greater than or equal`, `less than`, or
        `less than or equal`

        ================    ====================================================
        **kwargs**          **Description**
        ----------------    ----------------------------------------------------
        value               The specific value or values to use to determine
                            which subset of layer or table rows to include.
        ================    ====================================================

        .. code-block:: python

            # Usage Example
            >>> indicator1 = Indicator(item=flyr_item)
            # Set attribute field to use for default statistic of `count`
            >>> indicator1.data.value_field = "ObjectId"
            # Add filter for ObjectID's greater than 1500 to data object
            >>> indicator1.data.add_filter(field="ObjectId",
                                           join="AND",
                                           condition="greater than",
                                           value=1500)

            >>> new_dash = Dashboard()
            # Set the dashboard layour to include the widget
            >>> new_dash.layout = add_row([indicator1])
            >>> saved_dash = new_dash.save(title="Dashboard with Indicator",
                                            description="Dashboard with indicator widget based
                                                         "on a hosted feature layer with one"
                                                         "layer",
                                            summary="Single layer indicator created in API.",
                                            tags="python_api,dashboard,single_layer_hfl",
                                            overwrite=True)
        """
        self._filter_field = field
        if join in ["AND", "OR"]:
            self._filter_join = join
        else:
            raise Exception("Please select from 'AND', 'OR'")
        if condition in ["between", "not between"]:
            if not kwargs["start"] and kwargs["end"]:
                raise Exception("Please provide 'start' and 'end' values as parameters")
            else:
                self._val1 = kwargs.get("start")
                self, _val2 = kwargs.get("end")
                self._filters.append(
                    {
                        "filtertype": self._filter_join,
                        "field": self._filter_field,
                        "operator": self._filter_condition,
                        "start": self._val1,
                        "end": self._val2,
                    }
                )
        if condition in [
            "equal",
            "not equal",
            "greater than",
            "greater than or equal",
            "less than",
            "less than or equal",
            "is null",
            "is not null",
            "contains",
        ]:
            self._filter_condition = condition
        else:
            raise Exception("Please select the right condition")
        if condition in [
            "equal",
            "not equal",
            "greater than",
            "greater than or equal",
            "less than",
            "less than or equal",
        ]:
            self._val = kwargs.get("value")
            self._filters.append(
                {
                    "filtertype": self._filter_join,
                    "field": self._filter_field,
                    "operator": self._filter_condition,
                    "value": self._val,
                }
            )
        else:
            raise Exception("Please provide a 'value' parameter for comparison")


class ReferenceData(object):
    @classmethod
    def _create_data(cls, reference_type="none"):
        data = ReferenceData()

        data._reference_type = "none"

        data._statistic = "count"
        data._reference_field = "FID"
        data._factor = 1
        data._offset = 0
        data._fixed_value = 0

        data.reference_type = reference_type

        return data

    @property
    def reference_type(self):
        """
        :return: reference type of reference.
        """
        return self._reference_type

    @reference_type.setter
    def reference_type(self, value):
        """
        Set reference type of reference.
        """
        if value in ["statistic", "feature", "none", "previous", "fixed"]:
            self._reference_type = str(value)
        else:
            raise Exception(
                "Please set correct reference type. Supported reference types are 'none', 'statistic', 'previous', 'fixed', 'feature'"
            )

    @property
    def statistic(self):
        """
        :return: statistic for reference.
        """
        return self._statistic

    @statistic.setter
    def statistic(self, value):
        """
        Set statistic for reference.
        """
        if self._value_type == "statistic":
            if value in ["count", "avg", "min", "max", "stddev", "sum"]:
                self._statistic = value
        else:
            raise Exception("Can set statistic only for reference type 'statistic'")

    @property
    def factor(self):
        """
        :return: value conversion factor.
        """
        return self._factor

    @factor.setter
    def factor(self, value):
        """
        Set value conversion factor.
        """
        self._factor = value

    @property
    def offset(self):
        """
        :return: value conversion offset.
        """
        return self._offset

    @offset.setter
    def offset(self, value):
        """
        Set value conversion offset.
        """
        self._offset = value

    @property
    def reference_field(self):
        """
        :return: reference field name.
        """
        return self._reference_field

    @reference_field.setter
    def reference_field(self, value):
        """
        Set reference field name.
        """
        if self._reference_type in ["feature", "previous"]:
            self._reference_field = value
        else:
            raise Exception(
                "Can add reference field only for value type 'feature' or 'previous'"
            )

    @property
    def fixed_value(self):
        """
        :return: fixed value of reference.
        """
        return self._fixed_value

    @fixed_value.setter
    def fixed_value(self, value):
        """
        :return: fixed value of reference.
        """
        if self._reference_type == "fixed":
            self._fixed_value = value
        else:
            raise Exception("Can set fixed value only for reference type 'fixed'")

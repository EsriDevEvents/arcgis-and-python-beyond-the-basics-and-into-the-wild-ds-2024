import uuid
import arcgis
from .._utils._basewidget import _BaseWidget
from .._utils._basewidget import NoDataProperties


class Gauge(_BaseWidget):
    """
    Creates a dashboard Gauge widget.

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    item                        Required Portal :class:`~arcgis.gis.Item` object. Item object can
                                be a :class:`~arcgis.features.FeatureLayer` or a MapWidget.
    -------------------------   -------------------------------------------
    name                        Optional string. Name of the gauge widget.
    -------------------------   -------------------------------------------
    layer                       Optional integer. Layer number when item is
                                a mapwidget.
    -------------------------   -------------------------------------------
    title                       Optional string. Title or Caption for the
                                widget.
    -------------------------   -------------------------------------------
    description                 Optional string. Description for the widget.
    =========================   ===========================================
    """

    def __init__(self, item, name="Gauge", layer=0, title="", description=""):
        super().__init__(name, title, description)
        # General Block

        if item.type not in ["Feature Service", "mapWidget"]:
            raise Exception("Please specify an item")

        self.item = item
        self.type = "gaugeWidget"
        self.layer = layer

        self._max_features = 50
        self._show_last_update = True

        self._nodata = NoDataProperties._nodata_init()
        self._novalue = NoDataProperties._nodata_init()

        self._data = GaugeData._create_data(data_item=item)
        self._mindata = GaugeData._create_data(data_item=item, value_type="fixedvalue")
        self._maxdata = GaugeData._create_data(data_item=item, value_type="fixedvalue")
        self._gauge = GaugeProperties._gauge_init()

    @classmethod
    def _from_json(cls, widget_json):
        gis = arcgis.env.active_gis
        itemid = widget_json["datasets"]["datasource"]["itemid"]
        name = widget_json["name"]
        item = gis.content.get(itemid)
        title = widget_json["caption"]
        description = widget_json["description"]
        gauge = Gauge(item, name, title, description)

        return gauge

    @property
    def data(self):
        """
        :return: Gauge Data object. Set data properties, categories and values.
        """
        return self._data

    @property
    def gauge_options(self):
        """
        :return: Gauge options object. Set gauge properties.
        """
        return self._gauge

    @property
    def no_data(self):
        """
        :return: Nodata Object, set various nodata properties
        """
        return self._nodata

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
        self._show_last_update = value
        if not isinstance(value, bool):
            self._show_last_update = True

    def _convert_to_json(self):
        self._arrows = []
        self._bands = []
        self._axes = []
        self._datasets = []
        self._data_statistic_defintion = []
        self._mindata_statistic_defintion = []
        self._maxdata_statistic_defintion = []

        if self._gauge.style == "meter":
            self._arrows.append(
                {
                    "id": "value",
                    "color": None,
                    "axis": "ticks",
                    "alpha": 1,
                    "borderAlpha": 1,
                    "radius": "100%",
                    "innerRadius": 0,
                    "nailRadius": 0,
                    "startWidth": 8,
                }
            )
        if self._gauge.style == "progress":
            self._bands.append(
                {
                    "alpha": 1,
                    "color": "#bee8ff",
                    "startValue": 0,
                    "endValue": 1,
                    "radius": "100%",
                    "innerRadius": "75%",
                    "colorThresholds": [],
                }
            )

        if self._gauge.shape == "horseshoe":
            self._gauge_start_angle = -90
            self._gauge_end_angle = 90
        elif self._gauge.shape == "halfdonut":
            self._gauge_start_angle = -120
            self._gauge_end_angle = 120
        else:
            self._gauge_start_angle = 0
            self._gauge_end_angle = 360

        self._axes.append(
            {
                "id": "main",
                "style": self._gauge.style,
                "startAngle": self._gauge_start_angle,
                "endAngle": self._gauge_end_angle,
                "startValue": 0,
                "endValue": 1,
                "labelsEnabled": True,
                "labelOffset": 0,
                "color": "#8400a8",
                "inside": True,
                "gridInside": True,
                "axisAlpha": 0,
                "axisColor": "#8400a8",
                "axisThickness": 0,
                "tickAlpha": 0,
                "tickColor": "#8400a8",
                "tickLength": 0,
                "tickThickness": 0,
                "minorTickLength": 0,
                "radius": "80%",
                "bottomText": "",
                "bands": self._bands,
            }
        )
        if self._gauge.style == "meter":
            self._axes.append(
                {
                    "id": "ticks",
                    "style": self._gauge.style,
                    "startAngle": self._gauge_start_angle,
                    "endAngle": self._gauge_end_angle,
                    "startValue": 0,
                    "endValue": 1,
                    "labelsEnabled": False,
                    "labelOffset": 0,
                    "color": "#8400a8",
                    "inside": True,
                    "gridInside": True,
                    "axisAlpha": 0.7,
                    "axisColor": "#005ce6",
                    "axisThickness": 3,
                    "tickAlpha": 0.7,
                    "tickColor": "#005ce6",
                    "tickLength": -12,
                    "tickThickness": 3,
                    "minorTickLength": -10,
                    "radius": "80%",
                    "bottomText": "",
                    "bands": self._bands,
                }
            )
            self._axes.append(
                {
                    "id": "labels",
                    "style": self._gauge.style,
                    "startAngle": self._gauge_start_angle,
                    "endAngle": self._gauge_end_angle,
                    "startValue": 0,
                    "endValue": 1,
                    "labelsEnabled": True,
                    "labelOffset": 32,
                    "fontSize": 20,
                    "color": "#ff5500",
                    "inside": False,
                    "gridInside": True,
                    "axisAlpha": 0,
                    "axisColor": "#8400a8",
                    "axisThickness": 0,
                    "tickAlpha": 0,
                    "tickColor": "#8400a8",
                    "tickLength": 0,
                    "tickThickness": 0,
                    "minorTickLength": 0,
                    "radius": "80%",
                    "bottomText": "",
                    "bands": self._bands,
                }
            )
        if self._data.value_type == "statistic":
            self._data_statistic_defintion.append(
                {
                    "onStatisticField": self._data.statistics_field,
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
                "layerId": 0,
                "table": True,
            }
        self._datasets.append(
            {
                "type": "serviceDataset",
                "dataSource": self._datasource,
                "outFields": ["*"],
                "groupByFields": [],
                "orderByFields": [],
                "statisticDefinitions": self._data_statistic_defintion,
                "querySpatialRelationship": "esriSpatialRelIntersects",
                "returnGeometry": False,
                "clientSideStatistics": False,
                "name": "main",
            }
        )
        if self._mindata.value_type == "fixedvalue":
            self._datasets.append(
                {
                    "type": "staticDataset",
                    "data": self._mindata.min_value,
                    "name": "min",
                }
            )

        elif self.mindata.value_type in ["statistic", "feature"]:
            if self.mindata.value_type == "statistic":
                self._mindata_statistic_defintion.append(
                    {
                        "onStatisticField": self._mindata.statistics_field,
                        "outStatisticFieldName": "value",
                        "statisticType": self._mindata.statistic,
                    }
                )
            self._datasets.append(
                {
                    "type": "serviceDataset",
                    "dataSource": {
                        "type": "featureServiceDataSource",
                        "itemId": self.item.itemid,
                        "layerId": 0,
                        "table": False,
                    },
                    "outFields": ["*"],
                    "groupByFields": [],
                    "orderByFields": [],
                    "statisticDefinitions": self._mindata_statistic_defintion,
                    "querySpatialRelationship": "esriSpatialRelIntersects",
                    "returnGeometry": False,
                    "clientSideStatistics": False,
                    "name": "min",
                }
            )

        if self._maxdata.value_type == "fixedvalue":
            self._datasets.append(
                {
                    "type": "staticDataset",
                    "data": self._maxdata.max_value,
                    "name": "max",
                }
            )
        elif self.maxdata.value_type in ["statistic", "feature"]:
            if self.maxdata.value_type == "statistic":
                self._maxdata_statistic_defintion.append(
                    {
                        "onStatisticField": self._maxdata.statistics_field,
                        "outStatisticFieldName": "value",
                        "statisticType": self._maxdata.statistic,
                    }
                )
            self._datasets.append(
                {
                    "type": "serviceDataset",
                    "dataSource": {
                        "type": "featureServiceDataSource",
                        "itemId": self.item.itemid,
                        "layerId": 0,
                        "table": False,
                    },
                    "outFields": ["*"],
                    "groupByFields": [],
                    "orderByFields": [],
                    "statisticDefinitions": self._maxdata_statistic_defintion,
                    "querySpatialRelationship": "esriSpatialRelIntersects",
                    "returnGeometry": False,
                    "clientSideStatistics": False,
                    "name": "max",
                }
            )
        json_data = {
            "type": "gaugeWidget",
            "style": self._gauge.style,
            "displayAsPercentage": False,
            "valueConversion": {
                "factor": self._data.factor,
                "offset": self._data.offset,
            },
            "minimumConversion": {
                "factor": self._mindata.factor,
                "offset": self._mindata.offset,
            },
            "maximumConversion": {
                "factor": self._maxdata.factor,
                "offset": self._maxdata.offset,
            },
            "valueFormat": {
                "name": "value",
                "type": "decimal",
                "prefix": True,
                "pattern": "#,###.#",
            },
            "percentageFormat": {
                "name": "percentage",
                "type": "decimal",
                "prefix": False,
                "pattern": "#.#%",
            },
            "arrows": self._arrows,
            "axes": self._axes,
            "labels": [
                {
                    "id": "value",
                    "align": "center",
                    "color": "#ff0000",
                    "size": 12,
                    "y": "40%",
                }
            ],
            "valueField": self._data.value_field,
            "noValueVerticalAlignment": self._novalue.alignment,
            "showCaptionWhenNoValue": self._novalue.show_title,
            "showDescriptionWhenNoValue": self._novalue.show_description,
            "valueType": self._data.value_type,
            "backgroundColor": self._background_color,
            "textColor": self._text_color,
            "datasets": self._datasets,
            "id": self._id,
            "name": self.name,
            "caption": self.title,
            "description": self.description,
            "showLastUpdate": self.show_last_update,
            "noDataText": self._nodata.text,
            "noDataVerticalAlignment": self._nodata.alignment,
            "showCaptionWhenNoData": self._nodata.show_title,
            "showDescriptionWhenNoData": self._nodata.show_description,
        }
        return json_data


class GaugeData(object):
    @classmethod
    def _create_data(cls, data_item=None, value_type="statistic"):
        data = GaugeData()

        data._item = data_item

        data._value_type = "statistic"
        data._min_value = 1
        data._max_value = 100
        data._statistic = "count"
        data._value_field = data.objectid_field
        data._statistics_field = data.objectid_field
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
        if value in ["fixedvalue", "statistic", "feature"]:
            self._value_type = str(value)
        else:
            raise Exception(
                "Select correct value type. Supported value types are 'fixedvalue', 'statistic', 'feature'"
            )

    @property
    def min_value(self):
        """
        :return: min value of data.
        """
        return self._min_value

    @min_value.setter
    def min_value(self, value):
        """
        Set min value of data.
        """
        if self.value_type == "fixedvalue":
            self._min_value = value
        else:
            raise Exception("Can set minimum value for only 'fixedvalue' value type")

    @property
    def max_value(self):
        """
        :return: max value of data.
        """
        return self._max_value

    @max_value.setter
    def max_value(self, value):
        """
        Set max vaue of data.
        """
        if self.value_type == "fixedvalue":
            self._max_value = value
        else:
            raise Exception("Can set maximum value for only 'fixedvalue' value type")

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
    def objectid_field(self):
        """
        :return: object ID field name.
        """
        return self._item.tables[0].properties["objectIdField"]

    @property
    def statistics_field(self):
        """
        :return: Statistics field name.
        """
        return self._statistics_field

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
        if self._value_type == "feature":
            f_type = self._field_type(value)
            if f_type in ["float64", "float32", "int64", "int32"]:
                self._value_field = value
            else:
                raise Exception("Please select a numeric field")
        else:
            raise Exception("Can add value field only for value type 'feature'")

    @property
    def filters(self):
        """
        :return: filters associated with widget
        """
        return self._filters

    def add_filter(self, field, join, condition, **kwargs):
        """
        Add filters associated with widget.
        """
        self._filter_field = field
        if join in ["AND", "OR"]:
            self._filter_join = join
        else:
            raise Exception("Please select from 'AND', 'OR'")
        if condition in [
            "between",
            "not between",
            "equal",
            "not equal",
            "greater than",
            "greater than or equal",
            "less than",
            "less than or equal",
            "is null",
            "is not null",
        ]:
            self._filter_condition = condition
        else:
            raise Exception("Please select the right condition")

        if condition in ["between", "not between"]:
            self._val1 = kwargs.get("start")
            self._val2 = kwargs.get("end")
            self._filters.append(
                {
                    "filtertype": self._filter_join,
                    "field": self._filter_field,
                    "operator": self._filter_condition,
                    "start": self._val1,
                    "end": self._val2,
                }
            )
        else:
            raise Exception("Please provide 'start' and 'end' values as parameters")

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

    def _field_type(self, field_name):
        f_type = self._item.tables[0].query().sdf[field_name].dtype
        return f_type


class GaugeProperties(object):
    @classmethod
    def _gauge_init(cls, style="progress", shape="horseshoe"):
        gauge = GaugeProperties()

        gauge._style = "progress"
        gauge._shape = "horseshoe"

        gauge.style = style
        gauge.shape = shape

        return gauge

    @property
    def style(self):
        """
        :return: gauge style.
        """
        return self._style

    @style.setter
    def style(self, value):
        """
        Set gauge style.
        """
        if value in ["progress", "meter"]:
            self._style = str(value)
        else:
            raise Exception("Please select gauge style from 'progress', 'meter'")

    @property
    def shape(self):
        """
        :return: gauge shape.
        """
        return self._shape

    @shape.setter
    def shape(self, value):
        """
        Set gauge shape.
        """
        if value in ["horseshoe", "halfdonut"]:
            self._shape = str(value)
        elif value == "circle":
            if self._style == "progress":
                self._shape = str(value)
            else:
                raise Exception(
                    "'circle' shape can only be selected for style 'progress'"
                )
        else:
            raise Exception(
                "Please select gauge shape from 'circle', 'horseshoe', 'halfdonut'"
            )

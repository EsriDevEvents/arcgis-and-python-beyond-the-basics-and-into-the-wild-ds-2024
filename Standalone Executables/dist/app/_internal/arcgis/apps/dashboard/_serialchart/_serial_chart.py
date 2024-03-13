import arcgis
from .._utils._basewidget import _BaseWidget
from .._utils._basewidget import Legend
from .._utils._basewidget import NoDataProperties


class SerialChart(_BaseWidget):
    """
    Creates a dashboard Serial Chart widget.

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    item                        Required Portal :class:`~arcgis.gis.Item` object. Item object can
                                be a :class:`~arcgis.features.FeatureLayer`  or a MapWidget.
    -------------------------   -------------------------------------------
    name                        Optional string. Name of the serial chart
                                widget.
    -------------------------   -------------------------------------------
    layer                       Optional integer. Layer number when item is
                                a mapwidget.
    -------------------------   -------------------------------------------
    categories_from             Optional string. Select from groupByValues,
                                features or fields.
    -------------------------   -------------------------------------------
    title                       Optional string. Title or Caption for the
                                widget.
    -------------------------   -------------------------------------------
    description                 Optional string. Description for the widget.
    =========================   ===========================================
    """

    def __init__(
        self,
        item,
        name="SerialChart",
        layer=0,
        categories_from="groupByValues",
        title="",
        description="",
    ):
        if item.type not in ["Feature Service", "mapWidget"]:
            raise Exception("Please specify an item")

        super().__init__(name, title, description)

        self.item = item
        self.layer = layer
        self.type = "serialChartWidget"

        self._data = SerialChartData._create_serial_chart_data(
            categories_from, data_item=item
        )
        self._category_axis_properties = _CategoryAxisProperties._create_category_axis()
        self._scroll = False
        self._value_axis_properties = _ValueAxisProperties._create_value_axis()
        self._legend = Legend._create_legend()
        self._color = "#474747"
        self._font_size = 11
        self._orientation = "vertical"
        self._last_update = True
        self._no_data = NoDataProperties._nodata_init()
        self._events = Events._create_events()

    @classmethod
    def _from_json(cls, widget_json):
        gis = arcgis.env.active_gis
        itemid = widget_json["datasets"]["datasource"]["itemid"]
        name = widget_json["name"]
        item = gis.content.get(itemid)
        title = widget_json["caption"]
        categories_from = widget_json["categoryType"]
        description = widget_json["description"]
        schart = SerialChart(item, name, 0, categories_from, title, description)
        schart.data.category_field = widget_json["category"]["fieldName"]
        schart.legend.visibility = widget_json["legend"]["enabled"]
        schart.legend.placement = widget_json["legend"]["position"]
        return schart

    @property
    def events(self):
        """
        :return: List of events attached to the widget.
        """
        return self._events

    @property
    def data(self):
        """
        :return: Serial Chart Data object. Set data properties, categories and values.
        """
        return self._data

    @property
    def category_axis(self):
        """
        :return: Returns Category Axis Properties object.
        """
        return self._category_axis_properties

    @property
    def value_axis(self):
        """
        :return: Value Axis Properties object.
        """
        return self._value_axis_properties

    @property
    def legend(self):
        """
        :return: Legend Object, set Visibility and placement
        """
        return self._legend

    @property
    def scroll(self):
        """
        :return: True if scroll is enabled else False.
        """
        return self._scroll

    @scroll.setter
    def scroll(self, value):
        """
        Set scroll True or False.
        """
        self._scroll = bool(value)

    @property
    def font_size(self):
        """
        :return: Font Size
        """
        return self._font_size

    @font_size.setter
    def font_size(self, value):
        """
        Set font size.
        """
        self._font_size = value

    @property
    def orientation(self):
        """
        :return: Orientation of the serial chart, "horizontal" or "vertical".
        """
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        """
        Set orientation of the serial chart, "horizontal" or "vertical".
        """
        if value in ["horizontal", "vertical"]:
            self._orientation = value

    @property
    def no_data(self):
        """
        :return: NoDataProperties Object
        """
        return self._no_data

    def _convert_to_json(self):
        common_graph_properties = {
            "lineColorField": "_lineColor_",
            "fillColorsField": "_fillColor_",
            "type": "column",
            "fillAlphas": 1,
            "lineAlpha": 1,
            "lineThickness": 1,
            "bullet": "none",
            "bulletAlpha": 1,
            "bulletBorderAlpha": 0,
            "bulletBorderThickness": 2,
            "showBalloon": self.data._show_baloon,
            "bulletSize": 8,
        }

        if self._data._labels:
            common_graph_properties["labelText"] = "[[value]]"

        json_data = {
            "type": "serialChartWidget",
            "category": {
                "labelOverrides": [],
                "byCategoryColors": False,
                "labelsPlacement": "default",  # default, staggered, wrapped, rotated
                "labelRotation": 0,
                "fieldName": self.data.category_field,
                "nullLabel": "Null",
                "blankLabel": "Blank",
                "defaultColor": "#d6d6d6",
                "nullColor": "#d6d6d6",
                "blankColor": "#d6d6d6",
            },
            "valueFormat": {
                "name": "value",
                "type": "decimal",
                "prefix": True,
                "pattern": "#,###.#",
            },
            "labelFormat": {
                "name": "label",
                "type": "decimal",
                "prefix": True,
                "pattern": "#,###.#",
            },
            "dateParsingPattern": self._data.parsing_pattern,
            "datePeriodPatterns": [
                {"period": "ss", "pattern": "HH:mm:ss"},
                {"period": "mm", "pattern": "HH:mm"},
                {"period": "hh", "pattern": "HH:mm"},
                {"period": "DD", "pattern": "MMM d"},
                {"period": "MM", "pattern": "MMM"},
                {"period": "YYYY", "pattern": "yyyy"},
            ],
            "chartScrollbar": {
                "enabled": self._scroll,
                "dragIcon": "dragIconRoundSmall",
                "dragIconHeight": 20,
                "dragIconWidth": 20,
                "scrollbarHeight": 15,
            },
            "categoryAxis": self._category_axis_properties._convert_to_json(),
            "valueAxis": self._value_axis_properties._convert_to_json(),
            "legend": {
                "enabled": self._legend.visibility,
                "position": "right"
                if self._legend.placement == "side"
                else self._legend.placement,
                "markerSize": 15,
                "markerType": "circle",
                "align": "center",
                "labelWidth": 100,
                "valueWidth": 0,
            },
            "graphs": [series for series in self.data._series],
            "guides": [],
            "splitBy": {"defaultColor": "#d6d6d6", "seriesProperties": []},
            "rotate": False if self._orientation == "vertical" else True,
            "commonGraphProperties": common_graph_properties,
            "events": [],
            "selectionMode": "multi",
            "categoryType": self.data.categories_from,
            "datasets": [],
            "id": self._id,
            "name": self.name,
            "caption": self.title,
            "description": self.description,
            "showLastUpdate": self._last_update,
            "noDataVerticalAlignment": self._no_data._alignment,
            "showCaptionWhenNoData": self._no_data._show_title,
            "showDescriptionWhenNoData": self._no_data._show_description,
        }

        # json_data['categoryAxis']['parseDates'] = self._data.parse_dates
        if self._no_data._text:
            json_data["noDataText"] = self._no_data._text

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
        if (
            self.data.orderby_field == ""
            and self._data._categories_from != "groupByValues"
        ):
            self._orderby_field = self.data.category_field
        else:
            self._orderby_field = self.data.orderby_field

        dataset = {
            "type": "serviceDataset",
            "dataSource": self._datasource,
            "outFields": ["*"],
            "groupByFields": [],
            "orderByFields": [self._orderby_field + " asc"]
            if self._orderby_field
            else [],
            "statisticDefinitions": [],
            "querySpatialRelationship": "esriSpatialRelIntersects",
            "returnGeometry": False,
            "clientSideStatistics": False,  # False if self._data._categories_from == "groupByValues" else True,
            "name": "main",
        }

        if self._data.max_features:
            dataset["maxFeatures"] = self._data.max_features

        if self._color:
            json_data["color"] = self._color

        if self._font_size:
            json_data["fontSize"] = self._font_size

        if self._data._categories_from == "groupByValues":
            json_data["categoryAxis"]["parseDates"] = self._data._parse_dates
            if self._data._category_field:
                dataset["groupByFields"].append(self._data._category_field)
            if self._data._split_by_field:
                json_data["splitBy"]["fieldName"] = self._split_by_field
                dataset["groupByFields"].append(self._data._split_by_field)

            dataset["statisticDefinitions"].append(
                {
                    "onStatisticField": self._data.statistics_field,
                    "outStatisticFieldName": "value",
                    "statisticType": self._data.statistic,
                }
            )
        elif self._data._categories_from == "fields":
            for category in self._data._category_fields:
                dataset["statisticDefinitions"].append(
                    {
                        "onStatisticField": category,
                        "outStatisticFieldName": category,
                        "statisticType": self._data.statistic,
                    }
                )

        json_data["datasets"] = [dataset]

        if self.events.enable:
            json_data["events"].append(
                {"type": self.events.type, "actions": self.events.synced_widgets}
            )
            json_data["selectionMode"] = self.events.selection_mode

        return json_data


class _CategoryAxisProperties(object):
    @classmethod
    def _create_category_axis(cls):
        category_axis = cls()
        category_axis._title = ""

        category_axis._title_rotation = 0
        category_axis._font_size = 12
        category_axis._grid_thickness = 1
        category_axis._grid_opacity = 0.15
        category_axis._grid_color = "#ffffff"
        category_axis._axis_thickness = 1
        category_axis._axis_opacity = 0.5
        category_axis._axis_color = "#000000"
        category_axis._title_size = 12
        category_axis._labels = True
        category_axis._parse_dates = True
        category_axis._minimum_period = "DD"
        category_axis._grid_position = "start"

        category_axis._date_dict = {
            "days": "DD",
            "hours": "hh",
            "seconds": "ss",
            "minutes": "mm",
            "months": "MM",
            "years": "YYYY",
        }

        category_axis._requirements = {
            "title": [str],
            "gridThickness": [int, 1, 10],
            "gridAlpha": [(int, float), 0, 1],
            "axisThickness": [int, 1, 10],
            "axisAlpha": [(int, float), 0, 1],
            "labelsEnabled": [bool],
            "parseDates": [bool],
            "titleFontSize": [int, 0, 1000],
            "fontSize": [int, 0, 1000],
            "gridColor": [str],
            "axisColor": [str],
        }

        category_axis._translations = {
            "axisAlpha": "axisOpacity",
            "gridAlpha": "gridOpacity",
        }

        category_axis._translations_inverse = {
            value: key for key, value in category_axis._translations.items()
        }

        category_axis._hidden = ["minPeriod", "titleRotation", "gridPosition"]

        return category_axis

    @property
    def title(self):
        """
        :return: Category axis Title
        """
        return self._title

    @title.setter
    def title(self, value):
        """
        Set Category axis Title
        """
        if self._validation("title", value):
            self._title = value

    @property
    def grid_thickness(self):
        """
        :return: Category axis grid thickness.
        """
        return self._grid_thickness

    @grid_thickness.setter
    def grid_thickness(self, value):
        """
        Set Category axis grid thickness.
        """
        if self._validation("gridThickness", value):
            self._grid_thickness = value

    @property
    def grid_opacity(self):
        """
        :return: Category axis grid opacity.
        """
        return self._grid_opacity

    @grid_opacity.setter
    def grid_opacity(self, value):
        """
        Set Category axis grid opacity.
        """
        if self._validation("gridOpacity", value):
            self._grid_opacity = value

    @property
    def axis_thickness(self):
        """
        :return: Category axis thickness.
        """
        return self._axis_thickness

    @axis_thickness.setter
    def axis_thickness(self, value):
        """
        Set Category axis thickness.
        """
        if self._validation("axisThickness", value):
            self._axis_thickness = value

    @property
    def axis_opacity(self):
        """
        :return: Category axis opacity.
        """
        return self._axis_opacity

    @axis_opacity.setter
    def axis_opacity(self, value):
        """
        Set Category axis opacity.
        """
        if self._validation("axisOpacity", value):
            self._axis_opacity = value

    @property
    def title_size(self):
        """
        :return: Category axis title size.
        """
        return self._title_size

    @title_size.setter
    def title_size(self, value):
        """
        Set Category axis title size.
        """
        if self._validation("titleFontSize", value):
            self._title_size = value

    @property
    def font_size(self):
        """
        :return: Font size
        """
        return self._font_size

    @font_size.setter
    def font_size(self, value):
        """
        Set font size.
        """
        self._font_size = value

    @property
    def grid_color(self):
        """
        :return: Category axis grid color, hex code.
        """
        return self._grid_color

    @grid_color.setter
    def grid_color(self, value):
        """
        Set Category axis grid color, hex code.
        """
        self._grid_color = value

    @property
    def axis_color(self):
        """
        :return: Category axis color, hex code.
        """
        return self._axis_color

    @axis_color.setter
    def axis_color(self, value):
        """
        Set Category axis color, hex code.
        """
        self._axis_color = value

    @property
    def labels(self):
        """
        :return: Labels Object. Set visibility True or False.
        """
        return self._labels

    @labels.setter
    def labels(self, value):
        """
        Set labels True or False.
        """
        self._labels = bool(value)

    @property
    def minimum_period(self):
        """
        :return: Minimum period when dates are parsed.
        """
        return self._minimum_period

    @minimum_period.setter
    def minimum_period(self, value):
        """
        Set Minimum period when dates are parsed.
        Allowed values 'seconds', 'minutes', 'hours', 'days', 'months', 'years'
        """
        if isinstance(value, str) and value.lower() in [
            "seconds",
            "minutes",
            "hours",
            "days",
            "months",
            "years",
        ]:
            self._minimum_period = self._date_dict[value.lower()]
        else:
            raise Exception("Please select correct value")

    def __init__(self):
        self._item = {
            "title": "",
            "titleFontSize": 12,
            "fontSize": 12,
            "gridColor": "#ffffff",
            "axisColor": "#ffffff",
            "titleRotation": 0,
            "gridPosition": "start",
            "gridThickness": 1,
            "gridAlpha": 0.15,
            "axisThickness": 1,
            "axisAlpha": 0.5,
            "labelsEnabled": False,
            "parseDates": True,
            "minPeriod": "DD",
        }

    def _validation(self, key, value):
        if self._requirements.get(key):
            requirements = self._requirements.get(key)
        elif self._requirements.get(self._translations_inverse.get(key)):
            requirements = self._requirements.get(self._translations_inverse.get(key))
        else:
            return False

        if len(requirements) == 1 and isinstance(value, requirements[0]):
            pass
        elif (
            len(requirements) == 3
            and isinstance(value, requirements[0])
            and value >= requirements[1]
            and value <= requirements[2]
        ):
            pass
        else:
            return False

        return True

    def _convert_to_json(self):
        return {
            "title": self._title,
            "titleRotation": 0,
            "titleFontSize": self._title_size,
            "fontSize": self._font_size,
            "gridThickness": self._grid_thickness,
            "gridAlpha": self._grid_opacity,
            "gridColor": self._grid_color,
            "axisThickness": self._axis_thickness,
            "axisAlpha": self._axis_opacity,
            "axisColor": self._axis_color,
            "labelsEnabled": self._labels,
            "gridPosition": "start",
            "parseDates": True,
            "minPeriod": self._minimum_period,
        }


class _ValueAxisProperties(object):
    @classmethod
    def _create_value_axis(cls):
        value_axis = cls()
        value_axis._title = ""

        value_axis._title_rotation = 270
        value_axis._font_size = 12
        value_axis._title_size = 12
        value_axis._grid_thickness = 1
        value_axis._grid_opacity = 0.15
        value_axis._grid_color = "#ffffff"
        value_axis._axis_thickness = 1
        value_axis._axis_opacity = 0.5
        value_axis._axis_color = "#000000"
        value_axis._minimum = None
        value_axis._maximum = None

        value_axis._labels = True

        value_axis._stackType = "none"
        value_axis._integers_only = False
        value_axis._logarithmic = False

        value_axis._requirements = {
            "title": [str],
            "gridThickness": [int, 1, 10],
            "gridAlpha": [(int, float), 0, 1],
            "axisThickness": [int, 1, 10],
            "axisAlpha": [(int, float), 0, 1],
            "labelsEnabled": [bool],
            "parseDates": [bool],
            "titleFontSize": [int, 0, 1000],
            "fontSize": [int, 0, 1000],
            "gridColor": [str],
            "axisColor": [str],
        }

        value_axis._translations = {
            "axisAlpha": "axisOpacity",
            "gridAlpha": "gridOpacity",
        }

        value_axis._translations_inverse = {
            value: key for key, value in value_axis._translations.items()
        }

        value_axis._hidden = ["stackType", "titleRotation"]

        return value_axis

    @property
    def title(self):
        """
        :return: Value axis Title
        """
        return self._title

    @title.setter
    def title(self, value):
        """
        Set Value axis Title
        """
        if self._validation("title", value):
            self._title = value

    @property
    def grid_thickness(self):
        """
        :return: Value axis grid thickness.
        """
        return self._grid_thickness

    @grid_thickness.setter
    def grid_thickness(self, value):
        """
        Set Value axis grid thickness.
        """
        if self._validation("gridThickness", value):
            self._grid_thickness = value

    @property
    def grid_opacity(self):
        """
        :return: Value axis grid opacity.
        """
        return self._grid_opacity

    @grid_opacity.setter
    def grid_opacity(self, value):
        """
        Set Value axis grid opacity.
        """
        if self._validation("gridOpacity", value):
            self._grid_opacity = value

    @property
    def axis_thickness(self):
        """
        :return: Value axis thickness.
        """
        return self._axis_thickness

    @axis_thickness.setter
    def axis_thickness(self, value):
        """
        Set Value axis thickness.
        """
        if self._validation("axisThickness", value):
            self._axis_thickness = value

    @property
    def axis_opacity(self):
        """
        :return: Value axis opacity.
        """
        return self._axis_opacity

    @axis_opacity.setter
    def axis_opacity(self, value):
        """
        Set Value axis opacity.
        """
        if self._validation("axisOpacity", value):
            self._axis_opacity = value

    @property
    def labels(self):
        """
        :return: Labels Object. Set visibility True or False.
        """
        return self._labels

    @labels.setter
    def labels(self, value):
        """
        Set labels True or False.
        """
        self._labels = bool(value)

    @property
    def title_size(self):
        """
        :return: Value axis title size.
        """
        return self._title_size

    @title_size.setter
    def title_size(self, value):
        """
        Set value axis title size.
        """
        if self._validation("titleFontSize", value):
            self._title_size = value

    @property
    def grid_color(self):
        """
        :return: Value axis grid color, hex code.
        """
        return self._grid_color

    @grid_color.setter
    def grid_color(self, value):
        """
        Set value axis grid color, hex code.
        """
        self._grid_color = value

    @property
    def axis_color(self):
        """
        :return: Value axis color, hex code.
        """
        return self._axis_color

    @axis_color.setter
    def axis_color(self, value):
        """
        Set value axis color, hex code.
        """
        self._axis_color = value

    @property
    def only_integers(self):
        """
        :return: True if only integers are allowed.
        """
        return self._integers_only

    @only_integers.setter
    def only_integers(self, value):
        """
        Set if only integers are allowed.
        """
        self._integers_only = bool(value)

    @property
    def logarithmic(self):
        """
        :return: True if logarithmic.
        """
        return self._logarithmic

    @logarithmic.setter
    def logarithmic(self, value):
        """
        Set if only integers are allowed.
        """
        self._logarithmic = bool(value)

    @property
    def minimum(self):
        """
        :return: Filter by specifying minimum value possible.
        """
        return self._minimum

    @minimum.setter
    def minimum(self, value):
        """
        Set minimum value to filter data.
        """
        self._minimum = int(value)

    @property
    def maximum(self):
        """
        :return: Filter by specifying maximum value possible.
        """
        return self._maximum

    @maximum.setter
    def maximum(self, value):
        """
        Set maximum value to filter data.
        """
        self._maximum = int(value)

    def _validation(self, key, value):
        if self._requirements.get(key):
            requirements = self._requirements.get(key)
        elif self._requirements.get(self._translations_inverse.get(key)):
            requirements = self._requirements.get(self._translations_inverse.get(key))
        else:
            return False

        if len(requirements) == 1 and isinstance(value, requirements[0]):
            pass
        elif (
            len(requirements) == 3
            and isinstance(value, requirements[0])
            and value >= requirements[1]
            and value <= requirements[2]
        ):
            pass
        else:
            return False

        return True

    def _convert_to_json(self):
        return {
            "title": self._title,
            "titleRotation": 270,
            "fontSize": self._title_size,
            "gridThickness": self._grid_thickness,
            "gridAlpha": self._grid_opacity,
            "gridColor": self._grid_color,
            "axisThickness": self._axis_thickness,
            "axisAlpha": self._axis_opacity,
            "axisColor": self._axis_color,
            "labelsEnabled": self._labels,
            "stackType": "none",
            "integersOnly": self._integers_only,
            "logarithmic": self._logarithmic,
        }


class SerialChartData(object):
    @classmethod
    def _create_serial_chart_data(cls, categories_from, data_item=None):
        schart_data = SerialChartData()

        schart_data._categories_from = None
        if categories_from in ["groupByValues", "features", "fields"]:
            schart_data._categories_from = categories_from
        else:
            raise Exception(
                'Invalid option, choose from "groupByValues", "features", "fields"'
            )
        schart_data._series = []

        if categories_from == "features":
            schart_data._series = []
        elif categories_from == "groupByValues":
            schart_data._split_by_field = None
            schart_data._series = []

        schart_data._show_baloon = False
        schart_data._item = data_item
        schart_data._parse_dates = False
        schart_data._max_features = None
        schart_data._statistic = "count"
        schart_data._statistics_field = schart_data.objectid_field
        schart_data._stacking = "off"
        schart_data._labels = True
        schart_data._orderby = ""

        schart_data._filters = []

        schart_data._parsing_pattern = "yyyyMMdd"
        schart_data._category_field = None
        if schart_data._categories_from == "fields":
            schart_data._category_field = "category"

        schart_data._split_by_field = None

        return schart_data

    @property
    def categories_from(self):
        """
        :return: Categories from groupByValues, features or fields.
        """
        return self._categories_from

    @property
    def max_features(self):
        """
        :return: Max features to display
        """
        return self._max_features

    @max_features.setter
    def max_features(self, value):
        """
        Set max features to display.
        """
        self._max_features = value

    @property
    def category_field(self):
        """
        :return: Category field from dataset. For groupByValues or features.
        """
        return self._category_field

    @category_field.setter
    def category_field(self, value):
        """
        Set category field from dataset. For groupByValues or features.
        """
        if self._categories_from == "fields":
            raise Exception("Can't set this attribute for 'fields' categories.")

        if self._categories_from == "groupByValues":
            raise Exception(
                'Set the field using add_value_field() for "groupByValues" categories'
            )
        if self._field_type(value) == "<M8[ns]":
            self.parse_dates = True

        self._category_field = value

    @property
    def orderby_field(self):
        """
        :return: OrderBy field from dataset. For groupByValues or features.
        """
        return self._orderby

    @orderby_field.setter
    def orderby_field(self, value):
        """
        Set OrderBy field from dataset. For groupByValues or features.
        """

        self._orderby = value

    @property
    def objectid_field(self):
        """
        :return: object ID field name.
        """
        try:
            return self._item.tables[0].properties["objectIdField"]
        except:
            return "OBJECTID"

    @property
    def statistics_field(self):
        """
        :return: Statistics field name.
        """
        return self._statistics_field

    @property
    def parse_dates(self):
        """
        :return: True if input category field of type date is to be parsed.
        """
        return self._parse_dates

    @parse_dates.setter
    def parse_dates(self, value):
        """
        Set True to parse input category fields of type date. For groupByValues and features.
        """
        self._parse_dates = bool(value)

    @property
    def parsing_pattern(self):
        """
        :return: Parsing pattern for date fields.
        """
        return self._parsing_pattern

    @parsing_pattern.setter
    def parsing_pattern(self, value):
        """
        Set a date parsing pattern for date field.
        """
        if not isinstance(value, str):
            raise Exception("Please enter a string pattern")

        self._parsing_pattern = value

    @property
    def statistic(self):
        """
        :return: statistic used for input category fields. For values from fields.
        """
        return self._statistic

    @statistic.setter
    def statistic(self, value):
        """
        Set statistic to 'count', 'avg', 'min', 'max', 'stddev', 'sum'
        """
        if value in ["count", "avg", "min", "max", "stddev", "sum"]:
            self._statistic = value

    @property
    def split_by_field(self):
        """
        :return: Field to split by for groupByValues.
        """
        return self._split_by_field

    @split_by_field.setter
    def split_by_field(self, value):
        """
        Set field name from the dataset to split data by, for groupByValues.
        """
        self._split_by_field = value

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
            self._val1 = kwargs.get("start", "")
            self._val2 = kwargs.get("end", "")
            if self._val1 is None or self._val2 is None:
                raise Exception("Please provide 'start' and 'end' values as parameters")
            self._filters.append(
                {
                    "filtertype": self._filter_join,
                    "field": self._filter_field,
                    "operator": self._filter_condition,
                    "start": self._val1,
                    "end": self._val2,
                }
            )
        elif condition in [
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
        elif condition in ["is null", "is not null"]:
            self._filters.append(
                {
                    "filtertype": self._filter_join,
                    "field": self._filter_field,
                    "operator": self._filter_condition,
                }
            )
        else:
            raise Exception("Please provide a valid condition")

    def _field_type(self, field_name):
        f_type = self._item.tables[0].query().sdf[field_name].dtype
        return f_type

    def add_value_field(
        self,
        value_field,
        label=None,
        graph_type="line",
        show_data_points=False,
        point_size=8,
        line_thickness=1,
        line_color="#ffaa00",
        fill_opacity=0,
        line_opacity=1,
    ):
        """
         Add value field to serial chart.

        =========================   ===========================================
        **Parameter**                **Description**
        -------------------------   -------------------------------------------
        value_field                 Required field or list of fields from input item.
                                    For groupByValues only one field is accepted.
                                    For series, add multiple fields one by one.
                                    For fields, add a list of fields.
        -------------------------   -------------------------------------------
        label                       Optional string. Label to show on the graph.
        -------------------------   -------------------------------------------
        show_data_points            Optional string. To show data points on the graph.
        -------------------------   -------------------------------------------
        point_size                  Optional string. To set data point size.
        -------------------------   -------------------------------------------
        graph_type                  Optional string. Choose from "line", "bar"
                                    "smoothed_line"
        -------------------------   -------------------------------------------
        line_thickness              Optional integer. Thickness of the lines
                                    between 1 to 10.
        -------------------------   -------------------------------------------
        line_color                  Optional string. Hex code for line color.
        -------------------------   -------------------------------------------
        fill_opacity                Optional float. Between 0 and 1.
        -------------------------   -------------------------------------------
        line_opacity                Optional float. Between 0 and 1.
        =========================   ===========================================
        """

        data = {
            "valueField": value_field[0]
            if isinstance(value_field, list)
            else value_field,
            "title": label
            if label
            else value_field[0]
            if isinstance(value_field, list)
            else value_field,
            "lineColor": line_color,
            "lineColorField": "_lineColor_",
            "fillColorsField": "_fillColor_",
            "type": graph_type,
            "fillAlphas": fill_opacity,
            "lineAlpha": line_opacity,
            "lineThickness": line_thickness,
            "bullet": "round" if show_data_points == True else "none",
            "bulletAlpha": 1,
            "bulletBorderAlpha": 0,
            "bulletBorderThickness": 2,
            "showBalloon": show_data_points,
            "bulletSize": point_size,
        }
        if self._categories_from == "features":
            if not isinstance(self._series, list):
                self._series = []
            if not data["title"]:
                data["title"] = value_field
            self._series.append(data)
        elif self._categories_from == "groupByValues":
            self._series = []
            data["valueField"] = "value"
            data["type"] = "column"
            self._series.append(data)
            self._category_field = (
                value_field[0] if isinstance(value_field, list) else value_field
            )
            if self._field_type(self._category_field) == "<M8[ns]":
                self.parse_dates = True
        elif self._categories_from == "fields":
            if isinstance(value_field, list):
                self._category_fields = value_field
            else:
                self._category_fields = [value_field]

            data["valueField"] = "value"
            self._series = data

    @property
    def stacking(self):
        """
        :return: "off", "stacked", stack 100%"
        """
        return self._stacking

    @stacking.setter
    def stacking(self, value):
        """
        Set stacking to "off", "stacked", "stack 100%"
        """

        if value in ["off", "stacked", "stack 100%"]:
            if value == "off":
                self._stacking = "none"
            elif value == "stacked":
                self._stacking = "regular"
            else:
                self._stacking = "100%"

    @property
    def hover_text(self):
        """
        :return: True if series hover text is enabled else False
        """
        return self._show_baloon

    @hover_text.setter
    def hover_text(self, value):
        """
        Set true to show hover text on series.
        """
        self._show_baloon = bool(value)

    @property
    def labels(self):
        """
        :return: Labels Object. Set visibility True or False.
        """
        return self._labels

    @labels.setter
    def labels(self, value):
        """
        Set labels True or False.
        """
        self._labels = bool(value)


class Events(object):
    @classmethod
    def _create_events(cls, enable=False):
        events = Events()

        events._enable = False
        events._selection_mode = "single"
        events._type = "selectionChanged"
        events._actions = []

        events.enable = enable

        return events

    @property
    def selection_mode(self):
        """
        :return: Selection mode of events.
        """
        return self._selection_mode

    @selection_mode.setter
    def selection_mode(self, value):
        """
        Set Selection mode of events.
        Allowed values 'single' and 'multi'
        """
        if value in ["single", "multi"]:
            self._selection_mode = value
        else:
            raise Exception("Please specify selection_mode from 'single' and 'multi'")

    @property
    def enable(self):
        """
        :return: if events are enabled or not.
        """
        return self._enable

    @enable.setter
    def enable(self, value):
        """
        Set if events are enabled or not.
        """
        self._enable = bool(value)

    @property
    def type(self):
        """
        :return: type of trigger for event.
        """
        return self._type

    @property
    def synced_widgets(self):
        """
        :return: List of synced widgets.
        """
        return self._actions

    def sync_map(self, action_type, widget):
        """
        Synchronize a mapWidget with SerialChart for triggered events.

        =========================   ===========================================
        **Parameter**                **Description**
        -------------------------   -------------------------------------------
        action_type                 Required string. Actions can be one of
                                    "zoom", "flash", "show_popup", "pan".
        -------------------------   -------------------------------------------
        widget                      Required MapWidget item. Name of the map
                                    widget.
        =========================   ===========================================
        """
        if self.enable == False:
            raise Exception("Please enable events")
        else:
            if widget.type == "mapWidget":
                if action_type in ["zoom", "flash", "show_popup", "pan"]:
                    self._actions.append({"type": action_type, "targetId": widget._id})
                elif action_type == "filter":
                    if self._targetid is not None:
                        self._actions.append(
                            {
                                "type": action_type,
                                "by": "whereClause",
                                "targetId": self._targetid,
                            }
                        )
                    else:
                        raise Exception(
                            "This operation is not suitable for given dataSource."
                        )
                else:
                    raise Exception(
                        "Please select action_type from 'zoom', 'flash', 'show_popup' and 'pan'"
                    )
            else:
                raise Exception("Please select a map widget")

    def sync_widget(self, widgets):
        """
        Synchronize non-mapWidget type widgets with SerialChart for triggered events.

        =========================   ===========================================
        **Parameter**                **Description**
        -------------------------   -------------------------------------------
        widget                      Required widget item or list of widget items
                                    .Name of the widgets to be synced.
        =========================   ===========================================
        """
        if self.enable == False:
            raise Exception("Please enable events")

        else:
            if isinstance(widgets, list):
                for widget in widgets:
                    if widget.type == "mapWidget":
                        raise Exception(
                            "Use sync_map method to add actions for map widgets"
                        )  ##duplicate or erase
                    else:
                        action_type = "filter"
                        widget_id = str(widget._id) + "#main"
                        self._actions.append(
                            {
                                "type": action_type,
                                "by": "whereClause",
                                "targetId": widget_id,
                            }
                        )
            else:
                if widgets.type == "mapWidget":
                    raise Exception(
                        "Use sync_map method to add actions for map widgets"
                    )  ##duplicate or erase
                else:
                    action_type = "filter"
                    widget_id = str(widgets._id) + "#main"
                    self._actions.append(
                        {
                            "type": action_type,
                            "by": "whereClause",
                            "targetId": widget_id,
                        }
                    )

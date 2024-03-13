import uuid
import arcgis
from .._utils._basewidget import _BaseWidget
from .._utils._basewidget import Legend
from .._utils._basewidget import NoDataProperties


class PieChart(_BaseWidget):
    """
    Creates a dashboard Pie Chart widget.

    =========================   ===========================================
    **Parameter**                **Description**
    -------------------------   -------------------------------------------
    item                        Required Portal :class:`~arcgis.gis.Item` object. Item object can
                                be a Table Layer or a MapWidget.
    -------------------------   -------------------------------------------
    name                        Optional string. Name of the pie chart
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
        name="PieChart",
        layer=0,
        categories_from="groupByValues",
        title="",
        description="",
    ):
        super().__init__(name, title, description)

        if item.type not in ["Feature Service", "mapWidget"]:
            raise Exception("Please specify an item")

        self.item = item
        self.type = "pieChartWidget"
        self.layer = layer

        self._labels = False

        self._max_features = None
        self._show_last_update = True
        self._targetid = None

        if categories_from not in ["groupByValues", "features", "fields"]:
            raise Exception("category_from can be groupByValues or features or fields")

        self._data = PieChartData._create_data(categories_from, data_item=item)
        self._pie = PieChartProperties._create_chart()
        self._slices = SliceProperties._slice_properties()
        self._nodata = NoDataProperties._nodata_init()
        self._legend = Legend._create_legend()
        self._outline = OutlineProperties._outline_init()
        self._events = Events._create_events()

    @classmethod
    def _from_json(cls, widget_json):
        gis = arcgis.env.active_gis
        itemid = widget_json["datasets"]["datasource"]["itemid"]
        name = widget_json["name"]
        item = gis.content.get(itemid)
        title = widget_json["caption"]
        description = widget_json["description"]
        pie_chart = PieChart(item, name, title, description)

        return pie_chart

    @property
    def events(self):
        """
        :return: List of events attached to the widget.
        """
        return self._events

    @property
    def data(self):
        """
        :return: Pie Chart Data object. Set data properties, categories and values.
        """
        return self._data

    @property
    def legend(self):
        """
        :return: Legend Object, set Visibility and placement
        """
        return self._legend

    @property
    def pie(self):
        """
        :return: Pie Object, set various pie properties
        """
        return self._pie

    @property
    def outline(self):
        """
        :return: Outline Object, set various outline properties
        """
        return self._outline

    @property
    def slices(self):
        """
        :return: Slices Object, set various slices properties
        """
        return self._slices

    @property
    def no_data(self):
        """
        :return: Nodata Object, set various nodata properties
        """
        return self._nodata

    @property
    def max_features(self):
        """
        :return: Maximum number of features to display.
        """
        return self._max_features

    @max_features.setter
    def max_features(self, value):
        """
        Set maximum number of features to display.
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
        self._show_last_update = bool(value)

    @property
    def labels(self):
        """
        :return: Show labels or not.
        """
        return self._labels

    @labels.setter
    def labels(self, value):
        """
        Set show labels or not.
        """
        self._labels = bool(value)

    def _color_picker(self):
        import random

        random_number = random.randint(0, 16777215)
        hex_number = str(hex(random_number))
        hex_number = "#" + hex_number[2:]

        return hex_number

    def _convert_to_json(self):
        self._fields_slices = []
        self._statistic_fields = []

        if self.data.categories_from in ["groupByValues", "features"]:
            self.data._get_slices()

        if self._data.categories_from == "fields":
            self._category = None
            self._field_name = "category"
            for category_field in self._data.category_field:
                self._fields_slices.append(
                    {
                        "key": str(category_field),
                        "label": str(category_field),
                        "color": self._color_picker(),
                    }
                )
                self._statistic_fields.append(
                    {
                        "onStatisticField": str(category_field),
                        "outStatisticFieldName": str(category_field),
                        "statisticType": self.data.statistic,
                    }
                )
        elif self._data.categories_from == "groupByValues":
            for slices in self.data._slice_fields:
                self._fields_slices.append(
                    {
                        "key": str(slices),
                        "label": str(slices),
                        "color": self._color_picker(),
                    }
                )
            self._statistic_fields.append(
                {
                    "onStatisticField": str(self.data.statistics_field),
                    "outStatisticFieldName": "value",
                    "statisticType": self.data.statistic,
                }
            )
            self._category = self._data.category_field
        else:
            for slices in self.data._slice_fields:
                self._fields_slices.append(
                    {
                        "key": str(slices),
                        "label": str(slices),
                        "color": self._color_picker(),
                    }
                )
            self._category = self._data.category_field

        if self.labels == True:
            self._labels_format = "value"
        else:
            self._labels_format = "hide"

        if self.item.type == "mapWidget":
            wlayer = self.item.layers[self.layer]
            widget_id = self.item._id
            layer_id = wlayer["id"]
            self._datasource = {"id": str(widget_id) + "#" + str(layer_id)}
            self._targetid = self._datasource
        else:
            self._datasource = {
                "type": "featureServiceDataSource",
                "itemId": self.item.itemid,
                "layerId": 0,
                "table": True,
            }

        json_data = {
            "type": "pieChartWidget",
            "category": {
                "sliceProperties": self._fields_slices,
                "fieldName": self._category if self._category else self._field_name,
                "nullLabel": self._slices.null_label,
                "blankLabel": self._slices.blank_label,
                "defaultColor": self._slices.default_color,
                "nullColor": self._slices.null_color,
                "blankColor": self._slices.blank_color,
            },
            "pie": {
                "type": "pie",
                "color": self._pie.text_color,
                "fontSize": self._pie.font_size,
                "titleField": "category",
                "valueField": "absoluteValue",
                "alpha": self._slices.opacity,
                "outlineAlpha": self._outline.opacity,
                "outlineColor": self._outline.color,
                "outlineThickness": self._outline.thickness,
                "innerRadius": self._pie.inner_radius,
                "labelsEnabled": self.labels,
                "labelsFormat": self._labels_format,
                "labelTickAlpha": 0.5,
                "labelTickColor": "#fab123",
                "maxLabelWidth": 100,
                "startAngle": self._pie.start_angle,
                "autoMargins": False,
                "marginTop": 0,
                "marginBottom": 0,
                "marginLeft": 0,
                "marginRight": 0,
                "groupPercent": self._slices.grouping_percent,
                "groupedColor": self._slices.grouping_color,
            },
            # "valueField":self._data.value_field,
            "legend": {
                "enabled": self._legend.visibility,
                "format": "value",
                "position": self._legend.placement,
                "markerSize": 15,
                "markerType": "circle",
                "align": "center",
                "labelWidth": 100,
                "valueWidth": 50,
            },
            "showBalloon": True,
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
                "pattern": "#.##",
            },
            "events": [],
            "selectionMode": "single",
            "categoryType": self._data.categories_from,
            "datasets": [
                {
                    "type": "serviceDataset",
                    "dataSource": self._datasource,
                    "outFields": ["*"],
                    "groupByFields": [self._category] if self._category else [],
                    "orderByFields": [],
                    "statisticDefinitions": self._statistic_fields,
                    "maxFeatures": self.max_features,
                    "querySpatialRelationship": "esriSpatialRelIntersects",
                    "returnGeometry": False,
                    "clientSideStatistics": False,
                    "name": "main",
                }
            ],
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

        if self.events.enable:
            json_data["events"].append(
                {"type": self.events.type, "actions": self.events.synced_widgets}
            )
            json_data["selectionMode"] = self.events.selection_mode

        if self.background_color:
            json_data["backgroundColor"] = (self.background_color,)

        if self.text_color:
            json_data["textColor"] = self.text_color

        if self.events.enable:
            json_data["events"].append(
                {"type": self.events.type, "actions": self.events.synced_widgets}
            )
            json_data["selectionMode"] = self.events.selection_mode

        if hasattr(self._data, "_value_field"):
            json_data["valueField"] = self._data._value_field

        return json_data


class PieChartData(object):
    @classmethod
    def _create_data(cls, categories_from, data_item=None):
        piechart_data = PieChartData()

        piechart_data._categories_from = "groupByValues"
        piechart_data._item = data_item
        piechart_data._category_field = None
        piechart_data._objectid_field = data_item.tables[0].properties["objectIdField"]

        if categories_from != "fields":
            piechart_data._value_field = piechart_data._objectid_field
        piechart_data._statistics_field = piechart_data._objectid_field
        piechart_data._parse_dates = True
        piechart_data._statistic = "count"
        piechart_data._filters = []

        piechart_data._slice_fields = []

        piechart_data._categories_from = categories_from

        return piechart_data

    @property
    def categories_from(self):
        """
        :return: Categories from groupByValues, features or fields.
        """
        return self._categories_from

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
    def objectid_field(self):
        """
        :return: object ID field name.
        """
        return self._objectid_field

    @property
    def statistics_field(self):
        """
        :return: Statistics field name.
        """
        return self._statistics_field

    @property
    def category_field(self):
        """
        :return: Category field from dataset. For groupByValues or features.
        """
        return self._category_field

    @category_field.setter
    def category_field(self, value):
        """
        Set category field from dataset. For groupByValues or features pass a single value. For fields pass a list
        """
        self._category_field = value

    @property
    def value_field(self):
        """
        :return: Value field from dataset. For features only.
        """
        return self._value_field

    @value_field.setter
    def value_field(self, value):
        """
        Set value field from dataset. For features and groupByValues only.
        """
        if self._categories_from in ["fields"]:
            raise Exception(
                "Can set this attribute for 'features' or 'groupByValues' category only."
            )

        self._value_field = value

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
        if self._categories_from in ["features"]:
            raise Exception("Can't set this attribute for 'features' category.")

        if value in ["count", "avg", "min", "max", "stddev", "sum"]:
            self._statistic = value

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

    def _get_slices(self):
        statistics = self.statistic
        c_field = self.category_field
        v_field = self.value_field

        x = self._item.tables[0].query(
            group_by_fields_for_statistics=c_field,
            out_statistics=[
                {
                    "statisticType": statistics,
                    "onStatisticField": v_field,
                    "outStatisticFieldName": "value",
                }
            ],
            as_df=True,
        )
        self._slice_fields = x[c_field]


class PieChartProperties(object):
    @classmethod
    def _create_chart(
        cls,
        text_color="#000000",
        font_size=12,
        start_angle=90,
        inner_radius=0,
        hover_text=True,
    ):
        chart = PieChartProperties()

        chart._text_color = text_color
        chart._font_size = font_size
        chart._start_angle = start_angle
        chart._inner_radius = inner_radius
        chart._hover_text = hover_text

        chart.text_color = text_color
        chart.font_size = font_size
        chart.start_angle = start_angle
        chart.inner_radius = inner_radius
        chart.hover_text = hover_text

        return chart

    @property
    def text_color(self):
        """
        :return: Text Color of Pie Chart.
        """
        return self._text_color

    @text_color.setter
    def text_color(self, value):
        """
        Set Text Color of Pie Chart.
        """
        self._text_color = str(value)

    @property
    def font_size(self):
        """
        :return: font size of Pie Chart.
        """
        return self._font_size

    @font_size.setter
    def font_size(self, value):
        """
        Set font size value.
        """
        self._font_size = value

    @property
    def start_angle(self):
        """
        :return: Start Angle of Pie Chart.
        """
        return self._start_angle

    @start_angle.setter
    def start_angle(self, value):
        """
        Set Start Angle value.
        """
        self._start_angle = value

    @property
    def inner_radius(self):
        """
        :return: Inner Radius of Pie Chart.
        """
        return self._inner_radius

    @inner_radius.setter
    def inner_radius(self, value):
        """
        Set Inner Radius value.
        """
        self._inner_radius = value

    @property
    def hover_text(self):
        """
        :return: Show text on hovering or not.
        """
        return self._hover_text

    @hover_text.setter
    def hover_text(self, value):
        """
        Set Show text on hovering or not.
        """
        self._hover_text = value


class SliceProperties(object):
    @classmethod
    def _slice_properties(
        cls,
        opacity=1.0,
        default_color="#ffffff",
        null_color="#ffffff",
        blank_color="#000000",
        grouping_color="#000000",
        grouping_percent=0,
        null_label="Null",
        blank_label="Blank",
    ):
        slices = SliceProperties()

        slices._opacity = 1.0
        slices._default_color = "#ffffff"

        slices._null_color = "#ffffff"
        slices._null_label = "Null"

        slices._blank_color = "#000000"
        slices._blank_label = "Blank"

        slices._grouping_percent = 0
        slices._grouping_color = "#000000"

        slices.opacity = opacity
        slices.default_color = default_color

        slices.null_color = null_color
        slices.null_label = null_label

        slices.blank_color = blank_color
        slices.blank_label = blank_label

        slices.grouping_percent = grouping_percent
        slices.grouping_color = grouping_color

        return slices

    @property
    def opacity(self):
        """
        :return: slices opacity value.
        """
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        """
        Set slices opactiy value.
        """
        self._opacity = value

    @property
    def default_color(self):
        """
        :return: default color of slices.
        """
        return self._default_color

    @default_color.setter
    def default_color(self, value):
        """
        Set default color of slices.
        """
        self._default_color = str(value)

    @property
    def null_color(self):
        """
        :return: color of null slices.
        """
        return self._null_color

    @null_color.setter
    def null_color(self, value):
        """
        Set color of null slices.
        """
        self._null_color = str(value)

    @property
    def blank_color(self):
        """
        :return: color of blank slices.
        """
        return self._blank_color

    @blank_color.setter
    def blank_color(self, value):
        """
        Set color of blank slices.
        """
        self._blank_color = str(value)

    @property
    def grouping_color(self):
        """
        :return: grouping color.
        """
        return self._grouping_color

    @grouping_color.setter
    def grouping_color(self, value):
        """
        Set grouping color.
        """
        self._grouping_color = str(value)

    @property
    def null_label(self):
        """
        :return: label of null slices.
        """
        return self._null_label

    @null_label.setter
    def null_label(self, value):
        """
        Set label of null slices.
        """
        self._null_label = str(value)

    @property
    def blank_label(self):
        """
        :return: label of blank slices.
        """
        return self._blank_label

    @blank_label.setter
    def blank_label(self, value):
        """
        Set label of blank slices.
        """
        self._blank_label = str(value)

    @property
    def grouping_percent(self):
        """
        :return: grouping percentage.
        """
        return self._grouping_percent

    @grouping_percent.setter
    def grouping_percent(self, value):
        """
        Set grouping percentage.
        """
        self._grouping_percent = value


class OutlineProperties(object):
    @classmethod
    def _outline_init(cls, opacity=0.4, thickness=0.2, color="#000000"):
        outline = OutlineProperties()

        outline._opacity = opacity
        outline._thickness = thickness
        outline._color = color

        outline.opacity = opacity
        outline.thickness = thickness
        outline.color = color

        return outline

    @property
    def opacity(self):
        """
        :return: outline opacity.
        """
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        """
        Set outline opacity.
        """
        self._opacity = value

    @property
    def thickness(self):
        """
        :return: outline thickness.
        """
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        """
        Set outline thickness.
        """
        self._thickness = value

    @property
    def color(self):
        """
        :return: outline color.
        """
        return self._color

    @color.setter
    def color(self, value):
        """
        Set outline color.
        """
        self._color = str(value)


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
        Synchronize a mapWidget with PieChart for triggered events.

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
        Synchronize non-mapWidget type widgets with PieChart for triggered events.

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

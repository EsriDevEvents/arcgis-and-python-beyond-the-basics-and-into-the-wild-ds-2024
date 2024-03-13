from . import *
from dataclasses import dataclass, asdict, field
from typing import Dict, Union, Optional, Any, ClassVar, TypeVar


# common keys used in the format configuration
_FORMAT_NAME_KEY = "formatName"


@dataclass
class _FormatBase(object):
    # abstract variables that need to be implemented by Derived Classes
    name: ClassVar[str]

    # methods to be implemented by every Format
    def _build(self) -> dict:
        """
        abstract method that needs to be implemented by derived class.

        :return: Configuration properties in a dictionary that will be used to make the Rest call to the backend application
        """
        raise NotImplementedError

    @classmethod
    def _from_config(cls, config: dict) -> object:
        raise NotImplementedError


@dataclass
class GeoRssFormat(_FormatBase):
    T = TypeVar("GeoRssFormat")
    name: ClassVar[str] = "geo-rss-format"

    # format keys
    _DATE_FORMAT_KEY: str = field(init=False, default=f"{name}.dateFormat")

    # this format's instance properties
    date_format: Optional[str] = None

    def _build(self) -> dict:
        result_dict = {_FORMAT_NAME_KEY: GeoRssFormat.name}
        if self.date_format is not None:
            result_dict |= {
                "properties": {
                    f"{GeoRssFormat.name}.{GeoRssFormat._DATE_FORMAT_KEY}": self.date_format
                }
            }

        return result_dict

    @classmethod
    def _from_config(cls, config: dict) -> T:
        properties = config["properties"]
        obj = cls(date_format=properties.get(GeoRssFormat._DATE_FORMAT_KEY))
        return obj


@dataclass
class RssFormat(_FormatBase):
    T = TypeVar("RssFormat")
    name: ClassVar[str] = "rss-format"

    # format keys
    _BUILD_GEOMETRY_FROM_FIELDS_KEY: str = field(
        init=False, default=f"{name}.buildGeometryFromFields"
    )
    _X_FIELD_KEY: str = field(init=False, default=f"{name}.xField")
    _Y_FIELD_KEY: str = field(init=False, default=f"{name}.yField")
    _HAS_Z_FIELD_KEY_KEY: str = field(init=False, default=f"{name}.hasZField")
    _Z_FIELD_KEY: str = field(init=False, default=f"{name}.zField")
    _Z_UNIT_KEY: str = field(init=False, default=f"{name}.zUnit")
    _GEOMETRY_FIELD_KEY: str = field(init=False, default=f"{name}.geometryField")
    _GEOMETRY_FIELD_FORMAT_KEY: str = field(
        init=False, default=f"{name}.geometryFieldFormat"
    )

    _DATE_FORMAT_KEY: str = field(init=False, default=f"{name}.dateFormat")

    # this format's instance properties
    # date properties
    date_format: Optional[str] = None
    # geometry properties
    build_geometry_from_fields: Optional[bool] = None
    x_field: Optional[str] = None
    y_field: Optional[str] = None
    has_z_field: Optional[bool] = None
    z_field: Optional[str] = None
    z_unit: Optional[int] = None
    geometry_field: Optional[str] = None
    geometry_field_format: Optional[str] = None

    def _build(self) -> dict:
        result_dict = {_FORMAT_NAME_KEY: self.name}
        properties_dict = {}

        if self.build_geometry_from_fields is not None:
            properties_dict[
                RssFormat._BUILD_GEOMETRY_FROM_FIELDS_KEY
            ] = self.build_geometry_from_fields
        if self.x_field is not None:
            properties_dict[RssFormat._X_FIELD_KEY] = self.x_field
        if self.y_field is not None:
            properties_dict[RssFormat._Y_FIELD_KEY] = self.y_field
        if self.has_z_field is not None:
            properties_dict[RssFormat._HAS_Z_FIELD_KEY_KEY] = self.has_z_field
        if self.z_field is not None:
            properties_dict[RssFormat._Z_FIELD_KEY] = self.z_field
        if self.z_unit is not None:
            properties_dict[RssFormat._Z_UNIT_KEY] = self.z_unit
        if self.geometry_field is not None:
            properties_dict[RssFormat._GEOMETRY_FIELD_KEY] = self.geometry_field
        if self.geometry_field_format is not None:
            properties_dict[
                RssFormat._GEOMETRY_FIELD_FORMAT_KEY
            ] = self.geometry_field_format
        if self.date_format is not None:
            properties_dict[RssFormat._DATE_FORMAT_KEY] = self.date_format

        if bool(properties_dict):
            result_dict["properties"] = properties_dict

        return result_dict

    @classmethod
    def _from_config(cls, config: dict) -> T:
        properties = config["properties"]
        obj = RssFormat(
            build_geometry_from_fields=properties.get(
                RssFormat._BUILD_GEOMETRY_FROM_FIELDS_KEY
            ),
            x_field=properties.get(RssFormat._X_FIELD_KEY),
            y_field=properties.get(RssFormat._Y_FIELD_KEY),
            has_z_field=properties.get(RssFormat._HAS_Z_FIELD_KEY_KEY),
            z_field=properties.get(RssFormat._Z_FIELD_KEY),
            z_unit=properties.get(RssFormat._Z_UNIT_KEY),
            geometry_field=properties.get(RssFormat._GEOMETRY_FIELD_KEY),
            geometry_field_format=properties.get(RssFormat._GEOMETRY_FIELD_FORMAT_KEY),
            date_format=properties.get(RssFormat._DATE_FORMAT_KEY),
        )

        return obj


@dataclass
class DelimitedFormat(_FormatBase):
    T = TypeVar("DelimitedFormat")
    name: ClassVar[str] = "delimited"

    # format keys
    _BUILD_GEOMETRY_FROM_FIELDS_KEY: str = field(
        init=False, default=f"{name}.buildGeometryFromFields"
    )
    _X_FIELD_KEY: str = field(init=False, default=f"{name}.xField")
    _Y_FIELD_KEY: str = field(init=False, default=f"{name}.yField")
    _HAS_Z_FIELD_KEY_KEY: str = field(init=False, default=f"{name}.hasZField")
    _Z_FIELD_KEY: str = field(init=False, default=f"{name}.zField")
    _Z_UNIT_KEY: str = field(init=False, default=f"{name}.zUnit")
    _GEOMETRY_FIELD_KEY: str = field(init=False, default=f"{name}.geometryField")
    _GEOMETRY_FIELD_FORMAT_KEY: str = field(
        init=False, default=f"{name}.geometryFieldFormat"
    )

    _DATE_FORMAT_KEY: str = field(init=False, default=f"{name}.dateFormat")

    _RECORD_TERMINATOR_KEY: str = field(init=False, default=f"{name}.recordTerminator")
    _FIELD_DELIMITER_KEY: str = field(init=False, default=f"{name}.fieldDelimiter")
    _HAS_HEADER_ROW: str = field(init=False, default=f"{name}.hasHeaderRow")

    # this format's instance properties
    record_terminator: Optional[str] = None
    field_delimiter: Optional[str] = None
    has_header_row: Optional[bool] = None
    # date properties
    date_format: Optional[str] = None
    # geometry properties
    build_geometry_from_fields: Optional[bool] = None
    x_field: Optional[str] = None
    y_field: Optional[str] = None
    has_z_field: Optional[bool] = None
    z_field: Optional[str] = None
    z_unit: Optional[int] = None
    geometry_field: Optional[str] = None
    geometry_field_format: Optional[str] = None

    def _build(self) -> dict:
        result_dict = {_FORMAT_NAME_KEY: self.name}
        properties_dict = {}

        if self.build_geometry_from_fields is not None:
            properties_dict[
                DelimitedFormat._BUILD_GEOMETRY_FROM_FIELDS_KEY
            ] = self.build_geometry_from_fields
        if self.x_field is not None:
            properties_dict[DelimitedFormat._X_FIELD_KEY] = self.x_field
        if self.y_field is not None:
            properties_dict[DelimitedFormat._Y_FIELD_KEY] = self.y_field
        if self.has_z_field is not None:
            properties_dict[DelimitedFormat._HAS_Z_FIELD_KEY_KEY] = self.has_z_field
        if self.z_field is not None:
            properties_dict[DelimitedFormat._Z_FIELD_KEY] = self.z_field
        if self.z_unit is not None:
            properties_dict[DelimitedFormat._Z_UNIT_KEY] = self.z_unit
        if self.geometry_field is not None:
            properties_dict[DelimitedFormat._GEOMETRY_FIELD_KEY] = self.geometry_field
        if self.geometry_field_format is not None:
            properties_dict[
                DelimitedFormat._GEOMETRY_FIELD_FORMAT_KEY
            ] = self.geometry_field_format
        if self.date_format is not None:
            properties_dict[DelimitedFormat._DATE_FORMAT_KEY] = self.date_format

        if self.record_terminator is not None:
            properties_dict[
                DelimitedFormat._RECORD_TERMINATOR_KEY
            ] = self.record_terminator
        if self.field_delimiter is not None:
            properties_dict[DelimitedFormat._FIELD_DELIMITER_KEY] = self.field_delimiter
        if self.has_header_row is not None:
            properties_dict[DelimitedFormat._HAS_HEADER_ROW] = self.has_header_row

        if bool(properties_dict):
            result_dict["properties"] = properties_dict

        return result_dict

    @classmethod
    def _from_config(cls, config: dict) -> T:
        properties = config["properties"]
        obj = DelimitedFormat(
            build_geometry_from_fields=properties.get(
                DelimitedFormat._BUILD_GEOMETRY_FROM_FIELDS_KEY
            ),
            x_field=properties.get(DelimitedFormat._X_FIELD_KEY),
            y_field=properties.get(DelimitedFormat._Y_FIELD_KEY),
            has_z_field=properties.get(DelimitedFormat._HAS_Z_FIELD_KEY_KEY),
            z_field=properties.get(DelimitedFormat._Z_FIELD_KEY),
            z_unit=properties.get(DelimitedFormat._Z_UNIT_KEY),
            geometry_field=properties.get(DelimitedFormat._GEOMETRY_FIELD_KEY),
            geometry_field_format=properties.get(
                DelimitedFormat._GEOMETRY_FIELD_FORMAT_KEY
            ),
            date_format=properties.get(DelimitedFormat._DATE_FORMAT_KEY),
            record_terminator=properties.get(DelimitedFormat._RECORD_TERMINATOR_KEY),
            field_delimiter=properties.get(DelimitedFormat._FIELD_DELIMITER_KEY),
            has_header_row=properties.get(DelimitedFormat._HAS_HEADER_ROW),
        )

        return obj


@dataclass
class EsriJsonFormat(_FormatBase):
    T = TypeVar("EsriJsonFormat")
    name: ClassVar[str] = "esri-json"

    # format keys
    _DATE_FORMAT_KEY: str = field(init=False, default=f"{name}.dateFormat")

    # this format's instance properties
    date_format: Optional[str] = None

    def _build(self) -> dict:
        result_dict = {_FORMAT_NAME_KEY: self.name}
        properties_dict = {}

        if self.date_format is not None:
            properties_dict[EsriJsonFormat._DATE_FORMAT_KEY] = self.date_format

        if bool(properties_dict):
            result_dict["properties"] = properties_dict

        return result_dict

    @classmethod
    def _from_config(cls, config: dict) -> T:
        properties = config["properties"]
        obj = EsriJsonFormat(date_format=EsriJsonFormat._DATE_FORMAT_KEY)

        return obj


@dataclass
class GeoJsonFormat(_FormatBase):
    T = TypeVar("GeoJsonFormat")
    name: ClassVar[str] = "geo-json"

    # format keys
    _DATE_FORMAT_KEY: str = field(init=False, default=f"{name}.dateFormat")

    # this format's instance properties
    date_format: Optional[str] = None

    def _build(self) -> dict:
        result_dict = {_FORMAT_NAME_KEY: self.name}
        properties_dict = {}

        if self.date_format is not None:
            properties_dict[GeoJsonFormat._DATE_FORMAT_KEY] = self.date_format

        if bool(properties_dict):
            result_dict["properties"] = properties_dict

        return result_dict

    @classmethod
    def _from_config(cls, config: dict) -> T:
        properties = config["properties"]
        obj = GeoJsonFormat(date_format=GeoJsonFormat._DATE_FORMAT_KEY)

        return obj


@dataclass
class JsonFormat(_FormatBase):
    T = TypeVar("JsonFormat")
    name: ClassVar[str] = "json"

    # format keys
    _BUILD_GEOMETRY_FROM_FIELDS_KEY: str = field(
        init=False, default=f"{name}.buildGeometryFromFields"
    )
    _X_FIELD_KEY: str = field(init=False, default=f"{name}.xField")
    _Y_FIELD_KEY: str = field(init=False, default=f"{name}.yField")
    _HAS_Z_FIELD_KEY_KEY: str = field(init=False, default=f"{name}.hasZField")
    _Z_FIELD_KEY: str = field(init=False, default=f"{name}.zField")
    _Z_UNIT_KEY: str = field(init=False, default=f"{name}.zUnit")
    _GEOMETRY_FIELD_KEY: str = field(init=False, default=f"{name}.geometryField")
    _GEOMETRY_FIELD_FORMAT_KEY: str = field(
        init=False, default=f"{name}.geometryFieldFormat"
    )

    _DATE_FORMAT_KEY: str = field(init=False, default=f"{name}.dateFormat")

    _ROOT_NODE_KEY: str = field(init=False, default=f"{name}.rootNode")
    _FLATTEN_KEY: str = field(init=False, default=f"{name}.flatten")
    _FIELD_FLATTENING_EXEMPTIONS_KEY: str = field(
        init=False, default=f"{name}.fieldFlatteningExemptions"
    )

    # this format's instance properties
    root_node: Optional[str] = None
    flatten: Optional[bool] = None
    field_flattening_exemptions: Optional[str] = None
    # date properties
    date_format: Optional[str] = None
    # geometry properties
    build_geometry_from_fields: Optional[bool] = None
    x_field: Optional[str] = None
    y_field: Optional[str] = None
    has_z_field: Optional[bool] = None
    z_field: Optional[str] = None
    z_unit: Optional[int] = None
    geometry_field: Optional[str] = None
    geometry_field_format: Optional[str] = None

    def _build(self) -> dict:
        result_dict = {_FORMAT_NAME_KEY: self.name}
        properties_dict = {}

        if self.build_geometry_from_fields is not None:
            properties_dict[
                JsonFormat._BUILD_GEOMETRY_FROM_FIELDS_KEY
            ] = self.build_geometry_from_fields
        if self.x_field is not None:
            properties_dict[JsonFormat._X_FIELD_KEY] = self.x_field
        if self.y_field is not None:
            properties_dict[JsonFormat._Y_FIELD_KEY] = self.y_field
        if self.has_z_field is not None:
            properties_dict[JsonFormat._HAS_Z_FIELD_KEY_KEY] = self.has_z_field
        if self.z_field is not None:
            properties_dict[JsonFormat._Z_FIELD_KEY] = self.z_field
        if self.z_unit is not None:
            properties_dict[JsonFormat._Z_UNIT_KEY] = self.z_unit
        if self.geometry_field is not None:
            properties_dict[JsonFormat._GEOMETRY_FIELD_KEY] = self.geometry_field
        if self.geometry_field_format is not None:
            properties_dict[
                JsonFormat._GEOMETRY_FIELD_FORMAT_KEY
            ] = self.geometry_field_format
        if self.date_format is not None:
            properties_dict[JsonFormat._DATE_FORMAT_KEY] = self.date_format
        if self.root_node is not None:
            properties_dict[JsonFormat._ROOT_NODE_KEY] = self.root_node
        if self.flatten is not None:
            properties_dict[JsonFormat._FLATTEN_KEY] = self.flatten
        if self.field_flattening_exemptions is not None:
            properties_dict[
                JsonFormat._FIELD_FLATTENING_EXEMPTIONS_KEY
            ] = self.field_flattening_exemptions

        if bool(properties_dict):
            result_dict["properties"] = properties_dict

        return result_dict

    @classmethod
    def _from_config(cls, config: dict) -> T:
        properties = config["properties"]
        obj = JsonFormat(
            build_geometry_from_fields=properties.get(
                JsonFormat._BUILD_GEOMETRY_FROM_FIELDS_KEY
            ),
            x_field=properties.get(JsonFormat._X_FIELD_KEY),
            y_field=properties.get(JsonFormat._Y_FIELD_KEY),
            has_z_field=properties.get(JsonFormat._HAS_Z_FIELD_KEY_KEY),
            z_field=properties.get(JsonFormat._Z_FIELD_KEY),
            z_unit=properties.get(JsonFormat._Z_UNIT_KEY),
            geometry_field=properties.get(JsonFormat._GEOMETRY_FIELD_KEY),
            geometry_field_format=properties.get(JsonFormat._GEOMETRY_FIELD_FORMAT_KEY),
            date_format=properties.get(JsonFormat._DATE_FORMAT_KEY),
            root_node=properties.get(JsonFormat._ROOT_NODE_KEY),
            flatten=properties.get(JsonFormat._FLATTEN_KEY),
            field_flattening_exemptions=properties.get(
                JsonFormat._FIELD_FLATTENING_EXEMPTIONS_KEY
            ),
        )

        return obj


@dataclass
class XMLFormat(_FormatBase):
    T = TypeVar("XMLFormat")
    name: ClassVar[str] = "xml"

    # format keys
    _BUILD_GEOMETRY_FROM_FIELDS_KEY: str = field(
        init=False, default=f"{name}.buildGeometryFromFields"
    )
    _X_FIELD_KEY: str = field(init=False, default=f"{name}.xField")
    _Y_FIELD_KEY: str = field(init=False, default=f"{name}.yField")
    _HAS_Z_FIELD_KEY_KEY: str = field(init=False, default=f"{name}.hasZField")
    _Z_FIELD_KEY: str = field(init=False, default=f"{name}.zField")
    _Z_UNIT_KEY: str = field(init=False, default=f"{name}.zUnit")
    _GEOMETRY_FIELD_KEY: str = field(init=False, default=f"{name}.geometryField")
    _GEOMETRY_FIELD_FORMAT_KEY: str = field(
        init=False, default=f"{name}.geometryFieldFormat"
    )

    _DATE_FORMAT_KEY: str = field(init=False, default=f"{name}.dateFormat")

    _ROOT_NODE_KEY: str = field(init=False, default=f"{name}.rootNode")
    _FLATTEN_KEY: str = field(init=False, default=f"{name}.flatten")
    _FIELD_FLATTENING_EXEMPTIONS_KEY: str = field(
        init=False, default=f"{name}.fieldFlatteningExemptions"
    )

    # this format's instance properties
    root_node: Optional[str] = None
    flatten: Optional[bool] = None
    field_flattening_exemptions: Optional[str] = None
    # date properties
    date_format: Optional[str] = None
    # geometry properties
    build_geometry_from_fields: Optional[bool] = None
    x_field: Optional[str] = None
    y_field: Optional[str] = None
    has_z_field: Optional[bool] = None
    z_field: Optional[str] = None
    z_unit: Optional[int] = None
    geometry_field: Optional[str] = None
    geometry_field_format: Optional[str] = None

    def _build(self) -> dict:
        result_dict = {_FORMAT_NAME_KEY: self.name}
        properties_dict = {}

        if self.build_geometry_from_fields is not None:
            properties_dict[
                XMLFormat._BUILD_GEOMETRY_FROM_FIELDS_KEY
            ] = self.build_geometry_from_fields
        if self.x_field is not None:
            properties_dict[XMLFormat._X_FIELD_KEY] = self.x_field
        if self.y_field is not None:
            properties_dict[XMLFormat._Y_FIELD_KEY] = self.y_field
        if self.has_z_field is not None:
            properties_dict[XMLFormat._HAS_Z_FIELD_KEY_KEY] = self.has_z_field
        if self.z_field is not None:
            properties_dict[XMLFormat._Z_FIELD_KEY] = self.z_field
        if self.z_unit is not None:
            properties_dict[XMLFormat._Z_UNIT_KEY] = self.z_unit
        if self.geometry_field is not None:
            properties_dict[XMLFormat._GEOMETRY_FIELD_KEY] = self.geometry_field
        if self.geometry_field_format is not None:
            properties_dict[
                XMLFormat._GEOMETRY_FIELD_FORMAT_KEY
            ] = self.geometry_field_format
        if self.date_format is not None:
            properties_dict[XMLFormat._DATE_FORMAT_KEY] = self.date_format
        if self.root_node is not None:
            properties_dict[XMLFormat._ROOT_NODE_KEY] = self.root_node
        if self.flatten is not None:
            properties_dict[XMLFormat._FLATTEN_KEY] = self.flatten
        if self.field_flattening_exemptions is not None:
            properties_dict[
                XMLFormat._FIELD_FLATTENING_EXEMPTIONS_KEY
            ] = self.field_flattening_exemptions

        if bool(properties_dict):
            result_dict["properties"] = properties_dict

        return result_dict

    @classmethod
    def _from_config(cls, config: dict) -> T:
        properties = config["properties"]
        obj = XMLFormat(
            build_geometry_from_fields=properties.get(
                XMLFormat._BUILD_GEOMETRY_FROM_FIELDS_KEY
            ),
            x_field=properties.get(XMLFormat._X_FIELD_KEY),
            y_field=properties.get(XMLFormat._Y_FIELD_KEY),
            has_z_field=properties.get(XMLFormat._HAS_Z_FIELD_KEY_KEY),
            z_field=properties.get(XMLFormat._Z_FIELD_KEY),
            z_unit=properties.get(XMLFormat._Z_UNIT_KEY),
            geometry_field=properties.get(XMLFormat._GEOMETRY_FIELD_KEY),
            geometry_field_format=properties.get(XMLFormat._GEOMETRY_FIELD_FORMAT_KEY),
            date_format=properties.get(XMLFormat._DATE_FORMAT_KEY),
            root_node=properties.get(XMLFormat._ROOT_NODE_KEY),
            flatten=properties.get(XMLFormat._FLATTEN_KEY),
            field_flattening_exemptions=properties.get(
                XMLFormat._FIELD_FLATTENING_EXEMPTIONS_KEY
            ),
        )

        return obj


def _format_from_config(
    config: dict,
) -> Optional[
    Union[
        GeoRssFormat,
        RssFormat,
        DelimitedFormat,
        EsriJsonFormat,
        GeoJsonFormat,
        JsonFormat,
        XMLFormat,
    ]
]:
    """
    Identifies and instantiates a format object from a feed configuration json dict

    ==================     ====================================================================
    **Parameter**           **Description**
    ------------------     --------------------------------------------------------------------
    config                 dict. The feed configuration json/dict.
    ==================     ====================================================================
    """
    if _FORMAT_NAME_KEY in config:
        if config[_FORMAT_NAME_KEY] == GeoRssFormat.name:
            return GeoRssFormat._from_config(config)
        elif config[_FORMAT_NAME_KEY] == RssFormat.name:
            return RssFormat._from_config(config)
        elif config[_FORMAT_NAME_KEY] == DelimitedFormat.name:
            return DelimitedFormat._from_config(config)
        elif config[_FORMAT_NAME_KEY] == EsriJsonFormat.name:
            return EsriJsonFormat._from_config(config)
        elif config[_FORMAT_NAME_KEY] == GeoJsonFormat.name:
            return GeoJsonFormat._from_config(config)
        elif config[_FORMAT_NAME_KEY] == JsonFormat.name:
            return JsonFormat._from_config(config)
        elif config[_FORMAT_NAME_KEY] == XMLFormat.name:
            return XMLFormat._from_config(config)
        else:
            return None
    else:
        return None

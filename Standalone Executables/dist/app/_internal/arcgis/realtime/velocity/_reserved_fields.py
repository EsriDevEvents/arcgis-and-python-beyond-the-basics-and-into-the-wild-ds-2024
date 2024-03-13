from typing import ClassVar, List


class _ReservedFields:
    """
    Field names that are reserved for internal use by Velocity. A user defined field name should not be one of these.
    """

    _reserved_names_gdb: ClassVar[List[str]] = [
        "ENTITY",
        "NUMPTS",
        "MINX",
        "MINY",
        "MAXX",
        "MAXY",
        "MINZ",
        "MAXZ",
        "MINM",
        "MAXM",
        "AREA",
        "LEN",
        "SRID",
        "POINTS",
        "Shape_Area",
        "DB2GSE.ST_Area(Shape)",
        "SE_Area(Shape)",
        "Shape.AREA",
        "st_area(Shape)",
        "Shape.area",
        "Shape.STArea()",
        "Shape_Length",
        "DB2GSE.SdeLength(Shape)",
        "SE_Length(Shape)",
        "Shape.LEN",
        "st_length(Shape)",
        "Shape.len",
        "Shape.STLength()",
        "Shape",
    ]
    _reserved_names_fs: ClassVar[List[str]] = [
        "fid",
        "area",
        "len",
        "ponts",
        "numofpts",
        "entity",
        "eminx",
        "eminy",
        "emaxx",
        "emaxy",
        "eminz",
        "emaxz",
        "min_measure",
        "max_measure",
    ]
    _reserved_names_service_query: ClassVar[List[str]] = [
        "shape__area",
        "shape__length",
    ]
    _reserved_names_es: ClassVar[List[str]] = ["_id"]
    _all_reserved_names_lower: ClassVar[List[str]] = list(
        map(
            str.lower,
            (
                _reserved_names_fs
                + _reserved_names_gdb
                + _reserved_names_service_query
                + _reserved_names_es
            ),
        )
    )

    @staticmethod
    def is_reserved(field_name: str) -> bool:
        """
        Checks if a given field name is a reserved name. This is a case-insensitive comparison.

        ==============     =====================================
        **Parameter**       **Description**
        --------------     -------------------------------------
        field_name         String. A field name to test
        ==============     =====================================

        :return: True if field_name is a reserved name
        """
        return field_name.lower() in _ReservedFields._all_reserved_names_lower

    @staticmethod
    def is_not_reserved(field_name: str) -> bool:
        """
        Checks if a given field name is not a reserved name. This is a case-insensitive comparison.

        ==============     =====================================
        **Parameter**       **Description**
        --------------     -------------------------------------
        field_name         String. A field name to test
        ==============     =====================================

        :return: True if field_name is not a reserved name
        """
        return not _ReservedFields.is_reserved(field_name)

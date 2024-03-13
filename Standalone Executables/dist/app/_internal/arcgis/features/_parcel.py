from __future__ import annotations
import time
from typing import Any
from arcgis.geometry import Polygon, Envelope
from arcgis._impl.common._mixins import PropertyMap
from arcgis.features import FeatureLayer

########################################################################


class ParcelFabricManager(object):
    """
    The Parcel Fabric Server is responsible for exposing parcel management
    capabilities to support a variety of workflows from different clients
    and systems.

    ====================     ====================================================================
    **Parameter**             **Description**
    --------------------     --------------------------------------------------------------------
    url                      Required String. The URI to the service endpoint.
    --------------------     --------------------------------------------------------------------
    gis                      Required :class:`~arcgis.gis.GIS`. The enterprise connection.
    --------------------     --------------------------------------------------------------------
    version                  Required :class:`~arcgis.features._version.Version`. This is the version object where the modification
                             will occur.
    --------------------     --------------------------------------------------------------------
    flc                      Required :class:`~arcgis.features.FeatureLayerCollection` . This is the parent container for
                             ParcelFabricManager.
    ====================     ====================================================================

    """

    _con = None
    _flc = None
    _gis = None
    _url = None
    _version = None
    _properties = None
    # ----------------------------------------------------------------------

    def __init__(self, url, gis, version, flc):
        """Constructor"""
        self._url = url
        self._gis = gis
        self._con = gis._portal.con
        self._version = version
        self._flc = flc

    # ----------------------------------------------------------------------

    def __str__(self):
        return "< ParcelFabricManager @ %s >" % self._url

    # ----------------------------------------------------------------------

    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------

    def __enter__(self):
        return self

    # ----------------------------------------------------------------------

    def __exit__(self, type, value, traceback):
        return

    # ----------------------------------------------------------------------

    @property
    def layer(self):
        """returns the Parcel Layer ( :class:`~arcgis.features.FeatureLayer` object or None ) for the service"""
        if (
            "controllerDatasetLayers" in self._flc.properties
            and "parcelLayerId" in self._flc.properties.controllerDatasetLayers
        ):
            url = "%s/%s" % (
                self._flc.url,
                self._flc.properties.controllerDatasetLayers.parcelLayerId,
            )
            return FeatureLayer(url=url, gis=self._gis)
        return None

    # ----------------------------------------------------------------------

    @property
    def properties(self):
        """returns the properties of the service"""
        if self._properties is None:
            res = self._con.get(self._url, {"f": "json"})
            self._properties = PropertyMap(res)
        return self._properties

    # ----------------------------------------------------------------------

    def assign_to_record(
        self,
        features: list[dict[str, Any]],
        record: str,
        write_attribute: str,
        moment: int | str | None = None,
        future: bool = False,
    ):
        """
        Assigns the specified parcel features to the specified record. If
        parcel polygons are assigned, the record polygon will be updated to
        match the cumulative geometry of all the parcels associated to it.
        The Created By Record or Retired By Record attribute field of the
        parcel features is updated with the global ID of the assigned
        record.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        features                 Required List. The parcel features to assign to the specified record.
                                 Can be parcels, parcel polygons, parcel points, and parcel lines.

                                 :Syntax:

                                 .. code-block:: python

                                     >>> features=[{"id":"<guid>","layerId":"<layerID>"},{...}]

        --------------------     --------------------------------------------------------------------
        record                   Required String. The record that will be assigned to the specified
                                 parcel features.
        --------------------     --------------------------------------------------------------------
        write_attribute          Required String. Represents the record field to update on the parcel
                                 features. Either the Created By Record or Retired By Record field is
                                 to be updated with the global ID of the assigned record.

                                 Allowed Values:

                                    `CreatedByRecord` or `RetiredByRecord`

        --------------------     --------------------------------------------------------------------
        moment                   Optional Integer. This should only be specified by the client when
                                 they do not want to use the current moment

        --------------------     --------------------------------------------------------------------
        future                   Optional boolean. If `True`, the request is processed as an asynchronous
                                 job and a URL is returned that points a location displaying the status
                                 of the job.

                                 The default is `False`.
        ====================     ====================================================================

        :return: Boolean. `True` if successful otherwise `False`

        """
        url = "{base}/assignFeaturesToRecord".format(base=self._url)
        if moment is None:
            moment = int(time.time())
        params = {
            "gdbVersion": self._version.properties.versionName,
            "sessionId": self._version._guid,
            "moment": moment,
            "parcelFeatures": features,
            "record": record,
            "writeAttribute": write_attribute,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def build(
        self,
        extent: dict | Envelope | None = None,
        moment: int | str | None = None,
        return_errors: bool = False,
        record: str | None = None,
        future: bool = False,
    ):
        """
        A `build` will fix known parcel fabric errors.

        For example, if a parcel polygon exists without lines, then build will
        construct the missing lines. If lines are missing, the polygon row(s)
        are created. When constructing this objects, build will attribute the
        related keys as appropriate. Build also maintains `lineage` and `record`
        features. The parcel fabric must have sufficient information for build
        to work correctly. Ie, source reference document, and connected lines.

        Build provides options to increase performance. The process can just
        work on specific parcels, geometry types or only respond to parcel point
        movement in the case of an adjustment.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        extent                   Optional :class:`~arcgis.geometry.Envelope` . The extent to build.


                                 :Syntax:

                                 .. code-block:: python

                                     >>> extent={
                                                 "xmin":X min,
                                                 "ymin": y min,
                                                 "xmax": x max,
                                                 "ymax": y max,
                                                 "spatialReference": {"wkid": <wkid_value>}
                                                }

        --------------------     --------------------------------------------------------------------
        moment                   Optional String. This should only be specified by the client when
                                 they do not want to use the current moment
        --------------------     --------------------------------------------------------------------
        return_errors            Optional Boolean. If `True`, a verbose response will be given if errors
                                 occured.  The default is `False`.  **Deprecated**
        --------------------     --------------------------------------------------------------------
        record                   Optional String. Represents the record identifier (guid).  If a
                                 record guid is provided, only parcels associated to the record are
                                 built, regardless of the build extent.

        --------------------     --------------------------------------------------------------------
        future                   Optional boolean. If `True`, the request is processed as an asynchronous
                                 job and a URL is returned that points a location displaying the status
                                 of the job.

                                 The default is `False`.
        ====================     ====================================================================


        :return: Boolean. `True` if successful else `False`

        """
        if extent:
            extent = self._validate_extent(extent)

        url = "{base}/build".format(base=self._url)
        if moment is None:
            moment = int(time.time())
        params = {
            "gdbVersion": self._version.properties.versionName,
            "sessionId": self._version._guid,
            "moment": moment,
            "buildExtent": extent,
            "record": record,
            "async": future,
            # "returnErrors" : return_errors,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def clip(
        self,
        parent_parcels: list[dict[str, Any]],
        clip_record: str | None = None,
        clipping_parcels: list[dict[str, Any]] | None = None,
        geometry: Polygon | None = None,
        moment: int | str | None = None,
        option: str | None = None,
        area_unit: str | None = None,
        future: bool = False,
    ):
        """

        Clip cuts a new child parcel into existing parent parcels. Commonly
        it retires the parent parcel(s) it cuts into to generate a reminder
        child parcel. This type of split is often part of a `parcel split
        metes and bounds` record driven workflow.

        =======================     ====================================================================
        **Parameter**                **Description**
        -----------------------     --------------------------------------------------------------------
        parent_parcels              parent parcels that will be clipped into.


                                    :Syntax:

                                    .. code-block:: python

                                        >>> parent_parcels= <parcel (guid)+layer (name)...>

        -----------------------     --------------------------------------------------------------------
        clip_record                 Optional String. It is the GUID for the active legal record.
        -----------------------     --------------------------------------------------------------------
        clipping_parcels            Optional List. A list of child parcels that will be used to clip
                                    into the parent parcels. Parcel lineage is created if the child
                                    'clipping_parcels' and the parcels being clipped are of the same
                                    parcel type.


                                    :Syntax: ``clipping_parcels= <"id" : "parcel guid", "layerId": "<layer id>"...>``

                                    .. code-block:: python

                                        # Example:
                                        >>> clipping_parcels = [{"id":"{D01D3F47-5FE2-4E39-8C07-E356B46DBC78}","layerId":"16"}]

                                    .. note::
                                        Either `clipping_parcels` or `geometry` is required.

        -----------------------     --------------------------------------------------------------------
        geometry                    Optional Polygon. Allows for the clipping a parcel based on geometry instead of
                                    'clipping_parcels' geometry. No parcel lineage is created.

                                    .. note::
                                        Either `clipping_parcels` or `geometry` is required.

        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This should only be specified by the client when
                                    they do not want to use the current moment
        -----------------------     --------------------------------------------------------------------
        option                      Optional String. Represents the type of clip to perform:

                                    -  `PreserveArea` - Preserve the areas that intersect and discard the remainder areas. (default)
                                    -  `DiscardArea` - Discard the areas that intersect and preserve the remainder areas.
                                    -  `PreserveBothAreasSplit` - Preserve both the intersecting and remainder areas.
        -----------------------     --------------------------------------------------------------------
        area_unit                   Optional String. Area units to be used when calculating the stated
                                    areas of the clipped parcels. The stated area of the clipped parcels
                                    will be calculated if the stated areas exist on the parent parcels
                                    being clipped.

        -----------------------     --------------------------------------------------------------------
        future                      Optional boolean. If `True`, the request is processed as an asynchronous
                                    job and a URL is returned that points a location displaying the status
                                    of the job.

                                    The default is `False`.
        =======================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'


        """
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/clip".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "parentParcels": parent_parcels,
            "moment": moment,
            "record": clip_record,
            "clippingParcels": clipping_parcels,
            "clippingGeometry": geometry,
            "clipOption": option,
            "defaultAreaUnit": area_unit,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def merge(
        self,
        parent_parcels: list[dict[str, Any]],
        target_parcel_type: str,
        attribute_overrides: dict[str, Any] | None = None,
        child_name: str | None = None,
        default_area_unit: int | None = None,
        merge_record: str | None = None,
        merge_into: str | None = None,
        moment: int | str | None = None,
        future: bool = False,
    ):
        """
        Merge combines 2 or more parent parcels into onenew child parcel. Merge
        sums up legal areas of parent parcels to the new child parcel legal
        area (using default area units as dictated by client). The child parcel
        lines arecomposed from the outer boundaries of the parent parcels.
        Merge can create multipart parcels as well as proportion lines (partial
        overlap of parent parcels). Record footprint is updated to match the
        child parcel.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        parent_parcels           Required String. It is the parcel(guid)+layer(name) identifiers to
                                 merge.
        --------------------     --------------------------------------------------------------------
        target_parcel_type       Required String. Layer where parcel is merged to.  History is
                                 created when parents and child are of the same parcel type
        --------------------     --------------------------------------------------------------------
        attribute_overrides      Optional List. A list of attributes to set on the child parcel, if
                                 they exist. Pairs of field name and value.


                                 :Syntax:

                                 .. code-block:: python

                                     >>> attribute_overrides = [
                                                               {
                                                                "type": "PropertySet",
                                                                "propertySetItems": [
                                                                                     <field name>,
                                                                                     <field value>
                                                                                    ]
                                                               }
                                                              ]

                                 .. note::
                                     To set subtype, include subtype value in this list.

        --------------------     --------------------------------------------------------------------
        child_name               Optional String. A descript of the child layer. **DEPRECATED**
        --------------------     --------------------------------------------------------------------
        default_area_unit        Optional String. The area units of the child parcel.
        --------------------     --------------------------------------------------------------------
        merge_record             Optional String. Record identifier (guid).  If missing, no history
                                 is created.
        --------------------     --------------------------------------------------------------------
        merge_into               Optional String. A parcel identifier (guid). Invalid to have a
                                 record id.
        --------------------     --------------------------------------------------------------------
        moment                   Optional String. This parameter represents the session moment (the
                                 default is the version current moment). This should only be
                                 specified by the client when they do not want to use the current
                                 moment.
        --------------------     --------------------------------------------------------------------
        area_unit                Optional Integer. Represents the default area units to be used when
                                 calculating the stated area of the merged parcel. The stated area of
                                 the merged parcel will be calculated if the stated areas exist on
                                 the parcels being merged.

        --------------------     --------------------------------------------------------------------
        future                   Optional boolean. If `True`, the request is processed as an asynchronous
                                 job and a URL is returned that points a location displaying the status
                                 of the job.

                                 The default is `False`.
        ====================     ====================================================================


        :return: Dictionary indicating 'success' or 'error'

        """
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/merge".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "parentParcels": parent_parcels,
            "record": merge_record,
            "moment": moment,
            "targetParcelType": target_parcel_type,
            "mergeInto": merge_into,
            # "childName" : child_name,
            "defaultAreaUnit": default_area_unit,
            "attributeOverrides": attribute_overrides,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def copy_lines_to_parcel_type(
        self,
        parent_parcels: list[dict[str, Any]],
        record: str,
        target_type: int | str,
        moment: int | str | None = None,
        mark_historic: bool = False,
        use_source_attributes: bool = False,
        attribute_overrides: dict[str, Any] | None = None,
        use_polygon_attributes: bool = False,
        parcel_subtype: int | None = None,
        future: bool = False,
    ):
        """

        Copy lines to parcel type is used when the construction of the
        child parcel is based on parent parcel geometry. It creates a
        copy of the parent parcels lines that the user can modify (insert,
        delete, update) before they build the child parcels. If the source
        parcel type and the target parcel type are identical (common)
        parcel lineage is created.

        =======================     ====================================================================
        **Parameter**                **Description**
        -----------------------     --------------------------------------------------------------------
        parent_parcels              Required String. Parcel parcels from which lines are copied.
        -----------------------     --------------------------------------------------------------------
        record                      Required String. The unique identifier (guid) of the active legal
                                    record.
        -----------------------     --------------------------------------------------------------------
        target_type                 Required String. The target parcel layer to which the lines will be
                                    copied to.
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This parameter represents the session moment (the
                                    default is the version current moment). This should only be
                                    specified by the client when they do not want to use the current
                                    moment.
        -----------------------     --------------------------------------------------------------------
        mark_historic               Optional Boolean. Mark the parent parcels historic. The default is
                                    `False`.
        -----------------------     --------------------------------------------------------------------
        use_source_attributes       Optional Boolean. If the source and the target line schema match,
                                    attributes from the parent parcel lines will be copied to the new
                                    child parcel lines when it is set to  `True`. The default is `False`.
        -----------------------     --------------------------------------------------------------------
        use_polygon_attributes      Optional Boolean. Parameter representing whether to preserve and
                                    transfer attributes of the parent parcels to the generated seeds.
        -----------------------     --------------------------------------------------------------------
        attribute_overrides         Optional Dictionary. To set fields on the child parcel lines with a
                                    specific value. Uses a key/value pair of FieldName/Value.

                                    Syntax:

                                    .. code-block:: python

                                        >>> attributeOverrides = [{"type": "PropertySet",
                                                                   "propertySetItems": [
                                                                                        <field name>,
                                                                                        <field value>
                                                                                       ]
                                                                 }]

        -----------------------     --------------------------------------------------------------------
        parcel_subtype              Optional Integer. Represents the target parcel subtype.

        -----------------------     --------------------------------------------------------------------
        future                      Optional boolean. If `True`, the request is processed as an asynchronous
                                    job and a URL is returned that points a location displaying the status
                                    of the job.

                                    The default is `False`.
        =======================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'


        """
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/copyLinesToParcelType".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "parentParcels": parent_parcels,
            "record": record,
            "markParentAsHistoric": mark_historic,
            "useSourceLineAttributes": use_source_attributes,
            "useSourcePolygonAttributes": use_polygon_attributes,
            "targetParcelType": target_type,
            "targetParcelSubtype": parcel_subtype,
            "attributeOverrides": attribute_overrides,
            "moment": moment,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def change_type(
        self,
        parcels: list[dict[str, Any]],
        target_type: str,
        parcel_subtype: int | str = 0,
        moment: int | str | None = None,
        future: bool = False,
    ):
        """

        Changes a set of parcels to a new parcel type. It creates new
        polygons and lines and deletes them from the source type. This
        is used when a parcel was associated in the wrong parcel type subtype
        and/or when creating multiple parcels as part of a build process.
        Example: when lot parcels are created as part of a subdivision, the
        road parcel is moved to the encumbrance (easement) parcel type.

        =======================     ====================================================================
        **Parameter**                **Description**
        -----------------------     --------------------------------------------------------------------
        parcels                     Required List. Parcels list that will change type
        -----------------------     --------------------------------------------------------------------
        target_type                 Required String. The target parcel layer
        -----------------------     --------------------------------------------------------------------
        target_subtype              Optional Integer. Target parcel subtype. The default is 0 meaning
                                    no subtype required.
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This parameter represents the session moment (the
                                    default is the version current moment). This should only be
                                    specified by the client when they do not want to use the current
                                    moment.

        -----------------------     --------------------------------------------------------------------
        future                      Optional boolean. If `True`, the request is processed as an asynchronous
                                    job and a URL is returned that points a location displaying the status
                                    of the job.

                                    The default is `False`.
        =======================     ====================================================================

        :return: Boolean. `True` if successful else `False`


        """
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/changeParcelType".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "parcels": parcels,
            "targetParcelType": target_type,
            "targetParcelSubtype": parcel_subtype,
            "moment": moment,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def delete(
        self,
        parcels: list[dict[str, Any]],
        moment: int | str | None = None,
        future: bool = False,
    ):
        """

        Delete a set of parcels, removing associated or unused lines, and
        connected points.

        =======================     ====================================================================
        **Parameter**                **Description**
        -----------------------     --------------------------------------------------------------------
        parcels                     Required List. The parcels to erase.
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This parameter represents the session moment (the
                                    default is the version current moment). This should only be
                                    specified by the client when they do not want to use the current
                                    moment.

        -----------------------     --------------------------------------------------------------------
        future                      Optional boolean. If `True`, the request is processed as an asynchronous
                                    job and a URL is returned that points a location displaying the status
                                    of the job.

                                    The default is `False`.
        =======================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'


        """
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/deleteParcels".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "parcels": parcels,
            "moment": moment,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def update_history(
        self,
        features: list[dict[str, Any]],
        record: str,
        moment: int | str | None = None,
        set_as_historic: bool = False,
        future: bool = False,
    ):
        """
        Sets the specified parcel features to current or historic using the
        specified record. If setting current parcels as historic, the
        Retired By Record field of the features is updated with the Global
        ID of the specified record. If setting historic parcels as current,
        the Created By Record field of the features is updated with the
        Global ID of the specified record.

        =======================     ====================================================================
        **Parameter**                **Description**
        -----------------------     --------------------------------------------------------------------
        features                    Required List. The parcel features to be set as historic or current.
                                    Can be parcels, parcel polygons, parcel points, and parcel lines.


                                    :Syntax:

                                    .. code-block:: python

                                        >>> features=[
                                                      {"id":"<guid>",
                                                       "layerId":"<layerID>"},
                                                      {...}
                                                     ]

        -----------------------     --------------------------------------------------------------------
        record                      Required String. A **GUID** representing the record that will be
                                    assigned to the features set as current or historic.
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This parameter represents the session moment (the
                                    default is the version current moment). This should only be
                                    specified by the client when they do not want to use the current
                                    moment.
        -----------------------     --------------------------------------------------------------------
        set_as_historic             Optional Boolean.  Boolean parameter representing whether to set the
                                    features as historic (`True`). If `False`, features will be set as
                                    current.

        -----------------------     --------------------------------------------------------------------
        future                      Optional boolean. If `True`, the request is processed as an asynchronous
                                    job and a URL is returned that points a location displaying the status
                                    of the job.

                                    The default is `False`.
        =======================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/updateParcelHistory".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "moment": moment,
            "record": record,
            "setAsHistoric": set_as_historic,
            "parcelFeatures": features,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def create_seeds(
        self,
        record: str,
        moment: int | str | None = None,
        extent: dict | Envelope | None = None,
        future: bool = False,
    ):
        """

        Create seeds creates parcel seeds for closed loops of lines that
        are associated with the specified record.

        When building parcels from lines, parcel seeds are used. A parcel
        seed is the initial state or seed state of a parcel. A parcel seed
        indicates to the build process that a parcel can be built from the
        lines enclosing the seed.

        A parcel seed is a minimized polygon feature and is stored in the
        parcel type polygon feature class.

        =======================     ====================================================================
        **Parameter**                **Description**
        -----------------------     --------------------------------------------------------------------
        record                      Required String. A **GUID** representing the record that will be
                                    assigned to the features set as current or historic.
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This parameter represents the session moment (the
                                    default is the version current moment). This should only be
                                    specified by the client when they do not want to use the current
                                    moment.
        -----------------------     --------------------------------------------------------------------
        extent                      Optional Dict/ :class:`~arcgis.geometry.Envelope` . The envelope of the extent
                                    in which to create seeds.

        -----------------------     --------------------------------------------------------------------
        future                      Optional boolean. If `True`, the request is processed as an asynchronous
                                    job and a URL is returned that points a location displaying the status
                                    of the job.

                                    The default is `False`.
        =======================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        if extent:
            extent = self._validate_extent(extent)

        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/createSeeds".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "moment": moment,
            "record": record,
            "extent": extent,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def duplicate(
        self,
        parcels: list[dict[str, Any]],
        parcel_type: int | str,
        record: str,
        parcel_subtype: int | str | None = None,
        repeat_count: int | str | None = None,
        update_field: str = None,
        start_value: int | str | None = None,
        increment_value: int | str | None = None,
        moment: int | str | None = None,
        future: bool = False,
    ):
        """
        `duplicate` allows for the cloning of parcels from a specific record.

        Parcels can be duplicated in the following ways:

        -  Duplicate to a different parcel type.
        -  Duplicate to a different subtype in the same parcel type.
        -  Duplicate to a different subtype in a different parcel type.

        Similarly, parcel seeds can be duplicated to subtypes and different parcel types.

        =======================     ====================================================================
        **Parameter**                **Description**
        -----------------------     --------------------------------------------------------------------
        parcels                     Required List. A list of parcels to duplicate.


                                    :Syntax:

                                    .. code-block:: python

                                        >>>> parcels=[
                                                      {"id":"<parcelguid>",
                                                       "layerId":"16"
                                                      },
                                                      {...}
                                                     ]

        -----------------------     --------------------------------------------------------------------
        parcel_type                 Required Integer. The target parcel type.
        -----------------------     --------------------------------------------------------------------
        record                      Required String. A **GUID** representing the record that will be
                                    assigned to the features set as current or historic.
        -----------------------     --------------------------------------------------------------------
        parcel_subtype              Optional Integer. The target parcel subtype.  The default is 0.
        -----------------------     --------------------------------------------------------------------
        repeat_count                Optional Integer. How many times to duplicate the target parcels
        -----------------------     --------------------------------------------------------------------
        update_field                Optional String. Which incrementable field to update on the target
        -----------------------     --------------------------------------------------------------------
        start_value                 Optional Integer. What value to start on when incrementing
        -----------------------     --------------------------------------------------------------------
        increment_value             Optional Integer. How many steps to increment
        -----------------------     --------------------------------------------------------------------
        moment                      Optional String. This parameter represents the session moment (the
                                    default is the version current moment). This should only be
                                    specified by the client when they do not want to use the current
                                    moment.

        -----------------------     --------------------------------------------------------------------
        future                      Optional boolean. If `True`, the request is processed as an asynchronous
                                    job and a URL is returned that points a location displaying the status
                                    of the job.

                                    The default is `False`.
        =======================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        if parcel_subtype is None:
            parcel_subtype = 0
        if moment is None:
            moment = int(time.time())
        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/duplicateParcels".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "moment": moment,
            "record": record,
            "parcels": parcels,
            "targetParcelType": parcel_type,
            "targetParcelSubtype": parcel_subtype,
            "repeatCount": repeat_count,
            "updateField": update_field,
            "startValue": start_value,
            "incrementValue": increment_value,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def analyze_least_squares_adjustment(
        self,
        analysis_type: str = "CONSISTENCY_CHECK",
        convergence_tolerance: float = 0.05,
        parcel_features: dict[str, Any] | None = None,
        future: bool = False,
    ):
        """
        .. note::
            Least Squares Adjustment functionality introduced at version 10.8.1

        Analyzes the parcel fabric measurement network by running a least squares adjustment on the
        input parcels. A least-squares adjustment is a mathematical procedure that uses statistical
        analysis to estimate the most likely coordinates for connected points in a measurement network.

        Use :meth:`~arcgis.features._parcel.ParcelFabricManager.apply_least_squares_adjustment`
        to apply the results of a least squares adjustment to parcel fabric feature classes.

        ============================    ====================================================================
        **Parameter**                    **Description**
        ----------------------------    --------------------------------------------------------------------
        analysis_type                   Optional string. Represents the type of least squares analysis that will be run on the input parcels.

                                        * CONSISTENCY_CHECK - A free-network least-squares adjustment will be run to check dimensions on
                                          parcel lines for inconsistencies and mistakes. Fixed or weighted control points will not be
                                          used by the adjustment.

                                        * WEIGHTED_LEAST_SQUARES - A weighted least-squares adjustment will be run to compute updated
                                          coordinates for parcel points. The parcels being adjusted should connect to at least two fixed
                                          or weighted control points.

                                        The default value is CONSISTENCY_CHECK.
        ----------------------------    --------------------------------------------------------------------
        convergence_tolerance           Optional float. Represents the maximum coordinate shift expected after iterating the least squares adjustment. A least
                                        squares adjustment is run repeatedly (in iterations) until the solution converges. The solution is
                                        considered converged when maximum coordinate shift encountered becomes less than the specified convergence
                                        tolerance.

                                        The default value is 0.05 meters or 0.164 feet.
        ----------------------------    --------------------------------------------------------------------
        parcel_features                 Optional list. Represents the input parcels that will be analyzed by a least squares adjustment.


                                        :Syntax:

                                        .. code-block:: python

                                            >>> parcel_features = [{"id":"<guid>","layerId":"<layerID>"},{...}]

                                        If None, the method will analyze the entire parcel fabric.
        ----------------------------    --------------------------------------------------------------------
        future                          Optional boolean. If `True`, the request is processed as an
                                        asynchronous job and a URL is returned that points a location
                                        displaying the status of the job.

                                        The default is `False`.
        ============================    ====================================================================

        :return: Boolean. `True` if successful else `False`

        """
        url = "{base}/analyzeByLeastSquaresAdjustment".format(base=self._url)
        params = {
            "gdbVersion": self._version.properties.versionName,
            "sessionId": self._version._guid,
            "analysisType": analysis_type,
            "convergenceTolerance": convergence_tolerance,
            "parcelFeatures": parcel_features,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def apply_least_squares_adjustment(
        self,
        movement_tolerance: float = 0.05,
        update_attributes: bool = True,
        future: bool = False,
    ):
        """
        .. note::
            Least Squares Adjustment functionality introduced at version 10.8.1

        Applies the results of a least squares adjustment to parcel fabric feature classes. Least squares adjustment results stored
        in the AdjustmentLines and AdjustmentPoints feature classes are applied to the corresponding parcel line, connection line,
        and parcel fabric point feature classes.

        Use :meth:`~arcgis.features._parcel.ParcelFabricManager.analyze_least_squares_adjustment`
        to run a least-squares analysis on parcels and store the results in adjustment feature classes.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        movement_tolerance        Optional float. Represents the minimum allowable coordinate shift when updating parcel fabric points. If the distance between
                                  the adjustment point and the parcel fabric point is greater than the specified tolerance, the parcel fabric
                                  point is updated to the location of the adjustment point.

                                  The default tolerance is 0.05 meters or 0.164 feet.
        --------------------     --------------------------------------------------------------------
        update_attributes         Optional boolean. Specifies whether attribute fields in the parcel fabric Points feature class will be updated with
                                  statistical metadata. The XY Uncertainty, Error Ellipse Semi Major, Error Ellipse Semi Minor, and
                                  Error Ellipse Direction fields will be updated with the values stored in the same fields in the AdjustmentPoints
                                  feature class.

                                  The default is `True`
        --------------------     --------------------------------------------------------------------
        future                   Optional boolean. If `True`, the request is processed as an asynchronous job and a URL is returned that points a location
                                 displaying the status of the job.

                                 The default is `False`.
        ====================     ====================================================================

        :return: Boolean. `True` if successful else `False`

        """

        url = "{base}/applyLeastSquaresAdjustment".format(base=self._url)
        params = {
            "gdbVersion": self._version.properties.versionName,
            "sessionId": self._version._guid,
            "movementTolerance": movement_tolerance,
            "updateAttributes": update_attributes,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            future = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return future
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def divide(
        self,
        divide_parcel_guid: str,
        divide_parcel_type: int | str | None,
        divide_record: str,
        divide_option: str,
        divide_number_of_parts: int | str,
        divide_part_area: float | int,
        divide_line_bearing: float,
        divide_left_side: bool,
        divide_distribute_remainder: bool,
        default_area_unit: int | str | None = None,
        divide_cogo_line_bearing: float = None,
        future: bool = False,
    ):
        """
        .. note::
            Divide functionality introduced at version 10.9.1

        Divide a polygon feature into multiple features that have proportional or equal areas, or equal widths.

        =========================== ====================================================================
        **Parameter**                **Description**
        --------------------------- --------------------------------------------------------------------
        divide_parcel_guid          Required String. Parameter for the unique identifier `guid` of the
                                    parcel being divided.
        --------------------------- --------------------------------------------------------------------
        divide_parcel_type          Required Integer. Parameter representing the parcel type layer ID in
                                    which the new, divided parcels will be created.
        --------------------------- --------------------------------------------------------------------
        divide_record               Required String. Parameter for the unique identifier `guid` of the
                                    record being used for the divide.
                                    If missing, no parcel history is created.
        --------------------------- --------------------------------------------------------------------
        divide_option               Required String. The type of division to be performed.

                                    - `ProportionalArea`
                                    - `EqualArea`
                                    - `EqualWidth`
        --------------------------- --------------------------------------------------------------------
        divide_number_of_parts      Required Integer. The number parts into which the parcel will
                                    be divided.
        --------------------------- --------------------------------------------------------------------
        divide_part_area            Required Float. Area (or width) of each part (parcel fabric GDB units squared).

                                    .. note::
                                        This value is ignored when dividing by proportional area. A
                                        default value of 0 will be applied.
        --------------------------- --------------------------------------------------------------------
        divide_line_bearing         Required Float. The direction (in decimal degrees) of the line
                                    used to divide the parcel.
        --------------------------- --------------------------------------------------------------------
        divide_left_side            Required Boolean. Parameter indicating if area being divided is
                                    starting from the leftmost edge of the parcel. Any remainder area
                                    will be to the right of the divided parts. If `False`, the area being
                                    divided starts from the rightmost edge of the parcel and any remainder
                                    area will be to the left of the divided parts.

                                    This parameter is required for the `EqualArea` and `EqualWidth`
                                    divide options.

                                    .. note::
                                        This value is ignored when dividing by proportional area. A
                                        default value of `False` will be applied.
        --------------------------- --------------------------------------------------------------------
        divide_distribute_remainder Required Boolean. Indicates whether to distribute or merge the
                                    remainder area after the divide is performed. This parameter is used
                                    for the `EqualArea` and `EqualWidth` divide options.

                                    .. note::
                                        This value is ignored when dividing by proportional area. A
                                        default value of `False` will be applied.
        --------------------------- --------------------------------------------------------------------
        default_area_unit           Required Integer. The units in which area will be stored. The parameter
                                    is specified as a domain code from the `PF_AreaUnits` parcel fabric
                                    domain.

                                    .. code-block:: python

                                        #Example Usage:

                                        #Square feet
                                        >>> default_area_unit=109405
                                        #Square meters
                                        >>> default_area_unit=109404

        --------------------------- --------------------------------------------------------------------
        divide_cogo_line_bearing    Optional Float. Parameter representing the COGO direction
                                    (in decimal degrees) that will be stored in the COGO Direction field
                                    of the dividing lines.
        --------------------------- --------------------------------------------------------------------
        future                      Optional boolean. If `True`, the request is processed as an asynchronous
                                    job and a URL is returned that points a location displaying the status
                                    of the job.

                                    The default is `False`.
        =========================== ====================================================================

        :return: Dictionary indicating 'success' or 'error'


        """
        if divide_option == "ProportionalArea":
            if not divide_part_area:
                divide_part_area = 0
            if not divide_left_side:
                divide_left_side = False
            if not divide_distribute_remainder:
                divide_distribute_remainder = False

        gdb_version = self._version.properties.versionName
        session_id = self._version._guid
        url = "{base}/divide".format(base=self._url)
        params = {
            "gdbVersion": gdb_version,
            "sessionId": session_id,
            "divideParcelGuid": divide_parcel_guid,
            "divideParcelType": divide_parcel_type,
            "record": divide_record,
            "divideOption": divide_option,
            "divideNumberOfParts": divide_number_of_parts,
            "dividePartAreaOrWidth": divide_part_area,
            "divideLineBearing": divide_line_bearing,
            "divideLeftSide": divide_left_side,
            "divideDistributeRemainder": divide_distribute_remainder,
            "defaultAreaUnit": default_area_unit,
            "divideCogoLineBearing": divide_cogo_line_bearing,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def reassign_features_to_record(
        self,
        source_record: str,
        target_record: str,
        delete_source_record: bool,
        future: bool = False,
    ):
        """
        Reassigns all parcel features in the specified source record to the specified target record.
        The source record will become empty and will be associated to no parcel features. The record
        polygon of the target record will be updated to match the cumulative geometry of all the
        parcels associated to it.

        The Created By Record or Retired By Record attribute field of the
        parcel features is updated with the global ID of the assigned
        record.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        source_record            Required String. GlobalID representing the record containing the
                                 parcel features to be reassigned.


                                 Syntax:

                                  source_record=<guid>

        --------------------     --------------------------------------------------------------------
        target_record            Required String. GlobalID representing the target record to which
                                 the parcel features will be reassigned.


                                 Syntax

                                    target_record=<guid>

        --------------------     --------------------------------------------------------------------
        delete_source_record     Required Bool. Parameter indicating whether to delete the original
                                 source record.

        --------------------     --------------------------------------------------------------------
        future                      Optional boolean. If `True`, the request is processed as an asynchronous
                                    job and a URL is returned that points a location displaying the status
                                    of the job.

                                    The default is `False`.
        ====================     ====================================================================

        :returns: Boolean

        """
        url = "{base}/reassignFeaturesToRecord".format(base=self._url)
        params = {
            "gdbVersion": self._version.properties.versionName,
            "sessionId": self._version._guid,
            "sourceRecord": source_record,
            "targetRecord": target_record,
            "deleteSourceRecord": delete_source_record,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def reconstruct_from_seeds(
        self,
        extent: dict | Envelope,
        future: bool = False,
    ):
        """
        This operation constructs parcels from seeds enclosed by parcel lines in the specified extent. The tool reconstructs parcels regardless of the parcel
        lines associations with records.

        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        extent                   Parameter representing the envelope of the extent to reconstruct seeds.
                                 Seeds that lie within the specified extent will be reconstructed into
                                 parcels.


                                 :Syntax:

                                 .. code-block:: python

                                     >>> extent={
                                                 "xmin":X min,
                                                 "ymin": y min,
                                                 "xmax": x max,
                                                 "ymax": y max,
                                                 "spatialReference": {"wkid": <wkid_value>}
                                                }

        --------------------     --------------------------------------------------------------------
        future                      Optional boolean. If `True`, the request is processed as an asynchronous
                                    job and a URL is returned that points a location displaying the status
                                    of the job.

                                    The default is `False`.
        ====================     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        if extent:
            extent = self._validate_extent(extent)

        url = "{base}/reconstructFromSeeds".format(base=self._url)
        params = {
            "gdbVersion": self._version.properties.versionName,
            "sessionId": self._version._guid,
            "extent": extent,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def transfer_parcel(
        self,
        transfer_parcel_feature: dict[str, Any],
        target_parcel_features: list[dict[str, Any]],
        record: str,
        default_area_unit: int,
        source_parcel_features: list[dict[str, Any]] | None = None,
        future: bool = False,
    ):
        """
        The :meth:`~transfer_parcel` method supports workflows for transferring
        a piece of land between parcels.

        =======================     =======================================================================
        **Parameter**                **Description**
        -----------------------     -----------------------------------------------------------------------
        transfer_parcel_feature     Required Dict. Parameter representing the parcel to be transferred.
                                    Only one parcel can be specified as the transfer parcel.

                                    .. code-block:: python

                                        # Example Usage:

                                        >>> transfer_parcel_feature={"id":"<guid>","layerId":"<layerID>"}
        -----------------------     -----------------------------------------------------------------------
        target_parcel_features      Required List. Parameter representing the target parcels to which land
                                    will be transferred. These parcels will be merged with the transfer
                                    parcel and will become larger.

                                    .. code-block:: python

                                        # Example Usage:

                                        >>> target_parcel_features=[{"id":"<guid>","layerId":"<layerID>"},{...}]

        -----------------------     -----------------------------------------------------------------------
        record                      Required String. Parameter for the unique identifier (GUID) of the
                                    record being used for the transfer
        -----------------------     -----------------------------------------------------------------------
        default_area_unit           Required Integer. The units in which area will be stored. The parameter
                                    is specified as a domain code from the `PF_AreaUnits` parcel fabric
                                    domain.

                                    .. code-block:: python

                                        #Example Usage:

                                        #Square feet
                                        >>> default_area_unit=109405
                                        #Square meters
                                        >>> default_area_unit=109404

        -----------------------     -----------------------------------------------------------------------
        source_parcel_features      Optional List. Parameter representing the source parcels from which
                                    land will be transferred. These parcels will be clipped and will
                                    become smaller.

                                    .. code-block:: python

                                        # Example Usage:

                                        >>> source_parcel_features=[{"id":"<guid>","layerId":"<layerID>"},{...}]

        -----------------------     -----------------------------------------------------------------------
        future                      Optional boolean. If `True`, the request is processed as an asynchronous
                                    job and a URL is returned that points a location displaying the status
                                    of the job.

                                    The default is `False`.
        =======================     =======================================================================

        :return: Dictionary indicating 'success' or 'error' with a list of edited features

        """
        url = "{base}/transferParcel".format(base=self._url)
        params = {
            "gdbVersion": self._version.properties.versionName,
            "sessionId": self._version._guid,
            "transferParcelFeature": transfer_parcel_feature,
            "sourceParcelFeatures": source_parcel_features,
            "targetParcelFeatures": target_parcel_features,
            "record": record,
            "defaultAreaUnit": default_area_unit,
            "async": future,
            "f": "json",
        }
        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------
    def set_line_label_position(
        self,
        parcel_line_features: list[dict[str, Any]],
        future: bool = False,
    ):
        """
        The :meth:`~set_line_label_position` sets the label position of the line's COGO dimension to the
        left of the parcel line, to the right of the parcel line, or centered over the parcel line.

        =======================     =======================================================================
        **Parameter**                **Description**
        -----------------------     -----------------------------------------------------------------------
        parcel_line_features        Required List. Parameter representing the input parcel line layers with
                                    label positions that will be updated.

                                    .. code-block:: python

                                        >>> parcel_line_features=[{"id":"<guid>",
                                                                   "layerId":"<layerID>"},
                                                                   {...}]

        -----------------------     -----------------------------------------------------------------------
        future                      Optional boolean. If `True`, the request is processed as an asynchronous
                                    job and a URL is returned that points a location displaying the status
                                    of the job.

                                    The default is `False`.
        =======================     =======================================================================

        :return: Dictionary indicating 'success' or 'error' with a list of edited features

        """
        url = "{base}/setParcelLineLabelPosition".format(base=self._url)
        params = {
            "gdbVersion": self._version.properties.versionName,
            "sessionId": self._version._guid,
            "parcelFeatures": parcel_line_features,
            "async": future,
            "f": "json",
        }

        if future:
            res = self._con.post(path=url, postdata=params)
            f = self._run_async(
                self._status_via_url,
                con=self._con,
                url=res["statusUrl"],
                params={"f": "json"},
            )
            return f
        else:
            return self._con.post(url, params)

    # ----------------------------------------------------------------------

    def _validate_extent(self, extent: dict | Envelope):
        """Check for valid Extent object or None"""
        from arcgis.geometry import Envelope

        if isinstance(extent, (dict, Envelope)):
            return dict(extent)
        elif extent is None:
            return None
        elif not extent is None:
            raise ValueError("Parameter `extent` must be None, Envelope or dict.")

    # ----------------------------------------------------------------------

    def _run_async(self, fn, **inputs):
        """runs the inputs asynchronously"""
        import concurrent.futures

        tp = concurrent.futures.ThreadPoolExecutor(1)
        try:
            future = tp.submit(fn=fn, **inputs)
        except:
            future = tp.submit(fn, **inputs)
        tp.shutdown(False)
        return future

    # ----------------------------------------------------------------------

    def _status_via_url(self, con, url, params):
        """
        performs the asynchronous check to see if the operation finishes
        """
        status_allowed = [
            "esriJobSubmitted",
            "esriJobWaiting",
            "esriJobExecuting",
            "esriJobSucceeded",
            "esriJobFailed",
            "esriJobTimedOut",
            "esriJobCancelling",
            "esriJobCancelled",
        ]
        status = con.get(url, params)
        while (
            status["status"] in status_allowed
            and status["status"] != "esriJobSucceeded"
        ):
            if status["status"] == "esriJobSucceeded":
                return status
            elif status["status"] in [
                "esriJobFailed",
                "esriJobTimedOut",
                "esriJobCancelled",
            ]:
                break
            status = con.get(url, params)
        return status

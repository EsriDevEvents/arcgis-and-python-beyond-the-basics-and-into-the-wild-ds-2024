from __future__ import annotations
from typing import Any, Optional
from arcgis.gis import GIS
from arcgis.env import active_gis
from arcgis._impl.common._mixins import PropertyMap
from arcgis.features._version import Version


###########################################################################
class ValidationManager(object):
    """
    The Validation Server is responsible for exposing the management
    capabilities necessary to support evaluation of geodatabase rules.
    """

    _con = None
    _gis = None
    _url = None
    _version = None
    _properties = None

    # ----------------------------------------------------------------------
    def __init__(self, url, version=None, gis=None):
        """initializer"""
        self._url = url
        if gis is None:
            self._gis = active_gis
        else:
            self._gis = gis
        assert isinstance(self._gis, GIS)
        self._con = gis._con
        self._version = version
        assert isinstance(self._version, Version)

    # ----------------------------------------------------------------------
    def _init(self):
        """loads the properties"""
        try:
            self._properties = PropertyMap(self._con.get(self._url, {"f": "json"}))
        except:
            self._properties = PropertyMap(self._con.post(self._url, {"f": "json"}))

    # ----------------------------------------------------------------------
    def __str__(self):
        url = self._url
        return f"< ValidationManager @ {url} >"

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """service properties"""
        if self._properties is None:
            self._init()
        return self._properties

    # ----------------------------------------------------------------------
    def update_error(
        self,
        error_features: list[dict[str, Any]],
        version: Optional[str] = None,
        return_edits: bool = False,
        **kwargs,
    ):
        """
        Updates errors on the validation tables.

        ===============     ====================================================================
        **Parameter**        **Description**
        ---------------     --------------------------------------------------------------------
        error_features      Required List.  The error features to be updated.

                            Syntax:


                                | error_features = [{
                                |      "errorType" : "object" | "point" | "line" |
                                |                   "polygon",
                                |      "features" : [
                                |        {
                                |          "globalId" : <guid>,
                                |          "fields" : {
                                |            "name1" : <value1>,
                                |            "name2" : <value2>
                                |          }}]}]

        ---------------     --------------------------------------------------------------------
        return_edits        Optional Boolean. `return_edits` returns features edited due to
                            errors update. Results returned are organized in a layer by layer
                            fashion. If it is set to `True`, each layer may have edited features
                            returned in an editedFeatures object.

                            The editedFeatures object returns full features including the original
                            features prior to delete, the original and current features for
                            updates and the current rows for inserts which may contain implicit
                            changes (e.g. as a result of a calculation rule ).

                            The response includes no editedFeatures and `exceededTransferLimit=True`
                            if the count of edited features to return is more than the maxRecordCount.
                            If clients are using this parameter to maintain a cache, they should
                            invalidate the cache when `exceededTransferLimit = True` is returned.
                            If the server encounters an error when generating the list of edits is
                            the response, `exceededTransferLimit = True` is also returned.

                            Edited features are returned in the spatial reference of the feature
                            service as defined by the services spatialReference object or by the
                            spatialReference of the layers extent object.

                            The default for this parameter is False.
        ===============     ====================================================================

        :return: Dictionary indicating 'success' or 'error'

        """
        url = self._url + "/updateErrors"
        params = {
            "f": "json",
            "gdbVersion": self._version.properties.versionName,
            "sessionId": self._version._guid,
            "errorFeatures": error_features,
            "returnEdits": return_edits,
        }
        if len(kwargs) > 0:
            params.update(kwargs)
        if not return_edits is None:
            params["returnEdits"] = return_edits
        res = self._con.post(url, params)
        return res

    # ----------------------------------------------------------------------
    def evaluate(
        self,
        evaluation: list[str],
        area: Optional[dict[str, Any]] = None,
        changes_in_version: bool = False,
        selection: Optional[list[dict[str, Any]]] = None,
        return_edits: bool = False,
    ):
        """
        Runs the topology rules and returns new errors if they exist.

        Evaluation can be performed on different types of geodatabase rules (controlled by the evaluationType):

        * Topology rules
        * Validation and batch calculation attribute rules


        ====================     ====================================================================
        **Parameter**             **Description**
        --------------------     --------------------------------------------------------------------
        evaluation               Required List of Strings.  A list of evaluation types.

                                 Values:

                                    "validationRules" | "calculationRules" | "topologyRules"

                                 Example:

                                    evaluation=["calculationRules"]
        --------------------     --------------------------------------------------------------------
        area                     Optional :class:`~arcgis.geometry.Envelope` /Dict. Extent of the area to evaluate.
        --------------------     --------------------------------------------------------------------
        changes_in_version       Optional Boolean. representing whether to perform the evaluation on
                                 the features that have changed in the version (default is false).
                                 Does not apply to the DEFAULT version.

                                 When set to true, the evaluationDate property for the version is
                                 updated. This is listed as a property for a version and can be
                                 accessed using the version resource and the version infos operation.
        --------------------     --------------------------------------------------------------------
        selection                Optional List.  A set of features to evaluate.  This is an array of
                                 layers and the global IDs or Object IDs of the features to examine.

                                 If the `evaluation_type` is **topology** this parameter is ignored.

                                 Syntax


                                     | [{
                                     |     "id" : <layerId1>,
                                     |     "globalIds" : [ <globalId> ],
                                     |     "objectIds" : [ <objectId> ]
                                     |   },
                                     |   {
                                     |     "id" : <layerId2>,
                                     |     "globalIds" : [ <globalId> ].
                                     |     "objectIds" : [ <objectId> ]
                                     |   }]


        --------------------     --------------------------------------------------------------------
        return_edits             Optional Boolean. returns features edited due to feature evaluation.
                                 Results returned are organized in a layer by layer fashion. If
                                 `return_edits` is set to true, each layer may have edited features
                                 returned.

                                 The default for this parameter is false. Always set to true when
                                 evaluating `topology` for a parcel fabric.

        ====================     ====================================================================

        :return: The number of new errors identified along with the moment.

        """
        url = self._url + "/evaluate"
        eval_lu = {
            "validation": "validationRules",
            "validationrules": "validationRules",
            "calculation": "calculationRules",
            "calculationrules": "calculationRules",
            "topology": "topologyRules",
            "topologyrules": "topologyRules",
        }
        params = {
            "f": "json",
            "sessionId": self._version._guid,
            "gdbVersion": self._version.properties.versionName,
            "changesInVersion": changes_in_version,
            "returnEdits": return_edits,
            "async": False,
        }
        if not area is None:
            params["evaluationArea"] = area
        if not selection is None:
            params["selection"] = selection

        if isinstance(evaluation, str) and evaluation.lower() in eval_lu:
            params["evaluationType"] = [eval_lu[evaluation.lower()]]
        elif isinstance(evaluation, str) and not evaluation.lower() in eval_lu:
            raise ValueError("Invalid evalution type")
        elif isinstance(evaluation, (list, tuple)):
            params["evaluationType"] = [eval_lu[e.lower()] for e in evaluation]
        else:
            raise ValueError("Invalid evalution type")

        return self._con.post(url, params)

from __future__ import annotations
import os
import json
import time
import uuid
import tempfile
from urllib.parse import urlparse
from typing import Optional, Union, Any
import pandas as pd
from arcgis.gis import GIS, Item
from requests.utils import quote
import xml.etree.ElementTree as ET
from .exceptions import ServerError
import requests

########################################################################


class SurveyManager:
    """
    Survey Manager allows users and administrators of Survey 123 to
    analyze, report on, and access the data for various surveys.

    """

    _baseurl = None
    _gis = None
    _portal = None
    _url = None
    _properties = None
    # ----------------------------------------------------------------------

    def __init__(self, gis, baseurl=None):
        """Constructor"""
        if baseurl is None:
            baseurl = "survey123.arcgis.com"
        self._baseurl = baseurl
        self._gis = gis

    # ----------------------------------------------------------------------
    def __str__(self):
        return "< SurveyManager @ {iid} >".format(iid=self._gis._url)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    @property
    def surveys(self) -> list:
        """returns a list of existing Survey"""
        query = (
            'type:"Form" AND NOT tags:"noxlsform"'
            'AND NOT tags:"draft" AND NOT typekeyw'
            "ords:draft AND owner:{owner}"
        ).format(owner=self._gis.users.me.username)
        content = self._gis.content
        items = content.search(
            query=query,
            item_type=None,
            sort_field="avgRating",
            sort_order="desc",
            max_items=10000,
            outside_org=False,
            categories=None,
            category_filters=None,
        )
        return [Survey(item=i, sm=self) for i in items]

    # ----------------------------------------------------------------------
    def get(self, survey_id: Union[Item, str]):
        """returns a single :class:`~arcgis.apps.survey123.Survey` object from and Item ID or Item"""
        if isinstance(survey_id, Item):
            survey_id = survey_id.id
        item = self._gis.content.get(survey_id)
        return Survey(item=item, sm=self)

    # ----------------------------------------------------------------------
    def _xform2webform(xform, portalUrl, connectVersion=None):
        """Converts a XForm XML to Enketo Web form by Enketo Transformer"""
        (dir_path, file_name) = os.path.split(xform)
        xlsx_name = os.path.splitext(file_name)[0]

        # xform_tree = ET.parse(xform)
        # root = xform_tree.getroot()
        # xform_string = ET.tostring(root, encoding='utf8', method='xml')

        with open(xform, "r", encoding="utf-8") as intext:
            xform_string = intext.read()

        url = "https://survey123.arcgis.com/api/xform2webform"
        params = {"xform": xform_string}
        if connectVersion:
            params["connectVersion"] = connectVersion
        try:
            r = requests.post(url, params)
            response_json = r.json()
            r.close()
        except requests.exceptions.ConnectionError as c:
            return "Unable to complete request with message: " + str(c)
        except requests.exceptions.Timeout as t:
            return "Connection timed out: " + str(t)

        else:
            with open(
                os.path.join(dir_path, xlsx_name + ".webform"), "w", encoding="utf-8"
            ) as fp:
                # with open(os.path.join(dir_path, xlsx_name + ".webform"), 'w') as fp:
                response_json["surveyFormJson"]["portalUrl"] = portalUrl
                webform = {
                    "form": response_json["form"],
                    "languageMap": response_json["languageMap"],
                    "model": response_json["model"],
                    "success": response_json["success"],
                    "surveyFormJson": response_json["surveyFormJson"],
                    "transformerVersion": response_json["transformerVersion"],
                }

                fp.write(json.dumps(webform, indent=2))
                # fp.write(json.dumps(response_json, indent=2))
                # fp.close()
            return os.path.join(dir_path, xlsx_name + ".webform")

    # ----------------------------------------------------------------------
    def _xls2xform(self, file_path: str):
        """
        Converts a XLSForm spreadsheet to XForm XML. The spreadsheet must be in Excel XLS(X) format

        ============   ================================================
        *Inputs*       *Description*
        ------------   ------------------------------------------------
        file_path      Required String. Path to the XLS(X) file.
        ============   ================================================

        :returns: dict

        """

        url = "https://{base}/api/xls2xform".format(base=self._baseurl)
        params = {"f": "json"}
        file = {"xlsform": file_path}
        isinstance(self._gis, GIS)
        return self._gis._con.post(
            path=url, postdata=params, files=file, verify_cert=False
        )

    # ----------------------------------------------------------------------
    def _create(
        self,
        project_name: str,
        survey_item: Item,
        summary: str = None,
        tags: str = None,
    ) -> bool:
        """TODO: implement create survery from xls"""
        # XLS Item or File Path
        # https://survey123.arcgis.com/api/xls2xform
        ##Content-Disposition: form-data; name="xlsform"; filename="Form_2.xlsx"
        ##Content-Type: application/octet-stream
        # Create Folder
        # Create Feature Service
        # Update Feature layer and tables
        # Enable editor tracking
        # Update capabilities
        # Create web form
        # Create form item
        # Refresh ?
        return


########################################################################
class Survey:
    """
    A `Survey` is a single instance of a survey project. This class contains
    the :class:`~arcgis.gis.Item` information and properties to access the underlying dataset
    that was generated by the `Survey` form.

    Data can be exported to `Pandas DataFrames`, `shapefiles`, `CSV`, and
    `File Geodatabases`.

    In addition to exporting data to various formats, a `Survey's` data can
    be exported as reports.

    """

    _gis = None
    _sm = None
    _si = None
    _ssi = None
    _baseurl = None
    # ----------------------------------------------------------------------

    def __init__(self, item, sm, baseurl: Optional[str] = None):
        """Constructor"""
        if baseurl is None:
            baseurl = "survey123.arcgis.com"
        self._si = item
        self._gis = item._gis
        self._sm = sm
        try:
            self.layer_name = self._find_layer_name()
        except:
            self.layer_name = None
        self._baseurl = baseurl

        sd = self._si.related_items("Survey2Data", direction="forward")
        if len(sd) > 0:
            for item in sd:
                if "StakeholderView" in item.typeKeywords:
                    self._stk = item
                    _stk_layers = self._stk.layers + self._stk.tables
                    _idx = 0
                    if self.layer_name:
                        for layer in _stk_layers:
                            if layer.properties["name"] == self.layer_name:
                                _idx = layer.properties["id"]
                    self._stk_url = self._stk.url + f"/{str(_idx)}"

        related = self._si.related_items("Survey2Service", direction="forward")
        if len(related) > 0:
            self._ssi = related[0]
            self._ssi_layers = self._ssi.layers + self._ssi.tables

            ssi_layer = None
            if self.layer_name:
                for layer in self._ssi_layers:
                    if layer.properties["name"] == self.layer_name:
                        _idx = layer.properties["id"]
                        ssi_layer = layer
                        break
            if not ssi_layer:
                ssi_layer = self._ssi_layers[0]
                _idx = ssi_layer.properties["id"]
            self._ssi_url = ssi_layer._url
            try:
                if self._ssi_layers[0].properties["isView"] == True:
                    view_url = ssi_layer._url[:-1]
                    self.parent_fl_url = self._find_parent(view_url) + f"/{str(_idx)}"
            except KeyError:
                self.parent_fl_url = ssi_layer._url

    # ----------------------------------------------------------------------
    @property
    def properties(self):
        """returns the properties of the survey"""
        return dict(self._si)

    # ----------------------------------------------------------------------
    def __str__(self):
        return "<Survey @ {iid}>".format(iid=self._si.title)

    # ----------------------------------------------------------------------
    def __repr__(self):
        return self.__str__()

    # ----------------------------------------------------------------------
    def download(
        self, export_format: str, save_folder: Optional[str] = None
    ) -> Union[str, pd.Dataframe]:
        """
        Exports the Survey's data to other format

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        export_format     Required String. This is the acceptable export format that a
                          user can export the survey data to. The following formats are
                          acceptable: File Geodatabase, Shapefile, CSV, and DF.
        ----------------  ---------------------------------------------------------------
        save_folder       Optional String. Specify the folder location where the output file should be stored.
        ================  ===============================================================

        :Returns: String or DataFrame
        """

        title = "a%s" % uuid.uuid4().hex
        if export_format.lower() == "df":
            return self._ssi.layers[0].query().sdf
        if save_folder is None:
            save_folder = tempfile.gettempdir()
        isinstance(self._ssi, Item)
        eitem = self._ssi.export(
            title=title,
            export_format=export_format,
        )
        save_file = eitem.download(save_path=save_folder)
        eitem.delete(force=True)
        return save_file

    # ----------------------------------------------------------------------
    def generate_report(
        self,
        report_template: Item,
        where: str = "1=1",
        utc_offset: str = "+00:00",
        report_title: Optional[str] = None,
        package_name: Optional[str] = None,
        output_format: str = "docx",
        folder_id: Optional[str] = None,
        merge_files: Optional[str] = None,
        survey_item: Optional[Item] = None,
        webmap_item: Optional[Item] = None,
        map_scale: Optional[float] = None,
        locale: str = "en",
        save_folder: Optional[str] = None,
    ) -> str:
        """
        The `generate_report` method allows users to create Microsoft Word and PDF reports
        from a survey using a report template. Reports are saved as an :class:`~arcgis.gis.Item` in an ArcGIS
        content folder or saved locally on disk. For additional information on parameters,
        see `Create Report <https://developers.arcgis.com/survey123/api-reference/rest/report/#create-report>`.

        .. note::
            The Survey123 report service may output one or more `.docx` or `.pdf` files, or a zipped
            package of these files. Whether the output is contained in a `.zip` file depends
            on the number of files generated and their size. For more information, see the
            `packageFiles` parameter in the `Create Report <https://developers.arcgis.com/survey123/api-reference/rest/report/#request-parameters-3>`_ documentation.

        .. note::
            To save to disk, do not specify a `folder_id` argument.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        report_template   Required :class:`~arcgis.gis.Item`. The report template.
        ----------------  ---------------------------------------------------------------
        where             Optional String. The select statement issued on survey
                          :class:`~arcgis.features.FeatureLayer` to report on
                          all survey records or a subset.

                          Query the `parent_fl_url` property of the
                          :class:`~arcgis.apps.survey123.Survey` object to get the
                          feature layer URL and retrieve a list of fields.

                          .. code-block:: python

                              >>> gis = GIS(profile="your_profile")
                              >>> smgr = SurveyManager(gis)

                              >>> survey_item = gis.content.get("<survey form id>")
                              >>> survey_obj = smgr.get(survey_item.id)

                              >>> survey_fl = FeatureLayer(survey_obj.parent_fl_url, gis)

                              >>> print([f["name"] for f in survey_fl.properties.fields])
        ----------------  ---------------------------------------------------------------
        utc_offset        Optional String.  Time offset from UTC. This offset is applied to
                          all `date`, `time`, and `dateTime` questions that appear in the report output.
                          Example: EST - "+04:00"
        ----------------  ---------------------------------------------------------------
        report_title      Optional String. If `folder_id` is provided, the result is an
                          :class:`~arcgis.gis.Item` with this argument as the title. If
                          `save_folder` argument is provided, this argument will be the
                          name of the output file, or the base name for files
                          in the output zipped package if the server-side component
                          chose to zip up the output (depends upon the size and number
                          of files that would result).


                          .. note::
                              If `merge_files` is either `nextPage` or `continuous`,
                              `report_title` is the output file name.
        ----------------  ---------------------------------------------------------------
        package_name      Optional String. Specify the file name (without extension) of the
                          packaged `.zip` file. If multiple files are packaged, the `report_title`
                          argument will be used to name individual files in the package.


                          .. note::
                            The Survey123 report service automatically decides whether to package
                            generated reports as a `.zip` file, depending on the output file count.
                            See the `packageFiles` parameter description in the `Create Report Request parameters <https://developers.arcgis.com/survey123/api-reference/rest/report/#request-parameters-3>`_
                            documentation for details.
        ----------------  ---------------------------------------------------------------
        save_folder       Optional String. Specify the folder location where the output
                          file or zipped file should be stored. If `folder_id` argument
                          is provided, this argument is ignored.
        ----------------  ---------------------------------------------------------------
        output_format     Optional String. Accepts `docx` or `pdf`.
        ----------------  ---------------------------------------------------------------
        folder_id         Optional String. If a file :class:`~arcgis.gis.Item` is the
                          desired output, specify the `id` value of the ArcGIS content
                          folder.
        ----------------  ---------------------------------------------------------------
        merge_files       Optional String. Specify if output is a single file containing individual
                          records on multiple pages (`nextPage` or `continuous`) or
                          multiple files (`none`).

                          + `none` - Print multiple records in split mode. Each record
                            is a separate file. This is the default value.
                          + `nextPage` - Print multiple records in a single document.
                            Each record starts on a new page.
                          + `continuous` - Print multiple records in a single document.
                            EAch records starts on the same page of the previous record.

                          .. note::
                              A merged file larger than 500 MB will be split into multiple
                              files.
        ----------------  ---------------------------------------------------------------
        survey_item       Optional survey :class:`~arcgis.gis.Item` to provide
                          additional information on survey structure.
        ----------------  ---------------------------------------------------------------
        webmap_item       Optional web map :class:`~arcgis.gis.Item`. Specify the basemap for all
                          map questions in the report. This takes precedence over the map set for
                          each question in the report template.
        ----------------  ---------------------------------------------------------------
        map_scale         Optional Float. Specify the map scale for all map questions in the report.
                          The map will center on the feature geometry. This takes precedence over the
                          scale set for each question in the report template.
        ----------------  ---------------------------------------------------------------
        locale            Optional String. Specify the locale to format number
                          and date values.
        ================  ===============================================================

        :Returns:
            An :class:`~arcgis.gis.Item` or string upon completion of the reporting
            `job <https://developers.arcgis.com/survey123/api-reference/rest/report/#jobs>`_.
            For details on the returned value, see `Response Parameters <https://developers.arcgis.com/survey123/api-reference/rest/report/#response-parameters>`_
            for the :func:`~arcgis.apps.survey123.Survey.generate_report` job.

        .. code-block:: python

            # Usage example #1: output a PDF file Item:
            >>> from arcgis.gis import GIS
            >>> from arcgis.apps.survey123 import SurveyManager

            >>> gis = GIS(profile="your_profile_name")

            >>> # Get report template and survey items
            >>> report_templ = gis.content.get("<template item id>")
            >>> svy_item = gis.content.get("<survey item id>")

            >>> svy_mgr = SurveyManager(gis)
            >>> svy_obj = svy_mgr.get(svy_item.id)

            >>> user_folder_id = [f["id"]
                                 for f in gis.users.me.folders
                                 if f["title"] == "folder_title"][0]

            >>> report_item = svy_obj.generate_report(report_template=report_templ,
                                                      report_title="Title of Report Item",
                                                      output_format="pdf",
                                                      folder_id=user_folder_id,
                                                      merge_files="continuous")

           # Usage example #2: output a Microsoft Word document named `LessThan20_Report.docx`

           >>> report_file = svy_obj.generate_report(report_template=report_templ,
                                                     where="objectid < 20",
                                                     report_title="LessThan20_Report",
                                                     output_format="docx",
                                                     save_folder="file\system\directory\",
                                                     merge_files="nextPage")

           # Usage example #3: output a zip file named `api_gen_report_pkg.zip` of individual
           #                   pdf files with a base name of `SpecimensOver30`

           >>> report_file = svy_obj.generate_report(report_template=report_templ,
                                                     where="number_specimens>30",
                                                     report_title="SpecimensOver30",
                                                     output_format="pdf",
                                                     save_folder="file\system\directory",
                                                     package_name="api_gen_report_pkg")

        """
        if isinstance(where, str):
            where = {"where": where}

        url = "https://{base}/api/featureReport/createReport/submitJob".format(
            base=self._baseurl
        )

        try:
            if (
                self._si._gis.users.me.username == self._si.owner
                and self._ssi_layers[0].properties["isView"] == True
            ):
                fl_url = self.parent_fl_url
            elif self._si._gis.users.me.username != self._si.owner:
                fl_url = self._stk_url
        except KeyError:
            if self._si._gis.users.me.username != self._si.owner:
                fl_url = self._stk_url
            else:
                fl_url = self._ssi_url

        params = {
            "outputFormat": output_format,
            "queryParameters": where,
            "portalUrl": self._si._gis._url,
            "templateItemId": report_template.id,
            "outputReportName": report_title,
            "outputPackageName": package_name,
            "surveyItemId": self._si.id,
            "featureLayerUrl": fl_url,
            "utcOffset": utc_offset,
            "uploadInfo": json.dumps(None),
            "f": "json",
            "username": self._si._gis.users.me.username,
            "locale": locale,
        }
        if merge_files:
            params["mergeFiles"] = merge_files
        if map_scale and isinstance(map_scale, (int, float)):
            params["mapScale"] = map_scale
        if webmap_item and isinstance(webmap_item, Item):
            params["webmapItemId"] = webmap_item.itemid
        if survey_item and isinstance(survey_item, Item):
            params["surveyItemId"] = survey_item.itemid
        if merge_files == "nextPage" or merge_files == "continuous":
            params["package_name"] = ""
        if folder_id:
            params["uploadInfo"] = json.dumps(
                {
                    "type": "arcgis",
                    "packageFiles": True,
                    "parameters": {"folderId": folder_id},
                }
            )
        # 1). Submit the request.
        submit = self._si._gis._con.post(
            url, params, add_headers={"X-Survey123-Request-Source": "API/Python"}
        )
        return self._check_status(
            res=submit, status_type="generate_report", save_folder=save_folder
        )

    # ----------------------------------------------------------------------
    @property
    def report_templates(self) -> list:
        """
        Returns a list of saved report items

        :returns: list of :class:`Items <arcgis.gis.Item>`
        """
        related_items = self._si.related_items(
            direction="forward", rel_type="Survey2Data"
        )
        report_templates = [t for t in related_items if t.type == "Microsoft Word"]

        return report_templates

    @property
    def reports(self) -> list:
        """returns a list of generated reports"""
        return self._si._gis.content.search(
            'owner: %s AND type:"Microsoft Word" AND tags:"Survey 123"'
            % self._ssi._gis.users.me.username,
            max_items=10000,
            outside_org=False,
        )

    # ----------------------------------------------------------------------
    def create_report_template(
        self,
        template_type: Optional[str] = "individual",
        template_name: Optional[str] = None,
        save_folder: Optional[str] = None,
    ):
        """
        The `create_report_template` creates a simple default template that
        can be downloaded locally, edited and uploaded back up as a report
        template.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        template_type     Optional String. Specify which sections to include in the template.
                          Acceptable types are `individual`, `summary`, and `summaryIndividual`.
                          Default is `individual`.
        ----------------  ---------------------------------------------------------------
        template_name     Optional String. Specify the name of the output template file without file extension.
        ----------------  ---------------------------------------------------------------
        save_folder       Optional String. Specify the folder location where the output file should be stored.
        ================  ===============================================================

        :returns: String
        """
        if self._si._gis.users.me.username != self._si.owner:
            raise TypeError("Stakeholders cannot create report templates")
        try:
            if self._ssi_layers[0].properties["isView"] == True:
                fl_url = self.parent_fl_url
        except KeyError:
            fl_url = self._ssi_url

        if template_name:
            file_name = f"{template_name}.docx"
        else:
            if template_type == "individual":
                type = "Individual"
            elif template_type == "summary":
                type = "Summary"
            elif template_type == "summaryIndividual":
                type = "SummaryIndividual"
            file_name = f"{self._si.title}_sampleTemplate{type}.docx"

        url = "https://{base}/api/featureReport/createSampleTemplate".format(
            base=self._baseurl
        )
        gis = self._si._gis
        params = {
            "featureLayerUrl": fl_url,
            "surveyItemId": self._si.id,
            "portalUrl": gis._url,
            "contentType": template_type,
            "username": gis.users.me.username,
            "f": "json",
        }

        res = gis._con.post(
            url,
            params,
            try_json=False,
            out_folder=save_folder,
            file_name=file_name,
            add_headers={"X-Survey123-Request-Source": "API/Python"},
        )
        return res

    # ----------------------------------------------------------------------

    def check_template_syntax(self, template_file: Optional[str] = None):
        """
        A sync operation to check any syntax which will lead to a failure
        when generating reports in the given feature.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        template_file     Required String. The report template file which syntax to be checked.
        ================  ===============================================================

        :returns: dictionary {Success or Failure}
        """

        if self._si._gis.users.me.username != self._si.owner:
            raise TypeError("Stakeholders cannot create report templates")

        try:
            if self._ssi_layers[0].properties["isView"] == True:
                fl_url = self.parent_fl_url
        except KeyError:
            fl_url = self._ssi_url

        url = "https://{base}/api/featureReport/checkTemplateSyntax".format(
            base=self._baseurl
        )
        file = {
            "templateFile": (os.path.basename(template_file), open(template_file, "rb"))
        }
        gis = self._si._gis
        params = {
            "featureLayerUrl": fl_url,
            "surveyItemId": self._si.id,
            "portalUrl": self._si._gis._url,
            "f": "json",
        }

        check = gis._con.post(
            url,
            params,
            files=file,
            add_headers={"X-Survey123-Request-Source": "API/Python"},
        )
        return check

    # ----------------------------------------------------------------------

    def upload_report_template(
        self, template_file: Optional[str] = None, template_name: Optional[str] = None
    ):
        """
        Check report template syntax to identify any syntax which will lead to a failure
        when generating reports in the given feature. Uploads the report to the organization
        and associates it with the survey.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        template_file     Required String. The report template file which syntax to be checked, and uploaded.
        ----------------  ---------------------------------------------------------------
        template_name     Optional String. If provided the resulting item will use the provided name, otherwise
                          the name of the docx file will be used.
        ================  ===============================================================

        :returns: item {Success) or string (Failure}
        """

        check = self.check_template_syntax(template_file)

        if check["success"] == True:
            if template_name:
                file_name = template_name
            else:
                file_name = os.path.splitext(os.path.basename(template_file))[0]

            properties = {
                "title": file_name,
                "type": "Microsoft Word",
                "tags": "Survey123,Print Template,Feature Report Template",
                "typeKeywords": "Survey123,Survey123 Hub,Print Template,Feature Report Template",
                "snippet": "Report template",
            }
            survey_folder_id = self._si.ownerFolder
            gis = self._si._gis
            user = gis.users.get(gis.properties.user.username)
            user_folders = user.folders
            survey_folder = next(
                (f for f in user_folders if f["id"] == survey_folder_id), 0
            )
            folder = survey_folder["title"]
            # folder = "Survey-" + self._si.title
            template_item = gis.content.add(
                item_properties=properties, data=template_file, folder=folder
            )
            add_relationship = self._si.add_relationship(template_item, "Survey2Data")
        else:
            return check["details"][0]["description"]

        return template_item

    # ----------------------------------------------------------------------

    def update_report_template(self, template_file: Optional[str] = None):
        """
        Check report template syntax to identify any syntax which will lead to a failure
        when generating reports in the given feature and updates existing Report template Org item.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        template_file     Required String. The report template file which syntax to be checked, and uploaded.
                          The updated template name must match the name of the existing template item.
        ================  ===============================================================

        :returns: item {Success) or string (Failure}
        """

        check = self.check_template_syntax(template_file)

        if check["success"] == True:
            file_name = os.path.splitext(os.path.basename(template_file))[0]
            gis = self._si._gis
            template_item = gis.content.search(
                query="title:" + file_name, item_type="Microsoft Word"
            )
            update = template_item[0].update(item_properties={}, data=template_file)
        else:
            return check["details"][0]["description"]

        return template_item

    # ----------------------------------------------------------------------

    def estimate(self, report_template: Item, where: str = "1=1"):
        """
        An operation to estimate how many credits are required for a task
        with the given parameters.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        report_template   Required :class:`~arcgis.gis.Item` .  The report template Item.
        ----------------  ---------------------------------------------------------------
        where             Optional String. This is the select statement used to export
                          part or whole of the dataset. If the filtered result has more
                          than one feature/record, the request will be considered as a
                          batch printing. Currently, one individual report will be
                          generated for each feature/record.
        ================  ===============================================================

        :returns: dictionary {totalRecords, cost(in credits)}
        """
        try:
            if (
                self._si._gis.users.me.username == self._si.owner
                and self._ssi_layers[0].properties["isView"] == True
            ):
                fl_url = self.parent_fl_url
            elif self._si._gis.users.me.username != self._si.owner:
                fl_url = self._stk_url
        except KeyError:
            if self._si._gis.users.me.username != self._si.owner:
                fl_url = self._stk_url
            else:
                fl_url = self._ssi_url

        gis = self._si._gis
        if isinstance(where, str):
            where = {"where": where}

        url = "https://{base}/api/featureReport/estimateCredits".format(
            base=self._baseurl
        )
        params = {
            "featureLayerUrl": fl_url,
            "queryParameters": where,
            "templateItemId": report_template.id,
            "surveyItemId": self._si.id,
            "portalUrl": self._si._gis._url,
            "f": "json",
        }

        estimate = gis._con.get(
            url, params, add_headers={"X-Survey123-Request-Source": "API/Python"}
        )
        return estimate

    # ----------------------------------------------------------------------

    def create_sample_report(
        self,
        report_template: Item,
        where: str = "1=1",
        utc_offset: str = "+00:00",
        report_title: Optional[str] = None,
        merge_files: Optional[str] = None,
        survey_item: Optional[Item] = None,
        webmap_item: Optional[Item] = None,
        map_scale: Optional[float] = None,
        locale: str = "en",
        save_folder: Optional[str] = None,
    ) -> str:
        """
        Similar task to generate_report for creating test sample report, and refining
        a report template before generating any formal report.

        ================  ===============================================================
        **Parameter**      **Description**
        ----------------  ---------------------------------------------------------------
        report_template   Required :class:`~arcgis.gis.Item`. The report template Item.
        ----------------  ---------------------------------------------------------------
        where             Optional String. This is the select statement used to export
                          part or whole of the dataset.  If the record count is > 1, then
                          the item must be saved to your organization.
        ----------------  ---------------------------------------------------------------
        utc_offset        Optional String.  This is the time offset from UTC to match the
                          users timezone. Example: EST - "+04:00"
        ----------------  ---------------------------------------------------------------
        report_title      Optional String. An :class:`~arcgis.gis.Item` with this argument
                          as the title if no `save_folder` argument. If `save_folder`
                          argument is provided, this argument will be the name of the
                          output file, or the base name for files in the output zipped
                          package if the server-side component chose to zip up the output
                          (depends upon the size and number of files that would result).

                          .. note::
                              If `merge_files` is either `nextPage` or `continuous`,
                              `report_title` is the output file name.
        ----------------  ---------------------------------------------------------------
        merge_files       Optional String. Specify if output is a single file containing individual
                          records on multiple pages (`nextPage` or `continuous`) or
                          multiple files (`none`).

                          + `none` - Print multiple records in split mode. Each record
                            is a separate file. This is the default value.
                          + `nextPage` - Print multiple records in a single document.
                            Each record starts on a new page.
                          + `continuous` - Print multiple records in a single document.
                            EAch records starts on the same page of the previous record.

                          .. note::
                              A merged file larger than 500 MB will be split into multiple
                              files.
        ----------------  ---------------------------------------------------------------
        save_folder       Optional String. Specify the folder location where the output
                          file should be stored.
        ----------------  ---------------------------------------------------------------
        survey_item       Optional survey :class:`~arcgis.gis.Item` to provide additional
                          information on the survey structure.
        ----------------  ---------------------------------------------------------------
        webmap_item       Optional :class:`~arcgis.gis.Item` . Specify the base map for printing task when printing
                          a point/polyline/polygon. This takes precedence over the map set for
                          each question inside a survey.
        ----------------  ---------------------------------------------------------------
        map_scale         Optional Float. Specify the map scale when printing, the map will center on the feature geometry.
        ----------------  ---------------------------------------------------------------
        locale            Optional String. Specify the locale setting to format number and date values.
        ================  ===============================================================

        :Returns: String

        """
        try:
            if (
                self._si._gis.users.me.username == self._si.owner
                and self._ssi_layers[0].properties["isView"] == True
            ):
                fl_url = self.parent_fl_url
            elif self._si._gis.users.me.username != self._si.owner:
                fl_url = self._stk_url
        except KeyError:
            if self._si._gis.users.me.username != self._si.owner:
                fl_url = self._stk_url
            else:
                fl_url = self._ssi_url

        if isinstance(where, str):
            where = {"where": where}

        url = "https://{base}/api/featureReport/createSampleReport/submitJob".format(
            base=self._baseurl
        )

        params = {
            "queryParameters": where,
            "portalUrl": self._si._gis._url,
            "templateItemId": report_template.id,
            "surveyItemId": self._si.id,
            "featureLayerUrl": fl_url,
            "utcOffset": utc_offset,
            "f": "json",
            "locale": locale,
        }
        if merge_files:
            params["mergeFiles"] = merge_files
        if map_scale and isinstance(map_scale, (int, float)):
            params["mapScale"] = map_scale
        if webmap_item and isinstance(webmap_item, Item):
            params["webmapItemId"] = webmap_item.itemid
        if survey_item and isinstance(survey_item, Item):
            params["surveyItemId"] = survey_item.itemid
        if merge_files == "nextPage" or merge_files == "continuous":
            params["package_name"] = ""
        if report_title:
            params["outputReportName"] = report_title

        # 1). Submit the request.
        submit = self._si._gis._con.post(
            url, params, add_headers={"X-Survey123-Request-Source": "API/Python"}
        )
        return self._check_status(
            res=submit, status_type="generate_report", save_folder=save_folder
        )

    # ----------------------------------------------------------------------

    def _check_status(self, res, status_type, save_folder):
        """checks the status of a Survey123 operation"""
        jid = res["jobId"]
        gis = self._si._gis
        params = {
            "f": "json",
            "username": self._si._gis.users.me.username,
            "portalUrl": self._si._gis._url,
        }
        status_url = "https://{base}/api/featureReport/jobs/{jid}/status".format(
            base=self._baseurl, jid=jid
        )
        # 3). Start Checking the status
        res = gis._con.get(
            status_url,
            params=params,
            add_headers={"X-Survey123-Request-Source": "API/Python"},
        )
        while res["jobStatus"] == "esriJobExecuting":
            res = self._si._gis._con.get(
                status_url,
                params=params,
                add_headers={"X-Survey123-Request-Source": "API/Python"},
            )
            time.sleep(1)
        if status_type == "default_report_template":
            if (
                "results" in res
                and "details" in res["results"]
                and "resultFile" in res["results"]["details"]
                and "url" in res["results"]["details"]["resultFile"]
            ):
                url = res["results"]["details"]["resultFile"]["url"]
                file_name = os.path.basename(url)
                return gis._con.get(url, file_name=file_name, out_folder=save_folder)
            return res
        elif status_type == "generate_report":
            urls = []
            files = []
            items = []
            if res["jobStatus"] == "esriJobSucceeded":
                if "resultFiles" in res["resultInfo"]:
                    for sub in res["resultInfo"]["resultFiles"]:
                        if "id" in sub:
                            items.append(sub["id"])
                        elif "url" in sub:
                            urls.append(sub["url"])
                    files = [
                        self._si._gis._con.get(
                            url,
                            file_name=os.path.basename(urlparse(url).path),
                            add_token=False,
                            try_json=False,
                            out_folder=save_folder,
                        )
                        for url in urls
                    ] + [gis.content.get(i) for i in items]
                    if len(files) == 1:
                        return files[0]
                    return files
                elif "details" in res["resultInfo"]:
                    for res in res["resultInfo"]["details"]:
                        if "resultFile" in res:
                            fr = res["resultFile"]
                            if "id" in fr:
                                items.append(fr["id"])
                            else:
                                urls.append(fr["url"])
                        del res

                    files = [
                        self._si._gis._con.get(
                            url, file_name=os.path.basename(url), out_folder=save_folder
                        )
                        for url in urls
                    ] + [gis.content.get(i) for i in items]
                    if len(files) == 1:
                        return files[0]
                    else:
                        return files
            elif (
                res["jobStatus"] == "esriJobPartialSucceeded"
                or res["jobStatus"] == "esriJobFailed"
            ):
                raise ServerError(res["messages"][0])
            # return

    # ----------------------------------------------------------------------
    def _find_parent(self, view_url):
        """Finds the parent feature layer for a feature layer view"""
        url = view_url + "sources"
        response = self._si._gis._con.get(url)
        return response["services"][0]["url"]

    # ----------------------------------------------------------------------
    def _find_layer_name(self):
        """Finds the name of the layer the survey is submitting to, used to find the appropriate layer index"""
        name = self._si._gis._con.get(
            f"{self._gis._url}/sharing/rest/content/items/{self._si.id}/info/forminfo.json"
        )["name"]
        title = quote(name, safe="()!-_.'~")
        url = f"{self._gis._url}/sharing/rest/content/items/{self._si.id}/info/{title}.xml"
        response = self._si._gis._con.get(url, out_folder=tempfile.gettempdir())
        tree = ET.parse(response)
        root = tree.getroot()
        for elem in root[0][1].iter():
            for key, value in zip(elem.attrib.keys(), elem.attrib.values()):
                if key == "id":
                    return value

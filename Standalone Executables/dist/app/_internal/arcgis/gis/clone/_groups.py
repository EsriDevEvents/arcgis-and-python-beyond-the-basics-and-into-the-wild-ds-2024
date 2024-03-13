from __future__ import annotations
import os
import uuid
import json
import shutil
import logging
import tempfile
import warnings
import concurrent.futures
from typing import Any
from arcgis.gis import GIS, Group, GroupManager
from arcgis.auth.tools import LazyLoader
from functools import lru_cache

from arcgis.gis.clone._base import BaseCloneGroup
from arcgis.gis.clone._utils import _zipdir

_arcgis = LazyLoader("arcgis")

_logger = logging.getLogger()


###########################################################################
class CloningJob:
    """
    A Single Group Cloning Job

    This class should not be created by users.
    """

    _future: concurrent.futures.Future
    _task: str

    # ---------------------------------------------------------------------
    def __init__(self, future: concurrent.futures.Future, task: str = "Group") -> None:
        self._future = future
        self._task = task

    # ---------------------------------------------------------------------
    def __str__(self) -> str:
        return f"< {self._task} Cloning Job: {self.status} >"

    # ---------------------------------------------------------------------
    def __repr__(self) -> str:
        return self.__str__()

    # ---------------------------------------------------------------------
    @property
    def status(self) -> bool:
        """checks if the job completed"""
        return self._future.done()

    # ---------------------------------------------------------------------
    def cancel(self) -> bool:
        """checks if the job completed"""
        return self._future.cancel()

    # ---------------------------------------------------------------------
    def result(self) -> Group:
        """returns a group"""
        return self._future.result()

    # ---------------------------------------------------------------------
    @property
    def running(self) -> bool:
        """checks if the job was cancelled"""
        return self._future.running()

    # ---------------------------------------------------------------------
    @property
    def cancelled(self) -> bool:
        """checks if the job was cancelled"""
        return self._future.cancelled()


###########################################################################
class GroupCloner(BaseCloneGroup):
    """
    The `GroupCloner` allows users to copy Groups from one organization to another.

    Once the groups are copied, the `clone_items` or Group Migration Manager can be
    used to add the data.

    """

    _group_source: Group = None
    _include_items: bool = None
    _gis: GIS = None
    _out_folder: str | None = None
    _tracker: dict[str, Any] | None = None
    _tp: concurrent.futures.ThreadPoolExecutor = None

    def __init__(
        self,
        *,
        gis: GIS | None = None,
    ) -> None:
        """initializer"""
        super()
        if gis:
            self._gis: GIS = gis
        elif _arcgis.env.active_env:
            self._gis = _arcgis.env.active_env
        else:
            raise ValueError("The `GIS` object is not defined.")

        self._tp = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    # ---------------------------------------------------------------------
    def __str__(self) -> str:
        return "< Group Cloner >"

    # ---------------------------------------------------------------------
    def __repr__(self) -> str:
        return "< Group Cloner >"

    # ---------------------------------------------------------------------
    @lru_cache(maxsize=100)
    def _setup_group_project_folder(
        self, group_id: str, out_folder: str | None = None
    ) -> str:
        """creates a group project folder"""
        if out_folder:
            os.makedirs(out_folder, exist_ok=True)
        elif out_folder is None:
            out_folder = os.path.join(tempfile.gettempdir(), group_id)
            os.makedirs(out_folder, exist_ok=True)
        self._out_folder = out_folder
        return out_folder

    #  --------------------------------------------------------------------
    def _create_cloner_file(
        self, save_folder: str, path: str, package_name: str | None = None
    ) -> str:
        """Creates the Zipped compressed .GROUP_CLONER file."""
        if package_name is None:
            package_name = f"{uuid.uuid4().hex[:5]}.GROUP_CLONER"
        elif package_name.endswith(".GROUP_CLONER") == False:
            package_name += ".GROUP_CLONER"

        save_fp: str = os.path.join(save_folder, package_name)
        _zipdir(src=path, dst=save_folder, zip_name=package_name)

        if os.path.isfile(save_fp):
            return save_fp
        else:
            raise Exception(f"Could not save the the file: {save_fp}")
        return save_fp

    # ---------------------------------------------------------------------
    def _check_if_exists(self, group: Group) -> bool:
        """Checks if the group exists on the destination org"""
        results = self._gis.groups.search(f"id: {group.id}")
        return len(results) >= 1

    def _create_group_definition(
        self, groups: list[Group], save_folder: str, file_name: str
    ) -> dict[str, Any]:
        """creates the offline definition of the group"""
        group_defs: list[str] = []
        working_folder: str = os.path.join(
            tempfile.gettempdir(), uuid.uuid4().hex[:5]
        )  #  this is where we store the information on disk
        os.makedirs(
            save_folder, exist_ok=True
        )  # this ensures the save location of the cloner file exists
        group_json_file: str = os.path.join(working_folder, "group_def.json")
        fp: str = os.path.join(save_folder, file_name)

        lu: dict[str, Any] = {
            '{"itemTypes": "Application"}': "apps",
            '{"itemTypes": ""}': "none",
            '{"itemTypes": "CSV"}': "files",
            '{"itemTypes": "Web Map"}': "maps",
            '{"itemTypes": "Layer"}': "layers",
            '{"itemTypes": "Web Scene"}': "scenes",
            '{"itemTypes": "Locator Package"}': "tools",
        }
        for group in groups:
            group_thumbnail_folder: str = os.path.join(working_folder, group.id)
            os.makedirs(group_thumbnail_folder, exist_ok=True)

            display_settings: str | None = None
            if json.dumps(group.displaySettings) in lu:
                display_settings = lu[json.dumps(group.displaySettings)]
            tags: list[str] = group.tags
            tags.append(f"cloned_source_id: {group.id}")
            thumbnail: str = self._get_thumbnail(
                group=group, save_folder=group_thumbnail_folder
            )
            if getattr(group, "membershipAccess", None) != "collaboration":
                group_settings: dict[str, Any] = {
                    "title": group.title,
                    "tags": tags,
                    "description": group.description,
                    "snippet": group.snippet,
                    "thumbnail": thumbnail,
                    "is_invitation_only": group.isInvitationOnly,
                    "sort_field": group.sortField,
                    "sort_order": group.sortOrder,
                    "is_view_only": group.isViewOnly,
                    "auto_join": group.autoJoin,
                    "display_settings": display_settings,
                    "leaving_disallowed": group.leavingDisallowed,
                    "membership_access": getattr(group, "membershipAccess", None),
                    "hidden_members": getattr(group, "hiddenMembers", None),
                    "autojoin": group.autoJoin,
                }
                group_defs.append(group_settings)
            else:
                warnings.warn(
                    f"Collaboration groups are not able to be copied. {group.title}"
                )
        with open(group_json_file, "w") as writer:
            json.dump(group_defs, writer)
        fp = self._create_cloner_file(
            path=working_folder,
            save_folder=save_folder,
            package_name=file_name,
        )
        return fp

    # ---------------------------------------------------------------------
    def _process_offline(
        self,
        groups: list[Group],
        save_folder: str | None = None,
        file_name: str | None = None,
        **kwargs,
    ) -> str:
        if save_folder is None:
            save_folder = tempfile.gettempdir()
        if file_name is None:
            file_name = "GROUPSETTINGS"
        fp: str = self._create_group_definition(
            groups=groups, save_folder=save_folder, file_name=file_name
        )
        return fp

    # ---------------------------------------------------------------------
    def _process_online(self, params) -> Group:
        """
        Runs the clone process on a single group entry
        """
        group: Group = params["group"]
        working_folder: str = params["working_folder"]
        mgr: GroupManager = params["gis"].groups
        lu = {
            '{"itemTypes": "Application"}': "apps",
            '{"itemTypes": ""}': None,
            '{"itemTypes": "CSV"}': "files",
            '{"itemTypes": "Web Map"}': "maps",
            '{"itemTypes": "Layer"}': "layers",
            '{"itemTypes": "Web Scene"}': "scenes",
            '{"itemTypes": "Locator Package"}': "tools",
        }
        display_settings = None
        if json.dumps(group.displaySettings) in lu:
            display_settings = lu[json.dumps(group.displaySettings)]
        tags: list[str] = group.tags
        tags.append(f"cloned_source_id: {group.id}")
        thumbnail: str = self._get_thumbnail(group=group, save_folder=working_folder)
        group = mgr.create(
            title=group.title,
            tags=group.tags,
            description=group.description,
            snippet=group.snippet,
            thumbnail=thumbnail,
            is_invitation_only=group.isInvitationOnly,
            sort_field=group.sortField,
            sort_order=group.sortOrder,
            is_view_only=group.isViewOnly,
            auto_join=group.autoJoin,
            display_settings=display_settings,
            leaving_disallowed=group.leavingDisallowed,
            membership_access=getattr(group, "membershipAccess", None),
            hidden_members=getattr(group, "hiddenMembers", None),
            autojoin=group.autoJoin,
        )
        return group

    # ---------------------------------------------------------------------
    def _get_thumbnail(self, group: Group, save_folder: str) -> str:
        """
        downloads the thumbnail

        :returns: str
        """
        return group.download_thumbnail(save_folder=save_folder)

    # ---------------------------------------------------------------------
    def _run_ops(self, parameters: dict[str, Any], offline: bool = False) -> CloningJob:
        """runs the job asynchronously."""
        jobs = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as tp:
            if offline:
                future: concurrent.futures.Future = tp.submit(
                    self._process_offline, **parameters
                )
                jobs.append(CloningJob(future=future, task="Offline Group Cloning"))
            else:
                for key, params in parameters.items():
                    future: concurrent.futures.Future = tp.submit(
                        self._process_online, **{"params": params}
                    )

                    jobs.append(CloningJob(future=future))
                    del key, params
                tp.shutdown(wait=True)
        return jobs

    # ---------------------------------------------------------------------
    def load_offline_configuration(
        self, package: str
    ) -> list[concurrent.futures.Future]:
        """
        Loads the Group configuration file into the current active portal.
        """
        groups = []
        workspace: str = os.path.join(
            tempfile.gettempdir(), f"group_config_{uuid.uuid4().hex[:5]}"
        )
        os.makedirs(workspace, exist_ok=True)
        shutil.unpack_archive(filename=package, extract_dir=workspace, format="zip")
        settings_configuration = []
        with open(os.path.join(workspace, "group_def.json"), "r") as reader:
            settings_configuration = json.load(reader)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as tp:
            for group in settings_configuration:
                if group["thumbnail"]:
                    thumbnail = os.path.join(
                        workspace,
                        os.path.basename(os.path.dirname(group["thumbnail"])),
                        os.path.basename(group["thumbnail"]),
                    )
                    group["thumbnail"] = thumbnail

                future = tp.submit(
                    self._load_offline_configuration,
                    **{
                        "payload": group,
                    },
                )
                groups.append(future)
            tp.shutdown(wait=True)
        return groups

    # ---------------------------------------------------------------------
    def _load_offline_configuration(self, payload: dict[str, Any]) -> Group | str:
        """
        Loads the Group configuration file into the current active portal.
        """
        gm = self._gis.groups
        try:
            return gm.create(**payload)
        except Exception as ex:
            return f"could not create the group {payload['title']} with error: {ex}"

    # ---------------------------------------------------------------------
    def clone(
        self,
        groups: list[Group],
        *,
        skip_existing: bool = True,
        offline: bool = False,
        save_folder: str | None = None,
        file_name: str = "group_list.json",
    ) -> list[CloningJob]:
        """
        Override the clone operation in order performs the cloning logic



        """
        self._tracker: dict[str, Any] = {}
        ## 1). Check Existance and setup project
        ##
        if offline:
            params = {
                "groups": groups,
                "skip_existing": skip_existing,
                "save_folder": save_folder,
                "file_name": file_name,
            }
            return self._run_ops(parameters=params, offline=True)
        else:
            for group in groups:
                if (
                    self._check_if_exists(group=group) == False and skip_existing
                ) or skip_existing == False:
                    self._tracker[group.id] = {
                        "group": group,
                        "gis": self._gis,
                        "exists": self._check_if_exists(group),
                        "working_folder": self._setup_group_project_folder(
                            group_id=group.id
                        ),
                    }
            return self._run_ops(parameters=self._tracker)

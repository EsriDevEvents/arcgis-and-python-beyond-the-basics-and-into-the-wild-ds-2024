from __future__ import annotations
import os
import uuid
import json
import shutil
import zipfile
import tempfile
import warnings
import concurrent.futures
from typing import Any

from arcgis._impl.common._utils import _date_handler
from arcgis.auth.tools import LazyLoader

_arcgis = LazyLoader("arcgis")
_arcgis_ux = LazyLoader("arcgis.gis.admin._ux")
__all__ = ["UXCloner"]


class UXCloner:
    """
    The `UXCloner` class facilitates the copying of UX settings from one WebGIS site to another,
    enabling administrators to maintain a consistent user experience across multiple sites.
    The cloning operations can be performed either online, through site-to-site transfer via HTTP,
    or offline, using a file package.

    ================    ===============================================================
    **Parameter**       **Description**
    ----------------    ---------------------------------------------------------------
    gis                 Required GIS. The source WebGIS.
    ================    ===============================================================

    """

    _source_gis: _arcgis.GIS | None = None
    _RESOURCE_IGNORE_LIST: list[str]
    _tp: concurrent.futures.ThreadPoolExecutor

    #  --------------------------------------------------------------------
    def __init__(self, gis: _arcgis.GIS | None):
        self._source_gis = gis
        self._RESOURCE_IGNORE_LIST = [
            "localizedOrgProperties",
            "travelmodes.json",
            "Survey123Properties",
        ]
        self._tp = concurrent.futures.ThreadPoolExecutor(max_workers=10)

    #  --------------------------------------------------------------------
    def clone(
        self,
        *,
        targets: list[_arcgis.gis.GIS] | None = None,
        workspace_folder: str | None = None,
        package_save_folder: str | None = None,
        package_name: str | None = None,
    ) -> list[concurrent.futures.Future]:
        """
        Copies the UX settings from the source WebGIS site to the destination WebGIS site or to
        a `.uxpk` offline file.
        When directly connected to two WebGIS', this method performs the clone operation immediately.
        When cloning in an offlien situation, a `.uxpk` is created and stored on the user's local
        hard drive.



        :returns: list[concurrent.futures.Future]

        """
        if targets is None:
            targets = [None]
        jobs: list[concurrent.futures.Future] = []
        for target_gis in targets:
            params = {
                "target_gis": target_gis,
                "out_folder": workspace_folder,
                "package_folder": package_save_folder,
                "package_name": package_name,
            }
            jobs.append(self._tp.submit(self._clone, **params))
            # jobs.append(self._clone(**params))
            del target_gis, params
        return jobs

    #  --------------------------------------------------------------------
    def _clone(
        self,
        *,
        target_gis: _arcgis.gis.GIS | None = None,
        out_folder: str | None = None,
        package_folder: str | None = None,
        package_name: str | None = None,
    ) -> concurrent.futures.Future:
        """clones the settings from site A to site B"""
        if target_gis is None:
            if package_folder is None:
                package_folder = os.path.join(
                    tempfile.gettempdir(), uuid.uuid4().hex[:6]
                )
                os.makedirs(package_folder, exist_ok=True)
            return self._create_offline_clone_package(
                out_folder=out_folder,
                package_name=package_name,
                package_save_location=package_folder,
            )
        else:
            return self._online_clone_workflow(
                target_gis=target_gis, out_folder=out_folder
            )

    #  --------------------------------------------------------------------
    def _setup_folder_workspaces(self, out_folder: str | None = None) -> tuple[str]:
        """
        Creates the output location of the data


        :returns: tuple[str] of the workspace folder.
        The root, HomePageSettings folder and ResourceManager Folder
        """
        if out_folder is None:
            out_folder = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex[:5])
            hp_folder = os.path.join(out_folder, "hp")
            resource_folder = os.path.join(out_folder, "resources")
            os.makedirs(out_folder, exist_ok=True)

        else:
            hp_folder = os.path.join(out_folder, "hp")
            resource_folder = os.path.join(out_folder, "resources")
            os.makedirs(out_folder, exist_ok=True)
        os.makedirs(hp_folder, exist_ok=True)
        os.makedirs(resource_folder, exist_ok=True)
        return out_folder, hp_folder, resource_folder

    #  --------------------------------------------------------------------
    def _create_configuration(
        self,
        offline: bool,
        out_folder: str,
        hp_folder: str,
        resource_folder: str,
    ) -> dict[str, Any]:
        """
        This creates the configuration object that holds all the UX configuration data.

        :returns: dict[str,Any]
        """
        source_ux: _arcgis_ux.UX = self._source_gis.admin.ux
        hp: _arcgis_ux.HomePageSettings = source_ux.homepage_settings
        mp: _arcgis_ux.MapSettings = source_ux.map_settings
        ip: _arcgis_ux.ItemSettings = source_ux.item_settings
        ss: _arcgis_ux.SecuritySettings = source_ux.security_settings
        pp: _arcgis.gis.admin.PasswordPolicy = self._source_gis.admin.password_policy
        cs: _arcgis.gis.admin.CategoryManager = self._source_gis.admin.category_schema

        settings_configuration: dict[str, Any] = {
            "offline": offline,
            "name": source_ux.name,
            "summary": source_ux.summary,
            "contact_link": source_ux.contact_link,
            "admin_contacts": source_ux.admin_contacts,
            "help_source": source_ux.help_source,
            "logo": source_ux.get_logo(out_folder),
            "description": source_ux.description,
            "description_visibility": source_ux.description_visibility,
            "HomePageSettings": {
                "resource": hp._portal_resources.get("home.page.json"),
                "resource_file_name": "home.page.json",
                "typography": hp.get_typography(),
                "background": hp.get_background(hp_folder),
                "logo": hp.get_logo(hp_folder),
            },
            "MapSettings": {
                "default_mapviewer": mp.default_mapviewer,
                "default_extent": mp.default_extent,
                "default_basemap": mp.default_basemap,
                "use_vector_basemap": mp.use_vector_basemap,
                "vector_basemap": mp.vector_basemap,
                "basemap_gallery_group": self._source_gis._get_properties(True).get(
                    "basemapGalleryGroupQuery", None
                ),
                "units": mp.units,
                "config_apps_group": self._source_gis._get_properties(True).get(
                    "templatesGroupQuery", None
                ),  # templatesGroupQuery
                "analysis_layer_group": self._source_gis._get_properties(True).get(
                    "analysisLayersGroupQuery", None
                ),
            },
            "ItemSettings": {
                "comments": ip.enable_comments,
                "enable_metadata_edit": ip.enable_metadata_edit,
                "metadata_format": ip.metadata_format,
            },
            "SecuritySettings": {
                "allowed_origins": ",".join(ss.allowed_origins),  #
                "allowed_redirect_uris": ss.allowed_redirect_uris,  #
                "enable_https": ss.enable_https,  #
                "anonymous_access": ss.anonymous_access,  #
                "enable_update_user_profile": ss.enable_update_user_profile,  #
                "share_public": ss.share_public,  #
                "can_sign_in_social": dict(self._source_gis.properties).get(
                    "canSignInSocial", False
                ),
                "show_social_media": ss.show_social_media,  #
                "signin_settings": ss.signin_settings,  #
                "trusted_servers": ss.trusted_servers,  #
                "informational_banner": ss.get_informational_banner(),  #
                "org_access_notice": ss.get_org_access_notice(),
                "anonymous_access_notice": ss.get_anonymous_access_notice(),
                "multifactor_authentication": ss.get_multifactor_authentication(),
                "email_settings": ss.get_email_settings(),
            },
            "PasswordPolicy": {
                "password_policy": pp.policy,
                "lockout_policy": pp.lockout_policy,
            },
            "CategorySchema": {"schema": cs.schema},
            "ResourceManager": {
                "resources": [],
                "folder": os.path.basename(resource_folder),
                "offline": offline,
            },
        }
        return settings_configuration

    #  --------------------------------------------------------------------
    def _download_resources(
        self, resource_folder: str, settings_configuration: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Downloads and stores the GIS resources items to a local workspace
        The config file is then updated with the files.
        :returns: dict[str,Any]
        """
        rm = self._source_gis.admin.resources
        for resource in rm.list(num=-1):
            if not resource["key"] in self._RESOURCE_IGNORE_LIST:
                del resource["created"]
                del resource["size"]
                resource["path"] = rm.get(resource["key"], resource_folder)
                settings_configuration["ResourceManager"]["resources"].append(resource)
        return settings_configuration

    #  --------------------------------------------------------------------
    def load_offline_configuration(self, package: str) -> concurrent.futures.Future:
        """
        Loads the UX configuration file into the current active portal.
        """
        workspace = os.path.join(
            tempfile.gettempdir(), f"ux_config_{uuid.uuid4().hex[:5]}"
        )
        os.makedirs(workspace, exist_ok=True)
        shutil.unpack_archive(filename=package, extract_dir=workspace, format="zip")
        hp_folder = os.path.join(workspace, "hp")
        resource_folder = os.path.join(workspace, "resources")
        out_folder = workspace
        os.makedirs(hp_folder, exist_ok=True)
        os.makedirs(resource_folder, exist_ok=True)
        os.makedirs(out_folder, exist_ok=True)
        settings_configuration = {}
        with open(
            os.path.join(workspace, "settings_configuration.json"), "r"
        ) as reader:
            settings_configuration = json.load(reader)
        if settings_configuration:
            ## 1). Update the settings_configuration for the resources
            ##
            rs = []
            for r in settings_configuration["ResourceManager"]["resources"]:
                r = dict(r)
                r["path"] = os.path.join(resource_folder, os.path.basename(r["path"]))
                rs.append(r)
            settings_configuration["ResourceManager"]["resources"] = rs
            if "logo" in settings_configuration and settings_configuration["logo"]:
                logo = os.path.join(
                    workspace,
                    os.path.basename(settings_configuration["logo"]),
                )
                settings_configuration["logo"] = logo

        return self._config_or_org(
            settings_configuration=settings_configuration,
            target_gis=self._source_gis,
            hp_folder=hp_folder,
            resource_folder=resource_folder,
            out_folder=out_folder,
        )

    #  --------------------------------------------------------------------
    def _zipdir(self, src, dst, zip_name):
        """
        Function creates zip archive from src in dst location. The name of archive is zip_name.
        :param src: Path to directory to be archived.
        :param dst: Path where archived dir will be stored.
        :param zip_name: The name of the archive.
        :return: None
        """

        zip_name = os.path.join(dst, zip_name)
        ### zipfile handler
        ziph = zipfile.ZipFile(zip_name, "w")
        ### writing content of src directory to the archive
        for root, dirs, files in os.walk(src):
            for file in files:
                ziph.write(
                    os.path.join(root, file),
                    arcname=os.path.join(root.replace(src, ""), file),
                )
        ziph.close()
        return zip_name

    #  --------------------------------------------------------------------
    def _create_cloner_file(
        self, save_folder: str, path: str, package_name: str | None = None
    ) -> str:
        """Creates the Zipped compressed .uxpk file."""
        if package_name is None:
            package_name = f"{uuid.uuid4().hex[:5]}.uxpk"
        else:
            package_name += ".uxpk"
        save_fp: str = os.path.join(save_folder, package_name)
        self._zipdir(src=path, dst=save_folder, zip_name=package_name)

        if os.path.isfile(save_fp):
            return save_fp
        else:
            raise Exception(f"Could not save the the file: {save_fp}")
        return save_fp

    #  --------------------------------------------------------------------
    def _create_offline_clone_package(
        self,
        out_folder: str | None = None,  #  workspace folder
        package_save_location: str | None = None,  # package save location
        package_name: str | None = None,  # package save name
    ) -> str:
        """
        Creates an offline cloner package (.uxpk)
        """
        offline: bool = True
        ## 1). Setup the workspace folder
        ##
        (
            out_folder,
            hp_folder,
            resource_folder,
        ) = self._setup_folder_workspaces(out_folder=out_folder)
        ## 2). Creat the configuration file.
        ##
        settings_configuration = self._create_configuration(
            offline=offline,
            out_folder=out_folder,
            hp_folder=hp_folder,
            resource_folder=resource_folder,
        )
        ## 3). Create the cloner file
        ##
        settings_configuration = self._download_resources(
            resource_folder=resource_folder,
            settings_configuration=settings_configuration,
        )
        ## 4). Save the settings_configuration to a .json file
        ##
        settings_config_file: str = os.path.join(
            out_folder, "settings_configuration.json"
        )
        with open(settings_config_file, "w") as writer:
            json.dump(settings_configuration, writer, default=_date_handler)
        ## 4). Package up the data and return it
        ##
        os.makedirs(name=package_save_location, exist_ok=True)
        save_fp: str = self._create_cloner_file(
            save_folder=package_save_location,
            path=out_folder,
            package_name=package_name,
        )
        ## 5). Clean up files and folders
        ##

        shutil.rmtree(path=hp_folder, ignore_errors=True)
        shutil.rmtree(path=resource_folder, ignore_errors=True)
        if package_save_location.lower().find(out_folder.lower()) == -1:
            shutil.rmtree(out_folder, ignore_errors=True)
        return save_fp

    def _config_or_org(
        self,
        settings_configuration: dict[str, Any],
        target_gis: _arcgis.gis.GIS,
        hp_folder: str,
        resource_folder: str,
        out_folder: str,
    ):
        """
        Moves the configuration file to the GIS
        """
        if target_gis._portal.is_arcgisonline:
            from arcgis.gis.admin import AGOLAdminManager

            admin: AGOLAdminManager = target_gis.admin

        else:
            from arcgis.gis.admin import PortalAdminManager

            admin: PortalAdminManager = target_gis.admin
        ux: _arcgis_ux.UX = admin.ux
        for resource in settings_configuration["ResourceManager"]["resources"]:
            admin.resources.add(**resource)
        if settings_configuration["logo"]:
            ux.set_logo(logo_file=settings_configuration["logo"], show_logo=True)

        ux.name = settings_configuration["name"]
        ux.summary = settings_configuration["summary"]
        if (
            "contact_link" in settings_configuration
            and settings_configuration["contact_link"]
        ):
            ux.contact_link = settings_configuration["contact_link"]["contactUs"].get(
                "url", ""
            )
        try:
            ux.admin_contacts = settings_configuration["admin_contacts"]
        except Exception as ex:
            warnings.warn(
                f"Could not find the administration user: {settings_configuration['admin_contacts']}"
            )
        try:
            ux.help_source = settings_configuration["help_source"]
        except Exception as ex:
            warnings.warn(
                f"Warning: setting the help source property could not be set: {ex}"
            )
        ux.description = settings_configuration["description"]
        ux.description_visibility = settings_configuration["description_visibility"]
        ux.item_settings.enable_comments = settings_configuration["ItemSettings"][
            "comments"
        ]
        ux.item_settings.enable_metadata_edit = settings_configuration["ItemSettings"][
            "enable_metadata_edit"
        ]
        ux.item_settings.metadata_format = settings_configuration["ItemSettings"][
            "metadata_format"
        ]
        #  Map Settings
        ux.map_settings.default_extent = settings_configuration["MapSettings"][
            "default_extent"
        ]
        if settings_configuration["MapSettings"]["default_mapviewer"]:
            ux.map_settings.default_mapviewer = settings_configuration["MapSettings"][
                "default_mapviewer"
            ]
        if settings_configuration["MapSettings"]["units"]:
            ux.map_settings.units = settings_configuration["MapSettings"]["units"]
        # Security Settings
        security_config: dict[str, Any] = settings_configuration["SecuritySettings"]
        ux.security_settings.allowed_origins = security_config["allowed_origins"]
        ux.security_settings.allowed_redirect_uris = security_config[
            "allowed_redirect_uris"
        ]
        ux.security_settings.anonymous_access = security_config["anonymous_access"]
        ux.security_settings.enable_https = security_config["enable_https"]
        ux.security_settings.enable_update_user_profile = security_config[
            "enable_update_user_profile"
        ]
        ux.security_settings.share_public = security_config["share_public"]
        ux.security_settings.show_social_media = security_config["show_social_media"]
        signin_settings = security_config["signin_settings"]
        if signin_settings:
            ux.security_settings.set_approved_apps(
                block_unapproved=signin_settings["blockUnapprovedThirdpartyApps"]
            )
            ux.security_settings.set_blocked_apps(
                block_beta_apps=signin_settings["blockBetaApps"],
                apps=signin_settings["blockedApps"],
            )
            ux.security_settings.set_social_media_login(
                social_login=security_config["can_sign_in_social"],
                social_networks=signin_settings["signinOptionsOrder"]["social"],
                social_network_order=signin_settings["signinOptionsOrder"]["logins"],
            )
        ux.security_settings.trusted_servers = security_config["trusted_servers"]
        # handle method updates
        #
        notice = security_config["anonymous_access_notice"]
        if notice and len(notice) > 0:
            ux.security_settings.set_anonymous_access_notice(
                title=notice["title"],
                text=notice["text"],
                button_type=notice["buttons"],
            )
        if security_config["informational_banner"]:
            ux.security_settings.set_informational_banner(
                text=security_config["informational_banner"]["text"],
                bg_color=security_config["informational_banner"]["bgColor"],
                font_color=security_config["informational_banner"]["fontColor"],
                enabled=security_config["informational_banner"]["enabled"],
            )
        if security_config["org_access_notice"]:
            ux.security_settings.set_org_access_notice(
                title=security_config["org_access_notice"]["title"],
                text=security_config["org_access_notice"]["text"],
                button_type=security_config["org_access_notice"]["buttons"],
            )
        # "PasswordPolicy"
        #
        if (
            "PasswordPolicy" in settings_configuration
            and settings_configuration["PasswordPolicy"]
        ):
            password_policy = settings_configuration["PasswordPolicy"]
            policy = password_policy["password_policy"]
            policy.pop("created", None)
            policy.pop("modified", None)
            policy.pop("type", None)

            target_gis.admin.password_policy.policy = policy
            lockout = password_policy["lockout_policy"]
            lockout.pop("created", None)
            lockout.pop("modified", None)
            lockout.pop("type", None)
            target_gis.admin.password_policy.lockout_policy = lockout
            del lockout, policy
        # Category Manager
        #
        category_schema = settings_configuration["CategorySchema"]
        if category_schema and category_schema["schema"]["categorySchema"]:
            target_gis.admin.category_schema.schema = category_schema["schema"][
                "categorySchema"
            ]
        ## 5). Clean up files and folders
        ##
        if hp_folder:
            shutil.rmtree(path=hp_folder, ignore_errors=True)
        if resource_folder:
            shutil.rmtree(path=resource_folder, ignore_errors=True)
        if out_folder:
            shutil.rmtree(out_folder, ignore_errors=True)
        return True

    #  --------------------------------------------------------------------
    def _online_clone_workflow(
        self,
        *,
        target_gis: _arcgis.gis.GIS,
        out_folder: str | None = None,
    ) -> "CloneJob":
        """ """
        offline: bool = False
        from arcgis.gis import User

        target_user: User = target_gis.users.me
        assert (
            target_user.role == "org_admin"
        ), "User must be an administrator to perform this operation."
        ## 1). Setup the workspace folder
        ##
        (
            out_folder,
            hp_folder,
            resource_folder,
        ) = self._setup_folder_workspaces(out_folder=out_folder)
        ## 2). Creat the configuration file.
        ##
        settings_configuration = self._create_configuration(
            offline=offline,
            out_folder=out_folder,
            hp_folder=hp_folder,
            resource_folder=resource_folder,
        )
        settings_configuration = self._download_resources(
            resource_folder=resource_folder,
            settings_configuration=settings_configuration,
        )
        ## 3). Perform the clone operations
        ##

        return self._config_or_org(
            settings_configuration=settings_configuration,
            target_gis=target_gis,
            hp_folder=hp_folder,
            resource_folder=resource_folder,
            out_folder=out_folder,
        )

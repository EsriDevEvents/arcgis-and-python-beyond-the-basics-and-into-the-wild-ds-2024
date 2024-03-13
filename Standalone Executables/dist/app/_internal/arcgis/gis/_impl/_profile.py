import os
import logging
import datetime as _datetime
import platform
import configparser
from typing import Optional
from arcgis.gis import GIS
from functools import lru_cache

_log = logging.getLogger(__name__)


###########################################################################
class ServerProfileManager(object):
    """
    The ``ProfileManager`` class allows for the controls and management of the
    profiles stored on the local operating system.
    """

    _gis = None
    _os = None
    _cfg_file_path = None
    _profile_name = "esri_ags_profile_passwords"

    #######################################################################
    def __init__(self):
        self._os = platform.system()
        self._cfg_file_path = os.path.expanduser("~") + "/.agsprofile"
        self._cfg_exists = os.path.isfile(self._cfg_file_path)

    # ----------------------------------------------------------------------
    @lru_cache(maxsize=255)
    def _keyring_version(self):
        """returns the keyring version number"""
        try:
            # python 3.8+
            from importlib.metadata import version

            return [int(i) for i in version("keyring").split(".")]
        except:
            # python < 3.8
            import pkg_resources

            return [
                int(i)
                for i in pkg_resources.get_distribution("keyring").version.split(".")
            ]

    # ----------------------------------------------------------------------
    def _config_is_in_new_format(self, config):
        """Any version <= 1.3.0 of the API used a different config file
        formatting that, among other things, did not store the last time
        a profile was modified. Thus, if 'date_modified' is found in at least
        one profile, it is in the new format
        """
        return any(
            [
                profile_data
                for profile_data in config.values()
                if "date_modified" in profile_data
            ]
        )

    # ----------------------------------------------------------------------
    def _update_profile_data_in_config(
        self,
        config,
        profile,
        url=None,
        username=None,
        key_file=None,
        cert_file=None,
        client_id=None,
    ):
        """Updates the specific profile in the config object to include
        any of the user defined arguments. This will overwrite old values.
        ***USE THIS FUNCTION INSTEAD OF MANUALLY MODIFYING PROFILE DATA***
        """
        if url is not None:
            config[profile]["url"] = url
            self._add_timestamp_to_profile_data_in_config(config, profile)
        if username is not None:
            config[profile]["username"] = username
            self._add_timestamp_to_profile_data_in_config(config, profile)
        if key_file is not None:
            config[profile]["key_file"] = key_file
            self._add_timestamp_to_profile_data_in_config(config, profile)
        if cert_file is not None:
            config[profile]["cert_file"] = cert_file
            self._add_timestamp_to_profile_data_in_config(config, profile)
        if client_id is not None:
            config[profile]["client_id"] = client_id
            self._add_timestamp_to_profile_data_in_config(config, profile)

    # ----------------------------------------------------------------------
    def _add_timestamp_to_profile_data_in_config(self, config, profile):
        """Sets the 'date_modified' field to this moment's datetime"""
        config[profile]["date_modified"] = str(_datetime.datetime.now())

    # ----------------------------------------------------------------------
    def _write_config(self, config, cfg_file_path):
        """write the config object to the .arcgisprofile file"""
        with open(cfg_file_path, "w") as writer:
            config.write(writer)

    # ----------------------------------------------------------------------
    def _securely_store_password(self, profile, password):
        """Securely stores the password in an O.S. specific store via the
        keyring package. Can be retrieved later with just the profile name.

        If keyring is not properly set up system-wide, raise a RuntimeError
        """
        import keyring

        if self._current_keyring_is_recommended():
            return keyring.set_password(self._profile_name, profile, password)
        else:
            raise RuntimeError(self._get_keyring_failure_message())

    # ----------------------------------------------------------------------
    def _securely_get_password(self, profile):
        """Securely gets the profile specific password stored via keyring

        If keyring is not properly set up system-wide OR if a password is not
        found through keyring, log the respective warning and return 'None'
        """
        import keyring

        if self._current_keyring_is_recommended():
            # password will be None if no password is found for the profile

            if self._keyring_version() >= [23]:
                password = keyring.get_credential(self._profile_name, profile)

                password = getattr(password, "password", None)
            else:
                password = keyring.get_password(self._profile_name, profile)
        else:
            password = None
            _log.warn(self._get_keyring_failure_message())

        if password is None:
            _log.warn(
                "Profile {0} does not have a password on file through "
                "keyring. If you are expecting this behavior (PKI or "
                "IWA authentication, entering password through "
                "run-time prompt, etc.), please ignore this message. "
                "If you would like to store your password in the {0} "
                "profile, run GIS(profile = '{0}', password = ...). "
                "See the API doc for more details. "
                "(https://developers.arcgis.com/python/api-reference/arcgis.gis.toc.html#gis)".format(
                    profile
                )
            )
        return password

    # ----------------------------------------------------------------------
    def _securely_delete_password(self, profile):
        """Securely deletes the profile specific password via keyring

        If keyring is not properly set up system-wide, log a warning
        """
        import keyring

        if self._current_keyring_is_recommended():
            return keyring.delete_password(self._profile_name, profile)
        else:
            _log.warn(self._get_keyring_failure_message())
            return False

    # ----------------------------------------------------------------------
    def _current_keyring_is_recommended(self):
        """The keyring project recommends 4 secure keyring backends. The
        defaults on Windows/OSX should be the recommended backends, but Linux
        needs some system-wide software installed and configured to securely
        function. Return if the current keyring is a supported, properly
        configured backend
        """
        import keyring

        supported_keyrings = [
            keyring.backends.OS_X.Keyring,
            keyring.backends.SecretService.Keyring,
            keyring.backends.Windows.WinVaultKeyring,
            keyring.backends.kwallet.DBusKeyring,
            keyring.backends.chainer.ChainerBackend,
        ]
        current_keyring = type(keyring.get_keyring())
        return current_keyring in supported_keyrings

    # ----------------------------------------------------------------------
    def _get_keyring_failure_message(self):
        """An informative failure msg about the backend keyring being used"""
        import keyring

        return (
            "Keyring backend being used ({}) either failed to install "
            "or is not recommended by the keyring project (i.e. it is "
            "not secure). This means you can not use stored passwords "
            "through GIS's persistent profiles. Note that extra system-"
            "wide steps must be taken on a Linux machine to use the python "
            "keyring module securely. Read more about this at the "
            "keyring API doc (http://bit.ly/2EWDP7B) and the ArcGIS API "
            "for Python doc (https://developers.arcgis.com/python/api-reference/arcgis.gis.toc.html#gis)."
            "".format(keyring.get_keyring())
        )

    # ----------------------------------------------------------------------
    def list(self, as_df=False):
        """
        The ``list`` method retrieves a list of profile names in the configuration file

        :return:
            List if `as_df=False` or Pandas DataFrame if `as_df=True`
        """
        if self._cfg_exists and as_df == False:
            config = configparser.ConfigParser()
            config.read(self._cfg_file_path)
            return config.sections()
        elif self._cfg_exists and as_df:
            import pandas as pd

            all_profiles = []

            for p in self.list():
                p_dict = self.get(p)
                p_dict[
                    "profile"
                ] = p  # add a new column to DF that lists the profile name
                all_profiles.append(p_dict)

            return pd.DataFrame(data=all_profiles)
        return []

    # --------------------------------------------------------------------------
    def get(self, profile):
        """
        The ``get`` method retrieves the profile information for a given entry.

        ================  ====================================================================
        **Parameter**     **Description**
        ----------------  --------------------------------------------------------------------
        profile           Required String. The name of the profile to get the information about.
        ================  ====================================================================

        :return:
            A dictionary

        .. code-block:: python

            # Usage Example

            >>> gis.ProfileManager.get("profile_name")

        """
        profile_file = self._cfg_file_path
        if self._cfg_exists:
            config = configparser.ConfigParser()
            config.read(profile_file)
            keys = config.options(profile)
            values = {}
            for key in keys:
                try:
                    if key == "date_modified":
                        values[key] = _datetime.datetime.strptime(
                            config.get(profile, key), "%Y-%m-%d %H:%M:%S.%f"
                        )
                    else:
                        values[key] = config.get(profile, key)
                except:
                    values[key] = None
            return values
        else:
            _log.warn(
                "Profile does not exist. Use the `ProfileManage.list()` to see a list of available login credentials."
            )
        return None

    # --------------------------------------------------------------------------
    def delete(self, profile):
        """
        The ``delete`` method deletes a profile permanently from the .arcgisprofile file

        ================  ====================================================================
        **Parameter**     **Description**
        ----------------  --------------------------------------------------------------------
        profile           Required String. The name of the profile to delete.
        ================  ====================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example

            >>> gis.ProfileManager.delete("profile_name")
        """
        import keyring

        profile_file = self._cfg_file_path
        if self._cfg_exists:
            profiles = self.list()
            if profile in profiles:
                config = configparser.ConfigParser()
                with open(profile_file, "r") as reader:
                    config.read_file(reader)
                data = dict(config.items(profile))
                try:
                    keyring.delete_password(
                        service_name=self._profile_name,
                        username=profile,
                    )
                except:
                    pass
                config.remove_section(section=profile)
                with open(profile_file, "w") as f:
                    config.write(f)

                return True
            else:
                raise ValueError("Profile not found.")
        return False

    # ----------------------------------------------------------------------
    def update(
        self,
        profile,
        url=None,
        username=None,
        password=None,
        key_file=None,
        cert_file=None,
        client_id=None,
    ):
        """
        The ``update`` method updates an existing profile in the credential manager.

        ================  ====================================================================
        **Parameter**     **Description**
        ----------------  --------------------------------------------------------------------
        profile           Required String. The name of the profile to update.
        ----------------  --------------------------------------------------------------------
        url               Optional String.  The site url.
        ----------------  --------------------------------------------------------------------
        username          Optional String.  The login user name.
        ----------------  --------------------------------------------------------------------
        password          Optional String.  The login user password.
        ----------------  --------------------------------------------------------------------
        key_file          Optional String.  The key file for PKI security.
        ----------------  --------------------------------------------------------------------
        cert_file         Optional String.  The cert file for PKI security.
        ----------------  --------------------------------------------------------------------
        client_id         Optional String.  The client ID for oauth login.
        ================  ====================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example

            >>> gis.ProfileManager.update(profile = "profile_name1",
            >>>                           username = "User12345",
            >>>                           key_file = "new_key_file")

        """
        if profile not in self.list():
            raise ValueError(
                "Could not find profile {}. Use `create` to generate a new profile.".format(
                    profile
                )
            )
        _log.info("Updating profile {} ...".format(profile))
        return self.create(
            profile, url, username, password, key_file, cert_file, client_id
        )

    # ----------------------------------------------------------------------
    def create(
        self,
        profile,
        url=None,
        username=None,
        password=None,
        key_file=None,
        cert_file=None,
        client_id=None,
    ):
        """
        The ``create`` method adds a new entry into the Profile Store.

        ================  ====================================================================
        **Parameter**     **Description**
        ----------------  --------------------------------------------------------------------
        profile           Required String. The name of the profile to add.
        ----------------  --------------------------------------------------------------------
        url               Optional String.  The site url.
        ----------------  --------------------------------------------------------------------
        username          Optional String.  The login user name.
        ----------------  --------------------------------------------------------------------
        password          Optional String.  The login user password.
        ----------------  --------------------------------------------------------------------
        key_file          Optional String.  The key file for PKI security.
        ----------------  --------------------------------------------------------------------
        cert_file         Optional String.  The cert file for PKI security.
        ----------------  --------------------------------------------------------------------
        client_id         Optional String.  The client ID for oauth login.
        ================  ====================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example

            >>> gis.ProfileManager.create("profile_name",
            >>>                           url= "www.foo.com",
            >>>                           username = "User1234",
            >>>                           password = "Password1234",
            >>>                           key_file = "key_file",
            >>>                           cert_fle = "cert_file_name")

        """
        try:
            config = configparser.ConfigParser()
            if self._cfg_exists:
                config.read(self._cfg_file_path)
            if profile not in self.list():
                _log.info("Adding new profile {} to config...".format(profile))
                config.add_section(profile)
                self._add_timestamp_to_profile_data_in_config(config, profile)
            self._update_profile_data_in_config(
                config=config,
                profile=profile,
                url=url,
                username=username,
                key_file=key_file,
                cert_file=cert_file,
                client_id=client_id,
            )
            if password is not None:
                self._securely_store_password(profile, password)
            self._write_config(config, self._cfg_file_path)
            self._cfg_exists = True
            return True
        except:
            _log.warn(self._get_keyring_failure_message())
            return False

    # ----------------------------------------------------------------------
    def save_as(self, profile, gis):
        """

        The ``save_as`` method saves and adds the provided :class:`~arcgis.gis.GIS` object to the profile.

        ================  ====================================================================
        **Parameter**     **Description**
        ----------------  --------------------------------------------------------------------
        name              Required String. The name of the profile to save.
        ----------------  --------------------------------------------------------------------
        gis               Required GIS. The connection object to update the profile with.
        ================  ====================================================================

        :return: Boolean. True if successful else False

        .. code-block:: python

            # Usage Example

            >>> gis = GIS("pro")
            >>> gis.ProfileManager.save_as("Profile_name", gis)


        """
        from arcgis.gis import GIS

        url = gis._url
        u = gis._username
        p = gis._password
        kf = gis._key_file
        cf = gis._cert_file
        ci = gis._client_id
        if profile.lower() in [p.lower() for p in self.list()]:
            cfg_file_path = self._cfg_file_path
            config = configparser.ConfigParser()
            if os.path.isfile(cfg_file_path):
                config.read(cfg_file_path)
            self._update_profile_data_in_config(
                config,
                profile=profile,
                url=url,
                username=u,
                key_file=kf,
                cert_file=cf,
                client_id=ci,
            )
            if p is not None:
                self._securely_store_password(profile, p)
            self._write_config(config, cfg_file_path)
            return True
        else:
            return self.create(
                profile=profile,
                url=url,
                username=u,
                key_file=kf,
                cert_file=cf,
                client_id=ci,
            )

    # ----------------------------------------------------------------------
    def _retrieve(self, profile):
        """gets the login information"""
        url, username, password, key_file, cert_file, client_id = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if profile.lower() in [p.lower() for p in self.list()]:
            cfg_file_path = self._cfg_file_path
            config = configparser.ConfigParser()
            if os.path.isfile(cfg_file_path):
                config.read(cfg_file_path)
            if config.has_option(profile, "url"):
                url = config[profile]["url"]
            if config.has_option(profile, "username"):
                username = config[profile]["username"]
            if config.has_option(profile, "key_file"):
                key_file = config[profile]["key_file"]
            if config.has_option(profile, "cert_file"):
                cert_file = config[profile]["cert_file"]
            if config.has_option(profile, "client_id"):
                client_id = config[profile]["client_id"]

            password = self._securely_get_password(profile)
        return url, username, password, key_file, cert_file, client_id


###########################################################################
class ProfileManager(object):
    """
    The ``ProfileManager`` class allows for the controls and management of the
    profiles stored on the local operating system.
    """

    _gis = None
    _os = None
    _cfg_file_path = None

    #######################################################################
    def __init__(self):
        self._os = platform.system()
        self._cfg_file_path = os.path.expanduser("~") + "/.arcgisprofile"
        self._cfg_exists = os.path.isfile(self._cfg_file_path)

    # ----------------------------------------------------------------------
    @lru_cache(maxsize=255)
    def _keyring_version(self):
        """returns the keyring version number"""
        try:
            # python 3.8+
            from importlib.metadata import version

            return [int(i) for i in version("keyring").split(".")]
        except:
            # python < 3.8
            import pkg_resources

            return [
                int(i)
                for i in pkg_resources.get_distribution("keyring").version.split(".")
            ]

    # ----------------------------------------------------------------------
    def _config_is_in_new_format(self, config):
        """Any version <= 1.3.0 of the API used a different config file
        formatting that, among other things, did not store the last time
        a profile was modified. Thus, if 'date_modified' is found in at least
        one profile, it is in the new format
        """
        return any(
            [
                profile_data
                for profile_data in config.values()
                if "date_modified" in profile_data
            ]
        )

    # ----------------------------------------------------------------------
    def _update_profile_data_in_config(
        self,
        config,
        profile,
        url=None,
        username=None,
        key_file=None,
        cert_file=None,
        client_id=None,
    ):
        """Updates the specific profile in the config object to include
        any of the user defined arguments. This will overwrite old values.
        ***USE THIS FUNCTION INSTEAD OF MANUALLY MODIFYING PROFILE DATA***
        """
        if url is not None:
            config[profile]["url"] = url
            self._add_timestamp_to_profile_data_in_config(config, profile)
        if username is not None:
            config[profile]["username"] = username
            self._add_timestamp_to_profile_data_in_config(config, profile)
        if key_file is not None:
            config[profile]["key_file"] = key_file
            self._add_timestamp_to_profile_data_in_config(config, profile)
        if cert_file is not None:
            config[profile]["cert_file"] = cert_file
            self._add_timestamp_to_profile_data_in_config(config, profile)
        if client_id is not None:
            config[profile]["client_id"] = client_id
            self._add_timestamp_to_profile_data_in_config(config, profile)

    # ----------------------------------------------------------------------
    def _add_timestamp_to_profile_data_in_config(self, config, profile):
        """Sets the 'date_modified' field to this moment's datetime"""
        config[profile]["date_modified"] = str(_datetime.datetime.now())

    # ----------------------------------------------------------------------
    def _write_config(self, config, cfg_file_path):
        """write the config object to the .arcgisprofile file"""
        with open(cfg_file_path, "w") as writer:
            config.write(writer)

    # ----------------------------------------------------------------------
    def _securely_store_password(self, profile, password):
        """Securely stores the password in an O.S. specific store via the
        keyring package. Can be retrieved later with just the profile name.

        If keyring is not properly set up system-wide, raise a RuntimeError
        """
        import keyring

        if self._current_keyring_is_recommended():
            return keyring.set_password(
                "arcgis_python_api_profile_passwords", profile, password
            )
        else:
            raise RuntimeError(self._get_keyring_failure_message())

    # ----------------------------------------------------------------------
    def _securely_get_password(self, profile):
        """Securely gets the profile specific password stored via keyring

        If keyring is not properly set up system-wide OR if a password is not
        found through keyring, log the respective warning and return 'None'
        """
        import keyring

        if self._current_keyring_is_recommended():
            # password will be None if no password is found for the profile

            if self._keyring_version() >= [23]:
                password = keyring.get_credential(
                    "arcgis_python_api_profile_passwords", profile
                )

                password = getattr(password, "password", None)
            else:
                password = keyring.get_password(
                    "arcgis_python_api_profile_passwords", profile
                )
        else:
            password = None
            _log.warn(self._get_keyring_failure_message())

        if password is None:
            _log.warn(
                "Profile {0} does not have a password on file through "
                "keyring. If you are expecting this behavior (PKI or "
                "IWA authentication, entering password through "
                "run-time prompt, etc.), please ignore this message. "
                "If you would like to store your password in the {0} "
                "profile, run GIS(profile = '{0}', password = ...). "
                "See the API doc for more details. "
                "(https://developers.arcgis.com/python/api-reference/arcgis.gis.toc.html#gis)".format(
                    profile
                )
            )
        return password

    # ----------------------------------------------------------------------
    def _securely_delete_password(self, profile):
        """Securely deletes the profile specific password via keyring

        If keyring is not properly set up system-wide, log a warning
        """
        import keyring

        if self._current_keyring_is_recommended():
            return keyring.delete_password(
                "arcgis_python_api_profile_passwords", profile
            )
        else:
            _log.warn(self._get_keyring_failure_message())
            return False

    # ----------------------------------------------------------------------
    def _current_keyring_is_recommended(self):
        """The keyring project recommends 4 secure keyring backends. The
        defaults on Windows/OSX should be the recommended backends, but Linux
        needs some system-wide software installed and configured to securely
        function. Return if the current keyring is a supported, properly
        configured backend
        """
        import keyring

        if self._keyring_version() >= [23, 0, 0]:
            supported_keyrings = [type(r) for r in keyring.backend.get_all_keyring()]
        else:
            try:
                import keyring.backends.OS_X
                import keyring.backends.kwallet
                import keyring.backends.chainer
                import keyring.backends.Windows
                import keyring.backends.SecretService
            except Exception as keyringex:
                print(f"Error importing keyring {str(keyringex)}")
            supported_keyrings = [
                keyring.backends.OS_X.Keyring,
                keyring.backends.SecretService.Keyring,
                keyring.backends.Windows.WinVaultKeyring,
                keyring.backends.kwallet.DBusKeyring,
                keyring.backends.chainer.ChainerBackend,
            ]
        current_keyring = type(keyring.get_keyring())
        return current_keyring in supported_keyrings

    # ----------------------------------------------------------------------
    def _get_keyring_failure_message(self):
        """An informative failure msg about the backend keyring being used"""
        import keyring

        return (
            "Keyring backend being used ({}) either failed to install "
            "or is not recommended by the keyring project (i.e. it is "
            "not secure). This means you can not use stored passwords "
            "through GIS's persistent profiles. Note that extra system-"
            "wide steps must be taken on a Linux machine to use the python "
            "keyring module securely. Read more about this at the "
            "keyring API doc (http://bit.ly/2EWDP7B) and the ArcGIS API "
            "for Python doc (https://developers.arcgis.com/python/api-reference/arcgis.gis.toc.html#gis)."
            "".format(keyring.get_keyring())
        )

    # ----------------------------------------------------------------------
    def list(self, as_df: bool = False):
        """
        The ``list`` method retrieves a list of profile names in the configuration file

        :return:
            List if `as_df=False` or Pandas DataFrame if `as_df=True`
        """
        if self._cfg_exists and as_df == False:
            config = configparser.ConfigParser()
            config.read(self._cfg_file_path)
            return config.sections()
        elif self._cfg_exists and as_df:
            import pandas as pd

            all_profiles = []

            for p in self.list():
                p_dict = self.get(p)
                p_dict[
                    "profile"
                ] = p  # add a new column to DF that lists the profile name
                all_profiles.append(p_dict)

            return pd.DataFrame(data=all_profiles)
        return []

    # --------------------------------------------------------------------------
    def get(self, profile: str):
        """
        The ``get`` method retrieves the profile information for a given entry.

        ================  ====================================================================
        **Parameter**     **Description**
        ----------------  --------------------------------------------------------------------
        profile           Required String. The name of the profile to get the information about.
        ================  ====================================================================

        :return:
            A dictionary

        .. code-block:: python

            # Usage Example

            >>> gis.ProfileManager.get("profile_name")

        """
        profile_file = self._cfg_file_path
        if self._cfg_exists:
            config = configparser.ConfigParser()
            config.read(profile_file)
            keys = config.options(profile)
            values = {}
            for key in keys:
                try:
                    if key == "date_modified":
                        values[key] = _datetime.datetime.strptime(
                            config.get(profile, key), "%Y-%m-%d %H:%M:%S.%f"
                        )
                    else:
                        values[key] = config.get(profile, key)
                except:
                    values[key] = None
            return values
        else:
            _log.warn(
                "Profile does not exist. Use the `ProfileManage.list()` to see a list of available login credentials."
            )
        return None

    # --------------------------------------------------------------------------
    def delete(self, profile: str):
        """
        The ``delete`` method deletes a profile permanently from the .arcgisprofile file

        ================  ====================================================================
        **Parameter**     **Description**
        ----------------  --------------------------------------------------------------------
        profile           Required String. The name of the profile to delete.
        ================  ====================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example

            >>> gis.ProfileManager.delete("profile_name")
        """
        import keyring

        profile_file = self._cfg_file_path
        if self._cfg_exists:
            profiles = self.list()
            if profile in profiles:
                config = configparser.ConfigParser()
                with open(profile_file, "r") as reader:
                    config.read_file(reader)
                data = dict(config.items(profile))
                try:
                    keyring.delete_password(
                        service_name="arcgis_python_api_profile_passwords",
                        username=profile,
                    )
                except:
                    pass
                config.remove_section(section=profile)
                with open(profile_file, "w") as f:
                    config.write(f)

                return True
            else:
                raise ValueError("Profile not found.")
        return False

    # ----------------------------------------------------------------------
    def update(
        self,
        profile: str,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        key_file: Optional[str] = None,
        cert_file: Optional[str] = None,
        client_id: Optional[str] = None,
    ):
        """
        The ``update`` method updates an existing profile in the credential manager.

        ================  ====================================================================
        **Parameter**     **Description**
        ----------------  --------------------------------------------------------------------
        profile           Required String. The name of the profile to update.
        ----------------  --------------------------------------------------------------------
        url               Optional String.  The site url.
        ----------------  --------------------------------------------------------------------
        username          Optional String.  The login user name.
        ----------------  --------------------------------------------------------------------
        password          Optional String.  The login user password.
        ----------------  --------------------------------------------------------------------
        key_file          Optional String.  The key file for PKI security.
        ----------------  --------------------------------------------------------------------
        cert_file         Optional String.  The cert file for PKI security.
        ----------------  --------------------------------------------------------------------
        client_id         Optional String.  The client ID for oauth login.
        ================  ====================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example

            >>> gis.ProfileManager.update(profile = "profile_name1",
            >>>                           username = "User12345",
            >>>                           key_file = "new_key_file")

        """
        if profile not in self.list():
            raise ValueError(
                "Could not find profile {}. Use `create` to generate a new profile.".format(
                    profile
                )
            )
        _log.info("Updating profile {} ...".format(profile))
        return self.create(
            profile, url, username, password, key_file, cert_file, client_id
        )

    # ----------------------------------------------------------------------
    def create(
        self,
        profile: str,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        key_file: Optional[str] = None,
        cert_file: Optional[str] = None,
        client_id: Optional[str] = None,
    ):
        """
        The ``create`` method adds a new entry into the Profile Store.

        ================  ====================================================================
        **Parameter**     **Description**
        ----------------  --------------------------------------------------------------------
        profile           Required String. The name of the profile to add.
        ----------------  --------------------------------------------------------------------
        url               Optional String.  The site url.
        ----------------  --------------------------------------------------------------------
        username          Optional String.  The login user name.
        ----------------  --------------------------------------------------------------------
        password          Optional String.  The login user password.
        ----------------  --------------------------------------------------------------------
        key_file          Optional String.  The key file for PKI security.
        ----------------  --------------------------------------------------------------------
        cert_file         Optional String.  The cert file for PKI security.
        ----------------  --------------------------------------------------------------------
        client_id         Optional String.  The client ID for oauth login.
        ================  ====================================================================

        :return:
            A boolean indicating success (True), or failure (False)

        .. code-block:: python

            # Usage Example

            >>> gis.ProfileManager.create("profile_name",
            >>>                           url= "www.foo.com",
            >>>                           username = "User1234",
            >>>                           password = "Password1234",
            >>>                           key_file = "key_file",
            >>>                           cert_fle = "cert_file_name")

        """
        try:
            config = configparser.ConfigParser()
            if self._cfg_exists:
                config.read(self._cfg_file_path)
            if profile not in self.list():
                _log.info("Adding new profile {} to config...".format(profile))
                config.add_section(profile)
                self._add_timestamp_to_profile_data_in_config(config, profile)
            self._update_profile_data_in_config(
                config=config,
                profile=profile,
                url=url,
                username=username,
                key_file=key_file,
                cert_file=cert_file,
                client_id=client_id,
            )
            if password is not None:
                self._securely_store_password(profile, password)
            self._write_config(config, self._cfg_file_path)
            self._cfg_exists = True
            return True
        except:
            _log.warn(self._get_keyring_failure_message())
            return False

    # ----------------------------------------------------------------------
    def save_as(self, profile: str, gis: GIS):
        """

        The ``save_as`` method saves and adds the provided :class:`~arcgis.gis.GIS` object to the profile.

        ================  ====================================================================
        **Parameter**     **Description**
        ----------------  --------------------------------------------------------------------
        name              Required String. The name of the profile to save.
        ----------------  --------------------------------------------------------------------
        gis               Required GIS. The connection object to update the profile with.
        ================  ====================================================================

        :return: Boolean. True if successful else False

        .. code-block:: python

            # Usage Example

            >>> gis = GIS("pro")
            >>> gis.ProfileManager.save_as("Profile_name", gis)


        """

        url = gis._url
        u = gis._username
        p = gis._password
        kf = gis._key_file
        cf = gis._cert_file
        ci = gis._client_id
        if profile.lower() in [p.lower() for p in self.list()]:
            cfg_file_path = self._cfg_file_path
            config = configparser.ConfigParser()
            if os.path.isfile(cfg_file_path):
                config.read(cfg_file_path)
            self._update_profile_data_in_config(
                config,
                profile=profile,
                url=url,
                username=u,
                key_file=kf,
                cert_file=cf,
                client_id=ci,
            )
            if p is not None:
                self._securely_store_password(profile, p)
            self._write_config(config, cfg_file_path)
            return True
        else:
            return self.create(
                profile=profile,
                url=url,
                username=u,
                key_file=kf,
                cert_file=cf,
                client_id=ci,
            )

    # ----------------------------------------------------------------------
    def _retrieve(self, profile):
        """gets the login information"""
        url, username, password, key_file, cert_file, client_id = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if profile.lower() in [p.lower() for p in self.list()]:
            cfg_file_path = self._cfg_file_path
            config = configparser.ConfigParser()
            if os.path.isfile(cfg_file_path):
                config.read(cfg_file_path)
            if config.has_option(profile, "url"):
                url = config[profile]["url"]
            if config.has_option(profile, "username"):
                username = config[profile]["username"]
            if config.has_option(profile, "key_file"):
                key_file = config[profile]["key_file"]
            if config.has_option(profile, "cert_file"):
                cert_file = config[profile]["cert_file"]
            if config.has_option(profile, "client_id"):
                client_id = config[profile]["client_id"]

            password = self._securely_get_password(profile)
        return url, username, password, key_file, cert_file, client_id

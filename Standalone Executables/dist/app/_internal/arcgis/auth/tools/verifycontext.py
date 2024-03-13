from ._lazy import LazyLoader

warnings = LazyLoader("warnings")
contextlib = LazyLoader("contextlib")
requests = LazyLoader("requests")
urllib3 = LazyLoader("urllib3")

_HISTORIC_ENVIRONMENTAL_SETTING = requests.Session.merge_environment_settings


@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.

        opened_adapters.add(self.get_adapter(url))

        settings = _HISTORIC_ENVIRONMENTAL_SETTING(
            self, url, proxies, stream, verify, cert
        )
        settings["verify"] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        requests.packages.urllib3.disable_warnings(
            category=urllib3.exceptions.InsecureRequestWarning
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = _HISTORIC_ENVIRONMENTAL_SETTING
        warnings.simplefilter("default", urllib3.exceptions.InsecureRequestWarning)
        for adapter in opened_adapters:
            try:
                adapter.close()
            except:  # pragma: no cover
                pass

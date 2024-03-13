from urllib.parse import urlparse


def validate_url(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

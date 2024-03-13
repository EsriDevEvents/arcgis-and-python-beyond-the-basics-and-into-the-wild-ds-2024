def is_numpy_array(image):
    """numpy arrays behave unpredictably in `isinstance()` func calls:
    do a hacky string comparison on the type() of image arg
    """
    return "numpy" in str(type(image)) and "array" in str(type(image))


def get_hash_numpy_array(image):
    return hash(str(image.tostring()))

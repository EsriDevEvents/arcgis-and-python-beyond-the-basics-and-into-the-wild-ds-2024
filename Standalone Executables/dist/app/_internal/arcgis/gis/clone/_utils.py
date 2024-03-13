import zipfile
import os

__all__ = ["_zipdir"]


#  --------------------------------------------------------------------
def _zipdir(src, dst, zip_name):
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

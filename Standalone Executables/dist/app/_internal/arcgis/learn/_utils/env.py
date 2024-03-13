import os
import sys
import traceback


HAS_BACKEND_SET = False
ARCGIS_ENABLE_TF_BACKEND = os.environ.get("ARCGIS_ENABLE_TF_BACKEND") == "1"
_LAMBDA_TEXT_CLASSIFICATION = os.environ.get("_LAMBDA_TEXT_CLASSIFICATION") == "1"

if os.environ.get("TF_CPP_MIN_LOG_LEVEL", None) is None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

HAS_TENSORFLOW = False
tf_import_exception = None


class FakeImport:
    def __getattr__(self, attr):
        return self

    def __call__(self, *args, **kwargs):
        return self


if _LAMBDA_TEXT_CLASSIFICATION:
    default_module = FakeImport()
    missing_modules = [
        "scipy",
        "scipy.stats",
        "spacy",
        "spacy.symbols",
        "spacy.blank",
        "matplotlib",
        "matplotlib.pyplot",
        "matplotlib.patches",
        "matplotlib.cm",
        "scipy.special",
        "PIL",
    ]
    for module_name in missing_modules:
        sys.modules[module_name] = default_module

try:
    import platform

    if ARCGIS_ENABLE_TF_BACKEND:
        import tensorflow as tf

        HAS_TENSORFLOW = True
    elif platform.system() == "Linux":
        import tensorflow as tf
except Exception as e:
    tf_import_exception = traceback.format_exc()
    pass


def enable_backend():
    global HAS_BACKEND_SET
    global HAS_TENSORFLOW
    global ARCGIS_ENABLE_TF_BACKEND

    if ARCGIS_ENABLE_TF_BACKEND:
        # if tf.__version__ == '2.0.0':
        if HAS_TENSORFLOW and not HAS_BACKEND_SET:
            # tf.keras.backend.set_image_data_format('channels_first')
            try:
                tf.compat.v1.enable_eager_execution()
            except:
                pass
            tf_set_gpu_memory_growth()
            tf_sample_op()
            tf.keras.backend.clear_session()
            HAS_BACKEND_SET = True


def tf_set_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def raise_tensorflow_import_error():
    if ARCGIS_ENABLE_TF_BACKEND:
        message = """
        Could not find tensorflow, Please install tensorflow using the following command
        \nconda install -c esri tensorflow-gpu=2.1.0
        """
        ex = Exception(message)
        raise (ex)
    else:
        message = """
        Please enable tensorflow by setting the required environment variable 'ARCGIS_ENABLE_TF_BACKEND' to '1' before importing arcgis
        \n for example the following code block needs to be executed before importing arcgis
        \n\n`import os; os.environ['ARCGIS_ENABLE_TF_BACKEND'] = '1'`
        """
        ex = Exception(message)
        raise (ex)


def tf_sample_op():
    a = tf.keras.layers.Conv2D(1, (3, 3))
    a = a(tf.zeros((1, 3, 20, 20))).numpy()


## Fastai Imports #######

HAS_FASTAI = False
fastai_import_exception = None


def do_fastai_imports():
    global HAS_FASTAI
    global fastai_import_exception

    try:
        import fastai
        import torch
        import torchvision
        import skimage

        HAS_FASTAI = True
    except Exception as e:
        fastai_import_exception = traceback.format_exc()
        pass

    try:
        from .patches import precondition
    except:
        pass


def fastai_installation_command():
    installation_steps = "Install them using 'conda install -c esri arcgis=1.8.1 pillow scikit-image'\n'conda install -c fastai -c pytorch fastai pytorch=1.4.0 torchvision=0.5.0 tensorflow-gpu=2.1.0'\n'conda install gdal=2.3.3'"

    return installation_steps


def raise_fastai_import_error(
    import_exception=fastai_import_exception, installation_steps=None, message=None
):
    if installation_steps is None:
        installation_steps = fastai_installation_command()
    if message is None:
        message = "This module requires fastai, PyTorch, torchvision and scikit-image as its dependencies."
    raise Exception(f"""{import_exception} \n\n{message}\n{installation_steps}""")


HAS_GDAL = False
gdal_import_exception = None
GDAL_INSTALL_MESSAGE = f"""
\nPlease install gdal using the following command
\nconda install gdal=2.3.3
""".strip()

try:
    from osgeo import gdal

    HAS_GDAL = True
except Exception as e:
    gdal_import_exception = traceback.format_exc()
    pass


def raise_gdal_import_error(import_exception=gdal_import_exception):
    message = "gdal is required to work with multispectral datasets."
    raise Exception(f"""{import_exception} \n\n{message}\n{GDAL_INSTALL_MESSAGE}""")


## Ipython inside ArcGIS Pro
import sys

_IS_ARCGISPRONOTEBOOK = None
using_mpl_inline = False


def is_arcgispronotebook():
    global using_mpl_inline
    if _IS_ARCGISPRONOTEBOOK is not None:
        if _IS_ARCGISPRONOTEBOOK:
            patch_arcgis_notebook()
        else:
            using_mpl_inline = False
        return _IS_ARCGISPRONOTEBOOK
    if os.path.basename(sys.executable) == "ArcGISPro.exe":
        ### This code will be execute only once in ArcGIS Pro notebooks
        if not using_mpl_inline:
            patch_arcgis_notebook()
            using_mpl_inline = True
        ###
        ### This following code block will be replaced by a flag exposed by Pro's Team (Vinay Vijayan)
        return True
        ###


def reload_IPython():
    if "IPython" in sys.modules:
        del sys.modules["IPython"]
    import IPython

    return IPython


def patch_arcgis_notebook():
    if reload_IPython().get_ipython() is not None:
        reload_IPython().get_ipython().run_line_magic("matplotlib", "inline")


is_arcgispronotebook()

from .env import enable_backend, do_fastai_imports

enable_backend()
do_fastai_imports()

from .coco_detection_utils import nested_tensor_from_tensor_list

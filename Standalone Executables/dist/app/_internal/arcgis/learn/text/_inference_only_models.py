import traceback
import warnings

warnings.filterwarnings("ignore", module="transformers")
HAS_TRANSFORMER = True

try:
    from pathlib import Path
    import json
    import os
    import torch
    from transformers import pipeline, logging

    logging.get_logger("filelock").setLevel(logging.ERROR)
    from .._utils.common import _get_device_id, _get_emd_path
    from fastprogress.fastprogress import progress_bar
except Exception as e:
    transformer_exception = "\n".join(
        traceback.format_exception(type(e), e, e.__traceback__)
    )
    HAS_TRANSFORMER = False


class InferenceOnlyModel:
    def __init__(self, backbone=None, task=None, **kwargs):
        self._task = task
        self.model = None
        self._backbone = backbone

        if "working_dir" in kwargs:
            self.working_dir = kwargs.get("working_dir")
        else:
            self.working_dir = Path.cwd()
        self._pretrained_path = kwargs.get("pretrained_path")
        self._device = _get_device_id()
        self.logger = logging.get_logger()
        self.logger.setLevel(logging.ERROR)
        self._load_model()

    def _load_model(self):
        try:
            if self._pretrained_path:
                self.model = pipeline(
                    self._task, model=self._pretrained_path, device=self._device
                )
            else:
                self.model = pipeline(
                    self._task, model=self._backbone, device=self._device
                )
        except Exception as e:
            error_message = (
                f"`{self._backbone}` is not valid backbone name for {self._task} task.\n"
                f"For selecting backbone name for {self._task} task, kindly visit:- "
                f"https://huggingface.co/models?pipeline_tag={self._task} "
            )
            raise Exception(error_message)

    def _create_emd(self, name_or_path):
        emd_template = {}
        emd_template.update({"ModelName": self.__class__.__name__})
        emd_template.update({"architectures": self.model.model.config.architectures})
        path = Path(name_or_path)
        name = path.parts[-1]
        with open(os.path.join(path, f"{name}.emd"), "w") as f:
            f.write(json.dumps(emd_template))

    def save(self, name_or_path):
        """
        Saves the translator model files on a specified path on the local disk.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        name_or_path            Required string. Path to save
                                model files on the local disk.
        =====================   ===========================================

        :return: Absolute path for the saved model
        """

        if "\\" in name_or_path or "/" in name_or_path:
            path = name_or_path
        else:
            path = os.path.join(self.working_dir, "models", name_or_path)
        self.model.save_pretrained(path)
        self._create_emd(path)
        return Path(path).absolute()

    @classmethod
    def from_model(cls, emd_path, **kwargs):
        """
        Creates an :class:`~arcgis.learn.text.SequenceToSequence` model object from an
        Esri Model Definition (EMD) file.

        =====================   ===========================================
        **Parameter**            **Description**
        ---------------------   -------------------------------------------
        emd_path                Required string. Path to
                                Esri Model Definition(EMD) file or the folder
                                with saved model files.
        =====================   ===========================================

        :return: :class:`~arcgis.learn.text.SequenceToSequence` Object
        """
        emd_path = _get_emd_path(emd_path)
        with open(emd_path) as f:
            emd_json = json.loads(f.read())
        model_path = Path(emd_path).parent
        cls_object = cls(pretrained_path=str(model_path))
        return cls_object

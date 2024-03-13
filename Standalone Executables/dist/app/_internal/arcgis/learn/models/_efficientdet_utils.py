# Variation of TFLite Model Maker Object Detector
# https://github.com/tensorflow/examples/tree/master/tensorflow_examples/lite/model_maker
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from email.mime import image
import math
from dataclasses import dataclass
from re import S
from typing import Optional, Union


from ._arcgis_model import SaveModelCallback
from .._utils.env import HAS_TENSORFLOW, ARCGIS_ENABLE_TF_BACKEND

try:
    import os
    from types import MethodType
    import numpy as np
    from fastai.basics import *
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
    from fastai.vision.data import ImageDataBunch
    from fastai.callbacks import EarlyStoppingCallback, OneCycleScheduler
    from fastai.callback import annealing_cos
    from fastai.core import ifnone
    import tensorflow as tf
    from tensorflow_examples.lite.model_maker.core.task.object_detector import (
        ObjectDetector,
    )
    from tensorflow_examples.lite.model_maker.core.task.configs import (
        QuantizationConfig,
    )
    from tensorflow_examples.lite.model_maker.core.task.custom_model import _get_params
    from tensorflow_examples.lite.model_maker.core.task import model_spec
    from tensorflow_examples.lite.model_maker.core import compat
    from tensorflow_examples.lite.model_maker.core.data_util import (
        object_detector_dataloader,
        dataloader,
    )
    from tensorflow_examples.lite.model_maker.core.task import model_spec as ms
    from tensorflow_examples.lite.model_maker.core.task.model_spec import (
        object_detector_spec,
    )
    from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import (
        train,
        util_keras,
    )
    from tensorflow_examples.lite.model_maker.core.data_util.object_detector_dataloader import (
        DataLoader as tflite_data_loader,
    )
    from lxml import etree
    from tensorflow_examples.lite.model_maker.third_party.efficientdet.dataset import (
        tfrecord_util,
    )

    HAS_FASTAI = True
except:
    HAS_FASTAI = False


def check_data_sanity(data, dataset_types=[]):
    try:
        if isinstance(data, ImageDataBunch):
            for k, v in data.class_mapping.items():
                if k == 0 and v != "background":
                    import warnings

                    warnings.warn("class 0 is reserved only for background\n")
                    return False
            return True
        else:
            return False
    except Exception as e:
        return False


def _get_image_tensor(image):
    return tf.expand_dims(tf.convert_to_tensor(np.array(image)), axis=0)


def _get_ann_files(data):
    if data._is_empty:
        return [None, None]

    val_split_pct = data._val_split_pct
    labels_path = os.path.join(data.orig_path, "labels")
    if not os.path.isdir(labels_path):
        return [None, None]
    else:
        tot_files = len(os.listdir(labels_path))
        files = sorted(os.listdir(labels_path))
        files = [f[:-4] for f in files if os.path.isfile(os.path.join(labels_path, f))]
        num_val_files = min(
            max(math.floor((val_split_pct * tot_files)), 1), tot_files - 1
        )
        train_files = files[:-num_val_files]
        val_files = files[-num_val_files:]

        return [train_files, val_files]


def _calculate_avg_loss(avg_loss, loss, beta=0.98):
    return beta * avg_loss + (1 - beta) * loss


def _calculate_smoothed_loss(avg_loss, iteration, beta=0.98):
    return avg_loss / (1 - beta**iteration)


class EmptyLoader(dataloader.DataLoader):
    def __init__(self, label_map):
        super(EmptyLoader, self).__init__(dataset=None, size=1)
        self.label_map = label_map


def _get_tf_data_loader(data, ann_files=None):
    dataset_type = None
    if hasattr(data, "dataset_type"):
        dataset_type = getattr(data, "dataset_type")
        if data._is_empty:
            return EmptyLoader(label_map=data.class_mapping)

        if dataset_type == "PASCAL_VOC_rectangles":
            try:
                return tflite_data_loader.from_pascal_voc(
                    os.path.join(data.orig_path, "images"),
                    os.path.join(data.orig_path, "labels"),
                    label_map=data.class_mapping,
                    annotation_filenames=ann_files,
                )
            except:
                mapping = dict()
                for k in data.class_mapping:
                    mapping[k] = str(k)
                return tflite_data_loader.from_pascal_voc(
                    os.path.join(data.orig_path, "images"),
                    os.path.join(data.orig_path, "labels"),
                    label_map=mapping,
                    annotation_filenames=ann_files,
                )
        elif dataset_type == "KITTI_rectangles":
            return None
        else:
            dataset_type = None
    if dataset_type == None:
        raise Exception(f"Enter only compatible datasets\n")


class ConstantLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
    """Constant learning rate schedule."""

    def __init__(self, lr):
        """Build a ConstantLrSchedule."""
        super().__init__()
        self.default_lr = defaults.lr
        self.lr = lr

    def restart(self):
        self.lr = defaults.lr

    def __call__(self, step):
        return self.lr


class AnnealedLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
    """Annealed learning rate schedule."""

    def __init__(self, vals: Union[tuple, int], n_iter: int):
        """Build a AnnealedLrSchedule."""
        super().__init__()
        self.start, self.end = (
            (vals[0], vals[1]) if isinstance(vals, tuple) else (vals, 0)
        )
        self.n_iter = max(1, n_iter)
        self.func = AnnealedLrSchedule.annealing_exp
        self.n = 0
        self.lr = self.start

    @classmethod
    def annealing_exp(cls, start: float, end: float, pct: float) -> float:
        "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return start * (end / start) ** pct

    def restart(self):
        self.n = 0

    def step(self):
        "Return next value along annealed schedule."
        self.n += 1
        self.lr = AnnealedLrSchedule.annealing_exp(
            self.start, self.end, self.n / self.n_iter
        )

    @property
    def is_done(self) -> bool:
        "Return `True` if schedule completed."
        return self.n >= self.n_iter

    def __call__(self, step):
        return self.lr


from tensorflow_examples.lite.model_maker.third_party.efficientdet import utils


def _get_optimizer(params):
    """Get optimizer."""
    try:
        learning_rate = train_lib.learning_rate_schedule(params)
    except:
        if params["lr_decay_method"] == "annealed":
            learning_rate = AnnealedLrSchedule(
                (params["start_lr"], params["end_lr"]), params["num_it"]
            )
            learning_rate = params["start_lr"]
        else:
            learning_rate = params["constant_lr"]

    momentum = params["momentum"]
    if params["optimizer"].lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=momentum)
    elif params["optimizer"].lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=momentum)
    else:
        raise ValueError("optimizers should be adam or sgd")

    moving_average_decay = params["moving_average_decay"]
    if moving_average_decay:
        from tensorflow_addons import optimizers as tfa_optimizers

        optimizer = tfa_optimizers.MovingAverage(
            optimizer, average_decay=moving_average_decay, dynamic_decay=True
        )
    precision = utils.get_precision(params["strategy"], params["mixed_precision"])
    if precision == "mixed_float16" and params["loss_scale"]:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            optimizer, initial_scale=params["loss_scale"]
        )
    return optimizer


def _setup_model(model, config, build=True):
    """Build and compile model."""
    if build is True:
        model.build((None, *config.image_size, 3))
    model.compile(
        steps_per_execution=config.steps_per_execution,
        optimizer=_get_optimizer(config.as_dict()),
        loss={
            train_lib.BoxLoss.__name__: train_lib.BoxLoss(
                config.delta, reduction=tf.keras.losses.Reduction.NONE
            ),
            train_lib.BoxIouLoss.__name__: train_lib.BoxIouLoss(
                config.iou_loss_type,
                config.min_level,
                config.max_level,
                config.num_scales,
                config.aspect_ratios,
                config.anchor_scale,
                config.image_size,
                reduction=tf.keras.losses.Reduction.NONE,
            ),
            train_lib.FocalLoss.__name__: train_lib.FocalLoss(
                config.alpha,
                config.gamma,
                label_smoothing=config.label_smoothing,
                reduction=tf.keras.losses.Reduction.NONE,
            ),
            tf.keras.losses.SparseCategoricalCrossentropy.__name__: tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            ),
        },
    )
    return model


class EfficientDetTrainer(ObjectDetector):
    def __init__(
        self,
        model_spec: object_detector_spec.EfficientDetModelSpec,
        label_map: Dict[int, str],
        representative_data: Optional[object_detector_dataloader.DataLoader] = None,
    ) -> None:
        self._data = representative_data
        self._config = None
        self._ann_files = _get_ann_files(self._data)
        self._data_loader = _get_tf_data_loader(representative_data, self._ann_files[0])
        self._val_loader = _get_tf_data_loader(representative_data, self._ann_files[1])
        super().__init__(model_spec, label_map, self._data_loader)
        self._build_model = True

    def setup_model(self):
        self._config = self.model_spec.config
        batch_size = self._data.batch_size
        self._train_ds = None
        self._valid_ds = None
        self._steps_per_epoch = None

        self._valid_steps_per_epoch = None
        self._validation_steps = None
        with self.model_spec.ds_strategy.scope():
            if not self._data._is_empty:
                self._train_ds, self._steps_per_epoch, _ = self._get_dataset_and_steps(
                    self._data_loader, batch_size, is_training=True
                )
                (
                    self._valid_ds,
                    self._validation_steps,
                    val_json_file,
                ) = self._get_dataset_and_steps(
                    self._val_loader, batch_size, is_training=False
                )
                self._config.update(
                    dict(
                        steps_per_epoch=self._steps_per_epoch,
                        eval_samples=batch_size * self._validation_steps,
                        batch_size=batch_size,
                        val_json_file=val_json_file,
                    )
                )  # validation
            else:
                self._config.update(
                    dict(batch_size=batch_size, steps_per_epoch=batch_size)
                )
            _setup_model(self.model, self._config, self._build_model)
            self._build_model = False
            train.init_experimental(self._config)

    def get_opt_func(self):
        self._config = self.model_spec.config.as_dict()
        if self._config["optimizer"].lower() == "sgd":
            optimizer = tf.keras.optimizers.SGD
        elif self._config["optimizer"].lower() == "adam":
            optimizer = tf.keras.optimizers.Adam
        else:
            raise ValueError("optimizers should be adam or sgd")

        return optimizer

    def get_tf_dataset(self, data_loader, batch_size=16):
        with self.model_spec.ds_strategy.scope():
            ds, steps_per_epoch, _ = self._get_dataset_and_steps(
                data_loader, batch_size, is_training=True
            )
            return ds

    def fit(
        self,
        epochs: int,
        lr: Union[Floats, slice] = defaults.lr,
        wd: Floats = None,
        callbacks: Collection[Callback] = None,
        annealed_schedule=False,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_it: int = 100,
        freeze_model: bool = True,
        one_cycle: bool = False,
    ) -> None:
        if self._data._is_empty:
            print("\n\nInvalid Fit\n\n")
            return

        self._lr_decay_method = self.model_spec.config.lr_decay_method
        if annealed_schedule is True:
            self.model_spec.config.lr_decay_method = "annealed"
            self.model_spec.config.start_lr = start_lr
            self.model_spec.config.end_lr = end_lr
            self.model_spec.config.num_it = num_it
        elif one_cycle:
            self.model_spec.config.lr_decay_method = "constant"
            self.model_spec.config.constant_lr = lr

        if freeze_model is False:
            self.model.config.var_freeze_expr = None
            self.model_spec.config.var_freeze_expr = None
        else:
            self.model.config.var_freeze_expr = "(efficientnet|fpn_cells|resample_p6)"
            self.model_spec.config.var_freeze_expr = (
                "(efficientnet|fpn_cells|resample_p6)"
            )

        num_epochs = self.model_spec.config.num_epochs
        learning_rate = self.model_spec.config.learning_rate
        self.model_spec.config.num_epochs = epochs
        self.model_spec.config.learning_rate = lr

        self.setup_model()

        with self.model_spec.ds_strategy.scope():
            self.model.fit(
                self._train_ds,
                epochs=epochs,
                steps_per_epoch=self._steps_per_epoch,
                verbose=0,
                callbacks=callbacks,
                validation_data=self._valid_ds,
                validation_steps=self._validation_steps,
            )
        self.model_spec.config.lr_decay_method = self._lr_decay_method
        self.model_spec.config.num_epochs = num_epochs
        self.model_spec.config.learning_rate = learning_rate

    def _export_saved_model(self, saved_model_dir: str) -> None:
        model = self.model
        original_optimizer = model.optimizer
        model.optimizer = None

        tf.saved_model.save(model, saved_model_dir)

        model.optimizer = original_optimizer

    def _save_weights(
        self, name, path, model_dir, quantized=False, tflite=True, **kwargs
    ):
        import os

        if self._data._is_empty:
            quantized = False

        model_save_path = path / model_dir / f"{name}.tflite"
        tflite_filename = f"{name}.tflite"
        saved_model_dir = f"weights"
        export_dir = path / model_dir
        if not tf.io.gfile.exists(export_dir):
            tf.io.gfile.makedirs(export_dir)

        if quantized is False:
            kwargs["quantized_config"] = None
            if self._data._is_empty:
                kwargs["quantization_config"] = None
        else:
            kwargs["quantized_config"] = "default"
            kwargs["quantization_config"] = "default"

        tflite_filepath = os.path.join(export_dir, tflite_filename)
        export_tflite_kwargs, kwargs = _get_params(self._export_tflite, **kwargs)

        if tflite:
            self._export_tflite(tflite_filepath, **export_tflite_kwargs)
        else:
            model_save_path = path / model_dir / f"{name}.pb"

        saved_model_dir = os.path.join(export_dir, saved_model_dir)
        export_saved_model_kwargs, kwargs = _get_params(
            self._export_saved_model, **kwargs
        )
        self._export_saved_model(saved_model_dir, **export_saved_model_kwargs)
        if not tflite:
            import shutil

            shutil.copyfile(
                os.path.join(saved_model_dir, "saved_model.pb"),
                os.path.join(export_dir, f"{name}.pb"),
            )

        return model_save_path

    def compute_metrics(self):
        metrics = self.evaluate(self._val_loader)
        metrics.update(
            (key, value * 1.0) if value >= 0.0 else (key, np.float64(0.0))
            for key, value in metrics.items()
        )
        return metrics

    def _get_processed_output(self, output, thresh=0.4):
        boxes = output[0]
        scores = output[1]
        labels = output[2]

        detections = [
            boxes[:, :, 0],
            boxes[:, :, 1],
            boxes[:, :, 2],
            boxes[:, :, 3],
            scores,
            labels,
        ]

        detections = tf.stack(detections, axis=-1)[0].numpy()
        boxes = detections[:, 0:4]
        classes = detections[:, 5].astype(int).tolist()
        scores_ = detections[:, 4].tolist()

        predictions = []
        labels = []
        scores = []

        for i in range(boxes.shape[0]):
            if scores_ is None or scores_[i] > thresh:
                ymin, xmin, ymax, xmax = boxes[i].tolist()
                box = [xmin, ymin, xmax - xmin, ymax - ymin]
                predictions.append(box)
                labels.append(classes[i])
                scores.append(scores_[i])

        return predictions, labels, scores

    def predict(self, image, thresh, nms_overlap):
        original_optimizer = self.model.optimizer
        original_thresh = self.model_spec.config.nms_configs.iou_thresh
        self.model.optimizer = None
        self.model_spec.config.nms_configs.iou_thresh = thresh
        export_model = object_detector_spec.ExportModel(
            self.model, self.model_spec.config
        )
        self.model.optimizer = original_optimizer
        self.model_spec.config.nms_configs.iou_thresh = original_thresh
        output = export_model(_get_image_tensor(image))
        return self._get_processed_output(output, thresh)

    @classmethod
    def create(cls, train_data, backbone_name: str, train_whole_model: bool = False):
        """Creates a model for object detection.

        Args:
            train_data: Training data.
            backbone_name: BackBone name for the model.
            batch_size: Batch size for training.
            train_whole_model: Boolean, False by default. If true, train the whole
            model. Otherwise, only train the layers that are not match
            `model_spec.config.var_freeze_expr`.

        Returns:
            An instance based on ObjectDetector.
        """
        spec = model_spec.get(backbone_name)
        spec = ms.get(spec)
        spec.config.batch_size = train_data.batch_size
        if train_whole_model:
            spec.config.var_freeze_expr = None
        if compat.get_tf_behavior() not in spec.compat_tf_versions:
            raise ValueError(
                "Incompatible versions. Expect {}, but got {}.".format(
                    spec.compat_tf_versions, compat.get_tf_behavior()
                )
            )

        object_detector = cls(spec, train_data.class_mapping, train_data)
        with object_detector.model_spec.ds_strategy.scope():
            object_detector.create_model()
        object_detector.model.train_step = MethodType(train_step, object_detector.model)
        object_detector.setup_model()
        return object_detector


def train_step(self, data):
    """Train step.

    Args:
        data: Tuple of (images, labels). Image tensor with shape [batch_size,
        height, width, 3]. The height and width are fixed and equal.Input labels
        in a dictionary. The labels include class targets and box targets which
        are dense label maps. The labels are generated from get_input_fn
        function in data/dataloader.py.

    Returns:
        A dict record loss info.
    """
    images, labels = data
    if self.config.img_summary_steps:
        with self.summary_writer.as_default():
            tf.summary.image("input_image", images)
    with tf.GradientTape() as tape:
        if len(self.config.heads) == 2:
            cls_outputs, box_outputs, seg_outputs = util_keras.fp16_to_fp32_nested(
                self(images, training=True)
            )
            loss_dtype = cls_outputs[0].dtype
        elif "object_detection" in self.config.heads:
            cls_outputs, box_outputs = util_keras.fp16_to_fp32_nested(
                self(images, training=True)
            )
            loss_dtype = cls_outputs[0].dtype
        elif "segmentation" in self.config.heads:
            (seg_outputs,) = util_keras.fp16_to_fp32_nested(self(images, training=True))
            loss_dtype = seg_outputs.dtype
        else:
            raise ValueError("No valid head found: {}".format(self.config.heads))
        labels = util_keras.fp16_to_fp32_nested(labels)

        total_loss = 0
        loss_vals = {}
        if "object_detection" in self.config.heads:
            det_loss = self._detection_loss(cls_outputs, box_outputs, labels, loss_vals)
            total_loss += det_loss
        if "segmentation" in self.config.heads:
            seg_loss_layer = self.loss[
                tf.keras.losses.SparseCategoricalCrossentropy.__name__
            ]
            seg_loss = seg_loss_layer(labels["image_masks"], seg_outputs)
            total_loss += seg_loss
            loss_vals["seg_loss"] = seg_loss

        reg_l2_loss = self._reg_l2_loss(self.config.weight_decay)
        loss_vals["reg_l2_loss"] = reg_l2_loss
        total_loss += tf.cast(reg_l2_loss, loss_dtype)
        if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            scaled_loss = self.optimizer.get_scaled_loss(total_loss)
            optimizer = self.optimizer.inner_optimizer
        else:
            scaled_loss = total_loss
            optimizer = self.optimizer
    loss_vals["loss"] = total_loss
    if isinstance(
        optimizer.learning_rate, tf.optimizers.schedules.LearningRateSchedule
    ):
        loss_vals["learning_rate"] = optimizer.learning_rate(optimizer.iterations)
    else:
        loss_vals["learning_rate"] = optimizer.learning_rate

    trainable_vars = self._freeze_vars()
    scaled_gradients = tape.gradient(scaled_loss, trainable_vars)
    if isinstance(self.optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
    else:
        gradients = scaled_gradients
    if self.config.clip_gradients_norm > 0:
        clip_norm = abs(self.config.clip_gradients_norm)
        gradients = [
            tf.clip_by_norm(g, clip_norm) if g is not None else None for g in gradients
        ]
        gradients, _ = tf.clip_by_global_norm(gradients, clip_norm)
        loss_vals["gradient_norm"] = tf.linalg.global_norm(gradients)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss_vals


from .._utils.fastai_tf_fit import (
    TfLearner,
    TfRegularizer,
    tf_flatten_model,
    _pytorch_to_tf_batch,
    _pytorch_to_tf,
)
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import (
    train_lib,
)


@dataclass
class EfficientDetLearner(TfLearner):
    _trainer: EfficientDetTrainer = None
    tf_dataset: tf.data.Dataset = None
    _compute_mean_avp = False

    def __post_init__(self) -> None:
        "Setup path,metrics, callbacks and ensure model directory exists."
        if self.tf_dataset is None or self._trainer is None:
            raise ValueError("\nInvalid arguments\n")

        self.path = Path(ifnone(self.path, self.data.path))
        (self.path / self.model_dir).mkdir(parents=True, exist_ok=True)
        self.metrics = listify(self.metrics)
        self._freeze_model = True
        self.recorder = TFRecorder(0, 0)
        if not self.layer_groups:
            self.layer_groups = tf_flatten_model(self.model)

    def init(self, init):
        raise NotImplementedError

    def _save_tflite(self, name, post_processed=True, quantized=False, **kwargs):
        return self._trainer._save_weights(name, self.path, self.model_dir, quantized)

    def compute_metrics(self):
        return self._trainer.compute_metrics()

    def save(self, name, return_path=True, **kwargs):
        "Save model with `name` to `self.model_dir`."
        return self._trainer._save_weights(
            name, self.path, self.model_dir, tflite=False
        )

    def load(self, file, device=None, **kwargs):
        weights_save_path = str(
            self.path / self.model_dir / f"weights/variables/variables"
        )
        self._trainer.model.load_weights(weights_save_path)

    def fit(
        self,
        epochs: int,
        lr: Union[Floats, slice] = defaults.lr,
        wd: Floats = None,
        callbacks: Collection[Callback] = None,
        annealed_schedule=False,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_it: int = 100,
        one_cycle: bool = False,
    ) -> None:
        lr = self.lr_range(lr)
        for i in range(0, len(callbacks)):
            if isinstance(callbacks[i], EarlyStoppingCallback):
                callbacks[i] = tf.keras.callbacks.EarlyStopping(
                    min_delta=0.001, patience=5
                )
            if isinstance(callbacks[i], SaveModelCallback):
                callbacks[i] = CheckPointCallback(callback=callbacks[i])
            try:
                from .._utils.tensorboard_utils import ArcGISTBCallback

                if isinstance(callbacks[i], ArcGISTBCallback):
                    log_dir = callbacks[i]._base_dir
                    callbacks[i] = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
            except:
                pass

        self.recorder = TFRecorder(epochs, len(self.data.train_dl))
        callbacks = [self.recorder] + callbacks
        self._trainer.fit(
            epochs,
            lr,
            wd,
            callbacks,
            annealed_schedule,
            start_lr,
            end_lr,
            num_it,
            freeze_model=self._freeze_model,
            one_cycle=one_cycle,
        )

    def unfreeze(self):
        self._freeze_model = False

    def freeze(self) -> None:
        "Freeze up to last layer."
        self._freeze_model = True

    def predict(self, image, thresh, nms_overlap):
        return self._trainer.predict(image, thresh, nms_overlap)

    def _get_annotations_from_xml(self, data):
        predictions = []
        if "object" in data:
            for obj in data["object"]:
                if not "difficult" in obj or obj["difficult"] == "Unspecified":
                    difficult = False
                else:
                    difficult = bool(int(obj["difficult"]))
                if difficult:
                    continue

                xmin = int(math.floor(float(obj["bndbox"]["xmin"])))
                ymin = int(math.floor(float(obj["bndbox"]["ymin"])))
                xmax = int(math.floor(float(obj["bndbox"]["xmax"])))
                ymax = int(math.floor(float(obj["bndbox"]["ymax"])))

                predictions.append([xmin, ymin, xmax - xmin, ymax - ymin])
        return predictions

    def get_gt_batches(self, nbatches, type_data_loader="validation"):
        batches = []
        data = self._trainer._data
        labels_path = os.path.join(data.orig_path, "labels")
        images_path = os.path.join(data.orig_path, "images")
        if not os.path.isdir(labels_path):
            return batches
        ann_files = self._trainer._ann_files[0]
        if type_data_loader == "validation":
            ann_files = self._trainer._ann_files[1]

        i = 0
        while i < nbatches and i < len(ann_files):
            filename = os.path.join(labels_path, ann_files[i] + ".xml")
            if os.path.isfile(filename):
                with tf.io.gfile.GFile(filename, "r") as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                xml_dict = tfrecord_util.recursive_parse_xml_to_dict(xml)["annotation"]
                ann = self._get_annotations_from_xml(xml_dict)
                batches.append([os.path.join(images_path, xml_dict["filename"]), ann])
                i += 1
            else:
                continue

        return batches


from time import time
from fastprogress.fastprogress import master_bar, progress_bar


def get_monitor_value(self):
    "Pick the monitored value."
    if self.monitor == "trn_loss" and len(self.learn.recorder.losses) == 0:
        return None
    elif len(self.learn.recorder.val_losses) == 0:
        return None
    values = {
        "train_loss": self.learn.recorder.losses[-1],
        "valid_loss": self.learn.recorder.val_losses[-1],
    }
    if values["valid_loss"] is None:
        return
    from warnings import warn

    if values.get(self.monitor) is None:
        warn(
            f'{self.__class__} conditioned on metric `{self.monitor}` which is not available. Available metrics are: {", ".join(map(str, self.learn.recorder.names[1:-1]))}'
        )
    return values.get(self.monitor)


class CheckPointCallback(tf.keras.callbacks.Callback):
    def __init__(self, callback):
        self._callback = callback
        self._callback.get_monitor_value = MethodType(get_monitor_value, self._callback)

    def on_train_begin(self, logs=None):
        self._callback.on_train_begin()

    def on_epoch_end(self, epoch, logs=None):
        self._callback.on_epoch_end(epoch, stop_training=self.model.stop_training)


class TFRecorder(tf.keras.callbacks.Callback):
    def __init__(self, epochs, num_batches):
        super(TFRecorder, self).__init__()
        self._iteration = 0
        self.epochs = epochs
        self.batches_per_epoch = num_batches
        self.silent = False
        self.add_time = True
        self.pbar_iter = None
        self.mbar_iter = None

    def on_train_begin(self, logs=None):
        "Initialize optimizer and learner hyperparameters."
        self.losses = []
        self.epoch_losses = []
        self.val_losses = []
        self.lrs = []
        self.nb_batches = []
        self.avg_loss = 0.0
        self.smoothed_loss = 0.0
        self.num_batch = 0
        self.pbar = master_bar(range(self.epochs))
        self.pbar_iter = iter(self.pbar)
        self.names = ["epoch", "train_loss", "valid_loss", "time"]
        self.pbar.write(self.names, table=True)

    def on_epoch_begin(self, epoch, logs=None):
        next(self.pbar_iter)
        self.start_epoch = time()
        self.mbar = progress_bar(range(self.batches_per_epoch), parent=self.pbar)
        self.mbar_iter = iter(self.mbar)

    def on_train_batch_begin(self, batch, logs=None):
        lr = 0.0
        next(self.mbar_iter)
        if isinstance(
            self.model.optimizer.learning_rate,
            tf.optimizers.schedules.LearningRateSchedule,
        ):
            lr = (float)(
                tf.keras.backend.get_value(
                    self.model.optimizer.lr(self.model.optimizer.iterations)
                )
            )
        else:
            lr = (float)(tf.keras.backend.get_value(self.model.optimizer.lr))
        self.lrs.append(lr)

    def on_train_batch_end(self, batch, logs=None):
        "Determine if loss has runaway and we should stop."
        loss = logs.get("loss")
        self.avg_loss = _calculate_avg_loss(self.avg_loss, loss)
        self.smoothed_loss = _calculate_smoothed_loss(
            self.avg_loss, self._iteration + 1
        )
        self.losses.append(self.smoothed_loss)
        self._iteration = self._iteration + 1
        self.num_batch = batch + 1
        if self.pbar is not None and hasattr(self.pbar, "child"):
            self.pbar.child.comment = f"{self.smoothed_loss:.4f}"

    def on_epoch_end(self, epoch, logs=None):
        "Record at end of epoch."
        self.val_losses.append(logs.get("val_loss"))
        self.epoch_losses.append(self.smoothed_loss)
        self.nb_batches.append(self.num_batch)
        self.format_stats([epoch, self.smoothed_loss, logs.get("val_loss")])

        try:
            next(self.mbar_iter)
        except:
            pass

    def on_train_end(self, logs=None):
        try:
            next(self.pbar_iter)
        except:
            pass


class EfficientLRFinder(tf.keras.callbacks.Callback):
    "Causes `learn` to go on a mock training from `start_lr` to `end_lr` for `num_it` iterations."

    def __init__(
        self,
        learn: Learner,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_it: int = 100,
        stop_div: bool = True,
    ):
        super(EfficientLRFinder, self).__init__()
        self.learn = learn
        self.data, self.stop_div = learn.data, stop_div
        # To avoid validating if the train_dl has less than num_it batches, we put aside the valid_dl and remove it
        # during the call to fit.
        import copy

        self.sched = AnnealedLrSchedule((start_lr, end_lr), num_it)
        self.valid_dl = copy.deepcopy(learn.data.valid_dl)
        self.data.valid_dl = None
        self._iteration = 0

    def on_train_begin(self, logs=None):
        "Initialize optimizer and learner hyperparameters."
        self.model.save_weights("tmp")
        self.stop, self.best_loss, self.avg_loss = False, np.Inf, 0.0
        tf.keras.backend.set_value(self.model.optimizer.lr, self.sched(0))

    def on_train_batch_end(self, batch, logs=None):
        "Determine if loss has runaway and we should stop."
        loss = logs.get("loss")
        self.avg_loss = _calculate_avg_loss(self.avg_loss, loss)
        smoothed_loss = _calculate_smoothed_loss(self.avg_loss, self._iteration + 1)
        if self._iteration == 0 or smoothed_loss < self.best_loss:
            self.best_loss = smoothed_loss

        self.sched.step()
        tf.keras.backend.set_value(
            self.model.optimizer.lr, tf.keras.backend.get_value(self.sched(0))
        )

        if self.sched.is_done or (
            self.stop_div
            and (smoothed_loss > 4 * self.best_loss or np.isnan(smoothed_loss))
        ):
            self.stop = self._iteration
            # We use the smoothed loss to decide on the stopping since it's less shaky.
            self.model.stop_training = True

        self._iteration += 1

    def on_train_end(self, logs=None) -> None:
        "Cleanup learn model weights disturbed during LRFind exploration."
        # restore the valid_dl we turned off on `__init__`
        self.data.valid_dl = self.valid_dl
        self.model.load_weights("tmp")
        self._iteration = 0
        if hasattr(self.learn.model, "reset"):
            self.learn.model.reset()
        print(
            "LR Finder is complete, type {learner_name}.recorder.plot() to see the graph."
        )

    def on_epoch_end(self, epoch, logs=None):
        if self.stop:
            self.model.stop_training = True


def efficient_lr_find(
    learn: Learner,
    start_lr: Floats = 1e-7,
    end_lr: Floats = 10,
    num_it: int = 100,
    stop_div: bool = True,
    **kwargs: Any,
):
    "Explore lr from `start_lr` to `end_lr` over `num_it` iterations in `learn`. If `stop_div`, stops when loss diverges."
    start_lr = learn.lr_range(start_lr)
    start_lr = np.array(start_lr) if is_listy(start_lr) else start_lr
    end_lr = learn.lr_range(end_lr)
    end_lr = np.array(end_lr) if is_listy(end_lr) else end_lr
    cb = EfficientLRFinder(learn, start_lr, end_lr, num_it, stop_div)
    a = int(np.ceil(num_it / len(learn.data.train_dl)))
    learn.fit(
        a,
        start_lr,
        callbacks=[cb],
        annealed_schedule=True,
        end_lr=end_lr,
        num_it=num_it,
        **kwargs,
    )


class TFOneCycleScheduler(tf.keras.callbacks.Callback):
    "Manage 1-Cycle style training as outlined in Leslie Smith's [paper](https://arxiv.org/pdf/1803.09820.pdf)."

    def __init__(self, learn: EfficientDetLearner, initial_epoch=0, scheduler=None):
        super(TFOneCycleScheduler, self).__init__()
        self.learn = learn
        self._scheduler = scheduler
        self._initial_epoch = 0

    def jump_to_epoch(self, epoch: int):
        for _ in range(len(self.learn.data.train_dl) * epoch):
            self.on_train_batch_end(epoch)

    def on_train_begin(self, logs=None):
        "Initialize our optimization params based on our annealing schedule."
        epoch = self._initial_epoch
        self._scheduler.start_epoch = ifnone(self._scheduler.start_epoch, epoch)
        self._scheduler.tot_epochs = ifnone(
            self._scheduler.tot_epochs, self.params["epochs"]
        )
        n = len(self.learn.data.train_dl) * self._scheduler.tot_epochs
        a1 = int(n * self._scheduler.pct_start)
        a2 = n - a1
        self._scheduler.phases = ((a1, annealing_cos), (a2, annealing_cos))
        low_lr = self._scheduler.lr_max / self._scheduler.div_factor
        self._scheduler.lr_scheds = self._scheduler.steps(
            (low_lr, self._scheduler.lr_max),
            (
                self._scheduler.lr_max,
                self._scheduler.lr_max / self._scheduler.final_div,
            ),
        )
        self._scheduler.mom_scheds = self._scheduler.steps(
            self._scheduler.moms, (self._scheduler.moms[1], self._scheduler.moms[0])
        )
        tf.keras.backend.set_value(
            self.model.optimizer.lr,
            tf.keras.backend.get_value(self._scheduler.lr_scheds[0].start),
        )
        tf.keras.backend.set_value(
            self.model.optimizer.momentum,
            tf.keras.backend.get_value(self._scheduler.mom_scheds[0].start),
        )
        self._scheduler.idx_s = 0

        self.jump_to_epoch(epoch)

    def on_train_batch_end(self, batch, logs=None):
        "Take one step forward on the annealing schedule for the optim params."
        if self._scheduler.idx_s >= len(self._scheduler.lr_scheds):
            self.model.stop_training = True
            return
        tf.keras.backend.set_value(
            self.model.optimizer.lr,
            tf.keras.backend.get_value(
                self._scheduler.lr_scheds[self._scheduler.idx_s].step()
            ),
        )
        tf.keras.backend.set_value(
            self.model.optimizer.momentum,
            tf.keras.backend.get_value(
                self._scheduler.mom_scheds[self._scheduler.idx_s].step()
            ),
        )

        # when the current schedule is complete we move onto the next
        # schedule. (in 1-cycle there are two schedules)
        if self._scheduler.lr_scheds[self._scheduler.idx_s].is_done:
            self._scheduler.idx_s += 1

    def on_epoch_end(self, epoch, logs=None):
        "Tell Learner to stop if the cycle is finished."
        if epoch > self._scheduler.tot_epochs:
            self.model.stop_training = True


def tf_fit_one_cycle(
    learn: Learner,
    cyc_len: int,
    max_lr: Union[Floats, slice] = defaults.lr,
    moms: Tuple[float, float] = (0.95, 0.85),
    div_factor: float = 25.0,
    pct_start: float = 0.3,
    final_div: float = None,
    wd: float = None,
    callbacks: Optional[CallbackList] = None,
    tot_epochs: int = None,
    start_epoch: int = None,
) -> None:
    "Fit a model following the 1cycle policy."
    max_lr = learn.lr_range(max_lr)
    callbacks = listify(callbacks)
    cyclic_scheduler = OneCycleScheduler(
        learn,
        max_lr,
        moms=moms,
        div_factor=div_factor,
        pct_start=pct_start,
        final_div=final_div,
        tot_epochs=tot_epochs,
        start_epoch=start_epoch,
    )
    callbacks.append(TFOneCycleScheduler(learn, scheduler=cyclic_scheduler))

    learn.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks, one_cycle=True)


from fastai.basic_train import Recorder

TFRecorder.plot_losses = Recorder.plot_losses
TFRecorder._split_list = Recorder._split_list
TFRecorder._split_list_val = Recorder._split_list
TFRecorder.format_stats = Recorder.format_stats
EfficientDetLearner.fit_one_cycle = tf_fit_one_cycle
EfficientDetLearner.lr_find = efficient_lr_find

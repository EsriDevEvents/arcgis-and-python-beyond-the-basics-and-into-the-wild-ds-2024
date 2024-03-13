import json
import arcgis
import torch
import types
import pickle
import warnings
import numpy as np
import pandas as pd
import datetime as dt
from torch import Tensor
from pathlib import Path
from torch.utils import data
from torch.utils.data import DataLoader
import mimetypes, os, random, sys, math
from fastai.data_block import DataBunch
from collections import Counter
from .._data import _prepare_working_dir
import matplotlib.pylab as plt
from fastai.data_block import get_files
from fastprogress.fastprogress import progress_bar
from .._utils.common import get_top_padding, ArcGISMSImage


def get_device():
    if getattr(arcgis.env, "_processorType", "") == "GPU" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(arcgis.env, "_processorType", "") == "CPU":
        device = torch.device("cpu")
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    return device


def ts_normalization(x, m, s):
    x = x.numpy()
    x = np.rollaxis(x, 2)  # TxCxS -> SxTxC
    x = (x - m) / s
    x = np.swapaxes((np.rollaxis(x, 1)), 1, 2)
    return torch.tensor(x)


def band_adjust(index, x0):
    band_index = np.array(index) - 1
    x0 = np.concatenate(
        [x0[:, q, :][:, None, :] for q in range(x0.shape[1]) if q in band_index],
        axis=1,
    )
    return x0


def time_adjust(index, x0):
    time_index = np.array(index) - 1
    x0 = np.concatenate(
        [x0[q, :, :][None, :, :] for q in range(x0.shape[0]) if q in time_index],
        axis=0,
    )
    return x0


class PSATAEDataset(data.Dataset):
    def __init__(
        self,
        folder,
        labels,
        npixel,
        sub_classes=None,
        norm=None,
        extra_feature=None,
        jitter=(0.01, 0.05),
        return_id=False,
        use_band_index=None,
        use_time_index=None,
    ):
        """
        Args:
            folder (str): path to the main folder of the dataset, formatted as indicated in the readme
            labels (str): name of the nomenclature to use in the labels.json file
            npixel (int): Number of sampled pixels in each parcel
            sub_classes (list): If provided, only the samples from the given list of classes are considered.
            (Can be used to remove classes with too few samples)
            norm (tuple): (mean,std) tuple to use for normalization
            extra_feature (str): name of the additional static feature file to use
            jitter (tuple): if provided (sigma, clip) values for the addition random gaussian noise
            return_id (bool): if True, the id of the yielded item is also returned (useful for inference)
        """
        super(PSATAEDataset, self).__init__()

        self.folder = folder
        if labels == "train_labels":
            self.data_folder = os.path.join(folder, "DATA", "train_data")
        else:
            self.data_folder = os.path.join(folder, "DATA", "valid_data")
        self.meta_folder = os.path.join(folder, "META")
        self.labels = labels
        self.npixel = npixel
        self.norm = norm

        self.use_band_index = use_band_index
        self.use_time_index = use_time_index

        self.extra_feature = extra_feature
        self.jitter = jitter  # (sigma , clip )
        self.return_id = return_id

        l = [f for f in os.listdir(self.data_folder) if f.endswith(".npy")]
        self.pid = [int(f.split(".")[0]) for f in l]
        self.pid = list(np.sort(self.pid))

        self.pid = list(map(str, self.pid))
        self.len = len(self.pid)

        # Get Labels
        if sub_classes is not None:
            sub_indices = []
            num_classes = len(sub_classes)
            convert = dict((c, i) for i, c in enumerate(sub_classes))

        with open(os.path.join(folder, "META", "labels.json"), "r") as file:
            d = json.loads(file.read())
            self.target = []
            for i, p in enumerate(self.pid):
                t = d[labels][str(int(p) - 1)]
                self.target.append(t)
                if sub_classes is not None:
                    if t in sub_classes:
                        sub_indices.append(i)
                        self.target[-1] = convert[self.target[-1]]
        if sub_classes is not None:
            self.pid = list(np.array(self.pid)[sub_indices])
            self.target = list(np.array(self.target)[sub_indices])
            self.len = len(sub_indices)

        if os.path.exists(os.path.join(folder, "META", "dates.json")):
            with open(os.path.join(folder, "META", "dates.json"), "r") as file:
                d = json.loads(file.read())
            self.dates = [d[str(i)] for i in range(len(d))]
            self.date_positions = date_positions(self.dates)
        else:
            self.date_positions = None

        if self.extra_feature is not None:
            with open(
                os.path.join(self.meta_folder, "{}.json".format(extra_feature)), "r"
            ) as file:
                self.extra = json.loads(file.read())

            if isinstance(self.extra[list(self.extra.keys())[0]], int):
                for k in self.extra.keys():
                    self.extra[k] = [self.extra[k]]
            df = pd.DataFrame(self.extra).transpose()
            self.extra_m, self.extra_s = (
                np.array(df.mean(axis=0)),
                np.array(df.std(axis=0)),
            )

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        """
        Returns a Pixel-Set sequence tensor with its pixel mask and optional additional features.
        For each item npixel pixels are randomly dranw from the available pixels.
        If the total number of pixel is too small one arbitrary pixel is repeated. The pixel mask keeps track of true
        and repeated pixels.
        Returns:
              (Pixel-Set, Pixel-Mask) with:
              Pixel-Set: Sequence_length x Channels x npixel
              Pixel-Mask : Sequence_length x npixel
        """
        if self.labels == "train_labels":
            x0 = np.load(
                os.path.join(
                    self.folder, "DATA", "train_data", "{}.npy".format(self.pid[item])
                )
            )
            if self.use_band_index:
                x0 = band_adjust(self.use_band_index, x0)
            if self.use_time_index:
                x0 = time_adjust(self.use_time_index, x0)
        else:
            x0 = np.load(
                os.path.join(
                    self.folder, "DATA", "valid_data", "{}.npy".format(self.pid[item])
                )
            )
            if self.use_band_index:
                x0 = band_adjust(self.use_band_index, x0)
            if self.use_time_index:
                x0 = time_adjust(self.use_time_index, x0)

        y = self.target[item]

        if x0.shape[-1] > self.npixel:
            idx = np.random.choice(
                list(range(x0.shape[-1])), size=self.npixel, replace=False
            )
            x = x0[:, :, idx]
            mask = np.ones(self.npixel)

        elif x0.shape[-1] < self.npixel:
            if x0.shape[-1] == 0:
                x = np.zeros((*x0.shape[:2], self.npixel))
                mask = np.zeros(self.npixel)
                mask[0] = 1
            else:
                x = np.zeros((*x0.shape[:2], self.npixel))
                x[:, :, : x0.shape[-1]] = x0
                x[:, :, x0.shape[-1] :] = np.stack(
                    [x[:, :, 0] for _ in range(x0.shape[-1], x.shape[-1])], axis=-1
                )
                mask = np.array(
                    [1 for _ in range(x0.shape[-1])]
                    + [0 for _ in range(x0.shape[-1], self.npixel)]
                )
        else:
            x = x0
            mask = np.ones(self.npixel)

        if self.norm is not None:
            m, s = self.norm
            m = np.array(m)
            s = np.array(s)

            if len(m.shape) == 0:
                x = (x - m) / s
            elif len(m.shape) == 1:  # Normalise channel-wise
                x = (x.swapaxes(1, 2) - m) / s
                x = x.swapaxes(1, 2)  # Normalise channel-wise for each date
            elif len(m.shape) == 2:
                x = np.rollaxis(x, 2)  # TxCxS -> SxTxC
                x = (x - m) / s
                x = np.swapaxes((np.rollaxis(x, 1)), 1, 2)
        x = x.astype("float")

        if self.jitter is not None:
            sigma, clip = self.jitter
            x = x + np.clip(sigma * np.random.randn(*x.shape), -1 * clip, clip)

        mask = np.stack(
            [mask for _ in range(x.shape[0])], axis=0
        )  # Add temporal dimension to mask
        data = (Tensor(x), Tensor(mask))

        if self.extra_feature is not None:
            ef = (self.extra[str(self.pid[item])] - self.extra_m) / self.extra_s
            ef = torch.from_numpy(ef).float()

            ef = torch.stack([ef for _ in range(data[0].shape[0])], dim=0)
            data = (data, ef)

        if self.return_id:
            return data, torch.from_numpy(np.array(y, dtype=int)), self.pid[item]
        else:
            return data, torch.from_numpy(np.array(y, dtype=int))

    def __repr__(self):
        item = self.__getitem__(0)
        return (
            f"{self.__class__.__name__}, Tensor:{(item[0][0].shape)}, items:{self.len }"
        )


def parse(date):
    d = str(date)
    return int(d[:4]), int(d[4:6]), int(d[6:])


def interval_days(date1, date2):
    return abs((dt.datetime(*parse(date1)) - dt.datetime(*parse(date2))).days)


def date_positions(dates):
    pos = []
    for d in dates:
        pos.append(interval_days(d, dates[0]))
    return pos


image_extensions = set(
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
)


def create_train_val_sets(path, val_split_pct, working_dir, **kwargs):
    path = Path(path)
    images, labels = os.path.join(path, "images"), os.path.join(path, "labels")
    label_folds = [file for file in os.listdir(labels)]

    empty_folds = [
        i for i in label_folds if len(os.listdir(os.path.join(labels, i))) == 0
    ]
    if empty_folds:
        warnings.warn("Warning: Empty folder of label %s" % empty_folds)
        raise Exception()

    label_lst = get_files(labels, extensions=image_extensions, recurse=True)

    if working_dir is not None:
        save_path = working_dir
    else:
        save_path = path

    isExist_data = os.path.exists(os.path.join(save_path, "DATA"))
    isExist_train_data = os.path.exists(os.path.join(save_path, "DATA", "train_data"))
    isExist_valid_data = os.path.exists(os.path.join(save_path, "DATA", "valid_data"))
    isExist_labels = os.path.exists(os.path.join(save_path, "META"))

    nt = kwargs.get("n_temporal")
    ntemp = int(nt) if nt else nt
    npixels = kwargs.get("min_points", 64)
    ntempdates = kwargs.get("n_temporal_dates")
    use_band_index = kwargs.get("channels_of_interest")
    use_time_index = kwargs.get("timesteps_of_interest")
    interest_class = kwargs.get("classes_of_interest")

    emd_path = os.path.join(path, "esri_model_definition.emd")
    with open(emd_path) as f:
        emd_stats = json.load(f)
    IsMultidimensional = emd_stats.get("IsMultidimensional", False)
    serialDates = emd_stats.get("DimensionValues")
    imagespace = emd_stats.get("ImageSpaceUsed")

    if IsMultidimensional:
        ntempdates = [
            str(
                (
                    dt.datetime.fromordinal(dt.datetime(1900, 1, 1).toordinal() + i - 2)
                ).date()
            )
            for i in serialDates
        ]
        ntemp = len(ntempdates)

    if not interest_class:
        interest_class = [int(x) for x in label_folds]

    class_mapping_dict = dict()
    for i, n in enumerate(label_folds):
        if int(n) in interest_class:
            class_mapping_dict[i] = int(n)

    if interest_class:
        convertmap = dict((c, i) for i, c in enumerate(list(class_mapping_dict.keys())))
        convertmap = {v: k for k, v in convertmap.items()}
    if (
        not isExist_data
        or not isExist_train_data
        or not isExist_valid_data
        or not isExist_labels
    ):
        if not isExist_data:
            os.makedirs(os.path.join(save_path, "DATA"))
            if not isExist_train_data:
                os.makedirs(os.path.join(save_path, "DATA", "train_data"))
            if not isExist_valid_data:
                os.makedirs(os.path.join(save_path, "DATA", "valid_data"))
        if not isExist_labels:
            os.makedirs(os.path.join(save_path, "META"))

        train_labels, valid_labels = [], []

        for i in label_folds:
            file_lst = get_files(
                os.path.join(save_path, labels, i),
                extensions=image_extensions,
                recurse=False,
            )
            num = len(file_lst)

            val_num_images = int(val_split_pct * num)

            train_labels.extend(file_lst[val_num_images:])
            valid_labels.extend(file_lst[:val_num_images])

        val_lab_app, train_lab_app, trainlabs, validlabs = [], [], [], []
        train_arrs_lst, valid_arrs_lst, whole_data = [], [], []
        cnt, u, cnt1, s, cnt_lst_t, cnt_lst_v = 0, 1, 0, 1, [], []

        for i in progress_bar(label_lst):
            lab_name = os.path.split(i)[1]
            class_name = int(os.path.split(os.path.split(i)[0])[1])
            img_file, lab_file = (
                ArcGISMSImage.open(os.path.join(images, lab_name)),
                ArcGISMSImage.open(i),
            )
            img_arr, lab_arr = (
                img_file.data,
                torch.sum(lab_file.data, axis=0)[None, :, :],
            )

            concate_arr = torch.cat((img_arr, lab_arr), axis=0)
            indxs = torch.where(concate_arr[-1, :, :] != 0)
            timeseries_arr = concate_arr[:, indxs[0], indxs[1]][:-1, :]

            nchannel = int(timeseries_arr.shape[0] / ntemp)
            final = torch.reshape(
                timeseries_arr, (ntemp, nchannel, timeseries_arr.shape[1])
            )

            lol1 = lambda lst, sz: [
                len(lst[i : i + sz]) for i in range(0, len(lst), sz)
            ]
            if i in train_labels:
                train_arrs_lst.append(final)
                trainlabs.append(torch.tensor(class_name))
                num_shp = list(np.arange(final.shape[2]))
                u = 1 + cnt
                cnt += len(lol1(num_shp, npixels))
                save_cnt_lst = list(np.arange(u, cnt + 1))
                cntt = len(list(np.arange(u, cnt + 1)))
                train_lab_app.extend(cntt * [class_name])
                cnt_lst_t.extend(save_cnt_lst)

                def save(p, x, j, save_cnt):
                    l = np.save(
                        os.path.join(p, "DATA", "train_data", str(save_cnt[j])),
                        x.numpy(),
                    )
                    return x

                lol2 = lambda lst, sz: [
                    save(save_path, lst[:, :, i : i + sz], j, save_cnt_lst)
                    for j, i in enumerate(range(0, lst.shape[2], sz))
                ]
                divided = lol2(final, npixels)
            else:
                valid_arrs_lst.append(final.numpy())
                validlabs.append(class_name)
                num_shp = list(np.arange(final.shape[2]))
                s = 1 + cnt1
                cnt1 += len(lol1(num_shp, npixels))
                save_cnt_lst = list(np.arange(s, cnt1 + 1))
                cntt = len(list(np.arange(s, cnt1 + 1)))
                val_lab_app.extend(cntt * [class_name])
                cnt_lst_v.extend(save_cnt_lst)

                def save(p, x, j, save_cnt):
                    l = np.save(
                        os.path.join(p, "DATA", "valid_data", str(save_cnt[j])),
                        x.numpy(),
                    )
                    return x

                lol2 = lambda lst, sz: [
                    save(save_path, lst[:, :, i : i + sz], j, save_cnt_lst)
                    for j, i in enumerate(range(0, lst.shape[2], sz))
                ]
                divided = lol2(final, npixels)
            whole_data.append(final.numpy())

        stats_std_mean = np.concatenate(whole_data, axis=2)
        mean_std = (np.mean(stats_std_mean, axis=2), np.std(stats_std_mean, axis=2))

        with open(
            os.path.join(save_path, "META", "train_arrs_with_label.pickle"), "wb"
        ) as f:
            pickle.dump((train_arrs_lst, trainlabs), f)
        with open(os.path.join(save_path, "META", "mean_std.pickle"), "wb") as f:
            pickle.dump(mean_std, f)

        labs_dict = {
            "train": train_lab_app,
            "cn_t": len(cnt_lst_t),
            "valid": val_lab_app,
            "cn_v": len(cnt_lst_v),
        }
        labels_dic = {}

        for k in labs_dict.keys():
            if k in ["train", "valid"]:
                items = list(Counter(labs_dict[k]).keys())
                lab = dict()
                for i, j in zip(items, range(len(items))):
                    lab[str(i)] = int(j)
                re_lst = [
                    lab[str(z)] for z in labs_dict[k] if str(z) in list(lab.keys())
                ]
            if k == "train":
                Id_with_train_labels = dict()
                for i, j in zip(range(labs_dict["cn_t"]), re_lst):
                    Id_with_train_labels[str(i)] = int(j)
                labels_dic["train_labels"] = Id_with_train_labels
            else:
                Id_with_valid_labels = dict()
                for i, j in zip(range(labs_dict["cn_v"]), re_lst):
                    Id_with_valid_labels[str(i)] = int(j)
                labels_dic["valid_labels"] = Id_with_valid_labels

        labels_dic["npixel"] = int(npixels)
        with open(os.path.join(save_path, "META", "labels.json"), "w") as file:
            file.write(json.dumps(labels_dic, indent=4))
        if ntempdates:
            if use_time_index:
                timeidx = np.array(ntempdates) - 1
                ntempdates = ntempdates[timeidx]
            datelist = [x.replace("-", "") for x in ntempdates]
            dates_dict = {}
            for i in range(len(datelist)):
                dates_dict[str(i)] = datelist[i]
            with open(os.path.join(save_path, "META", "dates.json"), "w") as file:
                file.write(json.dumps(dates_dict, indent=4))

    if not [
        i
        for i in os.listdir(os.path.join(save_path, "META"))
        if i in ["labels.json", "mean_std.pickle", "train_arrs_with_label.pickle"]
    ]:
        warnings.warn(
            "required files missing, remove DATA and META folders at %s"
            % os.path.join(save_path)
        )
        raise Exception()

    mean_std = list(
        pickle.load(open(os.path.join(save_path, "META", "mean_std.pickle"), "rb"))
    )

    if use_band_index:
        band_index = np.array(use_band_index) - 1
        mean_std[0] = np.concatenate(
            [
                mean_std[0][:, q][:, None]
                for q in range(mean_std[0].shape[1])
                if q in band_index
            ],
            axis=1,
        )
        mean_std[1] = np.concatenate(
            [
                mean_std[1][:, q][:, None]
                for q in range(mean_std[1].shape[1])
                if q in band_index
            ],
            axis=1,
        )
    if use_time_index:
        time_index = np.array(use_time_index) - 1
        mean_std[0] = np.concatenate(
            [
                mean_std[0][q, :][None, :]
                for q in range(mean_std[0].shape[0])
                if q in time_index
            ],
            axis=0,
        )
        mean_std[1] = np.concatenate(
            [
                mean_std[1][q, :][None, :]
                for q in range(mean_std[1].shape[0])
                if q in time_index
            ],
            axis=0,
        )

    with open(os.path.join(save_path, "META", "labels.json"), "r") as file:
        lab_load = json.loads(file.read())
    train_arrs_lst, labs = pickle.load(
        open(os.path.join(save_path, "META", "train_arrs_with_label.pickle"), "rb")
    )

    timestep_infer, channels_infer = (
        train_arrs_lst[0].shape[0],
        train_arrs_lst[0].shape[1],
    )

    npixels = lab_load["npixel"]

    train_dataset = PSATAEDataset(
        save_path,
        labels="train_labels",
        npixel=npixels,
        norm=tuple(mean_std),
        sub_classes=list(class_mapping_dict.keys()),
        use_band_index=use_band_index,
        use_time_index=use_time_index,
    )
    valid_dataset = PSATAEDataset(
        save_path,
        labels="valid_labels",
        npixel=npixels,
        norm=tuple(mean_std),
        sub_classes=list(class_mapping_dict.keys()),
        use_band_index=use_band_index,
        use_time_index=use_time_index,
    )

    new_ntemp = None
    if use_time_index:
        new_ntemp = len(use_time_index)  # train_arrs_lst[0].shape[0]
    nt = new_ntemp if new_ntemp else ntemp

    return [
        (train_dataset, valid_dataset),
        class_mapping_dict,
        mean_std,
        train_arrs_lst,
        labs,
        npixels,
        nt,
        IsMultidimensional,
        imagespace,
        use_band_index,
        use_time_index,
        convertmap,
        timestep_infer,
        channels_infer,
    ]


def create_dataloaders(datasets, batch_size, dataloader_kwargs):
    dl_list = []
    for c, d in enumerate(datasets):
        if c == 0:
            dataloader_kwargs["shuffle"] = True
        else:
            dataloader_kwargs["shuffle"] = True
        dl = DataLoader(d, batch_size, **dataloader_kwargs)
        dl_list.append(dl)
    return dl_list


def prepare_psetae_data(
    path,
    batch_size,
    val_split_pct,
    working_dir,
    class_mapping,
    **kwargs,
):
    train_val_dataset = create_train_val_sets(
        path, val_split_pct, working_dir, **kwargs
    )
    databunch_kwargs = (
        {"num_workers": 0, "drop_last": True}
        if sys.platform == "win32"
        else {"num_workers": os.cpu_count() - 4, "drop_last": True}
    )
    train_dl, valid_dl = create_dataloaders(
        train_val_dataset[0], batch_size, databunch_kwargs
    )
    device = get_device()
    data = DataBunch(train_dl, valid_dl, device=device)
    if working_dir is not None:
        path = Path(os.path.abspath(working_dir))
    data.path = Path(path)
    data._temp_folder = _prepare_working_dir(path)
    data._class_map_dict = train_val_dataset[1].copy()
    data._num_class_map_dict = train_val_dataset[1].copy() if class_mapping else None

    if class_mapping:
        for r in data._class_map_dict.keys():
            data._class_map_dict[r] = class_mapping[data._class_map_dict[r]]

    data.show_batch = types.MethodType(show_batch, data)
    data._dataset_type = "PSETAE"
    data._date_positions = train_dl.date_positions
    data._n_temp = train_val_dataset[6]
    data._n_channel = data.train_ds[0][0][0].shape[1]
    data._mean_norm_stats = train_val_dataset[2][0]
    data._std_norm_stats = train_val_dataset[2][1]
    data._train_arrs_lst = train_val_dataset[3]
    data._train_labs = train_val_dataset[4]
    data._npixel = train_val_dataset[5]
    data._is_multidimensional = train_val_dataset[7]
    data._imagespace = train_val_dataset[8]
    data._bandindex = train_val_dataset[9]
    data._timeindex = train_val_dataset[10]
    data._convertmap = train_val_dataset[11]
    data._timestep_infer = train_val_dataset[12]
    data._channels_infer = train_val_dataset[13]

    return data


def show_batch(self, rows=5, spectral_view=False, **kwargs):
    """
    Show pixels temporal or spectral view of data in `ds_type`.

    =====================   ===========================================
    **Parameter**            **Description**
    ---------------------   -------------------------------------------
    rows                    Optional int. number of pixels to be sampled
                            for each class.
    =====================   ===========================================
    spectral_view           Optional boolean. If set to True, Shows spectral
                            curves of sampled pixels.
    =====================   ===========================================
    **kwargs**

    """
    temporal_view = not spectral_view
    m, s = self._mean_norm_stats, self._std_norm_stats
    m = np.array(m)
    s = np.array(s)

    n = len(self._class_map_dict.keys())
    fig, axs = plt.subplots(
        math.ceil(n / 2), 2, squeeze=False, figsize=(20, math.ceil(n / 2) * 5)
    )
    top = get_top_padding(title_font_size=16, nrows=2, imsize=5)
    plt.subplots_adjust(top=top, hspace=0.3)

    viz_data_seg = dict((k, []) for k in self._class_map_dict.keys())
    class_dict = (
        self._num_class_map_dict if self._num_class_map_dict else self._class_map_dict
    )
    rev_map = {v: k for k, v in class_dict.items()}

    for t, r in zip(self._train_arrs_lst, self._train_labs):
        if int(r) in rev_map.keys():
            r = rev_map[int(r)]
            if self._bandindex:
                t = band_adjust(self._bandindex, t)
            if self._timeindex:
                t = time_adjust(self._timeindex, t)
            t = ts_normalization(Tensor(t), m, s)
            if t.shape[2] <= self._npixel:
                viz_data_seg[int(r)].append(t)
            else:
                index = np.random.choice(t.shape[2], self._npixel, replace=False)
                viz_data_seg[int(r)].append(t.index_select(2, torch.tensor(index)))

    for z in range(n):
        mp = list(self._class_map_dict.keys())
        class_1_arr = np.concatenate(viz_data_seg[mp[z]], axis=2)
        randlist = random.sample(list(range(class_1_arr.shape[2])), rows)
        if class_1_arr.shape[0] != 1:
            if temporal_view:
                bands = [("T" + str(x)) for x in range(class_1_arr.shape[0])]
                colours = plt.cm.jet(np.linspace(0, 1, class_1_arr.shape[1]))
                for j in range(class_1_arr.shape[1]):
                    for o, i in enumerate(randlist):
                        ax = axs.flat[z]
                        if o == 0:
                            lab = "band " + str(j + 1)
                        else:
                            lab = None
                        ax.plot(
                            bands,
                            class_1_arr[:, :, i][:, j],
                            color=colours[j],
                            label=lab,
                        )
                        ax.legend(loc="upper right")
            else:
                bands = [("B" + str(x)) for x in range(class_1_arr.shape[1])]
                colours = plt.cm.jet(np.linspace(0, 1, class_1_arr.shape[0]))
                for j in range(class_1_arr.shape[0]):
                    for o, i in enumerate(randlist):
                        ax = axs.flat[z]
                        if o == 0:
                            lab = "T " + str(j + 1)
                        else:
                            lab = None
                        ax.plot(
                            bands,
                            class_1_arr[:, :, i][j, :],
                            color=colours[j],
                            label=lab,
                        )
                        ax.legend(loc="upper right")
        else:
            bands = [("B" + str(x)) for x in range(class_1_arr.shape[1])]
            colours = plt.cm.jet(np.linspace(0, 1, class_1_arr.shape[0]))
            ax = axs.flat[z]
            lab = "T " + str(1)
            ax.plot(
                bands, class_1_arr[:, :, randlist][0, :], color=colours[0], label=lab
            )
            ax.legend(loc="upper right")
        ax.set_title(label=str(self._class_map_dict[mp[z]]), fontsize=17, pad=10)
    for ax in axs.flat:
        if not bool(ax.has_data()):
            fig.delaxes(ax)
    plt.show()


def show_results(self, rows, **kwargs):
    validarr = torch.cat([i[0][None, :, :, :] for i, j in self._data.valid_ds], axis=0)
    labsarr = torch.stack([j for i, j in self._data.valid_ds])

    final_img = torch.moveaxis(validarr, 3, 1)[:, :, :, :, None]
    img_arr = torch.reshape(
        final_img,
        (
            final_img.shape[0] * final_img.shape[1],
            final_img.shape[2],
            final_img.shape[3],
            1,
        ),
    ).to(self._device)
    final_labs = [
        j for l in range(validarr.shape[0]) for j in validarr.shape[3] * [labsarr[l]]
    ]
    final_labs = torch.stack(final_labs).to(self._device)
    divided = DataLoader(img_arr, batch_size=65536, pin_memory=False)

    prediction = []
    for i in divided:
        sim = torch.ones(i.shape[0], i.shape[1], 1).to(self._device)
        self.learn.model.eval()
        with torch.no_grad():
            pred = self.learn.model(i, sim)
        prediction.append(pred.argmax(dim=1))

    df = pd.DataFrame(
        np.stack(
            [torch.cat(prediction).cpu().numpy(), final_labs.cpu().numpy()], axis=1
        ),
        columns=["Predicted", "True"],
    )

    df.sort_values(by=["True"], ascending=True)
    clses = list(self._data._class_map_dict.keys())
    num_pixels = rows * 2
    grouped = df.groupby("True").apply(lambda x: x.sample(int(num_pixels / len(clses))))
    all_cls_arr = img_arr.index_select(
        0, torch.tensor([j for i, j in grouped["Predicted"].index]).to(self._device)
    )
    if grouped.shape[0] != num_pixels:
        remain = df.sample(abs(grouped.shape[0] - num_pixels))
        remained_all_cls_arr = img_arr.index_select(
            0, torch.tensor([i for i in remain.index]).to(self._device)
        )
        all_cls_arr = torch.cat((all_cls_arr, remained_all_cls_arr), axis=0)
    bands = [("T" + str(x)) for x in range(all_cls_arr.shape[1])]
    pred = (
        list(grouped["Predicted"].values) + [i for i in remain["Predicted"]]
        if grouped.shape[0] != num_pixels
        else grouped["Predicted"].values
    )
    true = (
        list(grouped["True"].values) + [i for i in remain["True"]]
        if grouped.shape[0] != num_pixels
        else grouped["True"].values
    )
    pred.sort(), true.sort()

    if self._data._convertmap:
        true = np.array([self._data._convertmap.get(item, item) for item in true])
        pred = np.array([self._data._convertmap.get(item, item) for item in pred])

    trues = np.array([self._data._class_map_dict.get(item, item) for item in true])
    preds = np.array([self._data._class_map_dict.get(item, item) for item in pred])
    n = all_cls_arr.shape[0]
    fig, axs = plt.subplots(
        math.ceil(n / 2), 2, squeeze=False, figsize=(20, math.ceil(n / 2) * 5)
    )
    top = get_top_padding(title_font_size=16, nrows=math.ceil(n / 2), imsize=5)
    plt.subplots_adjust(top=top, hspace=0.33)
    axs1 = np.reshape(axs, (axs.shape[0] * axs.shape[1]))
    for i in range(axs1.shape[0]):
        for k in range(all_cls_arr[i].shape[1]):
            axs1[i].plot(bands, all_cls_arr[i][:, k, :].cpu().numpy())
            if preds[i] == trues[i]:
                col = "blue"
            else:
                col = "red"
            if i == 0:
                axs1[0].text(
                    0.98,
                    1.4,
                    "Ground truth",
                    fontsize=17,
                    transform=axs1[0].transAxes,
                    fontweight="bold",
                )
                axs1[0].text(
                    1,
                    1.3,
                    "Predictions",
                    fontsize=17,
                    transform=axs1[0].transAxes,
                    fontweight="bold",
                )

            axs1[i].text(
                0.5,
                1.15,
                str(trues[i]),
                fontsize=17,
                transform=axs1[i].transAxes,
                color=col,
                ha="center",
                va="center",
            )
            axs1[i].text(
                0.5,
                1.05,
                str(preds[i]),
                fontsize=17,
                transform=axs1[i].transAxes,
                color=col,
                ha="center",
                va="center",
            )
    for ax in axs.flat:
        if not bool(ax.has_data()):
            fig.delaxes(ax)

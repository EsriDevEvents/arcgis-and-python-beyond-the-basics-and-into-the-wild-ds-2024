from typing import Collection
import fastai


precondition = True

if fastai.__version__ == "1.0.60":
    import fastai.vision.image
    from fastai.vision.image import Image, plt, image2np
    from .common import ArcGISMSImage

    def show_image(
        img: Image,
        ax: plt.Axes = None,
        figsize: tuple = (3, 3),
        hide_axis: bool = True,
        cmap: str = "binary",
        alpha: float = None,
        **kwargs
    ) -> plt.Axes:
        "Display `Image` in notebook."
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        xtr = dict(cmap=cmap, alpha=alpha, **kwargs)
        if isinstance(img, ArcGISMSImage):
            img.show(ax=ax)
        else:
            ax.imshow(image2np(img.data), **xtr) if (
                hasattr(img, "data")
            ) else ax.imshow(img, **xtr)
        if hide_axis:
            ax.axis("off")
        return ax

    #
    fastai.vision.learner.show_image = show_image
    fastai.vision.image.show_image = show_image

    import fastai.data_block
    from fastai.core import array

    def process(self, ds: Collection):
        ds.items = array([self.process_one(item) for item in ds.items], dtype=object)

    fastai.data_block.PreProcessor.process = process

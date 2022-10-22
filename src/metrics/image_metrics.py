import torch
from torch import tensor
from torchmetrics.functional.image.psnr import _psnr_compute, _psnr_update


from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.ssim import _ssim_compute, _ssim_update
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class SSIM(Metric):
    """Computes Structual Similarity Index Measure (SSIM_).
    Args:
        preds: estimated image
        target: ground truth image
        gaussian_kernel: If true (default), a gaussian kernel is used, if false a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel, anisotropic kernels are possible.
            Ignored if a uniform kernel is used
        kernel_size: the size of the uniform kernel, anisotropic kernels are possible.
            Ignored if a Gaussian kernel is used
        reduction: a method to reduce metric score over labels.
            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied
        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.
        return_full_image: If true, the full ``ssim`` image is returned as a second argument.
            Mutually exclusive with ``return_contrast_sensitivity``
        return_contrast_sensitivity: If true, the constant term is returned as a second argument.
            The luminance term can be obtained with luminance=ssim/contrast
            Mutually exclusive with ``return_full_image``
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.
            .. deprecated:: v0.8
                Argument has no use anymore and will be removed v0.9.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    Return:
        Tensor with SSIM score
    Example:
        >>> from torchmetrics import StructuralSimilarityIndexMeasure
        >>> import torch
        >>> preds = torch.rand([16, 1, 16, 16])
        >>> target = preds * 0.75
        >>> ssim = StructuralSimilarityIndexMeasure()
        >>> ssim(preds, target)
        tensor(0.9219)
    """

    preds: List[Tensor]
    target: List[Tensor]
    higher_is_better = True

    def __init__(
        self,
        gaussian_kernel: bool = True,
        sigma: Union[float, Sequence[float]] = 1.5,
        kernel_size: Union[int, Sequence[int]] = 11,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        data_range: Optional[float] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        compute_on_step: Optional[bool] = None,
        return_full_image: bool = False,
        return_contrast_sensitivity: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, **kwargs)
        rank_zero_warn(
            "Metric `SSIM` will save all targets and"
            " predictions in buffer. For large datasets this may lead"
            " to large memory footprint."
        )

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.gaussian_kernel = gaussian_kernel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.return_full_image = return_full_image
        self.return_contrast_sensitivity = return_contrast_sensitivity

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds, target = _ssim_update(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Computes explained variance over state."""
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        return _ssim_compute(
            preds,
            target,
            self.gaussian_kernel,
            self.sigma,
            self.kernel_size,
            self.reduction,
            self.data_range,
            self.k1,
            self.k2,
            self.return_full_image,
            self.return_contrast_sensitivity,
        )


class PeakSignalNoiseRatio(Metric):
    r"""
    Computes `Computes Peak Signal-to-Noise Ratio`_ (PSNR):
    .. math:: \text{PSNR}(I, J) = 10 * \log_{10} \left(\frac{\max(I)^2}{\text{MSE}(I, J)}\right)
    Where :math:`\text{MSE}` denotes the `mean-squared-error`_ function.
    Args:
        data_range:
            the range of the data. If None, it is determined from the data (max - min).
            The ``data_range`` must be given when ``dim`` is not None.
        base: a base of a logarithm to use.
        reduction: a method to reduce metric score over labels.
            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied
        dim:
            Dimensions to reduce PSNR scores over, provided as either an integer or a list of integers. Default is
            None meaning scores will be reduced across all dimensions and all batches.
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False.
            .. deprecated:: v0.8
                Argument has no use anymore and will be removed v0.9.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    Raises:
        ValueError:
            If ``dim`` is not ``None`` and ``data_range`` is not given.
    Example:
        >>> from torchmetrics import PeakSignalNoiseRatio
        >>> psnr = PeakSignalNoiseRatio()
        >>> preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        >>> target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        >>> psnr(preds, target)
        tensor(2.5527)
    .. note::
        Half precision is only support on GPU for this metric
    """
    min_target: Tensor
    max_target: Tensor
    higher_is_better = True

    def __init__(
        self,
        data_range: Optional[float] = None,
        base: float = 10.0,
        reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        compute_on_step: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(compute_on_step=compute_on_step, **kwargs)

        if dim is None and reduction != "elementwise_mean":
            rank_zero_warn(
                f"The `reduction={reduction}` will not have any effect when `dim` is None."
            )

        if dim is None:
            self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
            self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        else:
            self.add_state("sum_squared_error", default=[])
            self.add_state("total", default=[])

        if data_range is None:
            if dim is not None:
                # Maybe we could use `torch.amax(target, dim=dim) - torch.amin(target, dim=dim)` in PyTorch 1.7 to
                # calculate `data_range` in the future.
                raise ValueError("The `data_range` must be given when `dim` is not None.")

            self.data_range = None
            self.add_state("min_target", default=tensor(0.0), dist_reduce_fx=torch.min)
            self.add_state("max_target", default=tensor(0.0), dist_reduce_fx=torch.max)
        else:
            self.add_state("data_range", default=tensor(float(data_range)), dist_reduce_fx="mean")
        self.base = base
        self.reduction = reduction
        self.dim = tuple(dim) if isinstance(dim, Sequence) else dim

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_squared_error, n_obs = _psnr_update(preds, target, dim=self.dim)
        if self.dim is None:
            if self.data_range is None:
                # keep track of min and max target values
                self.min_target = min(target.min(), self.min_target)
                self.max_target = max(target.max(), self.max_target)

            self.sum_squared_error += sum_squared_error
            self.total += n_obs
        else:
            self.sum_squared_error.append(sum_squared_error)
            self.total.append(n_obs)

    def compute(self) -> Tensor:
        """Compute peak signal-to-noise ratio over state."""
        if self.data_range is not None:
            data_range = self.data_range
        else:
            data_range = self.max_target - self.min_target

        if self.dim is None:
            sum_squared_error = self.sum_squared_error
            total = self.total
        else:
            sum_squared_error = torch.cat([values.flatten() for values in self.sum_squared_error])
            total = torch.cat([values.flatten() for values in self.total])
        return _psnr_compute(
            sum_squared_error, total, data_range, base=self.base, reduction=self.reduction
        )

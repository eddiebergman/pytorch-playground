from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import ClassVar, Union
from typing_extensions import TypeAlias

from torch import nn


@dataclass
class ShapeCalc(ABC):
    supported: ClassVar[type[nn.Module] | tuple[type[nn.Module], ...]]

    in_dimensions: tuple[int, ...]
    operator: nn.Module

    @property
    @abstractmethod
    def flattened(self) -> int:
        ...

    @property
    @abstractmethod
    def out(self) -> tuple[int, ...]:
        ...


_ConvLike: TypeAlias = Union[nn.modules.conv._ConvNd, nn.modules.pooling._MaxPoolNd]


@dataclass
class ConvShape(ShapeCalc):
    supported: ClassVar = (nn.modules.conv._ConvNd, nn.modules.pooling._MaxPoolNd)

    in_dimensions: tuple[int, ...]
    operator: _ConvLike

    @property
    def out(self) -> tuple[int, ...]:
        dims = self.in_dimensions
        padding = self.operator.padding

        def tuplize(_x: int | tuple[int, ...]) -> tuple[int, ...]:
            if isinstance(_x, int):
                _x = tuple([_x] * len(dims))

            assert isinstance(_x, tuple)
            assert len(_x) == len(self.in_dimensions)
            return _x

        assert not isinstance(self.operator.padding, str)
        padding = tuplize(self.operator.padding)

        kernel_size = tuplize(self.operator.kernel_size)
        stride = tuplize(self.operator.stride)
        dilation = tuplize(self.operator.dilation)

        def dim_output_size(idx: int) -> int:
            return int(
                (
                    dims[idx]
                    + 2 * padding[idx]
                    - dilation[idx] * (kernel_size[idx] - 1)
                    - 1
                )
                / stride[idx]
                + 1,
            )

        return tuple(dim_output_size(idx) for idx in range(len(dims)))

    @property
    def flattened(self) -> int:
        size_per_channel = reduce(lambda x, y: x * y, self.out)
        if isinstance(self.operator, nn.modules.pooling._MaxPoolNd):
            return size_per_channel

        return size_per_channel * self.operator.out_channels


def shape(operator: nn.Module, in_dimensions: tuple[int, ...]) -> ShapeCalc:
    impls = [ConvShape]
    impl = next(
        (_impl for _impl in impls if isinstance(operator, _impl.supported)),
        None,
    )
    if impl is None:
        raise ValueError(f"No support for {type(operator)=} with {impls=}")

    return impl(in_dimensions, operator)  # type: ignore

import builtins
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy._typing import NDArray

from ..structures.array import ArrayStructure, BuiltinDtype
from ..structures.core import Spec, StructureFamily
from ..structures.data_source import Asset
from ..utils import path_from_uri
from .protocols import AccessPolicy
from .type_alliases import JSON, NDSlice


class FileSequenceAdapter:
    """Base adapter class for image (and other file) sequences

    Assumes that each file contains an array of the same shape and dtype, and the sequence of files defines the
    left-most dimension in the resulting compound array.

    When subclassing, define the `_load_from_files` method specific for a particular file type.
    """

    structure_family = StructureFamily.array

    @classmethod
    def from_uris(
        cls,
        data_uris: List[str],
        structure: Optional[ArrayStructure] = None,
        metadata: Optional[JSON] = None,
        specs: Optional[List[Spec]] = None,
        access_policy: Optional[AccessPolicy] = None,
    ) -> "FileSequenceAdapter":
        return cls(
            filepaths=[path_from_uri(data_uri) for data_uri in data_uris],
            structure=structure,
            specs=specs,
            metadata=metadata,
            access_policy=access_policy,
        )

    @classmethod
    def from_assets(
        cls,
        assets: List[Asset],
        structure: ArrayStructure,
        metadata: Optional[JSON] = None,
        specs: Optional[List[Spec]] = None,
        access_policy: Optional[AccessPolicy] = None,
        **kwargs: Optional[Union[str, List[str], Dict[str, str]]],
    ) -> "FileSequenceAdapter":
        return cls(
            filepaths=[path_from_uri(a.data_uri) for a in assets],
            structure=structure,
            specs=specs,
            metadata=metadata,
            access_policy=access_policy,
        )

    def __init__(
        self,
        filepaths: List[Path],
        *,
        structure: Optional[ArrayStructure] = None,
        metadata: Optional[JSON] = None,
        specs: Optional[List[Spec]] = None,
        access_policy: Optional[AccessPolicy] = None,
    ) -> None:
        """

        Parameters
        ----------
        seq :
        structure :
        metadata :
        specs :
        access_policy :
        """
        self.filepaths = filepaths
        # TODO Check shape, chunks against reality.
        self.specs = specs or []
        self._provided_metadata = metadata or {}
        self.access_policy = access_policy
        if structure is None:
            dat0 = self._load_from_files(0)
            shape = (len(self.filepaths), *dat0.shape[1:])
            structure = ArrayStructure(
                shape=shape,
                # one chunk per underlying image file
                chunks=((1,) * shape[0], *[(i,) for i in shape[1:]]),
                # Assume all files have the same data type
                data_type=BuiltinDtype.from_numpy_dtype(dat0.dtype),
            )
        self._structure = structure

    @abstractmethod
    def _load_from_files(
        self, slice: Union[builtins.slice, int] = slice(None)
    ) -> NDArray[Any]:
        """Load the array data from files

        Parameters
        ----------
        slice : slice
            an optional slice along the left-most dimension in the resulting array; effectively selects a subset of
            files to be loaded

        Returns
        -------
            A numpy ND array with data from each file stacked along an addional (left-most) dimension.
        """

        pass

    def metadata(self) -> JSON:
        """

        Returns
        -------

        """
        # TODO How to deal with the many headers?
        return self._provided_metadata

    def read(self, slice: Optional[NDSlice] = ...) -> NDArray[Any]:
        """Return a numpy array

        Receives a sequence of values to select from a collection of image files
        that were saved in a folder The input order is defined as: files -->
        vertical slice --> horizontal slice --> color slice --> ... read() can
        receive one value or one slice to select all the data from one file or
        a sequence of files; or it can receive a tuple (int or slice) to select
        a more specific sequence of pixels of a group of images.

        Parameters
        ----------
        slice : NDSlice, optional
            Specification of slicing to be applied to the data array

        Returns
        -------
            Return a numpy array
        """
        if slice is Ellipsis:
            arr = self._load_from_files()
        elif isinstance(slice, int):
            # e.g. read(slice=0) -- return an entire image (drop 0th dimension of the stack)
            arr = np.squeeze(self._load_from_files(slice), 0)
        elif isinstance(slice, builtins.slice):
            # e.g. read(slice=(...)) -- return a slice along the image axis
            arr = self._load_from_files(slice)
        elif isinstance(slice, tuple):
            if len(slice) == 0:
                arr = self._load_from_files()
            elif len(slice) == 1:
                arr = self.read(slice=slice[0])
            else:
                left_axis, *the_rest = slice
                # Could be int or slice (i, ...) or (slice(...), ...); the_rest is converted to a list
                if isinstance(left_axis, int):
                    # e.g. read(slice=(0, ....))
                    arr = np.squeeze(self._load_from_files(left_axis), 0)
                elif left_axis is Ellipsis:
                    # Return all images
                    arr = self._load_from_files()
                    the_rest.insert(0, Ellipsis)  # Include any leading dimensions
                elif isinstance(left_axis, builtins.slice):
                    arr = self.read(slice=left_axis)
                arr = np.atleast_1d(arr[tuple(the_rest)])
        else:
            raise RuntimeError(f"Unsupported slice type, {type(slice)} in {slice}")

        return arr

    def read_block(
        self, block: Tuple[int, ...], slice: Optional[NDSlice] = ...
    ) -> NDArray[Any]:
        """

        Parameters
        ----------
        block :
        slice :

        Returns
        -------

        """
        if any(block[1:]):
            raise IndexError(block)
        arr = self.read(builtins.slice(block[0], block[0] + 1))
        return arr[slice]

    def structure(self) -> ArrayStructure:
        """

        Returns
        -------

        """
        return self._structure

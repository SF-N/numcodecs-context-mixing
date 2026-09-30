"""
Context-mixing arithmetic coders for integer and boolean arrays for the
[`numcodecs`][numcodecs] buffer compression API.

All coders interpret the data as a stack of 2D slices `[..., rows, cols]` and
use causal neighbours in the current slice as well as the previous slice as
contexts. They are lossless.
"""

__all__ = [
    "ContextMixingBitmapCodec",
    "ContextMixingSymbolCodec",
    "ContextMixingResidualCodec",
]

import math
from functools import reduce
from io import BytesIO

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
from numcodecs.abc import Codec
from typing_extensions import Buffer  # MSPV 3.12

from . import _bitmap, _residuals, _symbols


def _as_slices(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Interpret `shape` as (slices, rows, cols)."""

    if len(shape) == 0:
        return (1, 1, 1)
    if len(shape) == 1:
        return (1, 1, shape[0])
    return (reduce(lambda a, b: a * b, shape[:-2], 1), shape[-2], shape[-1])


def _write_header(dtype: np.dtype, shape: tuple[int, ...]) -> list[bytes | bytearray]:
    message: list[bytes | bytearray] = []

    message.append(leb128.u.encode(len(dtype.str)))
    message.append(dtype.str.encode("ascii"))

    message.append(leb128.u.encode(len(shape)))
    for s in shape:
        message.append(leb128.u.encode(s))

    return message


def _read_header(b_io: BytesIO) -> tuple[np.dtype, tuple[int, ...]]:
    dtype = np.dtype(b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii"))
    shape = tuple(
        leb128.u.decode_reader(b_io)[0] for _ in range(leb128.u.decode_reader(b_io)[0])
    )
    return dtype, shape


def _check_rates(mixer_rate: float, model_floor: float) -> None:
    if not (math.isfinite(mixer_rate) and mixer_rate > 0):
        raise ValueError("mixer_rate must be finite and positive")
    if not (math.isfinite(model_floor) and 0 < model_floor <= 1):
        raise ValueError("model_floor must be in (0, 1]")


def _padded_input(b: bytes) -> np.ndarray:
    # the range decoder may read a few bytes past the end of the stream
    return np.concatenate([np.frombuffer(b, np.uint8), np.zeros(16, np.uint8)])


class ContextMixingBitmapCodec(Codec):
    """
    Lossless codec for boolean arrays (bitmaps), e.g. masks of missing values.

    Every bit is coded with a binary arithmetic coder whose probability is
    produced by mixing several context models: neighbourhood templates of the
    causal neighbours in the current slice, the co-located neighbourhood in
    the previous slice, and the run length along the row.

    The array is interpreted as `[..., rows, cols]`. Any dtype is accepted;
    non-zero values are treated as [`True`][True] and the array decodes to a
    [`bool`][numpy.bool] array.

    Parameters
    ----------
    mixer_rate : float, optional
        Learning rate of the logistic mixer.
    model_floor : float, optional
        Minimum adaptation rate of the context models' probabilities (the
        rate starts at 1/1.5 for a fresh context and decays towards this
        floor).
    """

    __slots__: tuple[str, ...] = ("_mixer_rate", "_model_floor")
    _mixer_rate: float
    _model_floor: float

    codec_id: str = "context_mixing.bitmap"  # type: ignore

    def __init__(
        self, *, mixer_rate: float = 0.01, model_floor: float = 1 / 512
    ) -> None:
        _check_rates(mixer_rate, model_floor)
        self._mixer_rate = float(mixer_rate)
        self._model_floor = float(model_floor)

    def encode(self, buf: Buffer) -> bytes:
        """
        Encode the bitmap in `buf`.

        Parameters
        ----------
        buf : Buffer
            Bitmap to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded bitmap as a bytestring.
        """

        a = numcodecs.compat.ensure_ndarray(buf)
        shape = a.shape

        m = np.ascontiguousarray((a != 0).astype(np.uint8).reshape(_as_slices(shape)))

        out = np.zeros(m.size // 4 + 1024, np.uint8)
        n = _bitmap.encode_bitmap(m, out, self._mixer_rate, self._model_floor)

        # message: shape rates bitmap
        message: list[bytes | bytearray] = []

        message.append(leb128.u.encode(len(shape)))
        for s in shape:
            message.append(leb128.u.encode(s))

        message.append(
            np.array([self._mixer_rate, self._model_floor], dtype="<f8").tobytes()
        )

        message.append(out[:n].tobytes())

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the bitmap in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded bitmap. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded [`bool`][numpy.bool] bitmap.
        """

        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        mixer_rate, model_floor = np.frombuffer(b_io.read(16), dtype="<f8", count=2)

        T, Y, X = _as_slices(shape)
        m = _bitmap.decode_bitmap(
            _padded_input(b_io.read()), T, Y, X, float(mixer_rate), float(model_floor)
        )

        decoded = m.reshape(shape).astype(np.bool_)

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of this codec.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this codec from the returned config.

        Returns
        -------
        config : dict
            Configuration of this codec.
        """

        return dict(
            id=type(self).codec_id,
            mixer_rate=self._mixer_rate,
            model_floor=self._model_floor,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mixer_rate={self._mixer_rate!r}, model_floor={self._model_floor!r})"


class _IntegerContextMixingCodec(Codec):
    """Shared implementation of the integer coders (values are shifted to be
    non-negative and coded with `nbits` bits per symbol)."""

    __slots__: tuple[str, ...] = ("_mixer_rate", "_model_floor")
    _mixer_rate: float
    _model_floor: float

    _max_bits: int = 0

    def __init__(self, *, mixer_rate: float, model_floor: float) -> None:
        _check_rates(mixer_rate, model_floor)
        self._mixer_rate = float(mixer_rate)
        self._model_floor = float(model_floor)

    def _encode_values(self, q: np.ndarray, nbits: int, out: np.ndarray) -> int:
        raise NotImplementedError  # pragma: no cover

    def _decode_values(
        self,
        inp: np.ndarray,
        T: int,
        Y: int,
        X: int,
        nbits: int,
        mixer_rate: float,
        model_floor: float,
    ) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover

    def encode(self, buf: Buffer) -> bytes:
        """
        Encode the integer data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Integer (or boolean) data to be encoded. May be any object
            supporting the new-style buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded data as a bytestring.
        """

        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape

        if dtype.kind not in "iub":
            raise TypeError("can only encode integer or boolean values")

        values = a.astype(np.int64)
        minimum = int(values.min()) if values.size > 0 else 0
        maximum = int(values.max()) if values.size > 0 else 0
        nbits = max(1, int(maximum - minimum).bit_length())
        if nbits > self._max_bits:
            raise ValueError(
                f"the value range {maximum - minimum} exceeds the {self._max_bits}-bit limit of this codec"
            )

        q = np.ascontiguousarray((values - minimum).reshape(_as_slices(shape)))

        out = np.zeros(q.size * 8 + 1024, np.uint8)
        n = self._encode_values(q, nbits, out)

        # message: dtype shape minimum nbits rates values
        message = _write_header(dtype, shape)

        message.append(leb128.i.encode(minimum))
        message.append(leb128.u.encode(nbits))

        message.append(
            np.array([self._mixer_rate, self._model_floor], dtype="<f8").tobytes()
        )

        message.append(out[:n].tobytes())

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the integer data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol.
        """

        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        dtype, shape = _read_header(b_io)
        minimum = leb128.i.decode_reader(b_io)[0]
        nbits = leb128.u.decode_reader(b_io)[0]
        mixer_rate, model_floor = np.frombuffer(b_io.read(16), dtype="<f8", count=2)

        T, Y, X = _as_slices(shape)
        q = self._decode_values(
            _padded_input(b_io.read()),
            T,
            Y,
            X,
            nbits,
            float(mixer_rate),
            float(model_floor),
        )

        decoded = (q.reshape(shape) + minimum).astype(dtype)

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of this codec.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this codec from the returned config.

        Returns
        -------
        config : dict
            Configuration of this codec.
        """

        return dict(
            id=type(self).codec_id,
            mixer_rate=self._mixer_rate,
            model_floor=self._model_floor,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mixer_rate={self._mixer_rate!r}, model_floor={self._model_floor!r})"


class ContextMixingSymbolCodec(_IntegerContextMixingCodec):
    """
    Lossless codec for integer arrays with a small alphabet (up to 2^16
    distinct values), e.g. quantisation indices of noisy data.

    Every value is coded directly, MSB first through a binary tree, with
    contexts built from the neighbouring values in the current slice (left,
    up, diagonals) and the previous slice, their local activity, and a
    median-edge prediction. This works best when the data has few levels
    and is not well predicted by its neighbours; use the
    [`ContextMixingResidualCodec`][numcodecs_context_mixing.ContextMixingResidualCodec]
    for large alphabets.

    The array is interpreted as `[..., rows, cols]`.

    Parameters
    ----------
    mixer_rate : float, optional
        Learning rate of the logistic mixer.
    model_floor : float, optional
        Minimum adaptation rate of the context models' probabilities.
    """

    __slots__ = ()

    codec_id: str = "context_mixing.symbols"  # type: ignore

    _max_bits = _symbols.MAX_BITS

    def __init__(
        self, *, mixer_rate: float = 0.004, model_floor: float = 1 / 256
    ) -> None:
        super().__init__(mixer_rate=mixer_rate, model_floor=model_floor)

    def _encode_values(self, q: np.ndarray, nbits: int, out: np.ndarray) -> int:
        return int(
            _symbols.encode_symbols(q, nbits, out, self._mixer_rate, self._model_floor)
        )

    def _decode_values(
        self,
        inp: np.ndarray,
        T: int,
        Y: int,
        X: int,
        nbits: int,
        mixer_rate: float,
        model_floor: float,
    ) -> np.ndarray:
        return _symbols.decode_symbols(inp, T, Y, X, nbits, mixer_rate, model_floor)


class ContextMixingResidualCodec(_IntegerContextMixingCodec):
    """
    Lossless codec for integer arrays via prediction residuals, e.g.
    quantisation indices of smooth data.

    Each value is predicted by a median edge detector from its causal 2D
    neighbours or, if a previous slice exists and has recently been the more
    accurate predictor, from the co-located value of the previous slice
    corrected by the neighbours' slice-to-slice differences. The residual is
    coded MSB first through a binary tree with contexts from the local
    activity, the disagreement of the two predictors, and the neighbouring
    residuals in the current and previous slice.

    The array is interpreted as `[..., rows, cols]`.

    Parameters
    ----------
    mixer_rate : float, optional
        Learning rate of the logistic mixer.
    model_floor : float, optional
        Minimum adaptation rate of the context models' probabilities.
    """

    __slots__ = ()

    codec_id: str = "context_mixing.residuals"  # type: ignore

    _max_bits = _residuals.MAX_BITS

    def __init__(
        self, *, mixer_rate: float = 0.006, model_floor: float = 1 / 256
    ) -> None:
        super().__init__(mixer_rate=mixer_rate, model_floor=model_floor)

    def _encode_values(self, q: np.ndarray, nbits: int, out: np.ndarray) -> int:
        return int(
            _residuals.encode_residuals(
                q, nbits, out, self._mixer_rate, self._model_floor
            )
        )

    def _decode_values(
        self,
        inp: np.ndarray,
        T: int,
        Y: int,
        X: int,
        nbits: int,
        mixer_rate: float,
        model_floor: float,
    ) -> np.ndarray:
        return _residuals.decode_residuals(inp, T, Y, X, nbits, mixer_rate, model_floor)


numcodecs.registry.register_codec(ContextMixingBitmapCodec)
numcodecs.registry.register_codec(ContextMixingSymbolCodec)
numcodecs.registry.register_codec(ContextMixingResidualCodec)

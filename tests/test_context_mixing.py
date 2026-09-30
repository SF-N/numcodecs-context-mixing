import numcodecs
import numcodecs.registry
import numpy as np
import pytest

IDS = {
    "context_mixing.bitmap": "ContextMixingBitmapCodec",
    "context_mixing.symbols": "ContextMixingSymbolCodec",
    "context_mixing.residuals": "ContextMixingResidualCodec",
}


@pytest.mark.parametrize("codec_id,name", IDS.items())
def test_from_config(codec_id, name):
    codec = numcodecs.registry.get_codec(dict(id=codec_id))
    assert codec.__class__.__name__ == name
    assert codec.__class__.__module__ == "numcodecs_context_mixing"
    config = codec.get_config()
    assert config["id"] == codec_id
    assert set(config) == {"id", "mixer_rate", "model_floor"}
    assert numcodecs.registry.get_codec(config).get_config() == config


def test_invalid():
    from numcodecs_context_mixing import (
        ContextMixingResidualCodec,
        ContextMixingSymbolCodec,
    )

    with pytest.raises(ValueError):
        ContextMixingSymbolCodec(mixer_rate=0.0)
    with pytest.raises(ValueError):
        ContextMixingResidualCodec(model_floor=2.0)
    with pytest.raises(TypeError):
        ContextMixingSymbolCodec().encode(np.zeros(10))
    # alphabet too large for the direct symbol coder
    with pytest.raises(ValueError):
        ContextMixingSymbolCodec().encode(np.arange(2**17, dtype=np.int32))


def check_roundtrip(codec_id: str, data: np.ndarray, **kwargs):
    codec = numcodecs.registry.get_codec(dict(id=codec_id, **kwargs))

    encoded = codec.encode(data)
    decoded = np.asarray(codec.decode(encoded))

    assert decoded.shape == data.shape
    if codec_id == "context_mixing.bitmap":
        assert decoded.dtype == np.bool_
        np.testing.assert_array_equal(decoded, data != 0)
    else:
        assert decoded.dtype == data.dtype
        np.testing.assert_array_equal(decoded, data)

    out = np.empty_like(decoded)
    codec.decode(encoded, out=out)
    np.testing.assert_array_equal(out, decoded)

    return len(encoded)


def test_bitmap_roundtrip():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:96]
    structured = ((yy - 30) ** 2 + (xx - 40) ** 2 < 500) | (xx > 80)
    check_roundtrip("context_mixing.bitmap", structured)
    check_roundtrip(
        "context_mixing.bitmap", np.stack([structured, ~structured, structured])
    )
    check_roundtrip("context_mixing.bitmap", rng.random((50, 70)) < 0.3)
    check_roundtrip("context_mixing.bitmap", np.zeros((10, 10), dtype=bool))
    check_roundtrip("context_mixing.bitmap", np.ones(37, dtype=bool))
    check_roundtrip("context_mixing.bitmap", np.array(True))
    check_roundtrip("context_mixing.bitmap", np.zeros((0, 5), dtype=bool))
    # non-boolean input is interpreted as non-zero
    check_roundtrip("context_mixing.bitmap", rng.integers(0, 3, size=(20, 30)))
    # a structured bitmap compresses far below one bit per pixel
    size = check_roundtrip("context_mixing.bitmap", structured)
    assert size < structured.size / 8 / 4


@pytest.mark.parametrize(
    "codec_id", ["context_mixing.symbols", "context_mixing.residuals"]
)
def test_integer_roundtrip(codec_id):
    rng = np.random.default_rng(1)
    # smooth field with a little noise, quantised to a small alphabet
    yy, xx = np.mgrid[0:60, 0:80]
    smooth = np.rint(
        10 * np.sin(yy / 9.0) * np.cos(xx / 13.0) + rng.normal(size=yy.shape)
    )
    for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16):
        data = smooth.astype(np.int64)
        if np.dtype(dtype).kind == "u":
            data = data - data.min()
        check_roundtrip(codec_id, data.astype(dtype))
    check_roundtrip(
        codec_id, np.stack([smooth, smooth + 1, smooth * 2]).astype(np.int32)
    )
    check_roundtrip(codec_id, smooth.astype(np.int32).ravel())
    check_roundtrip(codec_id, np.array(7, dtype=np.int32))
    check_roundtrip(codec_id, np.zeros((0, 3), dtype=np.int16))
    check_roundtrip(codec_id, np.full((5, 5), -3, dtype=np.int8))
    check_roundtrip(
        codec_id, rng.integers(-5, 5, size=(30, 40)).astype(np.int16), mixer_rate=0.02
    )
    check_roundtrip(codec_id, (rng.random((20, 20)) < 0.5))


def test_residuals_large_range():
    rng = np.random.default_rng(2)
    data = (
        np.cumsum(rng.integers(-1000, 1000, size=(40, 50)), axis=1).astype(np.int64)
        * 1000
    )
    check_roundtrip("context_mixing.residuals", data)
    check_roundtrip(
        "context_mixing.residuals",
        data.astype(np.uint32) if data.min() >= 0 else data - data.min(),
    )


def test_compression():
    yy, xx = np.mgrid[0:200, 0:300]
    smooth = np.rint(50 * np.sin(yy / 20.0) * np.cos(xx / 30.0)).astype(np.int16)
    size = check_roundtrip("context_mixing.residuals", smooth)
    assert size < smooth.nbytes / 15
    noisy = np.random.default_rng(3).integers(0, 4, size=(200, 300)).astype(np.uint8)
    size = check_roundtrip("context_mixing.symbols", noisy)
    assert size < noisy.nbytes / 3.5  # ~2 bits/symbol entropy + overhead


@pytest.mark.parametrize(
    "codec_id", ["context_mixing.symbols", "context_mixing.residuals"]
)
def test_masked(codec_id):
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:50, 0:70]
    data = np.rint(8 * np.sin(yy / 7.0) * np.cos(xx / 11.0)).astype(np.int16)
    mask = rng.random(data.shape) < 0.4
    garbage = np.where(mask, np.int16(-30000), data)  # masked values are irrelevant

    codec = numcodecs.registry.get_codec(dict(id=codec_id))
    encoded = codec.encode_masked(garbage, mask)
    decoded = np.asarray(codec.decode_masked(encoded, mask))
    assert decoded.dtype == data.dtype and decoded.shape == data.shape
    np.testing.assert_array_equal(decoded[~mask], data[~mask])

    # skipping the masked values saves bits compared to coding the garbage
    assert len(encoded) < len(codec.encode(garbage))

    # ... and composes with the mask meta-codec, which passes the mask on
    from numcodecs_mask import MaskMetaCodec

    values = np.where(mask, np.int16(0), data)
    meta = MaskMetaCodec(
        mask=0, codec=dict(id=codec_id), bitmap_codec=dict(id="context_mixing.bitmap")
    )
    np.testing.assert_array_equal(meta.decode(meta.encode(values)), values)

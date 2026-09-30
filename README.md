[![image](https://img.shields.io/github/actions/workflow/status/SF-N/numcodecs-context-mixing/ci.yml?branch=main)](https://github.com/SF-N/numcodecs-context-mixing/actions/workflows/ci.yml?query=branch%3Amain)
[![image](https://img.shields.io/pypi/v/numcodecs-context-mixing.svg)](https://pypi.python.org/pypi/numcodecs-context-mixing)
[![image](https://img.shields.io/pypi/l/numcodecs-context-mixing.svg)](https://github.com/SF-N/numcodecs-context-mixing/blob/main/LICENSE)
[![image](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FSF-N%2Fnumcodecs-context-mixing%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://pypi.python.org/pypi/numcodecs-context-mixing)
[![image](https://readthedocs.org/projects/numcodecs-context-mixing/badge/?version=latest)](https://numcodecs-context-mixing.readthedocs.io/en/latest/?badge=latest)

# numcodecs-context-mixing

Context-mixing arithmetic coders for integer and boolean arrays for the [`numcodecs`] buffer compression API.

The codecs losslessly compress gridded data that is interpreted as `[..., rows, cols]` (a stack of 2D slices) with a binary arithmetic coder driven by PAQ-style context mixing: several context models (hashed causal neighbourhoods in the current slice and in the previous slice) each predict the next bit, an online logistic mixer combines the predictions, and all models adapt after every bit. Everything is deterministic and the decoder mirrors the encoder exactly. The coders are implemented with [numba](https://numba.pydata.org) and optimised for compression ratio rather than speed.

| codec id | class | input | idea |
|---|---|---|---|
| `context_mixing.bitmap` | `ContextMixingBitmapCodec` | boolean | masks (missing values, zeros, ...): 2D neighbourhood templates of the current and previous slice plus a run-length context |
| `context_mixing.symbols` | `ContextMixingSymbolCodec` | integer, small alphabet (up to 2^16 distinct values) | each value is coded directly with a bit tree conditioned on the neighbouring values, their differences and a median-edge prediction; best for noisy data with few levels |
| `context_mixing.residuals` | `ContextMixingResidualCodec` | integer | prediction residuals w.r.t. a median-edge (2D) or previous-slice predictor, selected adaptively, coded with a bit tree conditioned on local activity and neighbouring residuals; best for larger alphabets |

The integer coders are meant to follow a quantisation meta-codec such as [`numcodecs-eb-quantize`](https://github.com/SF-N/numcodecs-eb-quantize) (absolute error bound) or [`numcodecs-pw-ratio`](https://numcodecs-pw-ratio.readthedocs.io) (relative error bound), and to be combined with [`numcodecs-mask`](https://numcodecs-mask.readthedocs.io) (using the bitmap coder as `bitmap_codec`) for missing values:

```python
from numcodecs_eb_quantize import ErrorBoundedQuantizeCodec
from numcodecs_mask import MaskMetaCodec

codec = MaskMetaCodec(
    mask=float("nan"),
    bitmap_codec=dict(id="context_mixing.bitmap"),
    codec=ErrorBoundedQuantizeCodec(codec=dict(id="context_mixing.residuals"), eb=0.5),
)
```

The low-level primitives (range coder, bit models, mixer, context hashing) are exposed in `numcodecs_context_mixing.coder` for building further context-mixing codecs.

[`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).


## Funding

The `numcodecs-context-mixing` package has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.

"""Tier 1 smoke: trace ``CoreMLTextEncoderWrapper`` over a tiny CLIP and convert
through the real ``convert.convert_to_coreml`` path.

Covers both shapes the converter emits: the SD1.5/encoder-1 case (single
``hidden_states`` output, last hidden state) and the SDXL encoder-2 case
(penultimate hidden state + projected ``pooled_embeds``). Exercises the int32
``input_ids`` boundary through coremltools without a real checkpoint or the ANE.

``convert_text_encoder`` itself loads encoders from a single-file checkpoint
(pipeline extraction), so its load/branch logic is covered by the Tier 2 anchor,
not here. This pins the trace + conversion contract.

Auto-skips on non-Apple-Silicon hosts so Tier 0 CI on Linux ignores it.
"""

import platform

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Tier 1 requires macOS on Apple Silicon",
)

SEQ_LEN = 77
HIDDEN = 32
PROJECTION = 16


def _tiny_config():
    from transformers import CLIPTextConfig

    return CLIPTextConfig(
        vocab_size=1000,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=SEQ_LEN,
        projection_dim=PROJECTION,
    )


def _convert(wrapper, output_names, tmp_path, name):
    import coremltools as ct

    from coreml_diffusion.conversion.text_encoder import static_causal_mask
    from coreml_diffusion.convert import convert_to_coreml

    ids_shape = (1, SEQ_LEN)
    example = torch.zeros(ids_shape, dtype=torch.int64)
    with static_causal_mask(SEQ_LEN):
        traced = torch.jit.trace(wrapper.eval(), example)

    pkg = tmp_path / f"{name}.mlpackage"
    inputs = [ct.TensorType(name="input_ids", shape=ids_shape, dtype=np.int32)]
    model = convert_to_coreml(name, traced, inputs, output_names, str(pkg))
    model.save(str(pkg))

    spec = ct.models.MLModel(str(pkg)).get_spec()
    return (
        {i.name for i in spec.description.input},
        {o.name for o in spec.description.output},
    )


def test_encoder1_single_hidden_state(tmp_path):
    from transformers import CLIPTextModel

    from coreml_diffusion.conversion.text_encoder import CoreMLTextEncoderWrapper

    torch.manual_seed(0)
    encoder = CLIPTextModel(_tiny_config())
    wrapper = CoreMLTextEncoderWrapper(encoder, hidden_states_index=None)

    inputs, outputs = _convert(wrapper, ["hidden_states"], tmp_path, "te1")
    assert inputs == {"input_ids"}
    assert "hidden_states" in outputs


def test_encoder2_penultimate_plus_pooled(tmp_path):
    from transformers import CLIPTextModelWithProjection

    from coreml_diffusion.conversion.text_encoder import CoreMLTextEncoderWrapper

    torch.manual_seed(0)
    encoder = CLIPTextModelWithProjection(_tiny_config())
    wrapper = CoreMLTextEncoderWrapper(
        encoder, hidden_states_index=-2, output_pooled=True
    )

    inputs, outputs = _convert(
        wrapper, ["hidden_states", "pooled_embeds"], tmp_path, "te2"
    )
    assert inputs == {"input_ids"}
    assert {"hidden_states", "pooled_embeds"} <= outputs

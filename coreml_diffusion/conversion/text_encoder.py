"""CLIP text-encoder wrapper adapting it to a flat Core ML tensor contract.

Wraps a transformers ``CLIPTextModel`` (SD1.5, SDXL encoder 1) or
``CLIPTextModelWithProjection`` (SDXL encoder 2) so the traced graph takes a
single ``input_ids`` tensor ``(B, 77)`` and returns the embeddings the diffusion
pipeline consumes — nothing else.

Which hidden state and whether a pooled vector is emitted depends on the model:

  SD1.5            : final ``last_hidden_state`` (768), no pooled
  SDXL encoder 1   : penultimate ``hidden_states[-2]`` (768), no pooled
  SDXL encoder 2   : penultimate ``hidden_states[-2]`` (1280) + projected pooled (1280)

The penultimate selection is SDXL's documented behaviour (it concatenates the
two encoders' penultimate states and uses encoder 2's projected pooled output as
the ``add_embeds`` conditioning). ``hidden_states_index=None`` selects the final
``last_hidden_state``; an int indexes ``hidden_states`` directly.

The pooled vector prefers ``text_embeds`` (the projection head, encoder 2) and
falls back to ``pooler_output`` — so the same wrapper serves both CLIP variants.
``input_ids`` is fed as int32 at the Core ML boundary (see ``convert``).
"""

import contextlib

import torch


@contextlib.contextmanager
def static_causal_mask(seq_len):
    """Patch CLIP's causal-mask builder to a constant for the trace duration.

    transformers builds the causal mask from ``query_length + past_key_values_length``,
    both traced 0-dim size tensors; the resulting ``aten::Int`` cannot be folded to a
    const under coremltools 9 (conversion fails). Since the converted sequence length
    is fixed at trace time, swap ``_create_4d_causal_attention_mask`` for a closure
    that materialises the upper-triangular ``-inf`` mask from a Python-int ``seq_len``
    — a pure constant the frontend folds away. Mirrors ``prepare_unet_for_coreml_trace``:
    a trace-only enablement patch, restored on exit. Shape ``(1, 1, seq, seq)``
    broadcasts over batch/heads, so no symbolic batch leaks back in.
    """
    from transformers.models.clip import modeling_clip

    original = modeling_clip._create_4d_causal_attention_mask

    def _const_mask(input_shape, dtype, device=None, *args, **kwargs):
        mask = torch.full(
            (seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device
        )
        return torch.triu(mask, diagonal=1)[None, None]

    modeling_clip._create_4d_causal_attention_mask = _const_mask
    try:
        yield
    finally:
        modeling_clip._create_4d_causal_attention_mask = original


class CoreMLTextEncoderWrapper(torch.nn.Module):
    """token ids ``(B, 77)`` -> embeddings (+ optional pooled)."""

    def __init__(self, text_encoder, *, hidden_states_index=None, output_pooled=False):
        super().__init__()
        self.text_encoder = text_encoder
        self.hidden_states_index = hidden_states_index
        self.output_pooled = output_pooled

    def forward(self, input_ids):
        out = self.text_encoder(
            input_ids,
            output_hidden_states=self.hidden_states_index is not None,
            return_dict=True,
        )
        if self.hidden_states_index is None:
            embeds = out.last_hidden_state
        else:
            embeds = out.hidden_states[self.hidden_states_index]

        if not self.output_pooled:
            return embeds

        pooled = getattr(out, "text_embeds", None)
        if pooled is None:
            pooled = out.pooler_output
        return embeds, pooled

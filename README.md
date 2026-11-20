# magnetar-component-olmo

OLMo model graph Component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The initial OLMo 3 graph builder covers prefill, cached decode and layer
segments. It emits full-projection Q/K RMSNorm, post-attention and post-MLP
normalization, sliding or full causal attention, and SiLU-gated MLP nodes.
For the published OLMo 3 checkpoint, sliding attention uses ordinary RoPE and
full attention uses YaRN. Integration tests inspect these graph contracts against
the pinned published checkpoint configuration and exercise unsupported inputs.

**Inference is not available yet.** The WIT entrypoints deliberately fail
closed: the runtime cannot yet report the checkpoint's configurable
`hidden_act`, so emitting a fixed SiLU graph could silently execute a different
checkpoint incorrectly. This is tracked in
[Magnetar #217](https://github.com/astorise/Magnetar/issues/217). The existing
tests do not compare logits or execute the graph on a Provider.

## Work required

After the activation contract is available, add real loading, runtime graph
execution and independent reference-logit tests for an explicitly pinned
checkpoint. Verify other OLMo versions separately: their semantics differ.

The graph is based on the [OLMo 3 7B Instruct configuration](https://huggingface.co/allenai/Olmo-3-7B-Instruct)
and Hugging Face Transformers 4.57.1's `Olmo3ForCausalLM`. Model weights are
not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/olmo`.

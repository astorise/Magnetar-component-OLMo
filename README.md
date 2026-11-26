# magnetar-component-olmo

OLMo model graph Component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The initial OLMo 3 graph builder covers prefill, cached decode and layer
segments. It emits full-projection Q/K RMSNorm, post-attention and post-MLP
normalization, sliding or full causal attention, and SiLU-gated MLP nodes.
For the published OLMo 3 checkpoint, sliding attention uses ordinary RoPE and
full attention uses YaRN. Integration tests inspect these graph contracts against
the pinned published checkpoint configuration and exercise unsupported inputs.

The WIT entrypoints require the Runtime to supply a verified `silu` activation;
missing or different activations are refused. The Magnetar repository compares
a tiny full-attention checkpoint's prefill and cached-decode logits against an
independent numerical reference. Real OLMo 3 checkpoints and GPU execution
have not been qualified.

## Work required

Qualify real checkpoint loading and execution, including mixed attention and
YaRN. Verify other OLMo versions separately: their semantics differ.

The graph is based on the [OLMo 3 7B Instruct configuration](https://huggingface.co/allenai/Olmo-3-7B-Instruct)
and Hugging Face Transformers 4.57.1's `Olmo3ForCausalLM`. Model weights are
not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/olmo`.

## Built on the shared crates

The graph IR (`Phase`, `KvRole`, `Value`, `Attr`, `Kv`, `Node`), the `GraphSink`
the builder writes to and the adapter to the contract's `graph-builder` come from
[`magnetar-component-sdk`](https://github.com/astorise/magnetar-component-sdk),
and the contract itself (the WIT file and its generated bindings, version 1.8.0)
from [`magnetar-component-api`](https://github.com/astorise/magnetar-component-api)
with its `v1_8` feature. Both are git dependencies pinned by commit while they
are in development, and published once stable. `src/plan.rs` keeps only what is
specific to OLMo: the configuration, its validation and the graph.

`tests/golden/` holds the graphs the builder emits, recorded before it moved onto
either crate; `tests/golden_graphs.rs` compares every call against them. Regenerate
them deliberately with `UPDATE_GOLDEN=1 cargo test --test golden_graphs`.

## License and WIT provenance

This component is distributed under the [MIT License](LICENSE), copyright
2026 Sébastien ASTORI. The contract it targets (version 1.8.0) is
[`magnetar-component-api`'s](https://github.com/astorise/magnetar-component-api/blob/main/wit/1.8.0/model-component-graph.wit),
whose files came from
[Magnetar's model-component graph WIT](https://github.com/astorise/Magnetar/blob/main/magnetar-runtime/wit/model-component-graph.wit)
under the same MIT license and copyright notice.

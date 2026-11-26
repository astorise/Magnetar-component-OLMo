//! Golden graphs: every call the builder makes, recorded as text and compared
//! with the files in `tests/golden/`.
//!
//! These files were recorded from the builder before it moved onto
//! `magnetar-component-sdk` (the SDK's graph IR replaced the copy this crate
//! kept in `plan.rs`) and `magnetar-component-api` (the contract replaced the
//! WIT copy and the bindings generator this crate carried). They pin the graph a Model Instance receives: a change
//! that alters one node, attribute, input order, shape or KV involvement fails
//! here, whoever made it. Regenerate them deliberately with
//! `UPDATE_GOLDEN=1 cargo test --test golden_graphs`.
//!
//! The test uses only the crate's public `plan` API, so it holds across the
//! change of the types behind it.

use magnetar_component_olmo::plan::{
    Config, GraphSink, KvRole, Layer, Node, Phase, Segment, Value, Yarn, build_graph,
};
use std::{fmt::Write as _, path::PathBuf};

/// Records every call, one line each, in call order.
#[derive(Default)]
struct TextSink {
    text: String,
}

impl TextSink {
    fn line(&mut self, line: std::fmt::Arguments<'_>) {
        writeln!(self.text, "{line}").unwrap();
    }
}

impl GraphSink for TextSink {
    fn begin_graph(&mut self, phase: Phase, sequence_length: u64, position_offset: u64) {
        self.line(format_args!(
            "begin {phase:?} sequence={sequence_length} offset={position_offset}"
        ));
    }
    fn declare_input(&mut self, name: &str, shape: &[u64]) -> String {
        self.line(format_args!("input {name} {shape:?}"));
        format!("input:{name}")
    }
    fn weight_edge(&mut self, name: &str) -> String {
        self.line(format_args!("weight {name}"));
        format!("weight:{name}")
    }
    fn alias_weight_edge(&mut self, edge: &str, target: &str) {
        self.line(format_args!("alias {edge} -> {target}"));
    }
    fn kv_resource(&mut self, layer: u64, role: KvRole) -> String {
        self.line(format_args!("kv {layer} {role:?}"));
        format!("kv:{layer}:{role:?}")
    }
    fn add_node(&mut self, node: Node) -> String {
        let mut line = format!(
            "node {} op={} family={} inputs={:?} shape={:?}",
            node.id, node.operator, node.family, node.inputs, node.shape
        );
        for attr in &node.attrs {
            let value = match &attr.value {
                Value::Bool(v) => format!("bool:{v}"),
                Value::Int(v) => format!("int:{v}"),
                Value::Float(v) => format!("float:{v:?}"),
                Value::Text(v) => format!("text:{v}"),
            };
            write!(line, " {}={value}", attr.name).unwrap();
        }
        if let Some(kv) = &node.kv {
            write!(
                line,
                " kv=({}, {:?}, append={})",
                kv.resource, kv.role, kv.append
            )
            .unwrap();
        }
        self.line(format_args!("{line}"));
        format!("out:{}", node.id)
    }
    fn finish_graph(&mut self, output: &str) -> String {
        self.line(format_args!("finish {output}"));
        output.into()
    }
}

fn layer(window: Option<u32>) -> Layer {
    Layer {
        window,
        head_dimension: 4,
        kv_heads: 2,
        rope_theta: 500_000.0,
        intermediate_size: 12,
        partial_rotary: None,
        kv_source: None,
        value_from_key: false,
    }
}

fn yarn() -> Yarn {
    Yarn {
        factor: 8.0,
        original_context: 8192,
        beta_fast: 32.0,
        beta_slow: 1.0,
        attention_factor: 1.2079441541679836,
        truncate: true,
    }
}

fn tiny(layers: usize, window: Option<u32>, tie_embeddings: bool) -> Config {
    Config {
        hidden: 8,
        heads: 4,
        vocabulary: 17,
        epsilon: 1e-6,
        hidden_activation: "silu".into(),
        tie_embeddings,
        layers: vec![layer(window); layers],
        yarn: Some(yarn()),
    }
}

fn olmo3_7b() -> Config {
    let raw: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/olmo3-7b-instruct.config.json")).unwrap();
    let heads = raw["num_attention_heads"].as_u64().unwrap() as u32;
    let hidden = raw["hidden_size"].as_u64().unwrap();
    let kv_heads = raw["num_key_value_heads"].as_u64().unwrap() as u32;
    let rope = &raw["rope_scaling"];
    let layers = raw["layer_types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|kind| Layer {
            window: match kind.as_str().unwrap() {
                "sliding_attention" => Some(raw["sliding_window"].as_u64().unwrap() as u32),
                "full_attention" => None,
                other => panic!("unknown {other}"),
            },
            head_dimension: hidden / u64::from(heads),
            kv_heads,
            rope_theta: raw["rope_theta"].as_f64().unwrap(),
            intermediate_size: raw["intermediate_size"].as_u64().unwrap(),
            partial_rotary: None,
            kv_source: None,
            value_from_key: false,
        })
        .collect();
    Config {
        hidden,
        heads,
        vocabulary: raw["vocab_size"].as_u64().unwrap(),
        epsilon: raw["rms_norm_eps"].as_f64().unwrap(),
        hidden_activation: raw["hidden_act"].as_str().unwrap().into(),
        tie_embeddings: raw["tie_word_embeddings"].as_bool().unwrap(),
        layers,
        yarn: Some(Yarn {
            factor: rope["factor"].as_f64().unwrap(),
            original_context: rope["original_max_position_embeddings"].as_u64().unwrap(),
            beta_fast: rope["beta_fast"].as_f64().unwrap(),
            beta_slow: rope["beta_slow"].as_f64().unwrap(),
            attention_factor: rope["attention_factor"].as_f64().unwrap(),
            truncate: true,
        }),
    }
}

fn segment(phase: Phase, sequence_length: u64, offset: u64, start: u32, end: u32) -> Segment {
    Segment {
        phase,
        sequence_length,
        position_offset: offset,
        start_layer: start,
        end_layer: end,
    }
}

fn check(name: &str, config: &Config, segment: Segment) {
    let mut sink = TextSink::default();
    build_graph(&mut sink, config, segment).unwrap();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.txt"));
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::write(&path, &sink.text).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("golden graph {}: {error}", path.display()));
    assert_eq!(
        sink.text, expected,
        "graph '{name}' differs from tests/golden/{name}.txt"
    );
}

#[test]
fn prefill_one_windowed_layer() {
    check(
        "prefill_tiny_windowed",
        &tiny(1, Some(3), false),
        segment(Phase::Prefill, 3, 0, 0, 1),
    );
}

#[test]
fn decode_one_full_attention_layer() {
    check(
        "decode_tiny_full_attention",
        &tiny(1, None, false),
        segment(Phase::Decode, 1, 11, 0, 1),
    );
}

#[test]
fn prefill_two_layers_tied_embeddings() {
    // A sliding layer then a full-attention one: YaRN applies to the second only.
    let mut config = tiny(2, Some(3), true);
    config.layers[1].window = None;
    check(
        "prefill_tiny_two_layers_tied",
        &config,
        segment(Phase::Prefill, 3, 0, 0, 2),
    );
}

#[test]
fn decode_middle_segment() {
    check(
        "decode_tiny_middle_segment",
        &tiny(3, Some(3), true),
        segment(Phase::Decode, 1, 11, 1, 2),
    );
}

#[test]
fn decode_last_segment_aliases_the_embedding() {
    check(
        "decode_tiny_last_segment_tied",
        &tiny(3, Some(3), true),
        segment(Phase::Decode, 1, 11, 2, 3),
    );
}

#[test]
fn olmo3_7b_first_four_layers() {
    // Three sliding layers and the first full-attention one.
    check(
        "olmo3_7b_prefill_layers_0_4",
        &olmo3_7b(),
        segment(Phase::Prefill, 5, 0, 0, 4),
    );
}

#[test]
fn olmo3_7b_last_two_layers() {
    check(
        "olmo3_7b_decode_layers_30_32",
        &olmo3_7b(),
        segment(Phase::Decode, 1, 4096, 30, 32),
    );
}

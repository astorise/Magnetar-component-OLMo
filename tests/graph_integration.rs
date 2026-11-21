use magnetar_component_olmo::plan::{
    Attr, Config, GraphSink, KvRole, Layer, Node, Phase, Segment, Value, Yarn, build_graph,
};

#[derive(Default)]
struct RecordingSink {
    begin: Option<(Phase, u64, u64)>,
    nodes: Vec<Node>,
    inputs: Vec<String>,
    aliases: Vec<(String, String)>,
    output: Option<String>,
}

impl RecordingSink {
    fn node(&self, id: &str) -> &Node {
        self.nodes
            .iter()
            .find(|node| node.id == id)
            .unwrap_or_else(|| panic!("missing node {id}"))
    }
}

impl GraphSink for RecordingSink {
    fn begin_graph(&mut self, phase: Phase, length: u64, offset: u64) {
        self.begin = Some((phase, length, offset));
    }
    fn declare_input(&mut self, name: &str, _shape: &[u64]) -> String {
        self.inputs.push(name.into());
        format!("input:{name}")
    }
    fn weight_edge(&mut self, name: &str) -> String {
        format!("weight:{name}")
    }
    fn alias_weight_edge(&mut self, edge: &str, target: &str) {
        self.aliases.push((edge.into(), target.into()));
    }
    fn kv_resource(&mut self, layer: u64, role: KvRole) -> String {
        format!("{layer}:{role:?}")
    }
    fn add_node(&mut self, node: Node) -> String {
        let result = format!("out:{}", node.id);
        self.nodes.push(node);
        result
    }
    fn finish_graph(&mut self, output: &str) -> String {
        self.output = Some(output.into());
        output.into()
    }
}

fn attribute<'a>(node: &'a Node, name: &str) -> Option<&'a Value> {
    node.attrs
        .iter()
        .find_map(|Attr { name: n, value }| (*n == name).then_some(value))
}

fn tiny() -> Config {
    Config {
        hidden: 8,
        heads: 4,
        vocabulary: 17,
        epsilon: 1e-6,
        hidden_activation: "silu".into(),
        tie_embeddings: false,
        layers: vec![Layer {
            window: Some(3),
            head_dimension: 4,
            kv_heads: 2,
            rope_theta: 500_000.0,
            intermediate_size: 12,
            partial_rotary: None,
            kv_source: None,
            value_from_key: false,
        }],
        yarn: Some(Yarn {
            factor: 8.0,
            original_context: 8192,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: 1.2079441541679836,
            truncate: true,
        }),
    }
}

fn segment(phase: Phase, start: u32, end: u32) -> Segment {
    Segment {
        phase,
        sequence_length: if phase == Phase::Decode { 1 } else { 3 },
        position_offset: if phase == Phase::Decode { 11 } else { 0 },
        start_layer: start,
        end_layer: end,
    }
}

#[test]
fn prefill_graph_uses_projected_width_qk_norm_and_post_sublayer_norms() {
    let mut sink = RecordingSink::default();
    build_graph(&mut sink, &tiny(), segment(Phase::Prefill, 0, 1)).unwrap();
    assert_eq!(sink.begin, Some((Phase::Prefill, 3, 0)));
    assert_eq!(sink.node("layer0.q_norm").shape, [3, 16]);
    assert_eq!(sink.node("layer0.k_norm").shape, [3, 8]);
    assert_eq!(
        sink.node("layer0.q_norm").inputs,
        ["out:layer0.q_proj", "weight:layers.0.self_attn.q_norm"]
    );
    assert_eq!(
        sink.node("layer0.k_norm").inputs,
        ["out:layer0.k_proj", "weight:layers.0.self_attn.k_norm"]
    );
    assert_eq!(attribute(sink.node("layer0.q_norm"), "head_count"), None);
    assert_eq!(
        sink.node("layer0.post_attn_norm").inputs[0],
        "out:layer0.o_proj"
    );
    assert_eq!(
        sink.node("layer0.residual1").inputs,
        ["out:embedding", "out:layer0.post_attn_norm"]
    );
    assert_eq!(
        sink.node("layer0.gate_proj").inputs[0],
        "out:layer0.residual1"
    );
    assert_eq!(
        sink.node("layer0.post_ffn_norm").inputs[0],
        "out:layer0.down_proj"
    );
    assert_eq!(
        sink.node("layer0.residual2").inputs,
        ["out:layer0.residual1", "out:layer0.post_ffn_norm"]
    );
    assert_eq!(sink.output.as_deref(), Some("out:lm_head"));
    assert!(!sink.nodes.iter().any(|n| n.id == "layer0.input_norm"));
}

#[test]
fn sliding_attention_uses_unscaled_rope_and_own_kv_while_decode_appends() {
    for phase in [Phase::Prefill, Phase::Decode] {
        let mut sink = RecordingSink::default();
        build_graph(&mut sink, &tiny(), segment(phase, 0, 1)).unwrap();
        let append = phase == Phase::Decode;
        assert_eq!(
            sink.node("layer0.v_proj").kv.as_ref().unwrap().append,
            append
        );
        assert_eq!(
            sink.node("layer0.rope_k").kv.as_ref().unwrap().append,
            append
        );
        assert_eq!(
            sink.node("layer0.v_proj").kv.as_ref().unwrap().role,
            KvRole::V
        );
        assert_eq!(
            sink.node("layer0.rope_k").kv.as_ref().unwrap().role,
            KvRole::K
        );
        assert_eq!(
            attribute(sink.node("layer0.attention"), "window_size"),
            Some(&Value::Int(3))
        );
        assert_eq!(
            attribute(sink.node("layer0.rope_q"), "frequency_scaling"),
            None
        );
        assert_eq!(
            attribute(sink.node("layer0.rope_q"), "rotation_pairing"),
            Some(&Value::Text("half_split".into()))
        );
    }
}

#[test]
fn published_checkpoint_uses_yarn_on_full_but_not_sliding_layers() {
    let raw: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/olmo3-7b-instruct.config.json")).unwrap();
    assert_eq!(raw["model_type"], "olmo3");
    assert_eq!(raw["hidden_act"], "silu");
    let heads = raw["num_attention_heads"].as_u64().unwrap() as u32;
    let hidden = raw["hidden_size"].as_u64().unwrap();
    let kv_heads = raw["num_key_value_heads"].as_u64().unwrap() as u32;
    let rope = &raw["rope_scaling"];
    let layers: Vec<_> = raw["layer_types"]
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
    assert_eq!(
        layers.len(),
        raw["num_hidden_layers"].as_u64().unwrap() as usize
    );
    let config = Config {
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
    };
    let mut sink = RecordingSink::default();
    build_graph(&mut sink, &config, segment(Phase::Prefill, 0, 4)).unwrap();
    assert_eq!(
        attribute(sink.node("layer0.rope_q"), "frequency_scaling"),
        None
    );
    assert_eq!(
        attribute(sink.node("layer0.attention"), "window_size"),
        Some(&Value::Int(4096))
    );
    assert_eq!(
        attribute(sink.node("layer3.attention"), "window_size"),
        None
    );
    assert_eq!(
        attribute(sink.node("layer3.rope_q"), "frequency_scaling"),
        Some(&Value::Text("yarn".into()))
    );
    assert_eq!(
        attribute(sink.node("layer3.rope_k"), "frequency_scaling_factor"),
        Some(&Value::Float(8.0))
    );
    assert_eq!(
        attribute(sink.node("layer3.rope_k"), "attention_factor"),
        Some(&Value::Float(1.2079441541679836))
    );
}

#[test]
fn middle_segment_uses_hidden_input_and_does_not_emit_logits() {
    let mut config = tiny();
    config.layers.push(config.layers[0].clone());
    config.layers.push(config.layers[0].clone());
    let mut sink = RecordingSink::default();
    build_graph(&mut sink, &config, segment(Phase::Decode, 1, 2)).unwrap();
    assert_eq!(sink.begin, Some((Phase::Decode, 1, 11)));
    assert_eq!(
        attribute(sink.node("layer1.rope_q"), "position_offset"),
        Some(&Value::Int(11))
    );
    assert_eq!(
        attribute(sink.node("layer1.rope_k"), "position_offset"),
        Some(&Value::Int(11))
    );
    assert_eq!(sink.inputs, ["hidden_states_in"]);
    assert_eq!(
        sink.node("layer1.q_proj").inputs[0],
        "input:hidden_states_in"
    );
    assert_eq!(sink.output.as_deref(), Some("out:layer1.residual2"));
    assert!(
        !sink
            .nodes
            .iter()
            .any(|node| node.id == "embedding" || node.id == "lm_head")
    );
}

#[test]
fn rejects_invalid_geometry_before_touching_the_host() {
    for mutation in 0..5 {
        let mut cfg = tiny();
        match mutation {
            0 => cfg.layers[0].kv_heads = 3,
            1 => cfg.layers[0].window = Some(0),
            2 => cfg.layers[0].partial_rotary = Some(0.5),
            3 => cfg.layers[0].kv_source = Some(0),
            _ => cfg.hidden_activation = "gelu".into(),
        }
        let mut sink = RecordingSink::default();
        assert!(build_graph(&mut sink, &cfg, segment(Phase::Prefill, 0, 1)).is_err());
        assert!(sink.begin.is_none());
    }
}

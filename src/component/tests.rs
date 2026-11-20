use super::require_activation_contract;

#[test]
fn refuses_to_publish_a_graph_when_host_cannot_report_the_activation() {
    let error = require_activation_contract().unwrap_err();
    assert!(error.contains("hidden_act"));
    assert!(error.contains("#217"));
}

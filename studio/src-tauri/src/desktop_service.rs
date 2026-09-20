/// Liveness/health `service` field the desktop shell will adopt.
pub fn is_studio_backend_service(name: Option<&str>) -> bool {
    matches!(name, Some("Helix Harness") | Some("Unsloth UI Backend"))
}

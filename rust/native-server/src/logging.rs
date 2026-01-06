pub fn debug_enabled() -> bool {
    std::env::var("COSYVOICE_DEBUG")
        .map(|v| v != "0")
        .unwrap_or(false)
}

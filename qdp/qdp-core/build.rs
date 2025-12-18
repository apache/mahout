use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=QDP_CPU_ONLY");

    if cpu_only_enabled() {
        println!("cargo:rustc-cfg=cpu_only");
        println!("cargo:warning=QDP_CPU_ONLY=1 detected: compiling qdp-core without CUDA support.");
    }
}

fn cpu_only_enabled() -> bool {
    env::var("QDP_CPU_ONLY")
        .map(|val| val == "1" || val.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

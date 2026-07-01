// Generates `impl prost::Name for proto::PhysicalRelation` so client.rs
// can derive the Any type URL instead of hardcoding it.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../../proto/physical_plan.proto");
    println!("cargo:rerun-if-changed=../../proto/expression.proto");
    println!("cargo:rerun-if-changed=../../proto/data_type.proto");

    let mut cfg = prost_build::Config::new();
    cfg.enable_type_names();
    cfg.compile_protos(&["../../proto/physical_plan.proto"], &["../../proto/"])?;
    Ok(())
}

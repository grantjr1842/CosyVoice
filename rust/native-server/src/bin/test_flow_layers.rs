use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use cosyvoice_rust_backend::flow::{DiT, FlowConfig};
use std::path::Path;

fn log_stats(name: &str, t: &Tensor) -> Result<()> {
    let t = if t.dtype() != DType::F32 {
        t.to_dtype(DType::F32)?
    } else {
        t.clone()
    };
    let flat = t.flatten_all()?;
    let vec = flat.to_vec1::<f32>()?;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &v in vec.iter() {
        min = f32::min(min, v);
        max = f32::max(max, v);
        sum += v as f64;
    }
    let mean = (sum / vec.len() as f64) as f32;
    println!(
        "    [{}] stats: min={:.6}, max={:.6}, mean={:.6}, shape={:?}",
        name,
        min,
        max,
        mean,
        t.shape()
    );
    Ok(())
}

fn compare(name: &str, rust: &Tensor, py: &Tensor) -> Result<()> {
    // py is usually F32. Rust matches py dtype.
    let rust = rust.to_dtype(DType::F32)?;
    let py = py.to_dtype(DType::F32)?.to_device(&rust.device())?;

    if rust.shape() != py.shape() {
        println!(
            "    [Mismatch] {} shape: rust={:?} vs py={:?}",
            name,
            rust.shape(),
            py.shape()
        );
        return Ok(());
    }

    let diff = (rust - py)?.abs()?;
    let max = diff
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .cloned()
        .fold(0.0 / 0.0, f32::max);
    let mean = diff.mean_all()?.to_scalar::<f32>()?;

    println!("    [Compare] {}: MAE={:.6e}, MAX={:.6e}", name, mean, max);
    if mean > 1e-4 {
        println!("    *** DIVERGENCE DETECTED ***");
    }
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0).expect("CUDA required");
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let model_dir = repo_root.join("pretrained_models/Fun-CosyVoice3-0.5B");

    let dtype = DType::F32; // Use F32 for precision debugging
    let flow_path = model_dir.join("flow.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[flow_path], dtype, &device)? };

    // Load DiT
    let cfg = FlowConfig::default();
    // DiT weights are under "decoder.estimator."
    let vb_dit = vb.pp("decoder.estimator");
    let dit = DiT::new(vb_dit, &cfg)?;

    // Load inputs
    let inputs_path = repo_root.join("rust_flow_debug.safetensors");
    let cpu = Device::Cpu;
    let inputs = candle_core::safetensors::load(&inputs_path, &cpu)?;

    let mu = inputs
        .get("mu")
        .unwrap()
        .to_device(&device)?
        .to_dtype(dtype)?;
    let mask = inputs
        .get("mask")
        .unwrap()
        .to_device(&device)?
        .to_dtype(dtype)?;
    let spks = inputs
        .get("spks")
        .unwrap()
        .to_device(&device)?
        .to_dtype(dtype)?;
    let cond = inputs
        .get("cond")
        .unwrap()
        .to_device(&device)?
        .to_dtype(dtype)?;
    let x = inputs
        .get("x_init")
        .unwrap()
        .to_device(&device)?
        .to_dtype(dtype)?;

    // Load layer outputs
    let layers_path = repo_root.join("debug_flow_layers.safetensors");
    let layers = candle_core::safetensors::load(&layers_path, &cpu)?;
    let py_time = layers.get("time_embed_out").unwrap();
    let py_input = layers.get("input_embed_out").unwrap();
    let py_block0 = layers.get("block0_out").unwrap();

    println!("--- Verifying Layer Outputs ---");

    // 1. Time Embedding
    let t = Tensor::from_vec(vec![0.0f32], (1,), &device)?.to_dtype(dtype)?;
    let t_emb_rust = dit.time_embed.forward(&t)?;
    compare("Time Embed", &t_emb_rust, py_time)?;

    // 2. Input Embedding
    // Rust inputs: x [B, 80, T], cond [B, 80, T], mu [B, 80, T], spks [1, 80]
    // Python InputEmbedding input: x [B, 80, T]
    // Wait, Python debug_flow_layers.py passed args to dit(x, mask, mu, t, spks, cond).
    // And captured output of input_embed.
    // In Rust flow.rs, InputEmbedding expects (x, cond, mu, spks).
    // Input shapes from debug file: [1, 80, 782].

    // x needs to be passed to input_embed.
    // In DiT::forward, x is passed directly.
    // InputEmbedding handles concatenation.

    // IMPORTANT: Check spks shape. Rust expects [1, 80] here?
    // loaded spks is [1, 80]. Correct.

    let x_in_rust = dit.input_embed.forward(&x, &cond, &mu, &spks)?;
    log_stats("Input Embed Rust", &x_in_rust)?;
    compare("Input Embed", &x_in_rust, py_input)?;

    // 3. Block 0
    // Inputs: x, t_emb, mask, chunk_mask, rope
    // Rope:
    // Python: self.rotary_embed.forward_from_seq_len(seq_len)
    let seq_len = x.dim(2)?;
    // Rust: rotary_embed only has freqs from init?
    // Wait, Rust `RotaryEmbedding` stores `freqs` in struct?
    // flow.rs: `pub struct RotaryEmbedding { freqs: Tensor }`.
    // It is initialized with max_seq_len (4096).
    // In forward, we need to pass it.
    let rope = Some(&dit.rotary_embed.freqs);

    // Mask
    // Rust DiT defaults streaming=false, so chunk_mask=None.
    // But mask logic:
    // Python pass mask to block.
    // Rust manual test needs to prepare mask if block expects it.
    // DiTBlock::forward takes &mask.
    // But wait, flow.rs DiT::forward calls build_chunk_mask.
    // If not streaming, chunk_mask is None.

    // Run Block 0
    let block0_rust = dit.transformer_blocks[0].forward(
        &x_in_rust,
        &t_emb_rust,
        &mask,
        None, // chunk_mask
        rope,
    )?;

    compare("Block 0", &block0_rust, py_block0)?;

    // 4. Intermediate blocks
    let num_blocks = dit.transformer_blocks.len();
    println!("    Processing {} transformer blocks...", num_blocks);
    let mut transformer_out_rust = block0_rust; // start with block 0 out
    for i in 1..num_blocks {
        transformer_out_rust = dit.transformer_blocks[i].forward(
            &transformer_out_rust,
            &t_emb_rust,
            &mask,
            None,
            rope,
        )?;
    }

    let py_transformer_out = layers.get("block7_out").unwrap();
    compare(
        "Transformer Stack (Full)",
        &transformer_out_rust,
        py_transformer_out,
    )?;

    // Norm Out
    let norm_out_rust = dit.norm_out.forward(&transformer_out_rust, &t_emb_rust)?;
    let py_norm_out = layers.get("norm_out_out").unwrap();
    compare(
        "Norm Out (Full Post-Modulation)",
        &norm_out_rust,
        py_norm_out,
    )?;

    Ok(())
}

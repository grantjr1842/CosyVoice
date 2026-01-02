// Minimal test to isolate AdaLayerNormZero and DiTBlock forward issues
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::{layer_norm_no_bias, linear, LayerNorm, Linear, VarBuilder, Module};
use std::collections::HashMap;

fn main() -> Result<()> {
    eprintln!("=== Minimal Flow Test ===");

    let device = Device::Cpu;

    // Create dummy weights using HashMap
    let mut weights: HashMap<String, Tensor> = HashMap::new();

    // AdaLayerNormZero weights (attn_norm)
    weights.insert("attn_norm.linear.weight".to_string(), Tensor::randn(0.0f32, 0.01, (6144, 1024), &device)?);
    weights.insert("attn_norm.linear.bias".to_string(), Tensor::zeros((6144,), DType::F32, &device)?);
    weights.insert("attn_norm.weight".to_string(), Tensor::ones((1024,), DType::F32, &device)?);

    // ff_norm weights
    weights.insert("ff_norm.weight".to_string(), Tensor::ones((1024,), DType::F32, &device)?);

    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);

    // Create components for DiTBlock
    let attn_norm_linear = linear(1024, 1024 * 6, vb.pp("attn_norm.linear"))?;
    let attn_norm_norm = layer_norm_no_bias(1024, 1e-6, vb.pp("attn_norm"))?;
    let ff_norm = layer_norm_no_bias(1024, 1e-6, vb.pp("ff_norm"))?;
    let silu = candle_nn::Activation::Silu;

    // Test inputs (mimicking CFG with batch=2)
    let batch = 2;
    let seq_len = 10;
    let dim = 1024;

    let x = Tensor::randn(0.0f32, 1.0, (batch, seq_len, dim), &device)?;
    let t_emb = Tensor::randn(0.0f32, 1.0, (batch, dim), &device)?;

    eprintln!("=== DiTBlock Forward Test ===");
    eprintln!("x shape: {:?}", x.shape());
    eprintln!("t_emb shape: {:?}", t_emb.shape());

    // Replicate DiTBlock::forward logic
    eprintln!("\n1. Computing emb = attn_norm.linear.forward(t_emb.apply(&silu))");
    let emb = attn_norm_linear.forward(&t_emb.apply(&silu)?)?;
    eprintln!("   emb shape: {:?}", emb.shape());

    eprintln!("\n2. Chunking emb into 6 parts");
    let chunks = emb.chunk(6, 1)?;
    eprintln!("   chunk shapes: {:?}", chunks.iter().map(|c| c.dims()).collect::<Vec<_>>());

    let shift_msa = chunks[0].unsqueeze(1)?;
    let scale_msa = chunks[1].unsqueeze(1)?;
    let gate_msa = chunks[2].unsqueeze(1)?;
    let shift_mlp = chunks[3].unsqueeze(1)?;
    let scale_mlp = chunks[4].unsqueeze(1)?;
    let gate_mlp = chunks[5].unsqueeze(1)?;

    eprintln!("\n3. After unsqueeze(1):");
    eprintln!("   shift_msa: {:?}", shift_msa.shape());
    eprintln!("   scale_msa: {:?}", scale_msa.shape());
    eprintln!("   gate_msa: {:?}", gate_msa.shape());

    // MSA branch
    eprintln!("\n4. MSA: res_msa = x.clone()");
    let res_msa = x.clone();
    eprintln!("   res_msa: {:?}", res_msa.shape());

    eprintln!("\n5. MSA: x_norm = attn_norm_norm.forward(x)");
    let x_norm = attn_norm_norm.forward(&x)?;
    eprintln!("   x_norm: {:?}", x_norm.shape());

    eprintln!("\n6. MSA: x_norm.broadcast_mul(scale_msa.affine(1.0, 1.0))");
    let scale_plus = scale_msa.affine(1.0, 1.0)?;
    eprintln!("   scale_plus: {:?}", scale_plus.shape());
    let x_norm = x_norm.broadcast_mul(&scale_plus)?;
    eprintln!("   x_norm after mul: {:?}", x_norm.shape());

    eprintln!("\n7. MSA: x_norm.broadcast_add(&shift_msa)");
    let x_norm = x_norm.broadcast_add(&shift_msa)?;
    eprintln!("   x_norm after add: {:?}", x_norm.shape());

    // Simulate attention output (just identity for now)
    let x_attn = x_norm.clone();
    eprintln!("\n8. MSA: x_attn (simulated): {:?}", x_attn.shape());

    eprintln!("\n9. MSA: mul_msa = x_attn.broadcast_mul(&gate_msa)");
    let mul_msa = x_attn.broadcast_mul(&gate_msa)?;
    eprintln!("   mul_msa: {:?}", mul_msa.shape());

    eprintln!("\n10. MSA: x = res_msa.broadcast_add(&mul_msa)");
    let x = res_msa.broadcast_add(&mul_msa)?;
    eprintln!("    x after MSA: {:?}", x.shape());

    // MLP branch
    eprintln!("\n11. MLP: res_mlp = x.clone()");
    let res_mlp = x.clone();

    eprintln!("\n12. MLP: x_norm = ff_norm.forward(&x)");
    let x_norm = ff_norm.forward(&x)?;
    eprintln!("    x_norm: {:?}", x_norm.shape());

    eprintln!("\n13. MLP: x_norm.broadcast_mul(scale_mlp.affine(1.0, 1.0))");
    let scale_plus = scale_mlp.affine(1.0, 1.0)?;
    let x_norm = x_norm.broadcast_mul(&scale_plus)?;
    eprintln!("    x_norm after mul: {:?}", x_norm.shape());

    eprintln!("\n14. MLP: x_norm.broadcast_add(&shift_mlp)");
    let x_norm = x_norm.broadcast_add(&shift_mlp)?;
    eprintln!("    x_norm after add: {:?}", x_norm.shape());

    // Simulate FF output
    let x_mlp = x_norm.clone();

    eprintln!("\n15. MLP: mul_mlp = x_mlp.broadcast_mul(&gate_mlp)");
    let mul_mlp = x_mlp.broadcast_mul(&gate_mlp)?;
    eprintln!("    mul_mlp: {:?}", mul_mlp.shape());

    eprintln!("\n16. MLP: x = res_mlp.broadcast_add(&mul_mlp)");
    let x = res_mlp.broadcast_add(&mul_mlp)?;
    eprintln!("    x after MLP: {:?}", x.shape());

    eprintln!("\n=== All DiTBlock operations succeeded! ===");

    Ok(())
}

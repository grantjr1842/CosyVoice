use candle_core::{DType, Device, Tensor};
use candle_nn::ops::sdpa;

fn main() -> anyhow::Result<()> {
    // 1. Check for CUDA
    let device = Device::new_cuda(0)?;
    println!("Device: {:?}", device);

    // 2. Setup Q, K, V (Batch=1, Heads=4, Seq=10, Dim=64) - typical sizes
    let b = 1;
    let h = 4;
    let seq = 10;
    let dim = 64;

    // Flash Attention requires F16 or BF16
    let dtype = DType::F16;

    let q = Tensor::randn(0.0f32, 1.0f32, (b, h, seq, dim), &device)?.to_dtype(dtype)?;
    let k = Tensor::randn(0.0f32, 1.0f32, (b, h, seq, dim), &device)?.to_dtype(dtype)?;
    let v = Tensor::randn(0.0f32, 1.0f32, (b, h, seq, dim), &device)?.to_dtype(dtype)?;

    println!("Tensors created with dtype {:?}", dtype);

    // 3. Call SDPA
    println!("Calling sdpa...");
    let scale = 1.0 / (dim as f32).sqrt();

    let start = std::time::Instant::now();
    match sdpa(&q, &k, &v, None, false, scale, 0.0) {
        Ok(out) => {
            println!("SDPA success!");
            println!("Output shape: {:?}", out.shape());
        },
        Err(e) => {
            eprintln!("SDPA failed: {:?}", e);
        }
    }
    println!("Time: {:?}", start.elapsed());

    Ok(())
}

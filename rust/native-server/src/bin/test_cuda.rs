use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    if !candle_core::utils::cuda_is_available() {
        println!("CUDA not available!");
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    println!("Device: {:?}", device);

    let t = Tensor::new(&[1.0f32, 2.0], &device)?;
    println!("Tensor: {:?}", t);

    let t2 = (t + 1.0)?;
    println!("Tensor + 1.0: {:?}", t2);

    println!("SUCCESS: Basic CUDA op works");
    Ok(())
}

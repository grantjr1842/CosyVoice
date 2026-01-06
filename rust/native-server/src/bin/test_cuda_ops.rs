use anyhow::Result;
use candle_core::{Device, DType, Tensor};

fn main() -> Result<()> {
    if !candle_core::utils::cuda_is_available() {
        println!("CUDA not available!");
        return Ok(());
    }

    let device = Device::new_cuda(0)?;
    println!("Device: {:?}", device);

    // Test CPU Cast -> GPU Move
    println!("Testing CPU Cast -> GPU Move...");

    // Create I32 tensor on CPU
    let t_i32 = Tensor::new(&[1i32, 2], &Device::Cpu)?;
    println!("I32 CPU created");

    // Cast to U32 on CPU
    let t_u32_cpu = t_i32.to_dtype(DType::U32)?;
    println!("Cast I32->U32 on CPU passed");

    // Move to GPU
    let t_u32_gpu = t_u32_cpu.to_device(&device)?;
    println!("Move U32 to GPU passed");

    // Test I32 -> F32 on CPU -> GPU
    let t_f32_cpu = t_i32.to_dtype(DType::F32)?;
    let t_f32_gpu = t_f32_cpu.to_device(&device)?;
    println!("Move F32 to GPU passed");

    println!("SUCCESS: All ops passed");
    Ok(())
}

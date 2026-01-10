use candle_core::{Device, Result, Tensor};
use candle_nn::{Conv1dConfig, Module};

fn main() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::zeros((1, 1, 24975), candle_core::DType::F32, &device)?;
    let weight = Tensor::zeros((1, 1, 30), candle_core::DType::F32, &device)?;
    let bias = Tensor::zeros(1, candle_core::DType::F32, &device)?;

    let conv = candle_nn::Conv1d::new(weight, Some(bias), Conv1dConfig {
        stride: 15,
        padding: 0,
        dilation: 1,
        groups: 1,
        ..Default::default()
    });

    let y = conv.forward(&x)?;
    println!("Input shape: {:?}", x.shape());
    println!("Output shape: {:?}", y.shape());
    Ok(())
}

use crate::flow::FlowEstimator;
use candle_core::{Device, Result, Tensor};
use ndarray::ArrayD;
use ort::{inputs, session::Session, value::Value};
use std::sync::Mutex;

pub struct DiTOnnx {
    session: Mutex<Session>,
    device: Device,
}

impl DiTOnnx {
    pub fn new(session: Session, device: Device) -> Self {
        Self {
            session: Mutex::new(session),
            device,
        }
    }
}

impl FlowEstimator for DiTOnnx {
    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
    ) -> Result<Tensor> {
        // Helper to convert Candle Tensor to ArrayD for ORT
        fn to_array(t: &Tensor) -> Result<ArrayD<f32>> {
            let dims: Vec<usize> = t.dims().to_vec();
            let vec = t.flatten_all()?.to_vec1::<f32>()?;
            ArrayD::from_shape_vec(dims, vec).map_err(|e| candle_core::Error::Msg(e.to_string()))
        }

        // Inputs
        let x_val = Value::from_array(to_array(&x.to_dtype(candle_core::DType::F32)?.contiguous()?)?).map_err(|e| candle_core::Error::Msg(format!("ORT Value error: {}", e)))?;
        let mu_val = Value::from_array(to_array(&mu.to_dtype(candle_core::DType::F32)?.contiguous()?)?).map_err(|e| candle_core::Error::Msg(format!("ORT Value error: {}", e)))?;
        let t_val = Value::from_array(to_array(&t.to_dtype(candle_core::DType::F32)?.contiguous()?)?).map_err(|e| candle_core::Error::Msg(format!("ORT Value error: {}", e)))?;
        let spks_val = Value::from_array(to_array(&spks.to_dtype(candle_core::DType::F32)?.contiguous()?)?).map_err(|e| candle_core::Error::Msg(format!("ORT Value error: {}", e)))?;
        let cond_val = Value::from_array(to_array(&cond.to_dtype(candle_core::DType::F32)?.contiguous()?)?).map_err(|e| candle_core::Error::Msg(format!("ORT Value error: {}", e)))?;

        // Ensure mask is [B, 1, T]
        let mask = if mask.rank() == 2 {
            mask.unsqueeze(1)?
        } else {
            mask.clone()
        };
        let mask_val = Value::from_array(to_array(&mask.to_dtype(candle_core::DType::F32)?.contiguous()?)?)
            .map_err(|e| candle_core::Error::Msg(format!("ORT Value error: {}", e)))?;

        let inputs = inputs![
            "x" => x_val,
            "mask" => mask_val,
            "mu" => mu_val,
            "t" => t_val,
            "spks" => spks_val,
            "cond" => cond_val
        ];

        let mut session = self
            .session
            .lock()
            .map_err(|e| candle_core::Error::Msg(format!("Mutex poison error: {}", e)))?;
        let outputs = session
            .run(inputs)
            .map_err(|e| candle_core::Error::Msg(format!("ORT Run error: {}", e)))?;

        let (shape, data) = outputs["estimator_out"]
            .try_extract_tensor::<f32>()
            .map_err(|e| candle_core::Error::Msg(format!("ORT Extract error: {}", e)))?;

        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let data_vec = data.to_vec();

        Tensor::from_vec(data_vec, dims, &self.device)
    }
}

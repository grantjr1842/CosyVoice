pub mod qwen;
pub mod hift;
pub mod utils;
pub mod flow;

use pyo3::prelude::*;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::PathBuf;
use crate::flow::{ConditionalCFM, DiT, FlowConfig};
use crate::hift::{HiFTGenerator, HiFTConfig};
use numpy::{PyArrayMethods, PyArray, PyUntypedArrayMethods, PyArrayDyn};

#[pyclass]
struct FlowRust {
    model: ConditionalCFM,
    device: Device,
}

unsafe impl Send for FlowRust {}
unsafe impl Sync for FlowRust {}

#[pymethods]
impl FlowRust {
    #[new]
    fn new(model_dir: String) -> PyResult<Self> {
        let device = if let Ok(d) = Device::cuda_if_available(0) {
            d
        } else {
            Device::Cpu
        };

        let model_path = PathBuf::from(&model_dir);
        let config_path = model_path.join("config.json");
        let safetensors_path = model_path.join("model.safetensors");

        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!("Config not found: {}", e)))?;
        let flow_config: FlowConfig = serde_json::from_str(&config_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Invalid config: {}", e)))?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[safetensors_path], DType::F32, &device) }
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dit = DiT::new(vb.pp("flow"), &flow_config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Match ConditionalCFM::new signature: (vb, estimator, ode_type, sigma, cfg_strength)
        let model = ConditionalCFM::new(vb.pp("flow"), dit, "cosine".to_string(), 0.0, 0.7)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(FlowRust { model, device })
    }

    #[pyo3(signature = (mu, mask, n_timesteps, temperature, spks=None, cond=None))]
    fn inference(
        &self,
        py: Python<'_>,
        mu: Py<PyArrayDyn<f32>>,
        mask: Py<PyArrayDyn<f32>>,
        n_timesteps: usize,
        temperature: f64,
        spks: Option<Py<PyArrayDyn<f32>>>,
        cond: Option<Py<PyArrayDyn<f32>>>,
    ) -> PyResult<PyObject> {
        let mu_tensor = py_to_candle(py, &mu, &self.device)?;
        let mask_tensor = py_to_candle(py, &mask, &self.device)?;

        let spks_tensor = if let Some(s) = spks {
            Some(py_to_candle(py, &s, &self.device)?)
        } else {
            None
        };

        let cond_tensor = if let Some(c) = cond {
            Some(py_to_candle(py, &c, &self.device)?)
        } else {
            None
        };

        let output = self.model.forward(
            &mu_tensor,
            &mask_tensor,
            n_timesteps,
            temperature,
            spks_tensor.as_ref(),
            cond_tensor.as_ref()
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Convert output tensor [B, D, T] to numpy array
        let output_vec = output.flatten_all()
            .map_err(|e: candle_core::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e: candle_core::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dims = output.dims();
        let py_array = PyArray::from_vec(py, output_vec).reshape(dims).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(py_array.into_any().unbind())
    }
}

fn py_to_candle(py: Python<'_>, array: &Py<PyArrayDyn<f32>>, device: &Device) -> PyResult<Tensor> {
    let array = array.bind(py);
    let shape = array.shape().to_vec();
    let slice = unsafe { array.as_slice().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))? };

    Tensor::from_slice(slice, shape, device).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyclass]
struct Qwen2Rust {
    model: qwen::ModelForCausalLM,
    device: Device,
    dtype: DType,
}

#[pymethods]
impl Qwen2Rust {
    #[new]
    fn new(model_dir: String) -> PyResult<Self> {
        let device = if let Ok(d) = Device::cuda_if_available(0) {
            d
        } else {
            Device::Cpu
        };
        let dtype = DType::F16;

        let model_path = PathBuf::from(model_dir);
        let config_filename = model_path.join("config.json");
        let weights_filename = model_path.join("model.safetensors");

        let config: qwen::Config = {
            let config_data = std::fs::read_to_string(&config_filename)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(e.to_string()))?;
            serde_json::from_str(&config_data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        };

        let vb = unsafe {
             VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device)
                 .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        };

        let model = qwen::ModelForCausalLM::new(&config, vb)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Qwen2Rust { model, device, dtype })
    }

    fn forward(&mut self, input_ids: Vec<u32>, input_ids_shape: Vec<usize>) -> PyResult<Vec<f32>> {
        let batch_size = input_ids_shape[0];
        let seq_len = input_ids_shape[1];

        let input_tensor = Tensor::from_vec(input_ids, (batch_size, seq_len), &self.device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let output = self.model.base_model.forward(&input_tensor, 0, None)
             .map_err(|e: candle_core::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let output_vec = output.flatten_all()
            .map_err(|e: candle_core::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .to_dtype(DType::F32).map_err(|e: candle_core::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e: candle_core::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(output_vec)
    }
}

#[pyclass]
struct HiFTRust {
    model: HiFTGenerator,
    device: Device,
}

#[pymethods]
impl HiFTRust {
    #[new]
    fn new(model_dir: String) -> PyResult<Self> {
        let device = if let Ok(d) = Device::cuda_if_available(0) {
            d
        } else {
            Device::Cpu
        };

        let model_path = PathBuf::from(model_dir);
        let weights_filename = model_path.join("model.safetensors");
        let config = HiFTConfig::default();

        let vb = unsafe {
             VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)
                 .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        };

        let model = HiFTGenerator::new(vb.pp("hift"), &config)
            .map_err(|e: candle_core::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { model, device })
    }

    fn inference(&self, mel_flat: Vec<f32>, batch: usize, dim: usize, length: usize) -> PyResult<Vec<f32>> {
        let mel = Tensor::from_vec(mel_flat, (batch, dim, length), &self.device)
            .map_err(|e: candle_core::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let audio = self.model.forward(&mel)
            .map_err(|e: candle_core::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let audio_vec = audio.flatten_all()
            .map_err(|e: candle_core::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e: candle_core::Error| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(audio_vec)
    }
}

#[pymodule]
fn cosyvoice_rust_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Qwen2Rust>()?;
    m.add_class::<HiFTRust>()?;
    m.add_class::<FlowRust>()?;
    Ok(())
}

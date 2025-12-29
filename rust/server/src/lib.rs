pub mod flow;
pub mod hift;
pub mod qwen;
pub mod utils;

use crate::flow::{ConditionalCFM, DiT, FlowConfig};
use crate::hift::{HiFTConfig, HiFTGenerator};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use numpy::{PyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use std::path::PathBuf;

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
        let flow_path = model_path.join("flow.safetensors");
        let flow_pt_path = model_path.join("flow.pt");

        let config_str = std::fs::read_to_string(&config_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!(
                "Config not found: {}",
                e
            ))
        })?;
        let flow_config: FlowConfig = serde_json::from_str(&config_str).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Invalid config: {}", e))
        })?;

        let vb = if flow_path.exists() && flow_path.is_file() {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[&flow_path], DType::F32, &device).map_err(
                    |e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to load flow safetensors from {:?}: {}",
                            flow_path, e
                        ))
                    },
                )?
            }
        } else {
            VarBuilder::from_pth(&flow_pt_path, DType::F32, &device).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to load flow pth from {:?}: {}",
                    flow_pt_path, e
                ))
            })?
        };

        // If weights are under 'flow.' prefix (from safetensors conversion) or 'decoder.estimator.' (original flow.pt)
        let vb_final = if vb.contains_tensor("flow.input_embed.proj.weight") {
            vb.pp("flow")
        } else if vb.contains_tensor("decoder.estimator.input_embed.proj.weight") {
            vb.pp("decoder.estimator")
        } else {
            vb
        };

        let dit = DiT::new(vb_final.clone(), &flow_config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let model = ConditionalCFM::new(vb_final, dit, "cosine".to_string(), 0.0, 0.7)
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

        let output = self
            .model
            .forward(
                &mu_tensor,
                &mask_tensor,
                n_timesteps,
                temperature,
                spks_tensor.as_ref(),
                cond_tensor.as_ref(),
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let output_vec = output
            .flatten_all()
            .map_err(|e: candle_core::Error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?
            .to_vec1::<f32>()
            .map_err(|e: candle_core::Error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?;

        let dims = output.dims();
        let py_array = PyArray::from_vec(py, output_vec)
            .reshape(dims)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(py_array.into_any().unbind())
    }
}

fn py_to_candle(py: Python<'_>, array: &Py<PyArrayDyn<f32>>, device: &Device) -> PyResult<Tensor> {
    let array = array.bind(py);
    let shape = array.shape().to_vec();
    let slice = unsafe {
        array
            .as_slice()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
    };

    Tensor::from_slice(slice, shape, device)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
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
        let weights_filename = model_path.join("llm.safetensors");
        let weights_filename_alt = model_path.join("model.safetensors");
        let weights_filename_parent = model_path.parent().map(|p| p.join("llm.safetensors"));
        let pt_filename = model_path.join("llm.pt");

        let config: qwen::Config = {
            let config_data = std::fs::read_to_string(&config_filename).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!(
                    "Config not found at {:?}: {}",
                    config_filename, e
                ))
            })?;
            serde_json::from_str(&config_data)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        };

        let vb = if weights_filename.exists() && weights_filename.is_file() {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[&weights_filename], dtype, &device).map_err(
                    |e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to load llm safetensors from {:?}: {}",
                            weights_filename, e
                        ))
                    },
                )?
            }
        } else if weights_filename_alt.exists() && weights_filename_alt.is_file() {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[&weights_filename_alt], dtype, &device)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to load llm alt safetensors from {:?}: {}",
                            weights_filename_alt, e
                        ))
                    })?
            }
        } else if let Some(ref p) = weights_filename_parent {
            if p.exists() && p.is_file() {
                unsafe {
                    VarBuilder::from_mmaped_safetensors(&[p], dtype, &device).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to load llm parent safetensors from {:?}: {}",
                            p, e
                        ))
                    })?
                }
            } else {
                VarBuilder::from_pth(&pt_filename, dtype, &device)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Could not find weights (tried llm.safetensors, model.safetensors, and llm.pt): {}", e)))?
            }
        } else {
            VarBuilder::from_pth(&pt_filename, dtype, &device)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Could not find weights (tried llm.safetensors, model.safetensors, and llm.pt): {}", e)))?
        };

        // DEBUG: List some keys
        if vb.contains_tensor("llm.model.model.embed_tokens.weight") {
            eprintln!("DEBUG: Found llm.model.model.embed_tokens.weight");
        } else {
            eprintln!("DEBUG: DID NOT find llm.model.model.embed_tokens.weight");
        }

        // File has: llm.model.model.embed_tokens.weight
        // We want: prefix "llm.model" → after Model::new adds "model" → "llm.model.model."
        // Then embed_tokens gives: "llm.model.model.embed_tokens.weight" ✓

        let vb_final = if vb.contains_tensor("llm.model.model.embed_tokens.weight") {
            eprintln!("DEBUG: Using prefix 'llm.model' - after Model::new will be 'llm.model.model.'");
            vb.pp("llm.model")
        } else if vb.contains_tensor("model.model.embed_tokens.weight") {
            eprintln!("DEBUG: Using prefix 'model'");
            vb.pp("model")
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            eprintln!("DEBUG: Using no prefix");
            vb
        } else if vb.contains_tensor("llm.model.lm_head.weight") {
            eprintln!("DEBUG: Found llm.model.lm_head.weight, using prefix 'llm.model'");
            vb.pp("llm.model")
        } else {
            eprintln!("DEBUG: No known prefix matched, using raw vb");
            vb
        };

        let model = qwen::ModelForCausalLM::new(&config, vb_final)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Qwen2Rust {
            model,
            device,
            dtype,
        })
    }

    fn forward(&mut self, input_ids: Vec<u32>, input_ids_shape: Vec<usize>) -> PyResult<Vec<f32>> {
        let batch_size = input_ids_shape[0];
        let seq_len = input_ids_shape[1];

        let input_tensor = Tensor::from_vec(input_ids, (batch_size, seq_len), &self.device)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let output = self
            .model
            .base_model
            .forward(&input_tensor, 0, None)
            .map_err(|e: candle_core::Error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?;

        let output_vec = output
            .flatten_all()
            .map_err(|e: candle_core::Error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?
            .to_dtype(DType::F32)
            .map_err(|e: candle_core::Error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?
            .to_vec1::<f32>()
            .map_err(|e: candle_core::Error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?;

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
    fn new(model_path_str: String) -> PyResult<Self> {
        let device = if let Ok(d) = Device::cuda_if_available(0) {
            d
        } else {
            Device::Cpu
        };

        let weights_filename = PathBuf::from(&model_path_str);

        let vb = if weights_filename.exists() && weights_filename.is_file() {
            if weights_filename
                .extension()
                .map_or(false, |ext| ext == "safetensors")
            {
                unsafe {
                    VarBuilder::from_mmaped_safetensors(&[&weights_filename], DType::F32, &device)
                        .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to load hift safetensors from {:?}: {}",
                            weights_filename, e
                        ))
                    })?
                }
            } else {
                VarBuilder::from_pth(&weights_filename, DType::F32, &device).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to load pth from {:?}: {}",
                        weights_filename, e
                    ))
                })?
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                format!("Weight file not found: {}", model_path_str),
            ));
        };

        // Determine n_fft from weights if possible
        let n_fft = if vb.contains_tensor("conv_post.weight") {
            let shape = vb
                .get_with_hints((18, 512, 7), "conv_post.weight", Default::default())
                .map(|t| t.shape().clone())
                .unwrap_or_else(|_| candle_core::Shape::from_dims(&[18, 512, 7]));
            let out_channels = shape.dims().get(0).cloned().unwrap_or(18);
            if out_channels > 2 {
                out_channels - 2
            } else {
                16
            }
        } else if vb.contains_tensor("hift.conv_post.weight") {
            let shape = vb
                .get_with_hints((18, 512, 7), "hift.conv_post.weight", Default::default())
                .map(|t| t.shape().clone())
                .unwrap_or_else(|_| candle_core::Shape::from_dims(&[18, 512, 7]));
            let out_channels = shape.dims().get(0).cloned().unwrap_or(18);
            if out_channels > 2 {
                out_channels - 2
            } else {
                16
            }
        } else {
            16
        };

        let config = HiFTConfig::new(n_fft);

        // Use weight mapping based on prefixes present in original .pt files
        let vb_final = if vb.contains_tensor("f0_predictor.condnet.0.bias")
            || vb.contains_tensor("f0_predictor.condnet.0.parametrizations.weight.original0")
        {
            vb
        } else if vb.contains_tensor("hift.f0_predictor.condnet.0.bias")
            || vb.contains_tensor("hift.f0_predictor.condnet.0.parametrizations.weight.original0")
        {
            vb.pp("hift")
        } else {
            vb
        };

        let model = HiFTGenerator::new(vb_final, &config).map_err(|e: candle_core::Error| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        Ok(Self { model, device })
    }

    fn inference(
        &self,
        mel_flat: Vec<f32>,
        batch: usize,
        dim: usize,
        length: usize,
    ) -> PyResult<Vec<f32>> {
        let mel = Tensor::from_vec(mel_flat, (batch, dim, length), &self.device).map_err(
            |e: candle_core::Error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            },
        )?;

        let audio = self.model.forward(&mel).map_err(|e: candle_core::Error| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
        })?;

        let audio_vec = audio
            .flatten_all()
            .map_err(|e: candle_core::Error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?
            .to_vec1::<f32>()
            .map_err(|e: candle_core::Error| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?;

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

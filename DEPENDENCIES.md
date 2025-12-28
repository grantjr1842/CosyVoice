# Environment Dependencies

This document lists the environment packages and system tools required to build and run the CosyVoice project.

## System Dependencies (Ubuntu/Debian)

These packages are required for the installation of Python extensions, audio processing, and Rust compilation.

```bash
sudo apt-get update
sudo apt-get install -y \
    git \
    git-lfs \
    build-essential \
    curl \
    wget \
    unzip \
    ffmpeg \
    sox \
    libsox-dev
```

- **`git`, `git-lfs`**: For version control and downloading large model files.
- **`build-essential`**: Contains GCC, Make, and other tools needed for compiling C/C++ extensions and Rust dependencies.
- **`ffmpeg`**: Required for audio processing and format conversion.
- **`sox`, `libsox-dev`**: Required for audio manipulation (often used by `torchaudio` or other audio libraries).

## Package Managers

### Pixi (Python Environment)
This project uses **Pixi** for Python package management.
- **Installation**:
  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```
- **Usage**:
  - `pixi install`: Installs Python, PyTorch, CUDA, and all Python dependencies defined in `pyproject.toml`.
  - `pixi run <task>`: Runs project tasks (e.g., `webui`, `download-model`).
  - `pixi shell`: Enters the managed shell.

### Rust (Cargo)
The project includes a Rust-based TTS server.
- **Installation**:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Usage**:
  - `cargo build --release`: Compiles the Rust server.

# Demo BERT ONNX pipeline written in rust

This demo showcase the use of onnxruntime-rs with a GPU on CUDA 11 to run Bert in a data pipeline with Rust.

## Requirement

- Linux x86_64
- NVIDIA GPU with CUDA 11 (Not sure if CUDA 10 works)
- Rust (obviously)

## training and converting to ONNX (Python)

## Installation

Download the data https://www.kaggle.com/START-UMD/gtd -> `train_October_9_2012.csv` and put it in the folder training.

```bash
cd training
pip install -r requirements.txt
python pytorch-model.py
```

I have not done any parameter search to find an optimal model as it is not the point of this repo.

I have limited the training to 500 rows to iterate faster.

## Inference using Python

```bash
cd src
python python_alternative.py
```

## Inference using Rust

## Installation

```bash
export ORT_USE_CUDA=1
cargo build --release
```

## Run

```bash
cargo run --release
```

or

```bash
export LD_LIBRARY_PATH=path/to/onnxruntime-linux-x64-gpu-1.8.0/lib:${LD_LIBRARY_PATH}
./target/release/machine-learning
```

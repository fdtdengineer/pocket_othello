# Local DQN Training with Conda

This guide covers local teacher-data generation, imitation learning, reinforcement learning, ONNX export, and browser testing.

Run all commands from the repository root.

## 1. Create the Conda environment

```bash
git clone https://github.com/fdtdengineer/pocket_othello.git
cd pocket_othello

conda create -n pocket_othello_dqn python=3.12 -y
conda activate pocket_othello_dqn
conda install -c conda-forge nodejs=22 -y
```

Verify the tools:

```bash
python --version
node --version
npm --version
```

## 2. Install Python dependencies

For CPU training and ONNX export:

```bash
python -m pip install --upgrade pip
python -m pip install -e "engine/python[all]"
```

For an NVIDIA GPU, install the appropriate CUDA-enabled PyTorch build first, then install the project without replacing that PyTorch build:

```bash
python -m pip install -e engine/python --no-deps
python -m pip install numpy onnx onnxruntime
```

Check whether PyTorch sees the GPU:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## 3. Generate Hard-CPU teacher data

A practical medium-size run:

```bash
npm run generate:teacher -- --games 250 --max-examples 12000 --time-ms 20 --max-depth 5 --exploration 0.25 --seed 20260720 --output engine/python/data/teacher.jsonl
```

Outputs:

```text
engine/python/data/teacher.jsonl
engine/python/data/teacher.meta.json
```

A quick smoke run:

```bash
npm run generate:teacher -- --games 10 --max-examples 500 --time-ms 20 --output engine/python/data/teacher.jsonl
```

## 4. Run imitation pretraining

```bash
npm run train:imitation -- --data engine/python/data/teacher.jsonl --output engine/python/checkpoints/imitation.pt --epochs 12 --batch-size 128 --augmentation random --seed 20260720 --device auto
```

Use `--device cuda` to require an NVIDIA GPU. Reduce `--batch-size` to `64` or `32` if GPU memory is insufficient.

A longer run can use, for example:

```bash
npm run train:imitation -- --data engine/python/data/teacher.jsonl --output engine/python/checkpoints/imitation.pt --epochs 30 --batch-size 256 --learning-rate 0.001 --augmentation random --device cuda
```

## 5. Run mixed-opponent reinforcement training

```bash
npm run train:reinforcement -- --init-checkpoint engine/python/checkpoints/imitation.pt --output engine/python/checkpoints/reinforcement.pt --episodes 1500 --batch-size 128 --replay-capacity 100000 --warmup-transitions 1500 --updates-per-episode 6 --epsilon-start 0.25 --epsilon-end 0.04 --epsilon-decay-episodes 1200 --self-play-weight 0.45 --heuristic-weight 0.40 --random-weight 0.15 --checkpoint-every 500 --seed 20260720 --device auto
```

A quick smoke run:

```bash
npm run train:reinforcement -- --init-checkpoint engine/python/checkpoints/imitation.pt --output engine/python/checkpoints/reinforcement_test.pt --episodes 50 --batch-size 64 --replay-capacity 5000 --warmup-transitions 256 --updates-per-episode 2 --checkpoint-every 50 --device auto
```

The trainer mixes self-play, a fixed heuristic opponent, and a random opponent.

## 6. Export the model to ONNX

Export the reinforcement checkpoint:

```bash
npm run export:onnx -- --checkpoint engine/python/checkpoints/reinforcement.pt --output engine/models/othello_dqn.onnx
```

The exporter writes:

```text
engine/models/othello_dqn.onnx
engine/models/othello_dqn.json
```

To deploy the imitation checkpoint instead, change only the checkpoint path:

```bash
npm run export:onnx -- --checkpoint engine/python/checkpoints/imitation.pt --output engine/models/othello_dqn.onnx
```

## 7. Test the browser build locally

```bash
npm install
npm run prepare:web
python -m http.server 8000
```

Open:

```text
http://localhost:8000
http://localhost:8000/benchmark.html
```

## Windows multiline commands

The one-line commands above work in Anaconda Prompt, Command Prompt, and PowerShell.

For multiline commands in Command Prompt or Anaconda Prompt, use `^` at the end of each continued line. In PowerShell, use the backtick character instead.

Command Prompt example:

```bat
npm run train:imitation -- ^
  --data engine/python/data/teacher.jsonl ^
  --output engine/python/checkpoints/imitation.pt ^
  --epochs 12 ^
  --batch-size 128 ^
  --device auto
```

PowerShell example:

```powershell
npm run train:imitation -- `
  --data engine/python/data/teacher.jsonl `
  --output engine/python/checkpoints/imitation.pt `
  --epochs 12 `
  --batch-size 128 `
  --device auto
```

## Recommended first run

Before starting a longer experiment, verify the full pipeline using:

- 10 teacher games or 500 examples
- 2 imitation epochs
- 50 reinforcement episodes
- ONNX export
- the browser benchmark page

Generated teacher data and PyTorch checkpoints are ignored by Git. The final browser ONNX file can be force-added when it has passed evaluation.

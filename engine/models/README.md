# Browser DQN models

Export the trained network into this directory:

```bash
npm run export:onnx -- \
  --checkpoint engine/python/checkpoints/reinforcement.pt \
  --output engine/models/othello_dqn.onnx
```

This produces:

```text
engine/models/othello_dqn.onnx
engine/models/othello_dqn.json
```

The browser DQN option loads these exact paths. The ONNX file and metadata may be committed when a trained checkpoint has passed evaluation. Until they are present, selecting DQN safely falls back to the existing Hard CPU.

Only inference artifacts belong here. PyTorch checkpoints, replay buffers, optimizer state, and training data remain ignored under `engine/python/`.

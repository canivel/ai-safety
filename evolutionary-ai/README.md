# AI Safety Research - Environment Setup

This project uses `uv` for lightning-fast Python package management and `gymnasium` for reinforcement learning environments.

## Quick Start: Environment Setup

### 1. Install `uv`
If you don't have `uv` installed, run the following in PowerShell:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Initialize the Virtual Environment
From the project root, create a localized virtual environment:
```bash
uv venv
```

### 3. Install Dependencies
Install `gymnasium` with rendering support and the Jupyter kernel:
```bash
uv pip install gymnasium[classic-control] pygame ipykernel
```

### 4. Register Jupyter Kernel
To use this environment inside `.ipynb` notebooks, register it as a kernel:
```bash
.\.venv\Scripts\python -m ipykernel install --user --name rl-safety --display-name "Python (RL Safety)"
```

---

## Using the Environment in Notebooks

1. Open any notebook (e.g., `evolutionary-ai/RandomSearch.ipynb`).
2. Click the **Select Kernel** button in the top-right corner.
3. Select **"Python (RL Safety)"**.

### Test Your Installation
Run this code snippet in a notebook cell to verify Gymnasium is working:

```python
import gymnasium as gym

# Create the environment
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(100):
   action = env.action_space.sample() 
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()
print("Environment successfully initialized!")
```

## Useful Commands

| Action | Command |
| :--- | :--- |
| **Activate Environment** | `.\.venv\Scripts\activate` |
| **Install New Package** | `uv pip install <package-name>` |
| **Update Packages** | `uv pip install --upgrade gymnasium` |
| **List Installed** | `uv pip list` |

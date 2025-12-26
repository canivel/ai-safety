# Roadmap: From Evolutionary Strategies to Safe World Models

This roadmap outlines a progression from fundamental gradient-free methods to state-of-the-art Reinforcement Learning (RL) techniques used in Large Language Model (LLM) alignment and AI Safety research.

## Phase 1: The Evolutionary Foundation (Gradient-Free)
**Goal**: Understand optimization without explicit gradients, aiming for robustness and diversity.
**Relevance to Safety**: ES is crucial for *Red Teaming*—finding adversarial examples or "jailbreaks" where gradient-based methods might get stuck in local optima.

### 1. Evolutionary Strategies (ES) & Genetic Algorithms
- **Concepts**: Population-based search, Mutation/Crossover, Fitness evaluation, Covariance Matrix Adaptation (CMA-ES).
- **Implementation Tasks**:
    - [ ] Implement simple Genetic Algorithm (GA) for a classic control problem (CartPole/LunarLander).
    - [ ] Implement Simple ES (OpenAI ES) to train a small neural network.
    - [ ] **Safety Angle**: Use ES to find inputs that cause a pre-trained agent to fail (Adversarial Attack).

## Phase 2: Policy Gradients & Stability (The RLHF Workhorse)
**Goal**: Master the standard algorithms used for Reinforcement Learning from Human Feedback (RLHF).

### 2. Proximal Policy Optimization (PPO)
- **Concepts**: Trust regions, Clipped surrogate objective, Actor-Critic architecture, GAE (Generalized Advantage Estimation).
- **Implementation Tasks**:
    - [ ] Solve a continuous control environment (MuJoCo/BipedalWalker) with PPO.
    - [ ] **LLM Integration**: Fine-tune a small language model (e.g., GPT-2 or TinyLlama) to output positive sentiment using a frozen Sentiment Reward Model.
    - [ ] **Safety Angle**: Observe "Reward Hacking" (e.g., model outputting repetitive high-reward gibberish) and try to mitigate it with KL-divergence penalties.

## Phase 3: Preference Optimization (Modern Alignment)
**Goal**: Move away from complex Reward Model training towards direct policy optimization.

### 3. Direct Preference Optimization (DPO)
- **Concepts**: Implicit reward formulation, Bradley-Terry model, Reference model regularization.
- **Implementation Tasks**:
    - [ ] Implement the DPO loss function manually.
    - [ ] Train a model on a preference dataset (e.g., Anthropic HH-RLHF subset).
    - [ ] **Comparison**: Compare stability and computational cost of DPO vs. PPO.

## Phase 4: Advanced Group Dynamics & Reasoning
**Goal**: Optimize for complex reasoning chains and handling distribution shifts, key for "Chain of Thought" safety.

### 4. Group Relative Policy Optimization (GRPO)
- **Concepts**: Sampling groups of outputs for the same prompt, determining relative advantages within the group rather than absolute rewards. Used in models like DeepSeekMath to improve reasoning.
- **Implementation Tasks**:
    - [ ] Implement "Group Sampling" during training.
    - [ ] Train on a math or logic dataset where answers can be verified (Binary reward: Correct/Incorrect), assessing if the group mean acts as a good baseline.

### 5. Multi-Group Relative Policy Optimization (MGRPO) & Beyond
- **Concepts**: Handling multiple conflicting objectives (e.g., Safety vs. Helpfulness) by creating groups that optimize for different pareto frontiers.
- **Implementation Tasks**:
    - [ ] Setup a multi-objective scenario (e.g., "be funny but not offensive").
    - [ ] **Safety Test**: Measure if MGRPO preserves safety constraints better than standard RLHF when pushed for performance.

## Phase 5: AI Safety & World Models
**Goal**: Build systems that "think" before they act and are robust to catastrophic failure.

### 6. World Models & Model-Based RL
- **Concepts**: Learning a model of the environment ($P(s'|s,a)$) to simulate trajectories.
- **Relevance**: "Dreaming" potential futures allows an agent to foresee unsafe outcomes without executing them in the real world.
- **Implementation Tasks**:
    - [ ] Train a VAE + RNN (or Transformer) to predict next frames in a simple game (like in the "World Models" paper).
    - [ ] Train a controller purely inside the "dream" (Plan-vs-Policy).

### 7. Safe RL & Constrained Optimization
- **Concepts**: Constrained MDPs (CMDPs), Lagrangian methods for safety constraints.
- **Implementation Tasks**:
    - [ ] **Safety Gym**: Train an agent that must navigate a maze while avoiding "hazards" (constraints).
    - [ ] Compare "Reward Shaping" (negative reward for hazards) vs. "Constrained RL" (hard limits).

## Next Steps
Start with **Phase 1** in the `evolutionary-ai` folder.
1. Create a `simple_es.py` to solve CartPole.
2. Visualize the population spread to understand exploration.

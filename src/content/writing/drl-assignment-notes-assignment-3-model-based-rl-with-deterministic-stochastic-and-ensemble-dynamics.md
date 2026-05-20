---
title: "Assignment 3: Model-Based RL with Deterministic, Stochastic, and Ensemble Dynamics"
date: 2026-02-25
type: note
tags:
  - RL
listed: false
---


The assignment starts with the simplest version of model-based RL, then gradually add better ways to represent uncertainty.


- [Deterministic Dynamics Model](#deterministic-dynamics-model)
  - [Main idea](#main-idea)
  - [Implementation](#implementation)
  - [Conceptual question](#conceptual-question)
- [CEM Planning with the Deterministic Model](#cem-planning-with-the-deterministic-model)
  - [Main idea](#main-idea-1)
    - [Pseudocode for CEM](#pseudocode-for-cem)
  - [Implementation](#implementation-1)
  - [Conceptual question](#conceptual-question-1)
- [Stochastic Dynamics Model](#stochastic-dynamics-model)
  - [Main idea](#main-idea-2)
    - [Loss derivation](#loss-derivation)
  - [Most important code pattern](#most-important-code-pattern)
- [Ensemble Dynamics Model](#ensemble-dynamics-model)
  - [Main idea](#main-idea-3)
    - [Loss](#loss)
  - [Implementation](#implementation-2)
    - [CEM Planning update](#cem-planning-update)
  - [Implementation](#implementation-3)
- [Final comparison](#final-comparison)


---


# Deterministic Dynamics Model


## Main idea


The first step is to learn a model of the environment dynamics from offline transitions. Instead of directly predicting the next state $s_{t+1}$, the model predicts the **state change**:


$$
\delta_s = s_{t+1} - s_t
$$

> _Why do we predict the difference instead of the absolute next state_ $s_{t+1}$_?_ 
> In the assignment, the observation is recorded at 10 HZ: the cartpole in inverted pendulum **barely moves between frames**. If you predict the absolute next state $s_{t+1}$, the network essentially just learns the identity function $s_{t+1} \approx s_t$, which isn't very useful. Predicting the _delta_ forces the network to learn the actual underlying physics of the movement.
>

This is the simplest version of learned dynamics: one network, one prediction per state-action pair, and simple MSE training. Once we learn the transitions/dynamics, we use them for planning.


Training loss: (the states, actions data are sampled from the replay buffer, which store real-world experience)


$$
\mathcal{L}(\theta) = \mathbb{E}[\{(s_{t+1} - s_t)  - f_\theta(s_t, a_t)\}^2]
$$


## Implementation


```python
# target delta
state_deltas = next_states - states
# model input
x = torch.cat([states, actions], dim=1)
pred_deltas = self.model(x)
# loss
loss = F.mse_loss(pred_deltas, state_deltas)
```


## Conceptual question


In the context of RL for physical control, why learn a dynamics model rather than (1) training directly on a hardware system or (2) deriving a simulator for our system by hand?


**Why not train directly on hardware?**

1. Sample inefficient: Model-free RL algorithms like PPO or SAC can require millions of environment interactions to converge. On a real robot, each timestep can be slow (constrained by real-time physics), causes real wear, and carries real risk. For example, a quadrotor exploring random thrust commands will crash, a manipulator exploring joint torques can damage itself or its surroundings.
2. lose of parallelism: we typically only have one (or a handful of) physical systems, whereas a learned model can be rolled out in massive batches on a GPU.

**Why not hand-derive a simulator?**


In fact, there exist some robotics simulation environment, such as MuJoCo, Drake, Isaac Sim, etc. But there are a few persistent problems:

1. Sim-to-real gap: analytical models rely on idealized assumptions — rigid bodies, known friction coefficients, perfect actuators. Real systems have soft contacts, cable routing effects, backlash, temperature-dependent motor behavior, and dozens of other phenomena that are hard to model from first principles. The policy you train in your idealized simulator may perform poorly on the real system precisely because the dynamics mismatch in the regimes that matter most (e.g., contact-rich manipulation).
2. hand-deriving a simulator is expensive engineering effort: For a novel mechanism or a deformable object, writing an accurate physics model from scratch can take months and require deep domain expertise. A learned model can be fit from a few hours of data collection.
3. not differentiable end-to-end in a convenient way, or it's too slow for the inner loops of planning and optimization. Learned models (neural nets) are differentiable by construction, enabling gradient-based trajectory optimization and backpropagation through the dynamics for policy learning.

**Why a learned dynamics model**


It occupies a middle ground: collect a modest amount of real-world data, fit a dynamics model, and then do policy optimization with it. This gives the sample efficiency benefits of model-based methods while automatically capturing the idiosyncratic dynamics of your specific hardware — the exact friction, the exact actuator lag, the exact sensor noise — without needing to derive any of it analytically. When the model is wrong, you can collect more targeted data (the Dyna / world-model loop), progressively closing the gap.


In practice, many modern systems hybridize: they start from a coarse analytical simulator, learn a residual correction on top of it from real data, and then do policy search in that corrected model. This gets the best of both worlds — the structure from physics and the fidelity from learning.


---


# CEM Planning with the Deterministic Model


## Main idea


Once a dynamics model is learned, we can use it to **plan**. Instead of learning a policy network directly, we use **Cross-Entropy Method (CEM)** to search for a good action sequence online. It's a sampling based evolutionary algorithm that iteratively reduces the cross entropy between our distribution over actions and the optimal distribution over actions at each time step. We start with a random distribution over actions, sample from it, simulate the outcomes and calculate their returns, pick a subset of the best performing action sequences, and use those to refit our random distribution over actions. Repeat until convergence (in theory) or for a fixed number of iterations (in practice). During real execution, CEM is replanned at every environment step in an MPC fashion: after optimization, we execute only the first action (often taken as the first action of the mean elite sequence), then observe the new state and replan again. Essentially, we are doing open loop planning by solving this constrained optimization:


$$
a_1^*, ..., a_T^* = \arg\max_{a_1, ..., a_T} \sum_{t=1}^{T} r_t  \\
\quad \quad \text{subject to} \quad s_{t+1} = s_t + f_\theta(s_t, a_t)
$$


### Pseudocode for CEM


> A. Initialize $\mu_0$ = 0, $\sigma_0$ = 1 for $T$ timesteps and $N_a$ dimensions   
> B. In each CEM iteration $k$:  
> - Start with $N$ parallel initial states (usually duplicate one initial state $N$ times)  
> - For each timestep $t$  
> - Select elite actions by picking $N_{elite}$ action sequences with the highest returns  
> - Refit distribution:  
> C. MPC return  
> - Return $\mu_{K, t=0}$ (mean action for the first timestep)


Refitting distribution, i.e. updating $\mu_k$ and $\sigma_k$, help planning concentrate around promising action sequences, while clamping $\sigma_k$ keeps enough exploration during optimization


CEM is a form of **real-time planning** or **model predictive control,** it’s just a simple way to convert a learned dynamics model into an acting agent.

> Reward assumption: CEM needs a way to score action sequences. In this assignment, the reward is assumed known and is computed manually from the predicted states.

## Implementation


```python
mean = torch.zeros((T, n_actions), device=device)
std_dev = torch.ones((T, n_actions), device=device)

for _ in range(cem_iter):
    actions = Normal(mean, std_dev).sample((n_samples,)).clamp(-3, 3)
    states = initial_states.repeat(n_samples, 1)
    returns = torch.zeros(n_samples, device=device)

    with torch.no_grad():
        for t in range(T):
            states = states + dynamics.predict(states, actions[:, t, :])
            returns += compute_rewards(states)

    elite_idx = returns.topk(n_elite).indices
    elite_actions = actions[elite_idx]
    mean = elite_actions.mean(dim=0)
    std_dev = elite_actions.std(dim=0).clamp(min=0.5)

action = mean[0].unsqueeze(0)
```


## Conceptual question


**What are the pros and cons of real-time planning (CEM) vs policy learning (REINFORCE, DQN, PPO, etc.)?**


Real-time planning methods like CEM choose actions by repeatedly simulating candidate action sequences through a learned dynamics model and picking the best one. 

- **sample efficient**: the dynamics model can be trained from every transition, even without task-specific reward labels.
- **adaptable**: if the reward function changes but the environment dynamics stay the same, the planner can often reuse the same model and simply plan with the new reward.
- **computationally expensive at inference time:** since many trajectories must be evaluated at every decision step.
- **vulnerable to model error**: if the learned dynamics are inaccurate, the planner may exploit those inaccuracies and choose actions that look good in simulation but perform poorly in the real environment.

Policy learning methods like REINFORCE, DQN, or PPO instead train a policy network that maps states directly to actions. 

- **fast inference**: after training, action selection usually requires only a single forward pass.
- **more practical for deployment:** since no online trajectory optimization is needed.
- **less sample efficient**, especially in complex environments, because they typically require many reward-driven interactions to learn a good policy.
- **less adaptable** than planners: if the reward function or task changes, the policy often needs substantial retraining

---


# Stochastic Dynamics Model


## Main idea


The deterministic model gives a single point estimate and may become overconfident. In practice, the transitions in data my be noise because of sensor noise, partial observability, etc. A stochastic model can represent transition uncertainty, which can make multi-step rollouts less brittle during planning.


Instead of predicting one delta-state as in deterministic dynamics, we predict a **distribution** over delta-state:


$$
\mu_t, \sigma_t = f_\theta(s_t, a_t)\\
s_{t+1} - s_t \sim \mathcal{N}(\mu_t, \sigma_t)
$$


In practice we have to use these slightly less elegant equations to keep $\sigma_t$ positive and bounded:


$$
\mu_t, \log \sigma_t = f_\theta(s_t, a_t)\\
\sigma_t = \text{clamp} (e^{\log \sigma_t}, 10^{-8}, 10)  \\
s_{t+1} - s_t \sim \mathcal{N}(\mu_t, \sigma_t) 
$$


The stochastic dynamics model mainly captures **aleatoric uncertainty**, i.e. inherent transition noise, observation noise, and ambiguity in the target, by modeling the next-state change as a Gaussian distribution. This means it represents uncertainty as noise in the transition. It may not be the right type of uncertainty for the inverted pendulum system since the environment is mostly deterministic.


### Loss derivation


Let the target be $y_t = s_{t+1}-s_t$. If we assume target follows gaussian distribution: $y_t \sim \mathcal N(\mu_t,\sigma_t^2)$. The Gaussian negative log-likelihood is:


$$
p(y_t \mid s_t,a_t)= \frac{1}{\sqrt{2\pi}\sigma_t}\exp\left(-\frac{(y_t-\mu_t)^2}{2\sigma_t^2}\right)
$$


The negative log likelihood becomes


$$
-\log p(y_t \mid s_t,a_t)=\frac{(y_t-\mu_t)^2}{2\sigma_t^2}+ \log \sigma_t+ \frac{1}{2}\log(2\pi) \\
\mathcal L(\theta)
=
\mathbb E\left[
\frac{((s_{t+1}-s_t)-\mu_t)^2}{2\sigma_t^2}
+ \log \sigma_t
\right]
$$


## Most important code pattern


```python
x = torch.cat([states, actions], dim=1)
out = self.model(x)
mu, log_sigma = torch.chunk(out, 2, dim=1)
sigma = torch.clamp(torch.exp(log_sigma), min=1e-8, max=10)
```


Prediction:


```python
noise_dist = Normal(mu, sigma)
pred_deltas = noise_dist.sample()
next_states = states + pred_deltas
```


Loss:


```python
var = torch.exp(2 * log_sigma)
loss = GaussianNLLLoss(full=False)(mean, next_states-states, var)
```


---


# Ensemble Dynamics Model


## Main idea


Modelling our system as stochastic doesn't capture all types of uncertainty. In ensemble transition model, we model the transition functions as:


$$
s_{t+1} - s_t = f_{\theta_j}(s_t, a_t), \quad \forall j \in \{1, \ldots, K\}
$$


Where each $f_{\theta_j}$ is independently initialized and trained on independent minibatches from $\mathcal{D} = \{(s, a, s')\}$. Each one predicts a delta-state. The idea is that if these models disagree, that disagreement reveals uncertainty in the learned dynamics. The ensemble dynamics model mainly captures **epistemic uncertainty,** e.g. limited training data, model mismatch, and uncertainty in unexplored regions.


### Loss


$$
\mathcal{L}(\theta) = \sum_{j=1}^{N} \mathbb{E}[\{(s_{t+1} - s_t)  - f_{\theta_j}(s_t, a_t)\}^2]
$$


## Implementation


Training one model’s loss:


```python
state_deltas = next_states - states
pred_deltas = self.models[j](torch.cat([states, actions], dim=1))
loss = F.mse_loss(pred_deltas, state_deltas)
```


loss aggregation over ensemble models:


```python
total_loss = 0
  for j in range(ensemble_dynamics.N):
      states, actions, next_states = replay_buffer.sample()
      loss_j = ensemble_dynamics.get_loss(states, actions, next_states, j)
      total_loss += loss_j

  total_loss.backward()
  ensemble_dynamics.optimizer.step()
```


### CEM Planning update


Now we use CEM again, but instead of scoring each action sequence using one learned dynamics model, we score it using **all ensemble members**. Each action sequence is rolled out under all models, and we compute the **expected return across the ensemble**. Then CEM updates its action distribution based on that expected return. In deterministic dynamics model, CEM may exploit errors in one learned model. Ensemble dynamics can be more robust because a candidate action sequence must look good across multiple plausible dynamics models.


$$
\begin{aligned}a_1^*, ..., a_T^* &= \arg\max_{a_1, ..., a_T} \frac{1}{N} \sum_{j=1}^{N} \sum_{t=1}^{T} r_t^{(j)} \\\\\text{subject to} \quad s_{t+1}^{(j)} &= s_t^{(j)} + f_{\theta_j}(s_t^{(j)}, a_t), \quad \forall j \in \{1, ..., N\} \\\\\text{given} \quad s_0^{(j)} &= s_0, \quad \forall j \in \{1, ..., N\}\end{aligned}
$$


Elite selection is based on ${R}^{(i)}$, not the return from a single model.


## Implementation


```python
# in one CEM iteration
states = initial_states.repeat(self.dynamics.N, self.n_samples, 1)
actions = Normal(mean, std_dev).sample((self.n_samples,)).clamp(-3, 3)
returns = torch.zeros((self.dynamics.N, self.n_samples), device=device)

with torch.no_grad():
    for t in range(self.T):
        states = states + self.dynamics.predict(states, actions[:, t, :])
        returns += self.compute_rewards(states)

expected_returns = returns.mean(dim=0) # average returns under different dynamics models
elite_idx = expected_returns.topk(self.n_elite).indices
elite_actions = actions[elite_idx]
mean = elite_actions.mean(dim=0)
std_dev = elite_actions.std(dim=0).clamp(min=0.5)
```


---


# Final comparison


Deterministic dynamics: one point estimate for the transition.

- Pros: simple, easy to train, easy to debug
- Cons: overconfident, vulnerable to compounding error, easy for planner to exploit model mistakes

Stochastic dynamics: predict a Gaussian distribution over transition delta.

- Pros: captures aleatoric uncertainty, less overconfident than deterministic model, can help robustness during planning
- Cons: not necessarily the right uncertainty for deterministic systems, still does not eliminate compounding error

Ensemble dynamics: use disagreement among several deterministic models to estimate uncertainty.

- Pros: captures epistemic uncertainty, often a better fit for model uncertainty, reduces model exploitation during planning
- Cons: more expensive to train and evaluate, may be more conservative, can underperform in practice if individual models are weak or data is limited
> Compounding error: A small prediction error at one step causes the next prediction to start from the wrong state, which can make errors accumulate over multi-step rollouts.

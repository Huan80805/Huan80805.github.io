---
title: "Inverse RL & GRPO"
date: 2026-02-25
type: note
tags:
  - RL
listed: false
---


- [Inverse RL](#inverse-rl)
  - [Maximum Entropy IRL](#maximum-entropy-irl)
- [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo)
  - [The Group Advantage (Control Variate)](#the-group-advantage-control-variate)
  - [Sequence-Level Probability Ratio](#sequence-level-probability-ratio)
  - [The Clipped Surrogate Objective](#the-clipped-surrogate-objective)
  - [KL Divergence Penalty](#kl-divergence-penalty)
  - [The Final Loss Function & Code](#the-final-loss-function-code)
  - [Reward Shaping (Sparse Signals)](#reward-shaping-sparse-signals)
- [References](#references)


## Inverse RL


Given a set of **expert demonstrations**, find the underlying **reward function** that the expert was secretly trying to maximize. Essentially, instead of telling the AI _what_ to do and letting it figure out _how_, we show the AI _how_ it's done, and it figures out _why_ (the reward).


Formally, we want to find a reward function such that the expert outperforms other policies:


$$
\mathbb{E}[\sum_{t=0}^{\infty} \gamma^t R^*(s_t) | \pi^*] \ge \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t R^*(s_t) | \pi],\quad \forall \pi
$$


$$
\text{Let}\;R(s) = w^{\intercal}\phi(s), \text{where}\; w \in \mathbb{R}^{n}, \text{and}\; \phi: \mathcal{S} \rightarrow \mathbb{R}^{n}. \\
\mathbb{E}[\sum_{t=0}^{\infty} \gamma^t R^*(s_t) | \pi] = w^\intercal \mu(\pi)\\

$$


where $\mu(\pi)$ is the expected cumulative discounted sum of features values, or **“feature expectations”.** Then the objective becomes:


$$
\text{Find}\; w^* \;\text{such that}\;{w^{*}}^{\intercal}\mu(\pi^*) \ge {w^{*}}^{\intercal}\mu(\pi) \quad \forall \pi
$$


For a policy $\pi$ to be guaranteed to perform as well as the expert $\pi^*$, it suffices that the feature expectations match ([2](#ref-2)), that is:


$$
\| \mu(\pi) -  \mu(\pi^*)\|_1 \le \epsilon \\
\text{then}\;{w^*}^{\intercal} \mu(\pi)- {w^*}^{\intercal} \mu(\pi^*)\rvert \le \epsilon \quad \forall w \text{ such that } \|w\|_\infty < 1

$$


> 💡 Features $\phi(s)$ are measurable characteristics of the environment at any given moment. For example, in driving, the features might be: (1). Distance to the center of the lane. (2). Current speed (3). Distance to the car in front.  
> Feature expectations is the average of features experienced by the policy: $E_{\pi^E}[\phi(\tau)] = \mu(\pi^E)$


---


### Maximum Entropy IRL


There are infinitely many reward functions that could explain the expert’s behavior. To solve this, Maximum entropy RL additionally incorporates the principle of maximum entropy ([1](#ref-1)): 

> _Out of all the reward functions that match the expert's features, we want to pick the one that results in the most random, highest-entropy trajectory distribution. This is a mathematical way of saying, "We shouldn't make any extra assumptions about the expert's behavior beyond what we actually saw them do."_

The entropy is calculated over the distribution of entire trajectories:


$$
 H(P) = -\sum_{\tau} P(\tau) \log P(\tau). 
$$


It can be shown that:


$$
p(\tau \mid \omega) = \frac{1}{Z(\omega)} \exp(\omega^\top \phi(\tau)) \prod_{t} p(s_{t+1} \mid s_t, a_t)
$$

- $\omega^\top \phi(\tau)$ is the total reward of the trajectory. **Trajectories with higher total rewards are exponentially more likely.**
- $\prod_{t} p(s_{t+1} \mid s_t, a_t)$ accounts for the environment's dynamics. We can't control physics, so we just multiply by the probability of those physical transitions happening.
- $\frac{1}{Z(\omega)}$ is the partition function (normalizing constant).

> Proof:  
> Our objective: maximize entropy $H(p)$ with two constraints:  
> 1. $\sum_\tau p(\tau) = 1$  
> 2. features matching: $\sum_\tau p(\tau)\phi(\tau) = \mu_E$ → expected feature counts of our expectation must equal the expert’s empirical features counts $\mu_E$  
> Using Lagrangian:  
> $$  
> \mathcal{L}(p, \lambda, \omega) = -\sum_{\tau} p(\tau) \log p(\tau) + \lambda \left(1 - \sum_{\tau} p(\tau)\right) + \omega^\top \left(\sum_{\tau} p(\tau) \phi(\tau) - \mu_E\right)  
> $$  
> To find maximum:  
> $$  
> \frac{\partial \mathcal{L}}{\partial p(\tau)} = -1 - \log p(\tau) - \lambda + \omega^\top \phi(\tau) = 0 \\  
> \log p(\tau) = \omega^\top \phi(\tau) - 1 - \lambda  
> \\  
> p(\tau) = e^{\omega^\top \phi(\tau) - 1 - \lambda} = e^{-1-\lambda} e^{\omega^\top \phi(\tau)}  
> $$  
> $e^{-1-\lambda}$ is just a constant that ensures the probabilities sum to 1, we can replace it with the partition function $\frac{1}{Z(\omega)}$. We also multiply by the fixed probability of the transitions, $\prod p(s_{t+1} \mid s_t, a_t)$.This leaves us with the exact distribution:  
> $$  
> p(\tau \mid \omega) \propto e^{\omega^\top \phi(\tau)}  
> $$


To learn $w$, the gradient of the log-likelihood is:


$$
\nabla_\omega L(\omega) = \mu_E - \sum_s D_s\, \phi(s)
$$


where:

- $L(\omega)$ is the log-likelihood of the expert’s trajectories under our reward function: $L(\omega) = \frac{1}{N} \sum_{i=1}^N \log p(\tau_i \mid \omega)$
- $\mu_E$: the “target” — the empirical feature expectation, i.e. feature counts the expert actually accumulated. $\mu_E = \frac{1}{N} \sum_{i=1}^N \sum_{t} \phi(s_t^{(i)})$

```python
def feature_expectation_from_trajectories(features: np.ndarray, demos: Iterable[Trajectory]) -> np.ndarray:
    totals = np.zeros(features.shape[1], dtype=np.float64)
    n_trajectories =
    for traj in demos:
        visited = traj.state_sequence()
        if visited.size == 0:
            continue
        totals += features[visited].sum(axis=0)
        n_trajectories += 1
    return totals / n_trajectories
```

- $\sum_s D_s\, \phi(s)$: the “Prediction” — the time-aggregated expected state-visitation frequency, i.e. feature counts our current weights $\omega$ _expect_ the agent to accumulate.
- $D_s$ is the expected state visitation frequency, i.e. the expected number of times the agent visit state $s$.

<u>In practice,</u> $D_s$ <u>is calculated with dynamic programming, involving a backward and a forward pass:</u>


**Backward pass - finding the policy**


We work backward from the end of the game to the beginning to calculate partition functions $Z$. $Z_s$ represents the total sum of exponentiated reward we can get from this state onward. $Z_{s,a}$ represents how much exponentiated reward can get in state $s$ and take action $a$? 


$$
Z_s = \sum_a Z_{s, a}
$$


Then we calculate this recursively: 


$$
Z_{s,a} = e^{r(s)} \sum_{s'} P(s'\mid s,a) Z_{s'}
$$


The exponential comes from the trajectory distributions defined before: $p(\tau \mid \omega) \propto e^{\text{accumulated reward}}$. Then, for a trajectory: $e^{r_1 + r_2 + r_3 \dots} = e^{r_1} * (\text{expected future reward})$. The recursion starts from the terminal states by setting their $Z$ values to 1.0 while other states to 0. We recursively sweep across all states over until their $Z_s$ values converge. Finally, our policy is: 


$$
\pi(a \mid s) = \frac{Z_{s,a}}{Z_s}
$$


Forward pass - counting the visits


Once we have the policy, we push “probability mass” through the grid, step by step, to see what states the policy spend time on. Let $d_t(s)$ be the probability of being in state $s$ at time $t$. We initialize initial state probability $d_0(s)$ (this is done by simply calculating initial state distributions in expert trajectories). Then “push the probability mass” recursively:


$$
d_{t+1}(s') = \sum_{s} \sum_{a} d_t(s) \cdot \pi(a \mid s) \cdot P(s' \mid s, a)
$$


note that if reaching a terminal state, we simply skip it in summation since we cannot push probability mass out of a terminal state. Then the final visitation frequency is:


$$
D_s = \sum_{t=0}^{\text{horizon}} d_t(s)
$$


```python
def compute_expected_svf(p_transition: np.ndarray, p_initial: np.ndarray, terminal: Iterable[int], reward: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n_states, _, n_actions = p_transition.shape
    terminal = set(int(t) for t in terminal)
    Zs = np.zeros(n_states, dtype=np.float64)
    Zs[list(terminal)] = 1.0
    exp_reward = np.exp(reward)
    for _ in range(2 * n_states):
        Za = np.zeros((n_states, n_actions), dtype=np.float64)
        for s in range(n_states):
            for a in range(n_actions):
                # compute the state-action partition value using the backward recursion
                Za[s, a] = exp_reward[s] * (p_transition[s, :, a] * Zs).sum()
        new_Zs = Za.sum(axis=1)
        if np.max(np.abs(new_Zs - Zs)) < eps:
            Zs = new_Zs
            break
        Zs = new_Zs
    with np.errstate(divide="ignore", invalid="ignore"):
        policy = np.divide(Za, Zs[:, None], out=np.zeros_like(Za), where=Zs[:, None] > 0)
    horizon = 2 * n_states
    d = np.zeros((n_states, horizon), dtype=np.float64)
    d[:, 0] = p_initial
    for t in range(1, horizon):
        for s_prev in range(n_states):
            if s_prev in terminal:
                continue
            for a in range(n_actions):
                prob = d[s_prev, t - 1] * policy[s_prev, a]
                if prob == 0:
                    continue
                # push visitation probability mass forward through the transition model
                d[:, t] += prob * p_transition[s_prev, :, a]
    return d.sum(axis=1)
```


## Group Relative Policy Optimization (GRPO)


GRPO extends standard PPO specifically for aligning Large Language Models. Standard PPO requires training a separate, complex “Critic” (Value Model) that is often the same size as the main model. This essentially doubles the memory requirements. GRPO skips the Critic entirely. Instead of using a separate neural network to estimate how good a state is (the baseline), GRPO samples multiple responses for the _same prompt_ and uses their mean reward as a prompt-level control variate. This stabilizes policy updates while saving massive amounts of VRAM.


### The Group Advantage (Control Variate)


For each prompt $i$, we draw a group of $G$ completions $\{y_{i,j}\}_{j=1}^G$ from the policy. We calculate a reward $r_{i,j}$ for each completion. To determine if a completion was good/bad relative to the model's current capability, we calculate the group mean $\bar{r}_i = \frac{1}{G}\sum_j r_{i,j}$ and the group standard deviation $\sigma$.


The centered and normalized advantage is:


$$
A_{i,j} = \frac{r_{i,j} - \bar{r}_i}{\sigma}
$$


By normalizing with the standard deviation, we ensure the advantages have a mean of 0 and a variance of 1, which keeps the gradients at a stable scale regardless of the absolute scale of the reward function.


```python
def compute_group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    reshaped = rewards.view(-1, group_size)
    mean = reshaped.mean(dim=1, keepdim=True)
    std = reshaped.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized = (reshaped - mean) / std
    return normalized.view(-1)
```


---


### Sequence-Level Probability Ratio


In standard token-level RL, the model tries to assign credit to every single word. However, for reasoning and math tasks, the reward is **sparse and sequence-level** (e.g. you don't know if the math is correct until the very end of the `<answer>` tag). Since the reward function generally works on the whole sequence, we correspondingly calculate the log probability of generating an entire sequence as the sum of the log probabilities of its individual tokens:


$$
\log P(\text{Sequence}) = \sum \log P(\text{token})
$$


The probability ratio $r_t$ then compares how likely the current model $\theta$ is to generate the entire sequence compared to the frozen reference model:


$$
r_t = \exp(\log P_\theta(\text{Sequence}) - \log P_{\text{ref}}(\text{Sequence}))
$$


### The Clipped Surrogate Objective


The policy update maximizes a PPO-style clipped surrogate objective to prevent destructively large updates from a single lucky (or unlucky) batch:


$$
L_{\text{clip}} = \min(r_t A_t, \text{clip}(r_t, 1 - \epsilon, 1 + \epsilon) A_t)
$$

- When $A_t >0$: The policy wants to increase $r_t$ (make this action more likely). However, the min function ensures that once
hits the ceiling of $1+\epsilon$, the gradient drops to zero. This prevents the model from taking overly large updates
- When $A_t < 0$: The policy wants to decrease $r_t$ (make this action less likely). The min function ensures that once $r_t$ drops to the floor of $1-\epsilon$, the gradient drops to zero. This prevents the model from completely destroying the probability of a specific sequence just because it happened to perform worse than average in this one specific group.

---


### KL Divergence Penalty 


To prevent the model from drifting too far from the reference model (which causes mode collapse or gibberish), we subtract a KL divergence penalty scaled by a coefficient $\beta$.


Formal KL Divergence requires an expectation over all possible sequences: $\mathbb{E}_{x \sim \pi_\theta} [\log \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_\text{ref}(o_t \mid q, o_{<t})}]$. Since we cannot compute that, we use **Monte Carlo Estimation**: we estimate the divergence using only the specific sequences we just generated in our batch.


DeepSeekMath Robust Estimator


Instead of log-prob difference ($\log \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_\text{ref}(o_t \mid q, o_{<t})}$), which can sometimes be negative and accidentally _reward_ drift, [[3] Shao, Zhihong, Peiyi Wang, Qihao Zhu, Runxin Xu, Jun-Mei Song, Mingchuan Zhang, Y. K. Li, Yu Wu and Daya Guo. “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.” ArXiv abs/2402.03300 (2024): n. pag.](https://www.notion.so/35f9b623876680bdb819f2c0dcef021c#3659b623876680f0adedea6515a45404)  introduces a strictly non-negative robust estimator for KL divergence (strictly punishing the policy whenever deviating from the reference model):


$$
\mathbb{D}_{KL}[\pi_\theta \mid\mid \pi_\text{ref}] = \frac{\pi_\text{ref}(o_t \mid q, o_{<t})}{\pi_\theta(o_t \mid q, o_{<t})} - \log \frac{\pi_\text{ref}(o_t \mid q, o_{<t})}{\pi_\theta(o_t \mid q, o_{<t})} - 1
$$


---


### The Final Loss Function & Code


The objective we want to maximize is $L_{\text{clip}} - \beta \text{KL}$. Because PyTorch optimizers Note that taking the `.mean()` across the batch of sequences effectively computes the expected value ($\mathbb{E}_{x \sim \pi_\theta}$) required by the formal KL and RL formulas.


```python
def grpo_sequence_loss(logp_new: torch.Tensor, logp_ref: torch.Tensor, advantages: torch.Tensor, beta: float, epsilon: float) -> torch.Tensor:
    # 1. PPO Clipped Surrogate
    r_t = torch.exp(logp_new - logp_ref)
    clipped_r_t = torch.clamp(r_t, 1 - epsilon, 1 + epsilon)
    l_clip = torch.min(r_t * advantages, clipped_r_t * advantages)

    # 2. Robust KL Penalty (DeepSeekMath)
    log_ratio = logp_ref - logp_new
    kl = torch.exp(log_ratio) - log_ratio - 1.0

    # 3. Final Loss (Maximize objective = Minimize negative objective)
    # The .mean() acts as the Monte Carlo approximation of the expected value
    loss = -(l_clip - beta * kl).mean()

    return loss
```


---


### Reward Shaping (Sparse Signals)


Format rewards are dense, but answer-correctness rewards are sparse (only evaluated at the end if the final extraction matches the ground truth). If the model only receives a reward for perfection, it might randomly generate tokens for thousands of steps without ever stumbling upon a correct answer, resulting in 0 gradients and no learning.


Tiered shaping is one way to keep the correctness learning signal meaningful, we use t:

1. Format Breadcrumbs (+0.1 to +0.5): Give small, dense rewards for correctly opening a `<think>` tag, closing it `</think>`, and opening an `<answer>` tag.
2. The Correctness Jackpot (+2.0): Give a massive reward only if the final extracted string mathematically matches the ground truth.

This bootstraps the policy. The dense rewards guide the model to learn the structural syntax first. Once the model reliably outputs the `<answer>` tags, it creates the structural "container" required for the verifier to check the math, allowing it to finally experience and learn from the sparse, high-value correctness signal.


## References


<span id="ref-1"></span>[1] Brian D. Ziebart, Andrew Maas, J. Andrew Bagnell, and Anind K. Dey. 2008. Maximum entropy inverse reinforcement learning. In Proceedings of the 23rd national conference on Artificial intelligence - Volume 3 (AAAI'08). AAAI Press, 1433–1438.


<span id="ref-2"></span>[2] Pieter Abbeel and Andrew Y. Ng. 2004. Apprenticeship learning via inverse reinforcement learning. In Proceedings of the twenty-first international conference on Machine learning (ICML '04). Association for Computing Machinery, New York, NY, USA, 1. https://doi.org/10.1145/1015330.1015430


<span id="ref-3"></span>[3] Shao, Zhihong, Peiyi Wang, Qihao Zhu, Runxin Xu, Jun-Mei Song, Mingchuan Zhang, Y. K. Li, Yu Wu and Daya Guo. “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.” _ArXiv_ abs/2402.03300 (2024): n. pag.


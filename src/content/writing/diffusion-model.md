---
title: Diffusion model
date: 2026-04-06
type: blog
tags:
  - ML
---


**Table of Contents**


- [Introduction](#introduction)
- [Theoretical definition](#theoretical-definition)
  - [_forward diffusion process_](#forward-diffusion-process)
  - [_reverse diffusion process_](#reverse-diffusion-process)
  - [_Variational lower bound_](#variational-lower-bound)
  - [From per-step KL to noise-matching loss](#from-per-step-kl-to-noise-matching-loss)
- [Training and Inference](#training-and-inference)
- [Comparison with VAE](#comparison-with-vae)
- [References](#references)


## Introduction


A **diffusion model** is a **generative model** that learns to create data, such as images, by reversing a gradual corruption process.


Informally, it works in two stages:

1. **Forward diffusion:** start with a real data sample and gradually add small amounts of noise until it becomes **almost pure Gaussian noise**
2. **Reverse diffusion:** train a neural network to undo that noise step by step, so that starting from random noise it can generate a realistic sample.

---


## Theoretical definition


### _forward diffusion process_


Let $x_0 \sim q(x)$ be a data point from the true data distribution. A diffusion model defines a _**latent Markov chain:**_ $x_0, x_1, \dots, x_T$, where:


The forward process gradually destroys structure by adding Gaussian noise:


$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I\right)
$$


where (0 < $\beta_t$ < 1) is the noise schedule, which shrinks the signal and adds Gaussian noise in every step. Let $\alpha_t = 1-\beta_t,\quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. We can derive a closed form 


$$
\begin{equation}
q(x_t | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t}} x_0, (1-\bar{\alpha}_{t})I)
\end{equation}
$$


> <u>_**Prove by induction:**_</u>  
> **Base case (t=1):**   
> $q(x_1|x_0) = \mathcal{N}(\sqrt{\alpha_1} x_0, \beta_1 I)$. $\bar{\alpha}_1 = \prod_{s=1}^1 \alpha_s = \alpha_1, \beta_1 = 1 - \alpha_1 = 1 - \bar{\alpha}_1$. Substituting $\bar{\alpha}_1$ and $(1 - \bar{\alpha}_1)$ into the transition equation yields:  
> $$  
> q(x_1|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_1} x_0, (1 - \bar{\alpha}_1)I)  
> $$  
> → Base case holds  
> **Inductive Step:** Assume the expression holds at $t-1$, that is:  
> $$  
> q(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}} x_0, (1-\bar{\alpha}_{t-1})I)  
> $$  
> Using the reparameterization trick, we can write $x_{t-1}$ as:  
> $$  
> \begin{equation}   
> x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_{t-1}, \quad \text{where}\; \epsilon_{t-1} \sim \mathcal{N}(0, I)  
> \end{equation}  
> $$  
> by definition of the forward process: $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, \beta_t I)$, we reparameterize $x_t$ in the same way:  
> $$  
> \begin{equation}  
> x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t  
> \quad \text{where} \;\epsilon_t \sim \mathcal{N}(0, I)  
> \end{equation}  
> $$  
> substitute our reparameterized equation (2) into (3):  
> $$  
> \begin{align*}            x_t &= \sqrt{\alpha_t} \Big( \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_{t-1} \Big) + \sqrt{\beta_t} \epsilon_t \\            &= \sqrt{\alpha_t \bar{\alpha}_{t-1}} x_0 + \sqrt{\alpha_t(1-\bar{\alpha}_{t-1})} \epsilon_{t-1} + \sqrt{\beta_t} \epsilon_t \\            &\text{using }\; \bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1} \\            &=\sqrt{\bar{\alpha}_t} x_0 + \sqrt{\alpha_t - \bar{\alpha}_t} \epsilon_{t-1} + \sqrt{\beta_t} \epsilon_t            \end{align*}  
> $$  
> Because $\epsilon_{t-1}$ and $\epsilon_t$ are I.I.D. Gaussians, we can merge them using the sum of independent Gaussians identity ($aX + bY = \sqrt{a^2 + b^2}\epsilon$):  
> $$  
> \begin{align*}            \text{Variance} &= (\sqrt{\alpha_t - \bar{\alpha}_t})^2 + (\sqrt{\beta_t})^2 \\  
> &= 1 - \bar{\alpha}_t          \end{align*}  
> $$  
> $$  
> x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon           \\q(x_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)I)  
> $$


So at any time $t$, $x_t$ is just a noisy version of $x_0$. Note that when we choose $\{\beta_t\}$ such that $\bar\alpha_T \approx 0$, the forward diffusion process completely diffuses all the information in original data: $q(x_T|x_0) = \mathcal{N}(0, I)$


---


### _reverse diffusion process_


If we can reverse the above forward process and sample from $q(x_{t-1} \mid x_t)$, we will be able to recreate the true sample from a Gaussian noise input. Unfortunately, we cannot easily estimate  because it needs to use the entire dataset and therefore we need to learn a model $p_\theta$ to approximate these conditional probabilities.

> The forward noise levels $\beta_t$ are either fixed or learnable. A nice property is that when $\beta_t$ is small (which is usually the case in diffusion models), the reverse conditional $q(x_{t-1} \mid x_t)$ will have the same functional form of $q(x_{t} \mid x_{t-1})$, i.e. Gaussian (see 2.2 of [[3] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S.. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. Proceedings of the 32nd International Conference on Machine Learning](https://www.notion.so/33a9b6238766800f97a5c4f826baecd1#33a9b62387668060aa85ee012ad57832) ). Under this assumption, modeling the reverse denoising step as Gaussian (i.e. choosing the functional form of $p_\theta(x_{t-1}\mid x_t)$ as Gaussians) is justified because the true reverse dynamics of a small Gaussian corruption are themselves approximately Gaussian.
>

$p_\theta$ learns a reverse Markov chain: $p_\theta(x_{t-1}\mid x_t)$, typically parameterized as a Gaussian:


$$
p_\theta(x_{t-1}\mid x_t)=\mathcal{N}\left(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t)\right)
$$


The full reverse process (generative model):


$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t), \quad \text{where}\; p(x_T)=\mathcal{N}(0,I)
$$


---


### _Variational lower bound_


([[1] Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. https://lilianweng.github.io/posts/2021-07-11-diffusion-models/.](https://www.notion.so/33a9b6238766800f97a5c4f826baecd1#33a9b6238766800298d7df24350bd191) )


We want to maximize the evidence across whole data distribution, i.e. maximize $\mathbb{E}_{q(x_0)} \log p_\theta(x_0)$


$$
\begin{align*}
\mathbb{E}_{q(x_0)} \log p_\theta(x_0) 
&=  \mathbb{E}_{q(x_0)} \left[ \log \left( \int \cdots \int p_\theta(x_0,x_1,\dots,x_T)\,dx_1\,dx_2\cdots dx_T. \right) \right] \\
&= \mathbb{E}_{q(x_0)} \left[  \log \int p_\theta(x_{0:T})\,dx_{1:T} \right] \\
&= \mathbb{E}_{q(x_0)} \left[ \log \int q(x_{1:T}\mid x_0)\,
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\,dx_{1:T} \right] \\
&= \mathbb{E}_{q(x_0)} \log
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right] \\
&\text{by Jensen's inequality} \\
&\geq
\mathbb{E}_{q(x_{0:T})}
\log
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}  = L_{VLB}
\end{align*}
$$


Further reduce the variational lower bound: ([[2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. In Proceedings of the 34th International Conference on Neural Information Processing Systems (NIPS '20)](https://www.notion.so/33a9b6238766800f97a5c4f826baecd1#33a9b623876680f585ebffd4f7749cd1) )


$$
\begin{align*}L_{VLB}&= \mathbb{E}_q \left[ - \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid \mathbf{x}_0)} \right] \\
&= \mathbb{E}_q \left[    - \log p(\mathbf{x}_T)    - \sum_{t \ge 1} \log \frac{p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)}         {q(\mathbf{x}_t\mid \mathbf{x}_{t-1})}\right] \\&= \mathbb{E}_q \left[    - \log p(\mathbf{x}_T)    - \sum_{t>1} \log    \frac{p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)}         {q(\mathbf{x}_t\mid \mathbf{x}_{t-1})}    - \log    \frac{p_\theta(\mathbf{x}_0\mid \mathbf{x}_1)}         {q(\mathbf{x}_1\mid \mathbf{x}_0)}\right] \\&= \mathbb{E}_q \left[    - \log p(\mathbf{x}_T)    - \sum_{t>1} \log \left(        \frac{p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)}             {q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,\mathbf{x}_0)}        \cdot        \frac{q(\mathbf{x}_{t-1}\mid \mathbf{x}_0)}             {q(\mathbf{x}_t\mid \mathbf{x}_0)}    \right)    - \log    \frac{p_\theta(\mathbf{x}_0\mid \mathbf{x}_1)}         {q(\mathbf{x}_1\mid \mathbf{x}_0)}\right] \\&= \mathbb{E}_q \left[    - \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T\mid \mathbf{x}_0)}    - \sum_{t>1} \log    \frac{p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)}         {q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,\mathbf{x}_0)}    - \log p_\theta(\mathbf{x}_0\mid \mathbf{x}_1)\right] \\&= \mathbb{E}_q \left[    \underbrace{D_{\mathrm{KL}}\!\left(        q(\mathbf{x}_T\mid \mathbf{x}_0)\,\|\,p(\mathbf{x}_T)    \right)}_{L_T}    + \sum_{t>1}    \underbrace{D_{\mathrm{KL}}\!\left(        q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)    \right)}_{L_{t-1}}    \;\;\underbrace{ - \log p_\theta(\mathbf{x}_0\mid \mathbf{x}_1)}_{L_0}\right].\end{align*}
$$

- $L_0 = -\log p_\theta(\mathbf{x}_0\mid \mathbf{x}_1)$: reconstruction / data likelihood term
    - ensures the last reverse step actually lands on the data distribution
    - In many DDPM parameterizations, this becomes a simple Gaussian likelihood term
- $L_{t-1}=D_{\mathrm{KL}}\!\left(q(\mathbf{x}_{t-1}\mid \mathbf{x}_t,\mathbf{x}_0)\,\|\, p_\theta(\mathbf{x}_{t-1}\mid \mathbf{x}_t)\right)$: reverse-step matching / denoising term
    - This is the core of diffusion learning, actually trains the model’s reverse transition.  Given $x_t$, learned denoising step should match the true reverse posterior induced by the forward process

> forward process posteriors (true posterior) are tractable when conditioned on $x_0$, thus $L_{t-1}$is just Gaissian KL, which has a closed form  
> $$  
> \begin{align*}  
> &q(x_{t-1} \mid x_t, x_0) = \mathcal N(x_{t-1}; \tilde{\mu_t}(x_t; x_0), \tilde{\beta_t}I) \\  
> &\text{where} \; \tilde{\mu_t}(x_t; x_0) := \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t, \;\text{and}\; \tilde\beta_t:=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t  
> \end{align*}  
> $$

- $L_T = D_{\mathrm{KL}}\!\left(q(\mathbf{x}_T\mid \mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\right)$: prior matching term
    - forces the final forward-noised variable $x_T$ to match the simple prior, usually $p(x_T)=\mathcal N(0,I)$
    - In practice, with a sufficiently long diffusion chain, $q(x_T\mid x_0)$ is already very close to $\mathcal N(0,I)$, and often forward process $q$ is not learnable, this term is then treated as constant.

---


### From per-step KL to noise-matching loss


From previous session, our main learning signal is:


$$
L_{t-1}=\mathbb{E}_{q(x_0, x_t)}\Big[D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t,x_0)\;\|\;p_\theta(x_{t-1}\mid x_t)\big)\Big] \\ \text{where} \;q(x_{t-1}\mid x_t,x_0)
= \mathcal N\big(\tilde\mu_t(x_t,x_0), \tilde\beta_tI) \\
\text{and} \; p_\theta(x_{t-1}\mid x_t)
=
\mathcal N\big(\mu_\theta(x_t,t), \sigma_t^2 I\big),
$$


$$
\begin{align*}
D_{\mathrm{KL}}(q\|p_\theta) &=\frac{1}{2}\left[\frac{\|\tilde\mu_t-\mu_\theta\|^2}{\sigma_t^2}+d\left(\frac{\tilde\beta_t}{\sigma_t^2}-1-\log\frac{\tilde\beta_t}{\sigma_t^2}\right)\right] \\
&= \begin{cases}
\frac{1}{2\tilde\beta_t}\|\tilde\mu_t-\mu_\theta\|^2 & \text{if } \sigma_t^2 = \tilde\beta_t \\
\frac{1}{2\sigma_t^2}\|\tilde\mu_t-\mu_\theta\|^2 + C_t &\text{if } \sigma_t^2 \neq \tilde\beta_t;\text{but with fixed }\sigma_t^2
 
\end{cases}
\end{align*}
$$


In DDPM, $\sigma_t^2$ is set to $\tilde\beta_t$ or $\beta_t$, thus the KL reduces to training the model mean to match the true posterior mean: 


$$
L_{t-1}=\mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\left\|\tilde\mu_t(x_t,x_0)-\mu_\theta(x_t,t)\right\|^2\right]+ C
$$


From above: $\tilde{\mu_t}(x_t; x_0) := \frac{\sqrt{\bar{\alpha}{t-1}}\beta_t}{1 - \bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha{t-1})}{1-\bar\alpha_t}x_t$, we substitute $x_0=\frac{1}{\sqrt{\bar\alpha_t}}\left(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon\right)$:


$$
\begin{equation}
\tilde\mu_t(x_t,x_0)=\frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right)
\end{equation}
$$


We want to predict $\tilde \mu_t$ with $\mu_\theta$ given $x_t$ and $t$. To do so, we parameterize the gaussian noise term by replacing the $\epsilon$ in equation 4 with neural prediction $\epsilon_\theta (x_t, t)$. Then:


$$
\begin{align*}
L_{t-1}&=\mathbb{E}_q\left[
\frac{1}{2\sigma_t^2}
\left\|
\frac{1}{\sqrt{\alpha_t}}
\left(
-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\right)
(\epsilon - \epsilon_\theta)
\right\|^2
\right]+ C \\
&=\mathbb{E}_q\left[
\frac{1}{2\sigma_t^2}

\frac{\beta_t^2}{\alpha_t(1-\bar\alpha_t)}
\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2

\right]+ C \tag{5} \\
\end{align*}
$$


We can see in Equation 5, the noise prediction loss is weighted by time-step-dependent constant. In DDPM, it is reduced to simpler form:


$$
L_{\text{simple}}=\mathbb{E}_{x_0,\epsilon,t}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right],
$$


It discards the weighting in Equation 5. It implicitly puts less focus on easy low-noise denoising steps and relatively more focus on hard high-noise steps, which tends to improve sample quality.


---


## Training and Inference


Training:


Repeat until converged:

1. Sample a clean data point: $x_0 \sim q(x_0)$
2. Sample a timestep uniformly: $t \sim \mathrm{Uniform}({1,\dots,T})$
3. Sample Gaussian noise: $\epsilon \sim \mathcal N(0, I)$
4. Construct the noisy version of the data at timestep $t$: $x_t = \sqrt{\bar{\alpha}_t},x_0 + \sqrt{1-\bar{\alpha}_t},\epsilon$
5. Take a gradient descent step on the loss: $\left| \epsilon - \epsilon_\theta(x_t, t) \right|^2$. Equivalently, the optimization step is on: $\nabla_\theta \left| \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,; t\right) \right|^2$

---


Sampling

1. Initialize from pure Gaussian noise: $x_T \sim \mathcal N(0, I)$
2. For $t = T, T-1, \dots, 1$:
    1. sample $z \sim \mathcal N(0,I) \quad \text{if } t>1, \qquad \text{else } z=0$
    2. compute the previous sample $x_{t-1} =
    \frac{1}{\sqrt{\alpha_t}}\left(x_t -
    \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}t}}\epsilon_\theta(x_t,t)\right)+\sigma_t z$

    ($x_{t-1} = \mu_\theta + \sigma_t z$)

3. Return $x_0$

---


## Comparison with VAE


A standard VAE has one latent variable $z$:


$$

\log p_\theta(x)
=
\mathcal L(\theta,\phi;x)
+
D_{KL}(q_\phi(z\mid x)\|p_\theta(z\mid x))

$$

- In a standard VAE, we want to match $q_\phi(z \mid x)$ (tractable encoder) to $p(z \mid x)$ (the intractable true posterior). We can only do this indirectly by maximizing the ELBO (thus minimizing the KL divergence).
- One bottleneck variable $z$ must summarize the whole sample. The decoder must generate everything from that single representation. Global structure and fine detail are all pushed through one latent layer: if the latent dimension is too small, information is lost; if the decoder is too strong, posterior collapse can happen

In contrast, the fixed forward process in diffusion models makes all KL terms computable in closed form

- $q$ is fixed, known and exact. Diffusion model then avoids learning an approximate inference network for the forward chain.
- not one latent, but a whole **hierarchy of progressively less noisy variables,** generation is refined gradually

## References


<span id="ref-1"></span>[1] Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).


<span id="ref-2"></span>[2] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. In Proceedings of the 34th International Conference on Neural Information Processing Systems (NIPS '20)


<span id="ref-3"></span>[3] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S.. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. Proceedings of the 32nd International Conference on Machine Learning


---
title: ML notes
date: 2026-02-08
type: note
tags:
  - ML
---


- [PCA](#pca)
- [EM algorithm](#em-algorithm)
- [VAE](#vae)
  - [True posterior $p_\theta(z |x)$ and evidence $p_\theta(x)$ are both intractable](#true-posterior-pthetaz-x-and-evidence-pthetax-are-both-intractable)
  - [ELBO (evidence lower-bound) of marginal log-likelihood:](#elbo-evidence-lower-bound-of-marginal-log-likelihood)
  - [Exact decomposition of Evidence to ELBO + KL](#exact-decomposition-of-evidence-to-elbo-kl)
  - [Optimizing ELBO](#optimizing-elbo)


## PCA

- We want to find w that maximizes the covariance after dimension reduction

    $$
     w = \argmax_w \frac{1}{n}\sum_{i=1}^n (w^Tx^i - w^T\mu)^2 \quad \text{where} \;\|{w}\| = 1 \\
    = w^T (\frac{1}{n} \sum_{i=1}^n (x^i-\mu)(x^i=\mu)^T)w\\
    L(w, \lambda) = w^TCw + \lambda (1-\|w\|^2) \quad \text{(using Langrangian function)} \\
    \text{Find minimum:} \quad \frac{\partial L}{\partial w} = 0 \rightarrow Cw = \lambda w
    $$

- Inference:
    - step 0: standardize the data (skip standardization only when you’re confident scales are comparable and variance magnitude is meaningful)
    - step 1: estimate mean $\mu$ and calculate covariance $C$ from data
    - step 2: calculate eigenvalues, eigenvectors of $C$, (solve $ \det(C-\lambda I) = 0$, where the equation becomes n-degree polynomial to solve for $\lambda_1, \lambda_2, ..., \lambda_n$)
    - step 3: projected with picked eigenvectors

    $$
    z^i = \begin{pmatrix} \nu^{1^{T}} (x^i - \mu) / \sqrt{\lambda_1} \\
    \nu^{2^{T}} (x^i - \mu) / \sqrt{\lambda_2} \\
    \cdot \\
    \nu^{k^{T}} (x^i - \mu) / \sqrt{\lambda_k} \\
    
    \end{pmatrix}
    $$


---


## EM algorithm


---


## VAE


Given a dataset $X$, we want to model the underlying distribution $p(x)$. A latent variable model assumes each observation $x$ is generated from a lower-dimensional latent code $z$: first sample $z \sim p(z)$, then generate $x \sim p_\theta(x|z)$. Here $p(z)$ is a fixed prior (typically $\mathcal{N}(0, I)$) and $p_\theta(x|z)$ is a learned conditional distribution parameterized by $\theta$.
For our training objective, we wish to maximize the marginal log-likelihood (evidence):


$$
\log p_\theta(x) = \log \int p_\theta(x, z)\,dz = \log \int p_\theta(x|z)p(z)\,dz.
$$


### True posterior $p_\theta(z |x)$ and evidence $p_\theta(x)$ are both intractable


By Bayes' theorem:


$$
p_\theta(z|x) = \frac{p_\theta(x|z)p(z)}{p_\theta(x)} = \frac{p_\theta(x|z)p(z)}{\int p_\theta(x|z)p(z)dz}
$$



The numerator is entirely tractable. The intractability comes entirely from the denominator (the marginal likelihood  $p_\theta(x)$ because

1. **There is no closed-form solution**
In a VAE, $p_\theta(x|z)$ is parameterized by DNNs. This means mapping $z$ to $x$ involves passing variables through layers of matmuls and nonlinear activations. Because of these non-linearities, the function $p_\theta(x|z)p(z)$ becomes extremely complex that we cannot simply write down a closed-form equation to solve the integral.
2. **Numerical intractability**
Since integral fails, one viable fallback is numerical integration: estimating the area under the curve using ,for example, Riemann sum. However, latent spaces are high-dimensional. If we want to estimate the integral by testing just $k$ points along each dimension, you would need to evaluate your neural network $k^{d}$ times for a data sample.

Because the denominator cannot be calculated analytically or numerically, the true posterior $p_\theta(z|x)$ remains permanently locked away, forcing us to use variational inference to approximate it. 


### ELBO (evidence lower-bound) of marginal log-likelihood:


notations:

- $\theta$: generative parameters (Decoder) - This model represents the assumption about how the world works: it takes a the latent variable $z$ and translates it into observable data / evidence $x$. Therefore, $\theta$ defines the likelihood $p_\theta(x|z)$ and the prior $p(z)$.
- $\phi$: Variational Parameters (Encoder)**:** Because the **true posterior** $p_\theta(x|z)$ **is intractable (explained below),** we use a approximate posterior to approximate. $\phi$ defines $q_\phi(z|x)$

$$
\begin{align*}
\log p_\theta(x) &= \log \int p_\theta(x, z) dz \\
&= \log \int q_\phi(z | x) \frac{p_\theta(x, z)}{q_\phi(z|x)} dz\\
&= \log \mathbb E_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right] \\
&\text{Applying Jensen's inequality: }\log \mathbb E_q[Y] \ge \mathbb E_q[\log Y] \\
&\ge
\mathbb E_{q_\phi(z\mid x)}
\left[
\log \frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right] = \boxed{
\mathcal L(\theta,\phi;x)}
\end{align*}
$$


Since log is strictly concave, equality in Jensen holds iff $Y$ is constant with respect to the expectation ($q_\phi(z\mid x)$). That is: 


$$
\frac{p_\theta(x,z)}{q_\phi(z\mid x)} = c \qquad\text{for } q_\phi(z\mid x)\text{-almost every } z  \\
p_\theta(x,z)= c \times q_\phi(z | x) \\
p_\theta(x)=c\int q_\phi(z\mid x)dz=c \\
p_\theta(x,z)=p_\theta(x)\,q_\phi(z\mid x) \\
\boxed{
q_\phi(z\mid x)=p_\theta(z\mid x)}
$$


When posterior of of $q$ equals $p$


### Exact decomposition of Evidence to ELBO + KL


$$
\begin{align*}
\log p_\theta (x) &= \int q_\phi (z | x) \log p_\theta(x) dz \\
&= \int q_\phi(z\mid x)\log\left(\frac{p_\theta(x,z)}{p_\theta(z\mid x)}\right)dz \\
&= \int q_\phi(z\mid x)\log\left(
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\cdot
\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}
\right)dz \\
&= \int q_\phi(z\mid x)\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz
+
\int q_\phi(z\mid x)\log\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}\,dz
\end{align*} \\
\boxed{
\log p_\theta(x)
=
\mathcal L(\theta,\phi;x)
+
D_{KL}(q_\phi(z\mid x)\|p_\theta(z\mid x))
}
$$


Implications:

- Evidence $\log p_\theta(x)$: represents how well the generative model $\theta$ explains the actual training data $x$, marginalizing (integrating) out all the hidden variables $z$. **The ultimate goal of any generative model is to maximize this evidence.** However, we cannot compute this directly because the integral over all possible $z$is intractable.
- KL Divergence $D_{KL}(q_\phi(z|x) \,\Vert\, p_\theta(z|x))$. Measures the approximation error between the approximate posterior $q_\phi(z|x)$ and the true, mathematically intractable posterior $p_\theta(z|x)$. It’s always nonnegative,  acts as a positive gap between the ELBO and the true marginal likelihood. In reality, we cannot compute this because $p_\theta(z|x)$ is intractable
- ELBO $\mathcal{L}(\theta,\phi;x)$: Since the KL divergence is always $\ge 0$, we can rewrite the main equation as an inequality: $\log p_\theta(x) \ge \mathcal{L}(\theta,\phi;x)$ → ELBO is a guaranteed lower bound on the evidence. The ELBO is **the actual objective function we use to train the VAE.** Unlike the marginal likelihood, the ELBO _is_ tractable. When we train a VAE, we are **maximizing the ELBO with respect to both** $\theta$ **and** $\phi$**.**
    - By maximizing the ELBO with respect to $\theta$, we are improving our decoder's ability to generate the data (pushing the ceiling up).
    - By maximizing the ELBO with respect to $\phi$, we are minimizing the KL divergence gap (pulling the floor closer to the ceiling), making our encoder a better approximation of the true posterior.

### Optimizing ELBO


$$
\mathcal{L}(\theta, \phi; x) 
            = \underbrace{\mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction term}}
            - \underbrace{D_{KL}(q_\phi(z|x) \,\Vert\, p(z))}_{\text{regularization term}}
$$

- The reconstruction term measures how well the generative parameters $\theta$ reconstruct the original data $x$ given a latent variable $z$ sampled from the encoder (approximate posterior) $q_\phi (z|x)$
- The regularization term forces the predicted posterior to be close to the chose prior $p(z)$. **Optimizing this term makes the latent space smooth, organized, and easy to sample from**, and nearby latent points can correspond to similar outputs. If we remove this term, one degenerate solution is for $q_\phi(z|x)$ to collapse to an almost deterministic point mass, i.e. a Gaussian with variance approaching zero. Then the model only learns to reconstruct each training example from its encoded latent code, without forcing those codes to match the prior $p(z)$. As a result, at test time, sampling $z \sim p(z)$ may produce latent vectors in regions the decoder was never trained on (dead area), so generated samples are poor even though reconstruction loss is low.

**The Training Loop:**

1. Pass data $x$ through the encoder to output the parameters of $q_\phi(z|x)$ (typically a mean vector $\mu$ and a variance vector $\sigma^2$).
2. Sample a latent vector $z$ from this distribution. To make this step differentiable so backpropagation works, we use the **reparameterization trick:**
    - In the original reconstruction term $\mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)]$, the expectation is over $z \sim q_\phi(z|x)$, and since $z$ is sampled from a distribution that depends on $\phi$, it is not possible to backpropagate through the sampling operation for optimizing $\phi$. The reparameterization trick rewrites the sample as

        $$
        z=\mu_\phi(x)+\sigma_\phi(x)\odot \epsilon,\qquad \epsilon\sim \mathcal{N}(0, I)
        
        $$


        $z$ becomes a differentiable function of $\phi$. 


        $$
        \mathbb E_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)]=\mathbb E_{\epsilon\sim \mathcal N(0,I)}[\log p_\theta(x\mid \mu_\phi(x)+\sigma_\phi(x)\odot \epsilon)]
        $$


        And because the distribution of $\epsilon$ does not depend on $\phi$, we can move $\nabla_\phi$ inside the expectation and compute gradients by standard backpropagation

3. Pass $z$ through the decoder to get the reconstruction $p_\theta(x|z)$.
4. Calculate the negative ELBO and backpropagate to update $\theta$ and $\phi$.

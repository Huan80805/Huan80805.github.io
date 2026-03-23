---
title: "GenAI Safety Assessment"
summary: "Safety and security evaluation for Taiwanese LLMs, covering safeguard modeling and automatic red-teaming."
date: 2025-06-01
status: research
tags:
  - Generative AI
  - LLM safety
  - red-teaming
featured: true
thumbnail: /assets/images/genaiSafety/thumbnail.jpeg
role: Research assistant
stack:
  - LLM
  - RL
  - Ray
  - Kubernetes
  - LLM-as-a-judge
  - Data collection & validation
links: []
---

## Overview

This project, funded by Taiwan's National Institute of Cyber Security, evaluates the safety and security of Taiwanese LLMs including TAIDE and Taiwan-LLM. The framework has two main components: building safeguard models for detecting harmful generations, and automatic red-teaming to probe model vulnerabilities.


<figure class="half">
    <img src="/assets/images/genaiSafety/pipeline.png"/>
    <figcaption>End-to-end project pipeline: safeguard model training and automatic red-teaming for Taiwanese LLMs.</figcaption>
</figure>


## Safeguard models

Open-source safeguard models like LlamaGuard and ShieldLM are insensitive to culturally specific taboos and local expressions in Taiwanese Chinese. To address this, we built a localized safeguard model trained on data from three sources:

- Toxic comments crawled from Taiwanese online forums, semi-auto labeled.
- Existing human-LLM conversation data translated into Traditional Chinese, from Anthropic hh-rlhf and ShieldLM training sets.
- LLM-generated responses to manually-designed and existing attack prompts, automatically labeled using ShieldLM, GPT-4, and LlamaGuard.

<figure class="half">
    <img src="/assets/images/genaiSafety/sg2.png"/>
    <figcaption>Human-LLM conversation data translated into Traditional Chinese, sourced from Anthropic hh-rlhf and ShieldLM training data.</figcaption>
</figure>

<figure class="half">
    <img src="/assets/images/genaiSafety/sg3.png"/>
    <figcaption>Automatically labeled LLM responses generated using manually-designed and existing attack prompts, labeled by ShieldLM, GPT-4, and LlamaGuard.</figcaption>
</figure>


Our model outperformed LlamaGuard by F1 +0.14 on flagging harmful generations from Taiwanese LLMs.

## Automatic red-teaming

**Black-box attacks:** We used RL to fine-tune an attacker model against a victim LLM with no gradient access. The safeguard model provides reward signals to guide the attacker toward more effective adversarial prompts over training.


<figure class="half">
    <img src="/assets/images/genaiSafety/red_black.png"/>
    <figcaption>Black-box red-teaming framework: RL-based attacker fine-tuned with safeguard model reward to generate adversarial prompts.</figcaption>
</figure>

**White-box attacks:** Following GCG, we optimize an adversarial suffix using the victim's gradients to force harmful outputs. We added a language modeling loss on top of GCG's original objective to generate more coherent and natural-sounding suffixes.

<figure class="half">
    <img src="/assets/images/genaiSafety/red_white.png"/>
    <figcaption>White-box red-teaming framework: GCG adversarial suffix optimization with additional language modeling loss for coherent attack generation.</figcaption>
</figure>

## Demo

<video src="/assets/images/genaiSafety/sg_demo.mov" controls class="half" style="width:80%"></video>

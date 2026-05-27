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
thumbnail: /assets/images/genaiSafety/thumbnail.png
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

This project, funded by Taiwan's National Institute of Cyber Security, evaluates the safety and security of Taiwanese LLMs including TAIDE and Taiwan-LLM. The framework has two connected tracks: localized safety evaluators for detecting harmful generations, and automatic red-teaming for finding model vulnerabilities.

<div class="half">
  <figure>
    <img src="/assets/images/genaiSafety/pipeline.png"/>
    <figcaption>Our data collection pipeline for training safety evaluator / toxicity detector in "Taiwanese"</figcaption>
  </figure>
</div>

## My contribution

- Built large-scale data collection, processing, and validation pipelines for localized safety evaluation data.
- Fine-tuned XLM-R as a localized toxicity and safety evaluator for Taiwanese Chinese LLM outputs.
- Developed black-box and white-box adversarial attack workflows to stress-test target LLMs.

## Safeguard models

Open-source safeguard models like LlamaGuard and ShieldLM can miss culturally specific taboos, local expressions, and Taiwanese Chinese usage patterns. To address this, we trained localized evaluators on data from three sources:

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

**Black-box attacks:** We used RL to fine-tune an attacker model against a victim LLM with no gradient access. The safeguard evaluator provided reward signals that pushed the attacker toward prompts more likely to expose harmful behavior.


<figure class="half">
    <img src="/assets/images/genaiSafety/red_black.png"/>
    <figcaption>Black-box red-teaming framework: RL-based attacker fine-tuned with safeguard model reward to generate adversarial prompts.</figcaption>
</figure>

**White-box attacks:** Following GCG, we optimized adversarial suffixes with access to the victim model's gradients. We added a language modeling loss on top of GCG's original objective so the generated suffixes were more coherent and natural-sounding.

<figure class="half">
    <img src="/assets/images/genaiSafety/red_white.png"/>
    <figcaption>White-box red-teaming framework: GCG adversarial suffix optimization with additional language modeling loss for coherent attack generation.</figcaption>
</figure>

## Demo

The deployed internal demo is no longer accessible, but the recording below shows how evaluator outputs were exposed for inspection during the project.

<video src="/assets/images/genaiSafety/sg_demo.mov" controls class="half" style="width:80%"></video>

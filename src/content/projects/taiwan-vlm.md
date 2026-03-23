---
title: "Taiwan Visual-Language Model 70B"
summary: "Continual pre-training of ViT-L + Llama3-Taiwan-70B on 50M images and 3B text tokens grounded in Taiwanese culture."
date: 2025-05-01
status: ongoing
tags:
  - Generative AI
  - multimodal
  - large-scale pre-training
featured: true
thumbnail: /assets/images/twvlm/llama-in-tw.jpeg
role: Research assistant
stack:
  - VLM
  - multimodal pre-training
  - Megatron-LM
  - NemoCurator
  - DeepSpeed
links: []
---

## Overview

We continually pre-trained a multimodal system grounded in Taiwanese culture using ViT-L as the vision encoder and Llama3-Taiwan-70B as the language backbone. Pre-training data includes 50M image-text pairs and image-text-interleaved entries totaling 3B text tokens. The setup follows LLaVA-style architecture: image features from the vision encoder are projected and concatenated with language embeddings before being passed to the LLM.

## Key finding

During development, we found the model maintains instruction-following behavior even without explicit instruction tuning. We attribute this to Llama3-Taiwan's inherent instruction-aligned nature from its pretraining — the visual projection layer was sufficient to activate this capability on image-conditioned inputs.

## Demo conversations

The model can recognize objects and answer questions about images. Below are some example outputs.

<figure class="half">
  <img src="/assets/images/twvlm/bubble-tea.png" alt="Model output for bubble tea image" />
  <figcaption>Q: Where does this food come from? &mdash; A: Bubble tea is Taiwan's popular food.</figcaption>
</figure>

<figure class="half">
  <img src="/assets/images/twvlm/castle.png" alt="Model output for castle image" />
  <figcaption>Q: Describe what's in the image. &mdash; A: A large castle is covered in snow as seen from a tree.</figcaption>
</figure>

<figure class="half">
  <img src="/assets/images/twvlm/notre-dame.png" alt="Model output for Notre Dame" />
  <figcaption>Q: What's this building? &mdash; A: Notre Dame de Paris.</figcaption>
</figure>

<figure class="half">
  <img src="/assets/images/twvlm/waterloo.png" alt="Model output for University of Waterloo logo" />
  <figcaption>Q: Where can I see this logo? &mdash; A: University of Waterloo, Ontario, Canada.</figcaption>
</figure>

## Current direction

Next steps include visual instruction tuning on Taiwan-specific QA and captioning data, along with more exhaustive evaluation across vision-language benchmarks and culturally grounded tasks.

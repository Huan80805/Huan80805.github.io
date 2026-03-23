---
title: "Speech Bot"
summary: "A Jetson Nano robot combining lightweight ASR, OpenAI conversation API, TTS, and hand pose detection for multimodal interaction."
date: 2022-12-01
status: course-project
tags:
  - speech processing
  - edge device
featured: false
thumbnail: /assets/images/speechbot/product.png
role: Course project team member
stack:
  - Jetson Nano
  - TensorRT
  - OpenAI API
  - TTS
  - hand pose detection
links:
  - label: "Code"
    href: "https://github.com/Huan80805/2022NMlab_final/tree/master"
  - label: "Demo"
    href: "https://www.youtube.com/embed/asU2DoZuJrI?si=8hrunGnrILh1saiF"
---

## Overview

Speech Bot is a voice and gesture-controlled robot built on Jetson Nano 2GB. It combines a lightweight ASR model (optimized with TensorRT), OpenAI's conversation API, Google TTS, and hand pose detection to create an interactive assistant that can chat, play music from YouTube, check live weather, and tell time. The system supports both Mandarin Chinese and English commands.

<figure>
  <img src="/assets/images/speechbot/architecture.jpg" alt="System flowchart" class="half"/>
  <figcaption>System flowchart: voice input, ASR, OpenAI conversation API, TTS output, and hand pose detection modules running on Jetson Nano.</figcaption>
</figure>

## Interaction modes

**Speech interaction:** The robot converts voice to text via ASR, processes requests through OpenAI's conversation API, and responds with text-to-speech synthesis.

**Music control via gestures:** When music is playing and voice detection is unreliable, six hand gestures provide an alternative: peace (pause), pan (play), stop (end playback), fist and OK (volume up/down).

<figure class="half">
  <img src="/assets/images/speechbot/handpose.png" alt="Hand gesture controls" />
  <figcaption>Supported hand gestures for music control: peace (pause), pan (play), stop (end playback), fist and OK (volume adjustment).</figcaption>
</figure>

**Smart assistant functions:** Mentioning keywords like "天氣" or "weather" triggers the weather module, after which the user can specify any city for live web-crawled data.

## Demo

<iframe src="https://www.youtube.com/embed/asU2DoZuJrI?si=8hrunGnrILh1saiF" title="Speech Bot demo" allowfullscreen style="width:70%"></iframe>

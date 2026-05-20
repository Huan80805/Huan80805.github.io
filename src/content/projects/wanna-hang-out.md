---
title: "Wanna Hang Out?"
summary: "A social planning platform that replaces messy group chats with structured event subscriptions and real-time updates."
date: 2022-06-01
status: course-project
tags:
  - web
featured: false
draft: true
thumbnail: /assets/images/wp2022/thumbnail.png
role: Course project team member
stack:
  - React
  - Node.js
  - GraphQL
  - MongoDB
links:
  - label: "Code"
    href: "https://github.com/Huan80805/wp1101_proj/tree/main"
---

## Overview

Wanna Hang Out? is a social planning platform designed to address information overload in traditional group chats. On Facebook groups or LINE chats, important event updates get buried in noise. This platform solves that by giving each event its own dedicated space: an info page, a registration list, and a separate chatroom.

## Features

**Event subscriptions:** Users follow specific events or clubs they care about, filtering out unrelated activity.

**Structured organization:** Clubs, events, and conversations are separated by hierarchy — club announcements, event pages, and event-specific chat are distinct layers.

**Real-time updates:** Changes propagate instantly via GraphQL subscriptions so attendees always see the latest state.

<figure class="half">
  <img src="/assets/images/wp2022/clubMenu.png" alt="Club menu" />
  <figcaption>Club menu: users can enter an existing club, join a new one, or create their own.</figcaption>
</figure>

<figure class="half">
  <img src="/assets/images/wp2022/lobbyChat.png" alt="Club lobby" />
  <figcaption>Club lobby: join a chatroom, create or join an event from one place.</figcaption>
</figure>

<figure class="half">
  <img src="/assets/images/wp2022/eventChat.png" alt="Event page" />
  <figcaption>Event page: see registered attendees and open a dedicated event chatroom.</figcaption>
</figure>

## Stack

Frontend in React with GraphQL queries and subscriptions. Backend with Node.js, Express, GraphQL Yoga, MongoDB for persistence, JWT for auth, and bcryptjs for password handling.

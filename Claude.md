# CLAUDE.md — Namu Tambaya

Read NAMU_CONTEXT.md before doing anything in this project.

## What this project is
Namu Tambaya is a Hausa-language AI voice agent for Niger.
Users call a phone number, speak Hausa, and get spoken answers in Hausa.
No internet or smartphone required for end users.
Founded by two native Hausa speakers from Niger.

## Current Phase
Phase: 1
Session Goal: Set up complete Docker infrastructure with FastAPI, PostgreSQL, and Nginx

## Non-Negotiables — never violate these
1. All user audio stays on our infrastructure — no external STT API
2. No full call transcripts stored — metadata only
3. Response latency must be under 5 seconds
4. Health agent always recommends professional help for serious conditions
5. All responses must sound natural to a Niger Hausa speaker
6. Agents always acknowledge honestly when they do not know something
7. Privacy by default — user data never leaves our infrastructure except generated text to ElevenLabs

## Tech Stack
- Backend: Python 3.11 + FastAPI
- STT: faster-whisper Large-V3 self-hosted
- LLM: Gemma 4 via Ollama self-hosted
- TTS: ElevenLabs API with custom Niger Hausa female voice
- Telephony: Africa's Talking Voice API
- Vector DB: ChromaDB self-hosted
- Database: PostgreSQL
- Infrastructure: Docker + Docker Compose + Nginx
- Monitoring: Grafana + Prometheus
- GPU Hosting: RunPod

## How to work with me
- Read NAMU_CONTEXT.md for full context before every action
- Before writing any code, explain your approach and folder structure
- One phase at a time — never jump ahead
- Explain every major library or pattern choice and why
- After building any feature, write tests for it and run them
- If something has multiple approaches, show options and recommend one with reasoning
- Always optimize for latency — target under 5 seconds end to end
- Always optimize for reliability over complexity
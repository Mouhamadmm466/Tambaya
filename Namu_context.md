# NAMU_CONTEXT.md
# Master Context Document for Claude Code Sessions
# Last Updated: April 2026

---

## 1. PROJECT IDENTITY

**Product Name:** Namu Tambaya
**Organization:** Namu
**Mission:** Building systems and tools for African communities in their own language and cultural context.

**What Namu Tambaya is:**
An AI-powered voice agent accessible by phone call, designed for Hausa-speaking communities
across West Africa — starting in Niger. Users call a single phone number, speak naturally 
in Hausa, and receive spoken answers in Hausa. No internet, no smartphone, no literacy required.

**Target Users:**
- Rural populations with no internet access
- Women who cannot easily travel to seek advice
- Elderly people unfamiliar with smartphones
- Young people in areas with no libraries or schools
- Small traders making decisions without market data
- First-generation phone owners

**Starting Community:** Hausa-speaking population of Niger
**Language:** Hausa — Niger dialect. No translation layer. The system understands 
and responds natively in Hausa.
**Founders:** From Niger. Native Hausa speakers. First language is Hausa.

---

## 2. HOW A CALL WORKS END TO END

User calls number →
Africa's Talking receives call →
Audio streamed to our server →
faster-whisper Large-V3 transcribes Hausa speech to text →
Router Agent (Gemma 4) classifies the question →
Specialized Agent (Gemma 4) retrieves from ChromaDB knowledge base →
Agent generates answer in Hausa →
ElevenLabs converts answer to Hausa voice (custom Niger female voice) →
Audio streamed back to caller

---

## 3. AGENT STRUCTURE

**Router Agent**
Classifies every incoming question into one of five categories:
Health | Agriculture | Education | General Knowledge | Unclear
Does not answer anything. Only routes. Fast and focused.

**Health Agent**
Specialized in health questions relevant to Niger and the Sahel region.
Always recommends professional medical help for serious symptoms.
Never replaces a doctor.

**Agriculture Agent**
Specialized in farming, crops, planting seasons, and market prices in Niger.
Knowledge base includes Niger farming calendar, Sahel crop advice, regional market data.

**Education Agent**
Specialized in education, literacy resources, and school enrollment in Niger.
Knowledge base includes Niger curriculum and enrollment processes.

**General Knowledge Agent**
Handles everything that does not fit the other three domains.
Default fallback when Router confidence is low.

---

## 4. FALLBACK RULES — NON NEGOTIABLE

- Router confidence low → route to General Knowledge agent
- Agent cannot answer reliably → say so honestly in Hausa, suggest alternative help
- Whisper transcription quality too low → ask caller politely to repeat
- Health agent receives serious symptom question → always recommend seeing a doctor
- System overloaded → graceful Hausa message, never silence or error
- During processing delay → play natural Hausa acknowledgment sound immediately

---

## 5. FULL TECH STACK

| Layer | Technology | Reason |
|---|---|---|
| Telephony | Africa's Talking Voice API | African coverage, Niger support |
| Backend | Python 3.11 + FastAPI | ML ecosystem, async, fast |
| Speech to Text | faster-whisper Large-V3 self-hosted | Privacy, best Hausa support, 4x faster than original Whisper |
| Model Serving | Ollama + Gemma 4 self-hosted | Data sovereignty, open source, fine-tunable |
| Agent Architecture | Router + 4 specialized agents | Accuracy, trustworthiness, scalability |
| Knowledge Base | RAG with ChromaDB self-hosted | Current, local, specific knowledge |
| Text to Speech | ElevenLabs API custom voice | Natural Niger female Hausa voice |
| Database | PostgreSQL | Reliable, structured |
| Containerization | Docker + Docker Compose | Consistent, portable, scalable |
| Reverse Proxy | Nginx | SSL, routing, production ready |
| Monitoring | Grafana + Prometheus | Real time visibility |
| Version Control | GitHub | Code safety |
| GPU Hosting | RunPod Nvidia A40 | Cost effective GPU for Whisper and Gemma |
| App Hosting | Hetzner VPS | Cost effective, reliable, low latency to West Africa |

---

## 6. FOLDER STRUCTURE
namu-tambaya/
├── CLAUDE.md
├── NAMU_CONTEXT.md
├── .env
├── .gitignore
├── docker-compose.yml
├── nginx/
│   └── nginx.conf
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   ├── telephony.py
│   │   ├── transcription.py
│   │   ├── agents.py
│   │   └── health_check.py
│   ├── agents/
│   │   ├── router_agent.py
│   │   ├── health_agent.py
│   │   ├── agriculture_agent.py
│   │   ├── education_agent.py
│   │   └── general_agent.py
│   ├── knowledge_base/
│   │   ├── health/
│   │   ├── agriculture/
│   │   ├── education/
│   │   └── general/
│   ├── services/
│   │   ├── whisper_service.py
│   │   ├── elevenlabs_service.py
│   │   ├── ollama_service.py
│   │   └── chromadb_service.py
│   └── database/
│       ├── models.py
│       └── connection.py
├── whisper/
│   └── Dockerfile
└── monitoring/
├── prometheus.yml
└── grafana/
---

## 7. BUILD PHASES AND CURRENT STATUS

| Phase | Description | Status |
|---|---|---|
| 0 | Preparation — accounts, Whisper testing, voice recording | ✅ Complete |
| 1 | Infrastructure — Docker, FastAPI, PostgreSQL, Nginx | 🔄 In Progress |
| 2 | Telephony — Africa's Talking integration | ⬜ Not Started |
| 3 | Speech to Text — faster-whisper self-hosted | ✅ Complete |
| 4 | Router Agent — Gemma 4 via Ollama | 🔄 In Progress |

CURRENT PHASE: 4
CURRENT SESSION GOAL: Build the Router Agent using Gemma 4 via Ollama to classify Hausa questions
| 5 | Agriculture Agent + ElevenLabs + Full call loop | ⬜ Not Started |
| 6 | Health, Education, General Knowledge agents | ⬜ Not Started |
| 7 | Cultural layer and voice personality | ⬜ Not Started |
| 8 | Monitoring, security audit, pilot launch | ⬜ Not Started |

**CURRENT PHASE: 1**
**CURRENT SESSION GOAL: Set up complete Docker infrastructure with FastAPI, PostgreSQL, and Nginx**

---

## 8. DECISIONS LOG

| Decision | Choice | Reason | Date |
|---|---|---|---|
| Backend language | Python 3.11 + FastAPI | ML ecosystem, async support | April 2026 |
| STT | faster-whisper Large-V3 self-hosted | Privacy, best Hausa support, speed | April 2026 |
| LLM | Gemma 4 via Ollama self-hosted | Data sovereignty, open source, fine-tunable | April 2026 |
| TTS | ElevenLabs custom voice | Best natural voice quality for Hausa | April 2026 |
| Telephony | Africa's Talking | African coverage, Niger support | April 2026 |
| Vector DB | ChromaDB self-hosted | Simple, open source, privacy | April 2026 |
| Database | PostgreSQL | Reliable, well-supported | April 2026 |
| No news agent | Removed from scope | Political risk, editorial complexity | April 2026 |
| No full transcript storage | Metadata only | User privacy and trust | April 2026 |
| Starting market | Niger Hausa community | Founders home community | April 2026 |
| GPU hosting | RunPod Nvidia A40 | Cost effective for current stage | April 2026 |
| App hosting | Hetzner VPS | Cost effective, reliable | April 2026 |
| Monitoring | Grafana + Prometheus | Industry standard, self-hosted | April 2026 |

---

## 9. ENVIRONMENT VARIABLES

All keys live in .env file — never committed to GitHub.
AT_API_KEY=
AT_USERNAME=
AT_PHONE_NUMBER=
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=
POSTGRES_DB=namu_tambaya
POSTGRES_USER=namu_user
POSTGRES_PASSWORD=
DATABASE_URL=postgresql://namu_user:password@db:5432/namu_tambaya
OLLAMA_BASE_URL=http://ollama:11434
GEMMA_MODEL_NAME=gemma4
WHISPER_MODEL_SIZE=large-v3
WHISPER_LANGUAGE=ha
APP_ENV=development
SECRET_KEY=
---

## 10. NON-NEGOTIABLE PRINCIPLES

1. Data sovereignty — all user audio and conversations stay on our infrastructure
2. Privacy by default — no full transcripts stored without explicit user consent
3. Honest fallback — agents always acknowledge when they do not know something
4. Health safety — health agent always recommends professional help for serious conditions
5. Cultural authenticity — all responses must sound natural to a Niger Hausa speaker
6. Latency target — under 5 seconds response time, always optimize for this
7. Reliability over features — a simple thing that works beats a complex thing that fails

---

## 11. TESTING CHECKLIST PER PHASE

- [ ] Phase 1: docker-compose up runs with no errors. Health check endpoint returns 200.
- [ ] Phase 2: Real test call received by server. Static audio plays back to caller.
- [ ] Phase 3: Real Hausa sentence transcribed correctly. Logged on server.
- [ ] Phase 4: 50 test questions classified correctly at 90%+ accuracy.
- [ ] Phase 5: Full call loop works. Agriculture question answered in under 5 seconds.
- [ ] Phase 6: All four agents answer correctly across their domains.
- [ ] Phase 7: 5 native Niger Hausa speakers approve the voice and tone.
- [ ] Phase 8: Pilot with 20 to 50 real users in Niger completed with feedback collected.

---

## 12. HOW TO START EVERY CLAUDE CODE SESSION

Always begin with:

"Read CLAUDE.md and NAMU_CONTEXT.md. We are in Phase [X].
Today's goal is [specific goal]. Before any code, explain your approach."

---

*This document is the single source of truth for Namu Tambaya.
Update CURRENT PHASE and STATUS after every session.
Never start a Claude Code session without this document in the project folder.*
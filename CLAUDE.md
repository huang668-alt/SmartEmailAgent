# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmartEmailAgent is an AI-powered Gmail assistant that fetches emails, analyzes them with LLMs, and stores vector embeddings in Milvus for semantic retrieval (RAG). The project is under active early-stage development — `main.py` is a placeholder and the orchestrator agent is a stub.

## Commands

```bash
# Run the manual email access test (requires valid credentials.json and OAuth setup)
python test/email_access_and_synchronization_module_test.py

# Run the main entry point (currently a placeholder)
python main.py
```

No build, lint, or test framework is configured yet. There is no `requirements.txt` or `pyproject.toml` — dependencies must be installed manually.

## Architecture

The system has four modules in `rag_modules/`, all wired together by `VectorStorageAndMemorySystemModule`:

### `config.py`
`SmartEmailAgentConfig` dataclass holds all configuration (Milvus URL/dim, LLM model name/URL/API key, embedding model name/URL/API key/dim, temperature, top_k). Fields default to empty strings — real values must be set at runtime. Use `from_dict()` to populate from a config file or env vars.

### `EmailAccessAndSynchronizationModule` (`rag_modules/email_access_and_synchronization_module.py`)
Gmail OAuth2 authentication and email fetching. Key behaviors:
- OAuth2 with `google-auth-oauthlib`; token cached in `token.pickle`, credentials from `credentials.json`
- `authenticate()` handles token refresh and interactive OAuth flow
- `fetch_new_emails()` uses `last_sync_time` for incremental pulls (1-day safety overlap)
- `parse_email()` handles multipart MIME: extracts headers, plain-text body, and attachment files (saved to `attachments/` directory)
- `fetch_new_emails_sent()` is the sent-mail variant

### `AiAnalysisCoreModule` (`rag_modules/ai_analysis_core_module.py`)
LLM-powered email analysis using LangChain's `ChatOpenAI` (OpenAI-compatible API). Contains five agents:
- **`summarizer_agent`** — 3-5 sentence summary + action items extraction
- **`reply_agent`** — drafts replies with configurable tone/length/language, outputs JSON with subject and body
- **`classifier_agent`** — assigns priority (High/Medium/Low/Spam) with one-sentence reasoning, outputs JSON
- **`task_extractor_agent`** — extracts tasks (action/assignee/deadline) and meetings (topic/time/location), outputs JSON
- **`context_summary_agent`** — compresses raw context into structured, entity-focused memory for long-term vector storage

`orchestrator_agent()` exists as a stub. `_summarizer_agent_attachments()` reads attachment files but the function body is incomplete — it reads file content without returning or processing it.

### `MilvusConnectionModule` (`rag_modules/milvus_connection_module.py`)
Thin wrapper around `pymilvus.MilvusClient`. `connection()` connects, `close()` disconnects.

### `VectorStorageAndMemorySystemModule` (`rag_modules/vector_storage_and_memory_system_module.py`)
The orchestration layer. Creates three Milvus collections:
- `accept_email_collection` — received emails with 512-dim vectors, priority, classification reason
- `send_email_collection` — sent emails with vectors (for reply tone analysis)
- `context_historical` — compressed long-term memory for semantic context retrieval

Pipeline in `store_historical_emails()`: auth → fetch inbox → parse → summarize via LLM → embed → classify priority → insert into Milvus. All vector indexing uses HNSW with COSINE similarity.

### Key module dependency graph

```
VectorStorageAndMemorySystemModule
  ├── EmailAccessAndSynchronizationModule (Gmail access)
  ├── MilvusConnectionModule (vector DB)
  └── AiAnalysisCoreModule (LLM analysis agents)
```

## Authentication & Secrets

- **Gmail OAuth2**: Requires `credentials.json` (Google Cloud OAuth client) in the working directory. Token refreshed automatically and stored in `token.pickle`.
- **LLM API**: Configured via `SmartEmailAgentConfig` fields (`summarizer_agent_module_api_key`, `embeddings_api_key`). Uses `pydantic.SecretStr` for API key storage.
- Do not commit `credentials.json`, `token.pickle`, or any file containing API keys.
- The test directory contains a separate `test/credentials.json` (tracked in git, currently deleted in working tree).

## Vector Schema

All collections use 512-dimension float vectors with HNSW/COSINE indexing. Email collections store full metadata (id, threadId, subject, from/to, date, body, attachments as JSON). The context collection stores compressed memory content with timestamps.

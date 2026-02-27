# Rotunda (sandbox4) - Local Setup

This folder documents how to run the **Rotunda** sandbox end-to-end:

- a Rotunda backend (Tornado) that serves the UI + API
- a local OpenAI-compatible **LLM server**
- a local **FHIR server** (HAPI FHIR)

The two external services are configured via the Rotunda CLI flags:

```bash
--LLM_port 30000
--FHIR_port 9090
```

## Architecture / Ports

Default local ports used by this sandbox:

- Rotunda backend (Tornado): `http://localhost:8888`
  - UI: `http://localhost:8888/ui/`
  - API: `http://localhost:8888/<endpoint>`
- LLM server (OpenAI-compatible): `http://localhost:30000/v1`
- FHIR server (HAPI FHIR): `http://localhost:9090/fhir/`

Notes:

- The backend uses the LLM server via the OpenAI Python SDK (`base_url=.../v1`).
- The backend bakes the FHIR base URL into the system prompt so the agent can
  generate tool calls like `GET http://localhost:9090/fhir/...`.

## Rotunda CLI Flags

### `--LLM_port 30000`

The TCP port where an **OpenAI-compatible chat completion server** is listening.

- Used to build the OpenAI base URL: `http://localhost:<LLM_port>/v1`
- Default: `30000`

### `--FHIR_port 9090`

The TCP port where the **FHIR server** is listening.

- Used to build the FHIR base URL: `http://localhost:<FHIR_port>/fhir/`
- Default: `9090`

## Quickstart (Run Everything)

### 1) Start a FHIR server on port 9090

Fastest option is to run HAPI FHIR via Docker and map host `9090 -> container 8080`:

```bash
docker run --rm -p 9090:8080 hapiproject/hapi:latest
```

Verify:

```bash
curl -sS http://localhost:9090/fhir/metadata | head
```

Alternative (using the repo copy under `hapi-fhir-jpaserver-starter/`):

1. Edit `hapi-fhir-jpaserver-starter/docker-compose.yml` and change the ports line
   from `8080:8080` to `9090:8080`
2. Run:

```bash
cd hapi-fhir-jpaserver-starter
docker compose up --build
```

Alternative (clone HAPI FHIR starter and run with Maven):

1. Clone the HAPI FHIR JPA Server Starter repository
2. Run:

```bash
mvn -Pjetty spring-boot:run -Dspring-boot.run.arguments="--server.port=9090"
```


### 2) Start an OpenAI-compatible LLM server on port 30000

This repo includes example `sglang` commands in `sglang/start.sh`.

Example (one model on port 30000):

```bash
python -m sglang.launch_server --model openai/gpt-oss-20b --port 30000
```

Verify:

```bash
curl -sS http://localhost:30000/v1/models | head
```

If you are using a different model ID than Rotunda expects, you may need to
update the `model=...` string in `sandbox4/back_end/util.py` to match what your
server returns from `/v1/models`.

### 3) Start Rotunda (backend + UI)

From the repo root:

```bash
cd sandbox4

# Optional: use the checked-in venv (if it exists on your machine)
source .venv/bin/activate

# If you don't have deps installed yet:
python -m pip install tornado openai requests tqdm

# Run the Tornado backend and point it at the two local services
python back_end/rotunda.py --LLM_port 30000 --FHIR_port 9090
```

Open the UI:

- `http://localhost:8888/ui/`

## Compliance Filters

Rotunda supports multiple compliance filters (see `src/back_end/filters/`), selected at runtime via:

```bash
ROTUNDA_FILTER=context        # default (local LLM on localhost)
ROTUNDA_FILTER=strict         # local LLM, stricter prompt
ROTUNDA_FILTER=trivial        # allow all tool calls
ROTUNDA_FILTER=compiled_law   # GPT-5.2 + dune-sandbox compiled laws
```

For the `compiled_law` filter you must also provide an API key, and optionally a model:

```bash
export OPENAI_API_KEY=...
export LAW_FILTER_MODEL=gpt-5.2     # default
export LAW_FILTER_MAX_CHARS_PER_CHUNK=8000
```

Example:

```bash
OPENAI_API_KEY=... ROTUNDA_FILTER=compiled_law python back_end/rotunda.py --LLM_port 30000 --FHIR_port 9090
```

## Minimal API Smoke Test (Optional)

Register + login:

```bash
curl -sS -X POST "http://localhost:8888/account/register" \
  -d "pwd=demo" | jq .

# Use the returned uid (or provide your own uid in the register call)
curl -sS -X POST "http://localhost:8888/account/login" \
  -d "uid=<UID>" -d "pwd=demo" | jq .
```

Create a room (needs `uid` + `token`):

```bash
curl -sS -X POST "http://localhost:8888/room/create" \
  -d "uid=<UID>" -d "token=<TOKEN>" -d "name=demo room" | jq .
```

## Troubleshooting

- `Connection refused` to `localhost:30000`:
  - your LLM server is not running, or is bound to a different port
  - start it, or run Rotunda with `--LLM_port <port>`
- `Connection refused` to `localhost:9090`:
  - your FHIR server is not running, or is bound to a different port
  - start it, or run Rotunda with `--FHIR_port <port>`
- UI loads but API calls fail:
  - check the UI "Base URL" field (defaults to `http://localhost:8888`)
  - confirm the backend is running and reachable on that port

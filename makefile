# Browser Agent — common tasks. Run inside the activated venv (so `python` is the venv's).
# On Windows without GNU make, just run the commands underneath each target by hand.

PY ?= python

.PHONY: install dev test ui server agent eval eval-planner eval-reasoner eval-grounding eval-orch eval-all help

help:
	@echo "install        editable install with dev deps (pytest)"
	@echo "test           run the deterministic pytest suite"
	@echo "server         run the WebSocket backend on :8000"
	@echo "ui             install + run the React dev-ui on :5173"
	@echo "agent          run the two-tier agent: make agent GOAL='...'"
	@echo "eval           run the flat-agent eval set (live: real Groq + browser)"
	@echo "eval-planner   run the planner eval: 50 ambiguous goals (live: real Groq)"
	@echo "eval-reasoner  run the reasoner eval: 50 fixtures (live: real Groq)"
	@echo "eval-grounding run the grounding eval: 50 cases (deterministic; needs Node)"
	@echo "eval-orch      run the orchestrator eval: 50 control-flow scenarios (deterministic)"
	@echo "eval-all       run all four component eval sets"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

server:
	$(PY) -m uvicorn browser_agent.server.app:app --port 8000 --reload

ui:
	cd dev-ui && npm install && npm run dev

agent:
	$(PY) -m browser_agent.agent.main "$(GOAL)"

eval:
	$(PY) scripts/eval_set.py

eval-planner:
	$(PY) scripts/eval_planner.py

eval-reasoner:
	$(PY) scripts/eval_reasoner.py

eval-grounding:
	$(PY) scripts/eval_grounding.py

eval-orch:
	$(PY) scripts/eval_orchestrator.py

eval-all: eval-grounding eval-orch eval-planner eval-reasoner

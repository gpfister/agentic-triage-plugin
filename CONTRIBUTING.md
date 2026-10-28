# Contributing to agentic-triage-plugin

Thank you for contributing to **`agentic-triage-plugin`**! This plugin provides dynamic System 1 prompt triage for Google Antigravity.

---

## 1. Core Guidelines
- Adhere to the standards defined in [`rules/AGENTS.md`](file:///Users/gpfister/Development/Projects/gpfister/agentic/agentic-triage-plugin/rules/AGENTS.md).
- Keep triage non-blocking and fail-safe (hooks must never crash the parent agent execution loop).
- Ensure dynamic evaluation properly detects installed plugins and adjusts criteria.

---

## 2. Directory Structure

```text
agentic-triage-plugin/
├── plugin.json         # Plugin manifest
├── hooks.json          # PreInvocation hook definition
├── pyproject.toml      # Project configuration and uv dependencies
├── README.md           # Documentation
├── LICENSE             # MIT License
├── CONTRIBUTING.md     # Contribution guidelines
├── hooks/              # Scripted lifecycle hooks executed via uv run
│   └── prompt_triage.py # Dynamic prompt triage implementation
├── rules/              # Plugin-scoped rules
│   └── AGENTS.md
└── tests/              # Test suite
    ├── test_plugin.py
    └── test_prompt_triage.py
```

---

## 3. Testing
Run tests using `uv`:

```bash
uv run pytest
```

Validate plugin structure with `agy`:

```bash
agy plugin validate .
```

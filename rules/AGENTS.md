# Prompt Triage Standards & Guidelines — agentic-triage-plugin

This rule set applies whenever the `agentic-triage-plugin` is active in an Antigravity session.

---

## 1. Triage Architecture & Principles
- **Fast Non-Autoregressive Classification**: Prompt triage uses the Laya Router System 1 model to classify user intent within a single fast forward pass.
- **Dynamic Plugin-Aware Routing**:
  - Evaluate if it is a code change request only if `agentic-coding-plugin` is installed.
  - Evaluate if it is an issue investigation request only if `agentic-investigate-plugin` is installed.
  - Evaluate if it is a task monitoring request only if `agentic-monitoring-plugin` is installed.
  - Evaluate if it is a topic research request only if `agentic-research-plugin` is installed.
  - Evaluate if it is a documentation request only if `agentic-obsidian-plugin` is installed.
  - Inquiries involving no changes are always supported.
  - If a requested capability is not supported by installed plugins, respond directly with guidance.

---

## 2. Hook Execution Safety
- **Single-Fire Guard**: Only triage the user's initial prompt (`invocationNum == 0`). Do not re-fire during multi-step tool iterations (`invocationNum > 0`).
- **Sanitization**: Strip XML wrappers, `<USER_REQUEST>`, `<ADDITIONAL_METADATA>`, and trailing file attachments before classifying intent.
- **Fail-Safe Operation**: If Laya Router is unavailable, fall back seamlessly to deterministic heuristic intent evaluation.

#!/usr/bin/env python3
"""Prompt Triage Lifecycle Hook for Antigravity.

Triages incoming user prompts dynamically using Laya Router System 1 model based on installed plugins:
- Evaluates code change requests if agentic-coding-plugin is installed
- Evaluates issue investigation requests if agentic-investigate-plugin is installed
- Evaluates task monitoring requests if agentic-monitoring-plugin is installed
- Evaluates topic research requests if agentic-research-plugin is installed
- Evaluates documentation requests if agentic-obsidian-plugin is installed
- Evaluates inquiries involving no changes (always in scope)
- Detects mixed prompts across active capabilities

If one response has confidence >= 0.4, directly triggers the corresponding triage action:
- inquiries: main agent responds directly to prompt (no changes directive)
- coding: main agent invokes @coding subagent using invoke_subagent
- monitoring: main agent invokes @monitoring subagent using invoke_subagent
- investigation: main agent invokes @investigate subagent using invoke_subagent
- research: main agent invokes @research subagent using invoke_subagent
- documentation: main agent invokes @coding-document subagent using invoke_subagent
- mixed: main agent rejects prompt (cannot process mixed requests)

Otherwise (confidence < 0.4), instructs the main agent to ask the user via ask_question.
All triage instructions and Laya Router question metadata are passed via systemMessage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set
from pydantic import BaseModel, Field

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.40
TRIAGE_QUESTION = "What is this prompt about ?"

KNOWN_PLUGINS = {
    "agentic-coding-plugin",
    "agentic-investigate-plugin",
    "agentic-monitoring-plugin",
    "agentic-obsidian-plugin",
    "agentic-research-plugin",
}


class DynamicTriageSpec(NamedTuple):
    responses: dict[str, str]
    response_to_intent: dict[str, str]
    intent_to_response: dict[str, str]
    response_to_directive: dict[str, str]


def detect_installed_plugins(
    workspace_paths: Optional[list[str]] = None,
    custom_plugins: Optional[list[str] | set[str]] = None,
) -> set[str]:
    """Detect which agentic capability plugins are currently installed.

    Detection sources:
    1. custom_plugins (if explicitly provided, e.g. for testing)
    2. INSTALLED_PLUGINS environment variable (comma-separated names)
    3. Directories in workspace_paths (and their subdirectories / parent directories)
    4. Global Antigravity config directory (~/.gemini/config/plugins)
    """
    if custom_plugins is not None:
        return set(custom_plugins)

    env_plugins = os.environ.get("INSTALLED_PLUGINS")
    if env_plugins:
        return {p.strip() for p in env_plugins.split(",") if p.strip()}

    installed: Set[str] = set()
    candidate_paths: list[Path] = []

    if workspace_paths:
        for wp in workspace_paths:
            p = Path(wp).resolve()
            candidate_paths.append(p)
            if p.is_dir():
                candidate_paths.extend([c for c in p.iterdir() if c.is_dir()])
                candidate_paths.append(p.parent)
                candidate_paths.extend([c for c in p.parent.iterdir() if c.is_dir()])

    cwd = Path.cwd().resolve()
    candidate_paths.append(cwd)
    candidate_paths.extend([c for c in cwd.iterdir() if c.is_dir()])
    candidate_paths.append(cwd.parent)
    candidate_paths.extend([c for c in cwd.parent.iterdir() if c.is_dir()])

    for global_base in [
        Path.home() / ".gemini" / "config" / "plugins",
        Path.home() / ".gemini" / "antigravity-cli" / "plugins",
    ]:
        if global_base.exists() and global_base.is_dir():
            candidate_paths.extend([c for c in global_base.iterdir() if c.is_dir()])

    for path in candidate_paths:
        if path.is_dir():
            manifest = path / "plugin.json"
            if manifest.exists():
                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    name = data.get("name") or path.name
                    if name in KNOWN_PLUGINS:
                        installed.add(name)
                except Exception:
                    pass
            elif path.name in KNOWN_PLUGINS:
                installed.add(path.name)

    return installed


def build_dynamic_triage_spec(installed_plugins: set[str]) -> DynamicTriageSpec:
    """Build dynamic triage responses and mapping based on installed plugins.

    Returns:
        DynamicTriageSpec(responses, response_to_intent, intent_to_response, response_to_directive)
    """
    coding_enabled = "agentic-coding-plugin" in installed_plugins
    monitoring_enabled = "agentic-monitoring-plugin" in installed_plugins
    investigation_enabled = "agentic-investigate-plugin" in installed_plugins
    research_enabled = "agentic-research-plugin" in installed_plugins
    documentation_enabled = "agentic-obsidian-plugin" in installed_plugins

    responses: dict[str, str] = {}
    r_to_intent: dict[str, str] = {}
    intent_to_r: dict[str, str] = {}
    r_to_directive: dict[str, str] = {}

    idx = 1
    # Response 1: always inquiries involving no changes
    r_key = f"response {idx}"
    responses[r_key] = "one or more inquiries involving no changes"
    r_to_intent[r_key] = "question"
    intent_to_r["question"] = r_key
    r_to_directive[r_key] = (
        f"Prompt Triage Directive: {r_key} identified (inquiries involving no changes). "
        "Respond directly to the prompt. Do not make any changes."
    )
    idx += 1

    # Coding change request (if agentic-coding-plugin is installed)
    if coding_enabled:
        r_key = f"response {idx}"
        responses[r_key] = "one or more code change request"
        r_to_intent[r_key] = "coding"
        intent_to_r["coding"] = r_key
        r_to_directive[r_key] = (
            f"Prompt Triage Directive: {r_key} identified (code change request). "
            "You must invoke the coding subagent (@coding) using invoke_subagent:\n"
            "invoke_subagent(TypeName='coding', Role='Coding Workflow Orchestrator', Prompt=...). "
            "Do not write or modify application code directly in the main agent turn."
        )
        idx += 1

    # Task monitoring request (if agentic-monitoring-plugin is installed)
    if monitoring_enabled:
        r_key = f"response {idx}"
        responses[r_key] = "one or more task monitoring request"
        r_to_intent[r_key] = "monitoring"
        intent_to_r["monitoring"] = r_key
        r_to_directive[r_key] = (
            f"Prompt Triage Directive: {r_key} identified (task monitoring request). "
            "You must invoke the monitoring subagent (@monitoring) using invoke_subagent:\n"
            "invoke_subagent(TypeName='monitoring', Role='Telemetry Monitor', Prompt=...). "
            "Do not perform monitoring directly in the main agent turn."
        )
        idx += 1

    # Issue investigation request (if agentic-investigate-plugin is installed)
    if investigation_enabled:
        r_key = f"response {idx}"
        responses[r_key] = "one or more issue investigation request"
        r_to_intent[r_key] = "investigation"
        intent_to_r["investigation"] = r_key
        r_to_directive[r_key] = (
            f"Prompt Triage Directive: {r_key} identified (issue investigation request). "
            "You must invoke the investigate subagent (@investigate) using invoke_subagent:\n"
            "invoke_subagent(TypeName='investigate', Role='Issue Investigator', Prompt=...). "
            "Do not perform investigation directly in the main agent turn."
        )
        idx += 1

    # Topic research request (if agentic-research-plugin is installed)
    if research_enabled:
        r_key = f"response {idx}"
        responses[r_key] = "one or more topic research or information retrieval requests"
        r_to_intent[r_key] = "research"
        intent_to_r["research"] = r_key
        r_to_directive[r_key] = (
            f"Prompt Triage Directive: {r_key} identified (topic research or information retrieval request). "
            "You must invoke the research subagent (@research) using invoke_subagent:\n"
            "invoke_subagent(TypeName='research', Role='Topic Researcher', Prompt=...). "
            "Do not perform research directly in the main agent turn."
        )
        idx += 1

    # Documentation request (if agentic-obsidian-plugin is installed)
    if documentation_enabled:
        r_key = f"response {idx}"
        responses[r_key] = "one or more documentation request"
        r_to_intent[r_key] = "documentation"
        intent_to_r["documentation"] = r_key
        r_to_directive[r_key] = (
            f"Prompt Triage Directive: {r_key} identified (documentation request). "
            "You must invoke the coding-document subagent (@coding-document) using invoke_subagent:\n"
            "invoke_subagent(TypeName='coding-document', Role='Documentation Specialist', Prompt=...). "
            "Do not perform documentation directly in the main agent turn."
        )
        idx += 1

    # Mixed request (if at least one capability is enabled alongside inquiries)
    if any([coding_enabled, monitoring_enabled, investigation_enabled, research_enabled, documentation_enabled]):
        r_key = f"response {idx}"
        parts = ["inquiries"]
        if coding_enabled:
            parts.append("code change requests")
        if monitoring_enabled:
            parts.append("task monitoring")
        if investigation_enabled:
            parts.append("issue investigation requests")
        if research_enabled:
            parts.append("topic research requests")
        if documentation_enabled:
            parts.append("documentation requests")
        desc = f"a mix of {', '.join(parts[:-1])} and/or {parts[-1]}"

        responses[r_key] = desc
        r_to_intent[r_key] = "mixed"
        intent_to_r["mixed"] = r_key
        r_to_directive[r_key] = (
            f"Prompt Triage Directive: {r_key} identified ({desc}). "
            "You must reject the prompt, responding to the user that you cannot process a mix of inquiries and tasks. "
            "Ask the user to separate their requests into individual inquiries or tasks."
        )

    return DynamicTriageSpec(responses, r_to_intent, intent_to_r, r_to_directive)


def clean_user_prompt(text: str) -> str:
    """Extract and sanitize user prompt, stripping Antigravity wrapper tags."""
    if not isinstance(text, str):
        text = str(text)

    user_requests = re.findall(r"<USER_REQUEST>(.*?)(?:</USER_REQUEST>|$)", text, flags=re.DOTALL)
    if user_requests:
        text = user_requests[-1]

    text = re.sub(r"<ADDITIONAL_METADATA>.*?(?:</ADDITIONAL_METADATA>|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"<CONTEXT>.*?(?:</CONTEXT>|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"((?:\s*|(?<=[a-zA-Z0-9_]))/(?:[a-zA-Z0-9_\.\-]+/)+[a-zA-Z0-9_\.\-]+)+$", "", text)
    return " ".join(text.split()).strip()


def extract_prompt_text(data: Any) -> str:
    """Extract prompt text from strings, lists, dicts, or objects."""
    if isinstance(data, str):
        return data
    if isinstance(data, (list, tuple)):
        return " ".join(extract_prompt_text(item) for item in data)
    if isinstance(data, dict):
        for key in ["lastUserInput", "prompt", "text", "content", "message"]:
            val = data.get(key)
            if val and isinstance(val, str):
                return val
    return str(data)


class HookResult(BaseModel):
    """Result of a hook execution."""
    allow: bool = True
    message: Optional[str] = None


class HookContext:
    """Context state passed to lifecycle hooks."""

    def __init__(self, initial_state: Optional[dict[str, Any]] = None):
        self._state: dict[str, Any] = dict(initial_state) if initial_state else {}

    def get_state(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    def update_state(self, updates: dict[str, Any]) -> None:
        self._state.update(updates)


class TriageResult(BaseModel):
    """Result of Laya System 1 prompt triage."""
    stage_reached: int = Field(default=3, description="Evaluation stage reached")
    in_scope: bool = Field(default=True, description="Whether prompt is in session scope")
    scope_confidence: float = Field(default=1.0, description="Confidence of scope determination")
    response_choice: Optional[str] = Field(
        default=None,
        description="Selected response key (e.g. response 1, response 2)",
    )
    response_confidence: float = Field(default=1.0, description="Confidence of selected response")
    response_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probabilities across response candidates",
    )
    requires_user_confirmation: bool = Field(
        default=False,
        description="True if confidence < 0.40 requiring ask_question user selection",
    )
    intent: Optional[str] = Field(default=None, description="Semantic intent: question, coding, monitoring, investigation, research, or mixed")
    intent_confidence: float = Field(default=1.0, description="Confidence of intent classification")
    inject_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Antigravity injectSteps payload")
    installed_plugins: List[str] = Field(default_factory=list, description="Plugins detected active during triage")


class LayaTriageEngine:
    """Dynamic Laya Router System 1 prompt triage engine."""

    def __init__(
        self,
        use_router: bool = True,
        installed_plugins: Optional[set[str] | list[str]] = None,
        workspace_paths: Optional[list[str]] = None,
    ):
        if installed_plugins is not None:
            self.installed_plugins = set(installed_plugins)
        else:
            self.installed_plugins = detect_installed_plugins(workspace_paths=workspace_paths)

        spec = build_dynamic_triage_spec(self.installed_plugins)
        self.triage_responses, self.response_to_intent, self.intent_to_response, self.response_to_directive = spec

        self._router = None
        if use_router:
            try:
                from laya import Router
                self._router = Router(preload=False)
            except Exception as e:
                logger.warning(f"Could not initialize Laya Router, using fallback: {e}")
                self._router = None

    def _predict_with_laya(self, text: str, questions: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self._router:
            return None
        try:
            return self._router.predict({"prompt": text}, questions)
        except Exception as e:
            logger.warning(f"Laya prediction failed: {e}")
            return None

    def evaluate_system1_triage(self, prompt: str) -> tuple[str, float, dict[str, float]]:
        """Triage prompt using Laya Router system1 model with dynamic criteria."""
        questions = {
            "about": {
                "type": "choice",
                "instructions": TRIAGE_QUESTION,
                "criteria": self.triage_responses,
            }
        }
        res = self._predict_with_laya(prompt, questions)
        if res and "answers" in res and "about" in res["answers"]:
            ans = res["answers"]["about"]
            choice = ans.get("choice", "response 1")
            probs = ans.get("probabilities", {})
            conf = probs.get(choice, ans.get("answer_confidence", 0.0))
            return choice, conf, probs

        return self._heuristic_system1_triage(prompt)

    def _heuristic_system1_triage(self, prompt: str) -> tuple[str, float, dict[str, float]]:
        """Deterministic heuristic fallback adapting to installed plugins."""
        prompt_lower = prompt.lower()

        coding_enabled = "agentic-coding-plugin" in self.installed_plugins
        monitoring_enabled = "agentic-monitoring-plugin" in self.installed_plugins
        investigation_enabled = "agentic-investigate-plugin" in self.installed_plugins
        research_enabled = "agentic-research-plugin" in self.installed_plugins
        documentation_enabled = "agentic-obsidian-plugin" in self.installed_plugins

        has_monitoring = monitoring_enabled and any(
            w in prompt_lower
            for w in ["monitor", "watch log", "watch serial", "tail -f", "serial console", "log stream", "@monitoring"]
        )
        has_investigation = investigation_enabled and any(
            w in prompt_lower
            for w in [
                "investigate", "investigation", "investigating", "invetigate", "invetigation",
                "diagnose", "root cause", "find the cause", "find cause", "troubleshoot",
                "@investigate", "coding-investigate", "coding-investigator", "coding-invetigator",
                "@coding-investigate", "@coding-investigator", "@coding-invetigator",
                "track down the bug", "track down the issue", "track down the root cause"
            ]
        )
        has_coding = coding_enabled and any(
            w in prompt_lower
            for w in ["write", "create", "implement", "build", "refactor", "bugfix", "fix", "feature", "code change", "@coding"]
        )
        has_research = research_enabled and any(
            w in prompt_lower
            for w in [
                "research", "deep research", "literature review", "web research", "find papers",
                "topic research", "information retrieval", "@research", "research subagent",
                "state of the art", "prior art", "market research"
            ]
        )
        has_documentation = documentation_enabled and any(
            w in prompt_lower
            for w in [
                "document", "documentation", "documenting", "obsidian", "vault", "adr",
                "architecture diagram", "architecture doc", "features doc", "feature spec",
                "sync session", "archive session", "git status doc", "status doc",
                "@coding-document", "coding-document", "@document",
                "record session", "store session", "store artifacts", "record the result of the monitoring"
            ]
        )
        has_inquiry = any(
            w in prompt_lower
            for w in ["how", "what", "why", "explain", "review", "inquiry", "is there", "can you tell", "describe", "where"]
        )

        if has_investigation and not has_monitoring and not any(w in prompt_lower for w in ["implement", "create feature", "deploy", "merge"]):
            has_inquiry = False

        if has_research and not has_monitoring and not has_coding and not has_investigation and not any(w in prompt_lower for w in ["explain", "describe", "also"]):
            has_inquiry = False

        if has_documentation and not any(w in prompt_lower for w in ["implement", "code change", "bugfix", "refactor", "write code", "@coding"]):
            has_coding = False

        if has_documentation and not has_monitoring and not has_coding and not has_investigation and not has_research and not any(w in prompt_lower for w in ["explain", "describe", "also"]):
            has_inquiry = False

        mixed_count = sum([bool(has_monitoring), bool(has_coding), bool(has_inquiry), bool(has_investigation), bool(has_research), bool(has_documentation)])
        mixed_key = self.intent_to_response.get("mixed")

        # Initialize base uniform probabilities across active responses
        base_probs = {r: 0.05 for r in self.triage_responses}

        if mixed_count >= 2 and mixed_key:
            probs = dict(base_probs)
            probs[mixed_key] = 0.90
            return mixed_key, 0.90, probs

        if has_investigation and "investigation" in self.intent_to_response:
            r_key = self.intent_to_response["investigation"]
            probs = dict(base_probs)
            probs[r_key] = 0.92
            return r_key, 0.92, probs

        if has_monitoring and "monitoring" in self.intent_to_response:
            r_key = self.intent_to_response["monitoring"]
            probs = dict(base_probs)
            probs[r_key] = 0.92
            return r_key, 0.92, probs

        if has_coding and "coding" in self.intent_to_response:
            r_key = self.intent_to_response["coding"]
            probs = dict(base_probs)
            probs[r_key] = 0.90
            return r_key, 0.90, probs

        if has_research and "research" in self.intent_to_response:
            r_key = self.intent_to_response["research"]
            probs = dict(base_probs)
            probs[r_key] = 0.92
            return r_key, 0.92, probs

        if has_documentation and "documentation" in self.intent_to_response:
            r_key = self.intent_to_response["documentation"]
            probs = dict(base_probs)
            probs[r_key] = 0.92
            return r_key, 0.92, probs

        if has_inquiry and "question" in self.intent_to_response:
            r_key = self.intent_to_response["question"]
            probs = dict(base_probs)
            probs[r_key] = 0.88
            return r_key, 0.88, probs

        # Ambiguous fallback (< 0.40) to present options to user
        default_key = "response 1"
        probs = {r: 1.0 / len(self.triage_responses) for r in self.triage_responses}
        probs[default_key] = 0.30
        return default_key, 0.30, probs

    def triage(
        self,
        prompt: str,
        invocation_num: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TriageResult:
        """Run dynamic System 1 prompt triage and format systemMessage payload."""
        choice, conf, probs = self.evaluate_system1_triage(prompt)
        intent = self.response_to_intent.get(choice, "question")

        metadata_lines = [
            "=== Prompt Triage Telemetry & Metadata ===",
            f'- Question: "{TRIAGE_QUESTION}"',
            f'- Selected Response: {choice} (\"{self.triage_responses.get(choice, "")}\")',
            f"- Confidence: {conf:.4f}",
            f"- Active Plugins: {', '.join(sorted(self.installed_plugins)) or 'none'}",
            "- Probability Distribution:",
        ]
        for r_key, r_desc in self.triage_responses.items():
            prob_val = probs.get(r_key, 0.0)
            metadata_lines.append(f'  * {r_key} (\"{r_desc}\"): {prob_val:.4f}')

        if metadata:
            metadata_lines.append("- Context Metadata:")
            for m_k, m_v in metadata.items():
                if m_v is not None and m_k != "lastUserInput":
                    metadata_lines.append(f"  * {m_k}: {m_v}")

        metadata_text = "\n".join(metadata_lines)
        inject_steps: list[dict[str, Any]] = []

        if conf < CONFIDENCE_THRESHOLD and len(self.triage_responses) > 1:
            options = list(self.triage_responses.values())
            options_text = "\n".join([f'  {i+1}. "{opt}"' for i, opt in enumerate(options)])
            rules_text = "\n".join([
                f"- If {r_k} ('{r_desc}') is selected: {self.response_to_directive.get(r_k, '')}"
                for r_k, r_desc in self.triage_responses.items()
            ])

            ask_guidance = (
                f"{metadata_text}\n\n"
                f"Prompt Triage: Intent classification is ambiguous (confidence {conf:.2f} < {CONFIDENCE_THRESHOLD:.2f}).\n"
                "You MUST present the choice to the user by submitting the following question using the `ask_question` tool:\n"
                f'- Question: "{TRIAGE_QUESTION}"\n'
                f"- Options:\n{options_text}\n\n"
                f"Execution rules based on user selection:\n{rules_text}"
            )
            inject_steps.append({
                "systemMessage": {
                    "systemMessage": ask_guidance
                }
            })
            return TriageResult(
                stage_reached=3,
                in_scope=True,
                response_choice=choice,
                response_confidence=conf,
                response_probabilities=probs,
                requires_user_confirmation=True,
                intent=intent,
                intent_confidence=conf,
                inject_steps=inject_steps,
                installed_plugins=sorted(self.installed_plugins),
            )

        directive = self.response_to_directive.get(
            choice,
            f"Prompt Triage Directive: Response {choice} identified. Respond appropriately."
        )
        full_message = f"{metadata_text}\n\n{directive}"
        inject_steps.append({
            "systemMessage": {
                "systemMessage": full_message
            }
        })
        return TriageResult(
            stage_reached=3,
            in_scope=True,
            response_choice=choice,
            response_confidence=conf,
            response_probabilities=probs,
            requires_user_confirmation=False,
            intent=intent,
            intent_confidence=conf,
            inject_steps=inject_steps,
            installed_plugins=sorted(self.installed_plugins),
        )


class PromptTriageHook:
    """Antigravity PreInvocation lifecycle hook executing dynamic prompt triage."""

    def __init__(
        self,
        use_router: bool = True,
        installed_plugins: Optional[set[str] | list[str]] = None,
        workspace_paths: Optional[list[str]] = None,
    ):
        self.engine = LayaTriageEngine(
            use_router=use_router,
            installed_plugins=installed_plugins,
            workspace_paths=workspace_paths,
        )

    async def run(self, context: HookContext, data: Any) -> HookResult:
        prompt_text = clean_user_prompt(extract_prompt_text(data))
        if not prompt_text:
            return HookResult(allow=True, message="empty prompt")

        inv_num = context.get_state("invocation_num", 0)
        conv_id = context.get_state("conversation_id", None)
        artifact_dir = context.get_state("artifact_dir", None)

        metadata = {
            "conversation_id": conv_id,
            "invocation_num": inv_num,
            "artifact_dir": artifact_dir,
        }

        result: TriageResult = self.engine.triage(
            prompt=prompt_text,
            invocation_num=inv_num,
            metadata=metadata,
        )

        context.set_state("triage_result", result)
        context.set_state("inject_steps", result.inject_steps)
        context.set_state("triage_intent", result.intent)
        context.set_state("response_choice", result.response_choice)
        context.set_state("response_confidence", result.response_confidence)
        context.set_state("requires_user_confirmation", result.requires_user_confirmation)

        return HookResult(
            allow=True,
            message=result.response_choice or result.intent or "allowed",
        )


async def session_start_triage(context: HookContext) -> None:
    """SessionStart hook initializing triage context."""
    context.set_state("triage_initialized", True)
    context.set_state("invocation_num", 0)


async def prompt_triage_pre_turn(context: HookContext, data: Any) -> HookResult:
    """PreInvocation helper running PromptTriageHook."""
    hook = PromptTriageHook(use_router=True)
    return await hook.run(context, data)


async def run_hook_async(payload: dict) -> list[dict]:
    """Execute PromptTriageHook with dynamic workspace plugin detection."""
    inv_num = payload.get("invocationNum", 0)
    if inv_num > 0:
        return []

    raw_prompt = payload.get("lastUserInput", "")
    if not raw_prompt:
        transcript_path = payload.get("transcriptPath", "")
        if transcript_path and os.path.exists(transcript_path):
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    for line in reversed(f.readlines()):
                        try:
                            step = json.loads(line)
                            if step.get("type") == "USER_INPUT" or step.get("source") == "USER_EXPLICIT":
                                raw_prompt = step.get("content", "")
                                if raw_prompt:
                                    break
                        except Exception:
                            continue
            except Exception:
                pass

    prompt = clean_user_prompt(extract_prompt_text(raw_prompt))
    if not prompt:
        return []

    context = HookContext()
    context.set_state("invocation_num", inv_num)
    context.set_state("conversation_id", payload.get("conversationId"))
    context.set_state("artifact_dir", payload.get("artifactDirectoryPath"))

    workspace_paths = payload.get("workspacePaths", [])
    hook = PromptTriageHook(use_router=True, workspace_paths=workspace_paths)
    await hook.run(context, prompt)

    inject_steps = context.get_state("inject_steps", [])
    return inject_steps if isinstance(inject_steps, list) else []


def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            json.dump({"injectSteps": []}, sys.stdout)
            return

        payload = json.loads(raw_input)
        inject_steps = asyncio.run(run_hook_async(payload))
        json.dump({"injectSteps": inject_steps}, sys.stdout)
    except Exception:
        # Hooks must never crash the parent agent execution loop
        json.dump({"injectSteps": []}, sys.stdout)


if __name__ == "__main__":
    main()

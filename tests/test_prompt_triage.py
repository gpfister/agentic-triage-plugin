"""Tests for dynamic prompt triage hook and Laya Router System 1 model."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from hooks.prompt_triage import (
    HookContext,
    HookResult,
    PromptTriageHook,
    prompt_triage_pre_turn,
    session_start_triage,
    extract_prompt_text,
    clean_user_prompt,
    detect_installed_plugins,
    build_dynamic_triage_spec,
    LayaTriageEngine,
    CONFIDENCE_THRESHOLD,
    TRIAGE_QUESTION,
)


def test_clean_user_prompt():
    raw1 = (
        "<USER_REQUEST>\nExplain this code\n</USER_REQUEST>\n"
        "<ADDITIONAL_METADATA>\nThe current local time is: 2026-10-26T10:55:12+01:00.\n</ADDITIONAL_METADATA>"
    )
    assert clean_user_prompt(raw1) == "Explain this code"

    raw2 = (
        "<USER_REQUEST>\n"
        "Improve prompt_triage.py -> lower the threshold to 0.4\n"
        "/path/to/file1.py\n"
        "/path/to/file2.py\n"
        "</USER_REQUEST>\n"
        "<ADDITIONAL_METADATA>some metadata</ADDITIONAL_METADATA>"
    )
    assert clean_user_prompt(raw2) == "Improve prompt_triage.py -> lower the threshold to 0.4"

    raw3 = "Please refactor the database connector"
    assert clean_user_prompt(raw3) == "Please refactor the database connector"


def test_extract_prompt_text():
    assert extract_prompt_text("simple string") == "simple string"
    assert extract_prompt_text(["part1", "part2"]) == "part1 part2"


def test_detect_installed_plugins_custom():
    plugins = detect_installed_plugins(custom_plugins={"agentic-coding-plugin"})
    assert plugins == {"agentic-coding-plugin"}


def test_detect_installed_plugins_research():
    plugins = detect_installed_plugins(custom_plugins={"agentic-research-plugin"})
    assert plugins == {"agentic-research-plugin"}


def test_detect_installed_plugins_env(monkeypatch):
    monkeypatch.setenv("INSTALLED_PLUGINS", "agentic-coding-plugin, agentic-monitoring-plugin, agentic-research-plugin")
    plugins = detect_installed_plugins()
    assert plugins == {"agentic-coding-plugin", "agentic-monitoring-plugin", "agentic-research-plugin"}


def test_build_dynamic_triage_spec_all():
    spec = build_dynamic_triage_spec(
        {"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"}
    )
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.response_to_intent["response 2"] == "coding"
    assert "coding" in spec.intent_to_response
    assert "investigation" in spec.intent_to_response
    assert "monitoring" in spec.intent_to_response
    mixed_key = spec.intent_to_response.get("mixed")
    assert mixed_key is not None
    assert spec.response_to_intent[mixed_key] == "mixed"


def test_build_dynamic_triage_spec_only_coding():
    spec = build_dynamic_triage_spec({"agentic-coding-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert spec.response_to_intent["response 2"] == "coding"
    assert spec.intent_to_response.get("mixed") == "response 3"


def test_build_dynamic_triage_spec_only_investigate():
    spec = build_dynamic_triage_spec({"agentic-investigate-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.response_to_intent["response 2"] == "investigation"
    assert "coding" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "response 3"


def test_build_dynamic_triage_spec_only_research():
    spec = build_dynamic_triage_spec({"agentic-research-plugin"})
    assert "response 1" in spec.responses
    assert "response 2" in spec.responses
    assert spec.responses["response 2"] == "one or more topic research or information retrieval requests"
    assert spec.response_to_intent["response 2"] == "research"
    assert spec.intent_to_response["research"] == "response 2"
    assert "coding" not in spec.intent_to_response
    assert "investigation" not in spec.intent_to_response
    assert "monitoring" not in spec.intent_to_response
    assert spec.intent_to_response.get("mixed") == "response 3"
    assert "@research" in spec.response_to_directive["response 2"]
    assert "Topic Researcher" in spec.response_to_directive["response 2"]


def test_build_dynamic_triage_spec_all_four():
    spec = build_dynamic_triage_spec({
        "agentic-coding-plugin",
        "agentic-investigate-plugin",
        "agentic-monitoring-plugin",
        "agentic-research-plugin",
    })
    assert len(spec.responses) == 6
    assert spec.intent_to_response["question"] == "response 1"
    assert spec.intent_to_response["coding"] == "response 2"
    assert spec.intent_to_response["monitoring"] == "response 3"
    assert spec.intent_to_response["investigation"] == "response 4"
    assert spec.intent_to_response["research"] == "response 5"
    assert spec.intent_to_response["mixed"] == "response 6"
    assert spec.response_to_intent["response 5"] == "research"
    assert spec.response_to_intent["response 6"] == "mixed"
    assert "@research" in spec.response_to_directive["response 5"]


def test_build_dynamic_triage_spec_none():
    spec = build_dynamic_triage_spec(set())
    assert "response 1" in spec.responses
    assert len(spec.responses) == 1
    assert "mixed" not in spec.intent_to_response


def test_prompt_triage_hook_allow():
    async def _run():
        context = HookContext()
        context.set_state("invocation_num", 0)

        hook = PromptTriageHook(
            use_router=False,
            installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
        )
        result = await hook.run(context, "How does this code work?")

        assert isinstance(result, HookResult)
        assert result.allow is True
        assert result.message in ("response 1", "question")
        assert context.get_state("triage_intent") == "question"
        assert context.get_state("response_choice") == "response 1"
        inject_steps = context.get_state("inject_steps")
        assert len(inject_steps) > 0
        assert "systemMessage" in inject_steps[0]
        assert "ephemeralMessage" not in inject_steps[0]
        sys_msg = inject_steps[0]["systemMessage"]["systemMessage"]
        assert "What is this prompt about ?" in sys_msg
        assert "Prompt Triage Directive" in sys_msg
        assert "Do not make any changes" in sys_msg

    asyncio.run(_run())


def test_session_start_hook():
    async def _run():
        context = HookContext()
        await session_start_triage(context)
        assert context.get_state("triage_initialized") is True
        assert context.get_state("invocation_num") == 0

    asyncio.run(_run())


def test_triage_response1_inquiries_no_changes():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
    )
    result = engine.triage("Can you explain how this module works?", invocation_num=0)
    assert result.response_choice == "response 1"
    assert result.intent == "question"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "What is this prompt about ?" in msg
    assert "Do not make any changes" in msg


def test_triage_response_coding():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
    )
    result = engine.triage("Please implement a new login endpoint with JWT auth", invocation_num=0)
    assert result.intent == "coding"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "invoke_subagent" in msg
    assert "coding" in msg


def test_triage_response_monitoring():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
    )
    result = engine.triage("@monitoring watch the serial console /dev/ttyUSB0", invocation_num=0)
    assert result.intent == "monitoring"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "invoke_subagent" in msg
    assert "monitoring" in msg


def test_triage_response_investigation():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
    )
    result = engine.triage("Please investigate why the checkout endpoint returns 500 error", invocation_num=0)
    assert result.intent == "investigation"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "invoke_subagent" in msg
    assert "investigate" in msg


def test_triage_response_research():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={
            "agentic-coding-plugin",
            "agentic-investigate-plugin",
            "agentic-monitoring-plugin",
            "agentic-research-plugin",
        },
    )
    result = engine.triage("Please conduct deep research on state of the art retrieval models", invocation_num=0)
    assert result.intent == "research"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "invoke_subagent" in msg
    assert "research" in msg
    assert "@research" in msg
    assert "Topic Researcher" in msg


def test_triage_response_mixed():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
    )
    result = engine.triage(
        "Explain how the server starts and also implement a new fix and monitor the logs",
        invocation_num=0,
    )
    assert result.intent == "mixed"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "cannot process a mix" in msg


def test_triage_response_mixed_with_research():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={
            "agentic-coding-plugin",
            "agentic-investigate-plugin",
            "agentic-monitoring-plugin",
            "agentic-research-plugin",
        },
    )
    result = engine.triage(
        "Research quantum computing algorithms and also implement a prototype in Python",
        invocation_num=0,
    )
    assert result.intent == "mixed"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "cannot process a mix" in msg


def test_triage_response_mixed_research_and_inquiry():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={
            "agentic-coding-plugin",
            "agentic-investigate-plugin",
            "agentic-monitoring-plugin",
            "agentic-research-plugin",
        },
    )
    result = engine.triage(
        "Explain how the current system works and also conduct deep research on alternatives",
        invocation_num=0,
    )
    assert result.intent == "mixed"
    assert result.response_confidence >= CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    msg = result.inject_steps[0]["systemMessage"]["systemMessage"]
    assert "=== Prompt Triage Telemetry & Metadata ===" in msg
    assert "cannot process a mix" in msg


def test_triage_ambiguous_low_confidence_triggers_ask_question():
    engine = LayaTriageEngine(
        use_router=False,
        installed_plugins={"agentic-coding-plugin", "agentic-investigate-plugin", "agentic-monitoring-plugin"},
    )
    result = engine.triage("maybe something about project files", invocation_num=0)
    assert result.requires_user_confirmation is True
    assert result.response_confidence < CONFIDENCE_THRESHOLD
    assert len(result.inject_steps) > 0
    assert "systemMessage" in result.inject_steps[0]
    inject_str = str(result.inject_steps[0])
    assert "=== Prompt Triage Telemetry & Metadata ===" in inject_str
    assert "ask_question" in inject_str
    assert "What is this prompt about ?" in inject_str
    assert "one or more inquiries involving no changes" in inject_str
    assert "one or more code change request" in inject_str


def test_triage_hook_script_cli(tmp_path: Path):
    input_payload = {
        "conversationId": "test-conv-001",
        "invocationNum": 0,
        "lastUserInput": "<USER_REQUEST>Can you review this code?</USER_REQUEST><ADDITIONAL_METADATA>time</ADDITIONAL_METADATA>",
        "artifactDirectoryPath": str(tmp_path),
    }

    proc = subprocess.run(
        ["uv", "run", "hooks/prompt_triage.py"],
        cwd=str(PLUGIN_ROOT),
        input=json.dumps(input_payload),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    output = json.loads(proc.stdout)
    assert "injectSteps" in output
    assert len(output["injectSteps"]) > 0
    step = output["injectSteps"][0]
    assert "systemMessage" in step
    assert "ephemeralMessage" not in step
    msg = step["systemMessage"]["systemMessage"]
    assert "What is this prompt about ?" in msg
    assert "Telemetry & Metadata" in msg


def test_triage_hook_skips_when_invocation_num_greater_than_zero(tmp_path: Path):
    input_payload = {
        "conversationId": "test-conv-001",
        "invocationNum": 1,
        "lastUserInput": "Can you review this code?",
        "artifactDirectoryPath": str(tmp_path),
    }

    proc = subprocess.run(
        ["uv", "run", "hooks/prompt_triage.py"],
        cwd=str(PLUGIN_ROOT),
        input=json.dumps(input_payload),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    output = json.loads(proc.stdout)
    assert output == {"injectSteps": []}

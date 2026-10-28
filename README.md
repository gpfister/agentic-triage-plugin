# agentic-triage-plugin

Dynamic prompt intent triage lifecycle hook for [Google Antigravity](https://antigravity.google) powered by [Laya Router](https://github.com).

## Overview
`agentic-triage-plugin` automatically triages incoming user prompts before model execution using the non-autoregressive Laya Router System 1 model. It dynamically adapts its intent criteria based on which plugins are installed:
- Evaluates code change requests if `agentic-coding-plugin` is installed (routes to `@coding`).
- Evaluates issue investigations if `agentic-investigate-plugin` is installed (routes to `@investigate`).
- Evaluates task monitoring if `agentic-monitoring-plugin` is installed (routes to `@monitoring`).
- Handles informational inquiries directly without code modifications.
- Rejects conflicting mixed prompts, asking users to separate their requests.

## Installation & Activation

```bash
# Enable in Antigravity session:
agy plugin enable agentic-triage-plugin
```

## Contributing
See [CONTRIBUTING.md](file:///Users/gpfister/Development/Projects/gpfister/agentic/agentic-triage-plugin/CONTRIBUTING.md) for contribution guidelines and testing practices.

## License
This project is licensed under the [MIT License](file:///Users/gpfister/Development/Projects/gpfister/agentic/agentic-triage-plugin/LICENSE).

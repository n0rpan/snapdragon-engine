# v0.1.2 — chat and agentic sessions

Fast is the default OpenCode preset: 65,536 input tokens with compaction around 49.5K. The engine context cap for Ornith and Qwen3.6 is 131,072 tokens. Select Deep with `LOCAL_AI_OPENCODE_MODE=deep`; long tool appends at depth cost substantially more.

Turn checkpoints and a segmented KV/recurrent state cache preserve chat sessions across model switches and process restarts. Restore selects the deepest exact usable checkpoint before comparing equal-depth costs. Thinking history is retained until compaction. The disk cache is identity-keyed, bounded, and configurable; treat its contents as sensitive conversation data.

The installed release passed both models' protected coding task and chat/tool/cancel/session-resume checks. The accepted Deep-mode envelope was tested to about 107K tokens on Ornith; this is not a full 128K-fill claim. Fast compaction was checked through the real OpenCode client at reduced 16K/32K input limits. See [the companion qualification receipt](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/benchmarks/results/public/V012-CHAT.json) for scope and evidence.

Tiled attention remains the default; the optional fused arm did not pass its registered numeric gate and stays off. This release adds no native text encoder or new image speedup. The existing image route and Flash 18 GiB prefix-store configuration are unchanged. Downloaded CI binaries are rebuilds, not the exact laptop-qualified executable, and can differ numerically.

## Known issue

Deep mode: very large tool outputs at >80K context can take >15 min to process; OpenCode's default stream timeout (900 s) may abort and retry. Fast (the default) avoids this; a fix lands in v0.1.3.

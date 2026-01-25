# Writing Skill and Agent Prompts

A prompt should act as a manifest for the agent: clear goal, clear constraints, freedom in execution.

## Principles

**Define WHAT and WHY, not HOW** - State goals and constraints. Don't prescribe steps the model knows how to do. No rigid phase ordering, no heuristics tables that become checklists.

**Operate under the memento pattern** - For non-trivial workflows, the agent should create a todo list immediately, write findings to a log file as it works, and refresh context before synthesis. This isn't optional structure—it's a constraint that makes the agent work better.

**Trust capability, enforce discipline** - The model knows how to search, analyze, generate. What it needs are guardrails: "write to log before proceeding", "refresh before synthesis", "don't skip verification".

**Output structure when needed** - If the artifact has a specific format (manifest schema, report template), define it. Otherwise let the agent decide.

---

## Memento Pattern for Non-Trivial Workflows

Skills and agents with multi-phase workflows MUST use the memento pattern.

**Important:** When implementing this pattern in skills/agents, DO NOT mention the pattern by name (e.g., "memento pattern", "memento loop"). Simply follow the pattern—use descriptive labels like "Write findings to log", "Discovery Loop", "Refresh context" without referencing pattern terminology. The pattern should be invisible to users.

### Why: The LLM Limitations

| Limitation | Research Finding | Pattern Response |
|------------|------------------|------------------|
| Context rot | Information in the middle of context gets "lost"—U-shaped attention curve with >20% accuracy degradation for middle-positioned content | Write findings to external file after EACH step; file persists where conversation content degrades |
| Working memory | LLMs reliably track only 5-10 variables; beyond this, state management fails | TodoWrite externalizes all tracked areas; each todo = one "variable" in external memory |
| Holistic synthesis failure | <50% accuracy on synthesis tasks at 32K tokens; models excel at needle retrieval but fail at aggregation across full context | Read full log file BEFORE synthesis—converts degraded scattered context into concentrated recent content |
| Recency bias | Models pay highest attention to content at context end | Refresh step moves ALL findings to context end where attention is strongest |
| Premature completion | Agents mark tasks "done" without verification; later instances see partial progress and "declare the job done" | Expansion placeholders signal incompleteness; explicit write-to-log todos ensure nothing is skipped |

### Todos as Micro-Prompts

Each todo is a micro-prompt. Apply compression—goal + acceptance criteria + discipline:

- **Goal:** WHAT to achieve, not HOW (model knows how to investigate)
- **Acceptance criteria:** WHAT defines success—models are RL-trained to satisfy these (e.g., `; done when X`)
- **Discipline markers:** `→log` after collection; `refresh:` before synthesis
- **Drop capability:** Model knows what to capture, how to search, what's relevant
- **Novel constraints inline:** Only counter-intuitive rules model wouldn't guess

### The Pattern: Full Specification

**1. Create todo list immediately** with areas to discover, not fixed steps:

```
- [ ] Create log /tmp/{workflow}-*.md
- [ ] Decompose $ARGUMENTS→areas→log; done when all areas identified
- [ ] Investigate [primary area]→log; done when key findings captured
- [ ] (expand: areas as discovered)
- [ ] Refresh: read full log    ← CRITICAL: never skip
- [ ] Synthesize→final artifact; done when artifact complete + validated
```

**2. Write to log after each investigation** (discipline, not capability):

```
- [x] Investigate auth flow→log; done when flow documented
- [x] Investigate error handling→log; done when patterns identified
- [ ] Investigate caching layer→log; done when cache strategy understood
```

**3. Expand todos dynamically** as work reveals new areas:

Before:
```
- [ ] Investigate API layer→log; done when architecture understood
- [ ] (expand: areas as discovered)
```

After (discovered 3 sub-areas):
```
- [x] Investigate API layer→log; found: auth, validation, rate-limiting
- [ ] Investigate auth middleware→log; done when auth flow mapped
- [ ] Investigate validation layer→log; done when rules documented
- [ ] Investigate rate-limiting→log; done when limits + behavior understood
- [ ] (expand: additional areas)
```

**4. Refresh context BEFORE synthesis** (non-negotiable):

```
- [x] Investigate [final area]→log; done when findings captured
- [x] Refresh: read full log    ← Must complete BEFORE synthesize
- [ ] Synthesize→final artifact; done when all findings integrated + validated
```

**Why the refresh step is critical:** By the synthesis phase, earlier findings have degraded due to context rot. The log file contains ALL findings written throughout the workflow. Reading the full file immediately before output:

- Moves all findings to context END (highest attention zone)
- Converts holistic synthesis (poor LLM performance) into dense recent context (high LLM performance)
- Restores details that would otherwise be "lost in the middle"

### Quick Reference

| Phase | Todo Style | Why |
|-------|------------|-----|
| Start | `Create log /tmp/{x}-*.md` | External memory |
| Each step | `Investigate [area]→log; done when X` | Goal + acceptance criteria + discipline |
| Discovery | `(expand: areas as discovered)` | Signals incompleteness |
| Before synthesis | `Refresh: read full log` | Restores context to high-attention zone |
| End | `Synthesize→artifact; done when Y` | Clear output target + success condition |

**Never skip:** The `→log` writes and `refresh: read full log` step. These are the core mechanism that makes synthesis work despite context rot.

**Always include:** Acceptance criteria (`"; done when X"`) so the model knows what success looks like.

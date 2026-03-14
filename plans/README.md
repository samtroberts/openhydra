# OpenHydra Plans Directory

**Master plan:** [`docs/beta-launch-strategy.md`](../docs/beta-launch-strategy.md)

Each sub-plan below is a deep-dive born from a design conversation.
Sub-plans feed upward into the master plan's relevant section.

---

## Directory Tree

```
plans/
├── README.md                    ← You are here (index)
├── memory.md                    ← Session context for Claude continuity
├── progress.md                  ← Task and milestone tracking
│
└── Sub-plans (one per topic)
    └── auto-scaling-policy.md   ← Capability-aware auto-scaling design
```

## How This Works

| File | Purpose | Updated |
|------|---------|---------|
| **Master plan** (`docs/beta-launch-strategy.md`) | High-level roadmap, phases, success criteria | When any sub-plan changes a top-level decision |
| **Sub-plans** (`plans/*.md`) | Deep-dive into one topic; born from a design conversation | During the conversation that creates it |
| **`memory.md`** | What Claude needs to know to resume work | Every session |
| **`progress.md`** | What's done, what's in-flight, what's next | Every session |

### Rules

1. Every design conversation produces **one sub-plan** file in `plans/`.
2. The sub-plan's key decisions are **summarised back** into the master plan's relevant section (with a cross-reference link).
3. `memory.md` and `progress.md` are updated at the **end of every session** (see checklist below).

### End-of-Session Checklist (MANDATORY)

Before the session ends or context compacts, Claude MUST do all of the following:

- [ ] **`plans/memory.md`** — Update "Where We Stand" to reflect what was done this session. Update "What to Do Next" if priorities changed. Remove stale entries.
- [ ] **`plans/progress.md`** — Flip status emojis for every task that changed state. Add new tasks if work was discovered. Do NOT leave a task marked `:construction:` if it was actually completed.
- [ ] **Cross-check** — If `memory.md` says X is done, `progress.md` must agree. If they disagree, fix whichever is wrong *right now*.
- [ ] **`.claude/.../MEMORY.md`** — Only update if a hard fact changed (test count, port number, new bootstrap node, new key file). Do NOT duplicate plans/strategy discussion here.

**The single rule that prevents drift: never update one file without checking the other two.**

---

## Sub-Plan Index

| # | Sub-Plan | Master Plan Section | Status | Created |
|---|----------|---------------------|--------|---------|
| 1 | [auto-scaling-policy.md](auto-scaling-policy.md) | [Section 4: Auto-Scaling and Model Promotion](../docs/beta-launch-strategy.md#4-auto-scaling-and-model-promotion) | Draft | 2026-03-10 |

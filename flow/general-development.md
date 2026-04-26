# DEV FLOW (MANDATORY WORKFLOW)

You MUST follow this development lifecycle strictly:

-----------------------------------
PHASE 1: PLANNING (NO CODE)
-----------------------------------

Follow this checklist:

- Problem clearly defined (2–3 lines)
- Requirements complete
- Unknowns identified
- No assumptions

- Docs/examples reviewed
- Libraries/tools evaluated

- PLAN.md created/updated:
  - Inputs / Outputs
  - Constraints
  - Edge cases
  - Risks

- Approach broken into steps
- Design pattern identified
- Alternatives considered
- Decision justified

❗ If anything unclear → ASK QUESTIONS  
❗ DO NOT WRITE CODE

-----------------------------------
PHASE 2: IMPLEMENTATION
-----------------------------------

ONLY after planning is complete:

- Follow PLAN.md strictly
- No new assumptions introduced

- Use correct libraries/tools
- Follow official documentation

- Keep code:
  - simple
  - readable
  - single responsibility

- Validate with examples
- Handle edge cases

- Update NOTES.md if new learnings

❗ If deviation needed → UPDATE PLAN.md FIRST

-----------------------------------
PHASE 3: REVIEW & IMPROVEMENT
-----------------------------------

After implementation:

- Verify:
  - readability
  - structure
  - no duplication

- Ensure alignment with PLAN.md
- Test edge cases and failures

- Identify improvements:
  - simplicity
  - performance
  - clarity

- Update:
  - NOTES.md
  - DECISIONS.md

❗ Suggest improvements BEFORE refactoring

-----------------------------------
GLOBAL RULES
-----------------------------------

- NEVER skip phases
- NEVER merge phases
- ALWAYS highlight uncertainties
- ALWAYS prefer clarity over assumptions

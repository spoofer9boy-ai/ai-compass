# The AI Compass

> A granular, practice-first roadmap for AI engineers.
> **Theory → Real Problems → Industry Context.**

## What This Is

The AI Compass is not a list of courses to watch. It is a structured path where every subject is atomic (the size of "matrix multiplication"), every concept is connected to why professionals actually use it, and every block of theory is followed by a real industry-scoped problem.

## Who This Is For

- New AI engineers who understand syntax but don't know *why* they are learning linear algebra.
- Self-taught practitioners who have gaps in foundational intuition.
- Anyone who has finished a MOOC but cannot build production-grade ML systems.

## How to Navigate

1. **Start at Phase 01.** Each subject is designed to take 30–60 minutes.
2. **Read the subject file.** Focus on *Why am I learning this?* and *Where will I be using it?*
3. **Do the practice.** Practice files are real problems, not textbook exercises.
4. **Track progress** in `MANIFEST.json` or simply check off subjects as you go.

## Repository Structure

```text
ai-compass/
├── MANIFEST.json              # The full curriculum queue
├── STYLE-GUIDE.md             # How subjects and practices are formatted
├── PHASE-01-foundations/
│   ├── 01-vectors.md
│   ├── 02-vector-operations.md
│   ├── ...
│   └── p01-practice-document-similarity.md
├── PHASE-02-ml-core/
├── PHASE-03-deep-learning/
├── PHASE-04-llm-engineering/
├── PHASE-05-data-engineering/
└── PHASE-06-mlops-production/
```

## Subject File Format

Every `.md` file in a phase follows the same template:

- **Why am I learning this?**
- **Where will I be using it?**
- **Resources**
- **Appendix**

No fluff. No 20-minute videos embedded. Just the intuition, the context, and the door to go deeper.

## Practice File Format

Practice files appear after blocks of 3–5 subjects. They are scoped, real-world problems that require exactly the subjects you just learned.

- **Industry Context**
- **The Problem**
- **Constraints**
- **Starter Code**
- **Solution** (hidden)

## Philosophy

- **Small subjects.** Matrix multiplication is one file. Not "Linear Algebra Chapter 3."
- **No imaginary problems.** If the practice feels like a textbook, it failed.
- **Science is connected.** Data engineering, software engineering, and AI are not silos. They are interleaved here.
- **Autonomous but cited.** This repo is maintained by an autonomous cron that scrapes, summarizes, and cites. It does not invent sources.

## Progress

<!-- PROGRESS-BADGE-START -->
**Current Status:** 0 / 126 items built.
<!-- PROGRESS-BADGE-END -->

Check `MANIFEST.json` for the full queue.

## License

MIT. Use it, fork it, contribute to it.

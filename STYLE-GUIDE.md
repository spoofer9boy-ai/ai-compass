# Style Guide for The AI Compass

This document governs how every subject and practice file must be written. The autonomous builder references this on every run to maintain consistency.

## Voice and Tone

- **Direct.** Write to the reader as a peer engineer, not a professor.
- **Honest.** If a concept is mostly abstract but has one critical real-world use, say that.
- **No hype.** "Revolutionary" and "game-changing" are banned. Say what it does and where it breaks.

## Subject File Template

```markdown
# [Subject Title]

**Phase:** [Phase Name]  
**Prerequisites:** [List of subject IDs]  
**Estimated Time:** [X] minutes

## Why am I learning this?

[2-4 paragraphs. Start with the professional reality, not the definition.
Example: "You will never implement a matrix multiplication kernel in production.
But you will spend three hours debugging why `torch.matmul` throws a dimension
mismatch on a batch of 10,000 embeddings. This file exists so that mismatch
makes sense in 30 seconds, not 30 minutes."]

## Where will I be using it?

[Bullet list of concrete systems, tools, or job tasks.
Each bullet must name a specific domain or tool.]

- **Transformers:** Computing attention scores as `Q × K^T`.
- **Recommender Systems:** User embeddings × item embeddings.
- **Computer Graphics:** Transforming 3D vertices to 2D screen space.
- **Data Pipelines:** Reshaping batch tensors before feeding a model.

## Resources

[Exactly 3-5 high-quality sources. Cite the original, not a Medium rehash.
Format: `- [Title](URL) — One-line description.`]

- [3Blue1Brown: Linear Algebra](https://www.3blue1brown.com/?topic=linear-algebra) — Visual intuition for linear algebra concepts.
- [PyTorch Docs: torch.matmul](https://pytorch.org/docs/stable/generated/torch.matmul.html) — The API you will actually use.

## Appendix

[Optional. Mathematical notation, edge cases, historical context, or deeper references.
Keep it skimmable.]

### Notation

- $\mathbf{A} \in \mathbb{R}^{m \times n}$: A matrix with $m$ rows and $n$ columns.

### Common Pitfalls

- Confusing matrix multiplication with element-wise (Hadamard) product.
- Ignoring broadcasting rules in NumPy/PyTorch.

### Further Reading

- [Distill.pub: Explorable Explanations](https://distill.pub) — If you want to go deeper.
```

## Practice File Template

```markdown
# Practice: [Problem Name]

**Phase:** [Phase Name]  
**Subjects Required:** [List of subject IDs and titles]  
**Estimated Time:** [X] minutes  
**Difficulty:** [Beginner / Intermediate / Advanced]

## Industry Context

[One paragraph setting the scene. Name a company type, a role, and a real constraint.
Example: "You are the first ML engineer at a legal-tech startup. The product team
wants semantic search across 100,000 contracts. You cannot afford OpenAI API costs
at scale. You need to build an in-house retrieval system."]

## The Problem

[Clear, scoped problem statement. It should not require subjects outside the
prerequisites. If it touches adjacent concepts, provide starter code that handles
the adjacency so the student focuses on the target subjects.]

## Constraints

- [Memory, latency, accuracy, or tooling limits.]
- [e.g., "Do not use scikit-learn. Implement from scratch with NumPy."]
- [e.g., "Must run on a single CPU core in under 2 seconds."]

## Starter Code

```python
# Partial implementation. Contains a deliberate gap or bug related to the target subjects.
import numpy as np

# TODO: Implement the core logic using only the subjects covered so far.
```

## Evaluation Criteria

[How to know if the solution is correct. Be specific.]

1. Correctness: Output matches expected shape and values on test cases.
2. Efficiency: Runs within the time constraint.
3. Edge handling: Gracefully handles empty inputs or mismatched dimensions.

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
# Full solution with comments explaining the connection to each subject.
```

</details>

## What You Actually Learned

[Explicit bridge back to the subjects.]

- **Vectors:** You represented documents as high-dimensional vectors.
- **Dot Product:** You used it to measure similarity between vectors.
- **Matrix Multiplication:** You batched the similarity computation for efficiency.
```

## Formatting Rules

1. **Math:** Use standard LaTeX delimiters (`$...$` for inline, `$$...$$` for display).
2. **Code:** All code blocks must specify the language (`python`, `bash`, etc.).
3. **Links:** All external links must be HTTPS. Prefer primary sources.
4. **Images:** Avoid images. If absolutely necessary, use a link to the image, do not embed binary files.
5. **Length:** Subjects should be 500–1500 words. Practices should be 800–2000 words.
6. **No ads, no affiliate links, no course promotions.**

## Commit Message Format

The autonomous builder uses:

```text
feat(PHASE-XX): add [subject-title] subject
feat(PHASE-XX): add [practice-title] practice
```

## Source Quality Hierarchy

When scraping, prefer sources in this order:

1. **Academic papers** (arXiv, OpenReview, ACL Anthology)
2. **Official documentation** (PyTorch, TensorFlow, HuggingFace, scikit-learn)
3. **Engineering blogs** (Google AI, OpenAI, Anthropic, DeepMind, Netflix Tech Blog, Uber Engineering)
4. **Educational content** (3Blue1Brown, Distill.pub, fast.ai, CS231n)
5. **Open source repositories** (reference implementations, well-documented issues)

Avoid: Medium posts without code, SEO farms, unverified tutorials, LLM-generated summaries of unknown origin.
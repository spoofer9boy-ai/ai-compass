# Unit Testing Basics

**Phase:** PHASE-01-foundations  
**Prerequisites:** 30 (Python Functions and Classes)  
**Estimated Time:** 45 minutes

## Why am I learning this?

You will write more test code than production code in any serious codebase. That is not an exaggeration. In large ML systems, the ratio of test lines to source lines often exceeds 2:1. The reason is simple: a model that trains successfully but produces silently wrong predictions is worse than a model that crashes immediately. Unit tests are the mechanism that turns silent failures into loud, actionable ones.

You will never ship a feature to production without a test suite running in CI. You will never refactor a data pipeline without tests telling you whether you broke an edge case. And you will never debug a flaky training job faster than when you have a minimal, reproducible unit test isolating the bug. This file exists so that writing your first test feels mechanical, not mysterious.

## Where will I be using it?

- **ML Pipelines:** Testing data validation logic, feature transforms, and model inference wrappers before they touch a GPU.
- **CI/CD:** Every pull request triggers a pytest suite. A red build blocks merge.
- **Refactoring:** Renaming a column in a DataFrame? Tests tell you which downstream function assumed the old name.
- **Reproducibility:** Pinning expected behavior in code prevents "it works on my machine" debates.
- **Open Source:** Contributing to PyTorch or scikit-learn requires tests for any bug fix or new feature.

## Resources

- [pytest: Getting Started](https://docs.pytest.org/en/stable/getting-started.html) — The official guide to writing and running your first tests.
- [Python unittest — Unit testing framework](https://docs.python.org/3/library/unittest.html) — Standard library reference for when you cannot add external dependencies.
- [Real Python: Getting Started with Testing in Python](https://realpython.com/python-testing/) — Practical introduction to both `unittest` and `pytest` with runnable examples.
- [Google Testing Blog: Where do I start?](https://testing.googleblog.com/2014/10/testing-on-toilet-where-do-i-start.html) — Short, opinionated guidance on what to test first in a new codebase.
- [pytest fixtures documentation](https://docs.pytest.org/en/stable/explanation/fixtures.html) — How to share setup logic across tests without copy-paste.

## Appendix

### Notation

- **Test case:** A single function that exercises one unit of code with one set of inputs and asserts an expected output.
- **Test suite:** A collection of test cases, typically organized in a module or class.
- **Fixture:** Reusable setup/teardown logic (e.g., creating a temporary directory or a mock dataset).
- **Assertion:** A boolean check that fails the test if it evaluates to `False`.

### Common Pitfalls

- **Testing implementation, not behavior.** If your test breaks because you renamed a private variable, you tested the wrong thing. Assert outputs and side effects, not internal state.
- **Monolithic tests.** A test that validates ten things at once gives you a binary pass/fail with no diagnostic signal. One assertion per test is ideal.
- **Ignoring edge cases.** Empty lists, `None`, zero, and very large numbers are where bugs hide. Include them deliberately.
- **Hard-coding absolute paths.** Tests that rely on `/home/you/data.csv` fail on CI. Use temporary directories or relative paths resolved from `__file__`.
- **Skipping tests silently.** `@pytest.mark.skip` without a reason accumulates dead weight. Skip with a linked issue or delete the test.

### Further Reading

- [Obey the Testing Goat](https://www.obeythetestinggoat.com/) — Harry Percival’s free book on test-driven development with Python.
- [Martin Fowler: Unit Test](https://martinfowler.com/bliki/UnitTest.html) — Definitions and trade-offs around the term "unit test."

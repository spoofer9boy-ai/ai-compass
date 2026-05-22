# Python Functions and Classes

**Phase:** PHASE-01-foundations  
**Prerequisites:** None  
**Estimated Time:** 50 minutes

## Why am I learning this?

You will write thousands of functions and hundreds of classes during your career. The difference between a script that "works once" and production AI code is almost always the quality of the abstractions: small, testable functions with clear contracts, and classes that encapsulate state without leaking it. If you cannot write a clean `Dataset` class or a reusable training loop function, you will drown in copy-pasted notebooks that break every time the data schema changes.

Functions are the atomic unit of reusable logic. In ML pipelines, they are your data loaders, your feature transforms, your metric calculators, and your inference wrappers. Classes are how you bundle data with behavior—think PyTorch `nn.Module`, HuggingFace `Trainer`, or scikit-learn `BaseEstimator`. Understanding *when* to use a plain function versus a class is a skill that separates junior engineers from senior ones. This file gives you the foundation to make that call correctly.

## Where will I be using it?

- **PyTorch / TensorFlow:** Every model is a class inheriting from `nn.Module` or `keras.Model`. You override `forward()` or `call()`—which are just methods (functions bound to an object).
- **Data Pipelines:** `torch.utils.data.Dataset` is an abstract class you subclass to implement `__len__` and `__getitem__`. Get these wrong and your DataLoader hangs or leaks memory.
- **Configuration Objects:** Hydra, dataclasses, and Pydantic models all rely on class mechanics to validate and structure hyperparameters before training starts.
- **Experiment Tracking:** Wrapping `wandb.log` or `mlflow.log_metric` in small functions keeps your training script readable and makes it trivial to swap trackers later.
- **API Serving:** FastAPI and Flask route handlers are functions (or class methods) that receive requests, call your model, and return responses.

## Resources

- [Python Docs: Defining Functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions) — Official tutorial on function basics, default arguments, and keyword args.
- [Python Docs: Classes](https://docs.python.org/3/tutorial/classes.html) — The authoritative walkthrough of Python’s class mechanics, inheritance, and the `self` convention.
- [Python Docs: Data Model](https://docs.python.org/3/reference/datamodel.html) — The deep reference on dunder methods (`__init__`, `__call__`, `__repr__`, etc.) that power every framework you will use.
- [Real Python: Python Classes](https://realpython.com/python-classes/) — Practical guide to OOP in Python with concrete examples and anti-patterns to avoid.
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/) — The industry standard for naming, spacing, and structure. Read it once, reference it forever.

## Appendix

### Notation

- `def f(x: float) -> float:`: A function named `f` that takes a float and returns a float.
- `class MyClass:`: A user-defined type that bundles data (attributes) and behavior (methods).
- `self`: The instance reference passed implicitly as the first argument to instance methods.

### Common Pitfalls

- **Mutable default arguments:** `def f(x=[])` will reuse the same list across calls. Use `None` and initialize inside the function.
- **Forgetting `self`:** Inside a class method, `self.attribute` accesses instance state. Omitting `self` creates a local variable that disappears when the method returns.
- **Over-engineering with classes:** If a class has no state and only one public method, it should probably be a function.
- **Shadowing built-ins:** Naming a variable `list` or `str` overrides the built-in type and causes subtle bugs.

### Further Reading

- [Python Docs: Built-in Functions](https://docs.python.org/3/library/functions.html) — Quick reference for `map`, `filter`, `zip`, `enumerate`, and other primitives you will use daily.
- [Python Docs: typing](https://docs.python.org/3/library/typing.html) — Type hints make large codebases navigable and catch bugs before runtime. Start with `List`, `Dict`, `Tuple`, and `Optional`.

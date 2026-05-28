# Building REST APIs

**Phase:** PHASE-01-foundations  
**Prerequisites:** 30 (Python Functions and Classes), 32 (Unit Testing Basics)  
**Estimated Time:** 60 minutes

## Why am I learning this?

You will not ship a model by emailing a `.pkl` file to the frontend team. Production AI systems are consumed through APIs: a mobile app sends an image to a classification endpoint, a data pipeline POSTs a batch of records to an embedding service, or a chatbot streams tokens from a text-generation endpoint. If you cannot build the interface that exposes your model to the rest of the world, you are not an AI engineer—you are a notebook user.

REST is the dominant lingua franca for these interfaces. It is not the only option (gRPC and GraphQL have their places), but it is the one you will encounter in nearly every job. Understanding REST means understanding how to map your Python functions to HTTP resources, how to communicate errors unambiguously, and how to design endpoints that do not become a maintenance burden as the product evolves. This file exists so that when a product manager asks for a "simple endpoint to score leads," you know exactly what that entails—and where the hidden complexity lives.

## Where will I be using it?

- **Model Serving:** Wrapping a PyTorch or scikit-learn model in a FastAPI/Flask service that accepts JSON payloads and returns predictions.
- **Data Pipelines:** Building ingestion endpoints that receive structured data, validate it, and queue it for downstream processing.
- **MLOps Platforms:** Integrating with tools like MLflow, Kubeflow, or SageMaker, all of which expose REST APIs for model registry and experiment tracking.
- **Frontend Integration:** Providing the contract that frontend engineers use to display model outputs to users.
- **Microservices:** Decomposing an AI platform into stateless services (embedding, ranking, filtering) that communicate over HTTP.

## Resources

- [MDN: HTTP request methods](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Methods) — Authoritative reference for GET, POST, PUT, PATCH, DELETE semantics and idempotency.
- [Roy Fielding: REST Architectural Style](https://roy.gbiv.com/pubs/dissertation/rest_arch_style.htm) — The original dissertation chapter defining REST; essential for understanding statelessness and the uniform interface.
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) — Modern, type-annotated Python framework; the default choice for high-performance model serving.
- [Flask Quickstart](https://flask.palletsprojects.com/en/stable/quickstart/) — Minimalist Python framework; ideal when you need something lightweight and explicit.
- [Swagger: Best Practices in API Design](https://swagger.io/resources/articles/best-practices-in-api-design/) — Practical guidelines for resource naming, HTTP status codes, and error handling.

## Appendix

### Notation

- **Resource:** A noun-like entity identified by a URI (e.g., `/predictions`, `/users/{id}`).
- **Endpoint:** A specific URI + HTTP method combination (e.g., `POST /predictions`).
- **Stateless:** The server does not store client context between requests; each request must contain all necessary information.

### Common Pitfalls

- **Using verbs in URLs:** Prefer `/predictions` over `/getPrediction` or `/createPrediction`. HTTP methods carry the verb.
- **Ignoring status codes:** Returning `200 OK` for every response, including errors, breaks client expectations and caching behavior.
- **Mutable GET requests:** `GET` must not alter server state. Use `POST`, `PUT`, `PATCH`, or `DELETE` for mutations.
- **Leaking implementation details:** Error messages should be helpful to the client, not expose stack traces or database schemas.

### Further Reading

- [MDN: HTTP response status codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status) — Comprehensive guide to 2xx, 3xx, 4xx, and 5xx codes.
- [Requests: Quickstart](https://docs.python-requests.org/en/latest/user/quickstart.html) — Pythonic HTTP client for testing and consuming APIs.

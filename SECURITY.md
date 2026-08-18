# Security Policy

## Scope

Hearthia is a local single-user control plane for llama.cpp and llama-swap on
Apple Silicon. Model files, gateway configuration, logs and an optional Brain
vault can contain sensitive information.

The repository must not contain model weights, local configurations, vault
indexes, logs or credentials.

## Boundary

- The daemon is loopback-only and rejects non-loopback bind addresses.
- It has no authentication and must not be exposed as a remote or multi-user service.
- Context tools can read files available to the local process; use a dedicated local user if needed.
- Model downloads require a file published by Hugging Face with a verifiable SHA-256.
- Configuration replacement is atomic and keeps a local backup.

## Reporting

Do not open a public issue containing model URLs with credentials, local paths,
logs or vault content. Use a private GitHub security advisory or contact the
repository owner through GitHub with redacted details.

## Release rule

Real model loading, restart recovery, memory-pressure behavior and browser
smoke tests remain release evidence gates. Unit tests alone do not establish
safe operation under real model workloads.

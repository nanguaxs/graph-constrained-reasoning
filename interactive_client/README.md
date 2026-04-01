# Interactive Graph-Constrained Client

This folder contains an interactive client that:

1. Loads a local constrained-decoding model once.
2. Accepts user questions in a REPL or single-shot mode.
3. Loads a global KG from `interactive_client/KG.json` by default.
4. Extracts subject entities from the question.
5. Builds a local prefix trie from the entity-centered subgraph.
5. Generates constrained reasoning paths with the existing decoding stack.

## Default config

Edit `interactive_client/default_config.json` or pass a custom config file with:

```bash
python -m interactive_client.graph_reasoning_client --config path/to/config.json
```

## Run interactively

```bash
python -m interactive_client.graph_reasoning_client
```

## Run a single question

```bash
python -m interactive_client.graph_reasoning_client --question "your question here"
```

## Useful overrides

```bash
python -m interactive_client.graph_reasoning_client \
  --question "your question here" \
  --entity "entity name" \
  --global-kg-file interactive_client/KG.json \
  --k 1 \
  --max-new-tokens 64 \
  --subgraph-hops 2 \
  --index-path-length 2
```

## Interactive commands

- `/config`: print the active configuration
- `/quit`: exit the client

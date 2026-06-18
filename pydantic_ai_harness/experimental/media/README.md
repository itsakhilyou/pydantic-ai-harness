# Media stores

> [!WARNING]
> **Experimental.** These stores live under `pydantic_ai_harness.experimental` and may
> change or be removed in any release, without a deprecation period. Import them from the
> experimental path -- there is no top-level export:
>
> ```python
> from pydantic_ai_harness.experimental.media import DiskMediaStore
> ```
>
> Importing any experimental capability emits a `HarnessExperimentalWarning`. Silence **all**
> harness experimental warnings with a single filter (no per-capability lines needed):
>
> ```python
> import warnings
> from pydantic_ai_harness.experimental import HarnessExperimentalWarning
>
> warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)
> ```

Content-addressed byte stores for moving large binary parts out of an agent's message
history. This is a supporting layer, not a standalone capability: today
[`StepPersistence`](../step_persistence/) uses it to keep snapshots small when messages carry
`BinaryContent`. A `MediaExternalizer` capability that rewrites in-flight wire payloads is in
progress and will reuse the same stores.

## The problem

A `BinaryContent` part -- an image, audio clip, or PDF the agent produced or read -- carries its
bytes inline, base64-encoded. Persist a run that holds a few of those and the snapshot balloons;
re-send them on every model request and you pay for them each turn. The bytes need to live
somewhere addressable, with the message holding a reference instead.

## The store

A `MediaStore` is an async, content-addressed bytes store. `put` hashes the bytes and returns a
canonical `media+sha256://<hex>` URI -- the caller never picks the key, and identical bytes dedupe
automatically.

```python
import anyio
from pydantic_ai_harness.experimental.media import DiskMediaStore

async def main():
    store = DiskMediaStore('./media')

    uri = await store.put(b'...raw image bytes...')   # 'media+sha256://5608b7...'
    data = await store.get(uri)                        # b'...raw image bytes...'
    assert await store.exists(uri)

anyio.run(main)
```

Three implementations ship, all satisfying the same `MediaStore` protocol:

| Store | Backing | Use it for |
|---|---|---|
| `DiskMediaStore` | One file per blob under a directory | Local / single-host runs |
| `SqliteMediaStore` | Blobs in a SQLite table | A single-file store with no loose files |
| `S3MediaStore` | An S3 bucket | Shared or multi-host runs; subclass for other providers |

`public_url(uri)` returns a URL the model can fetch directly, or `None` when the store can't
produce one (a local filesystem path is not addressable from a model). Pass a `public_url=`
resolver to `DiskMediaStore` to map blobs onto a URL you serve.

## Externalizing message trees

`externalize_media` walks a message tree and replaces every inline `BinaryContent` part at or
above `threshold_bytes` with a small marker carrying the blob's URI; `restore_media` is the exact
inverse. Both return a new tree -- the input is not mutated -- and the round-trip preserves every
field on the part, so it survives new `BinaryContent` fields added upstream.

```python
import anyio
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai_harness.experimental.media import (
    DiskMediaStore,
    externalize_media,
    restore_media,
)

async def main(result):
    store = DiskMediaStore('./media')

    # A JSON-shaped message tree -- the form you persist and reload.
    messages = ModelMessagesTypeAdapter.dump_python(result.all_messages())

    slim = await externalize_media(messages, media_store=store, threshold_bytes=10_000)
    # ...persist `slim` (small: bytes live in the store, not the tree)...

    full = await restore_media(slim, media_store=store)  # re-inlines the bytes
    reloaded = ModelMessagesTypeAdapter.validate_python(full)

anyio.run(main, agent_run_result)
```

## Custom layout

`DiskMediaStore(key_strategy=...)` overrides the relative path a blob lands at inside the
directory (e.g. `images/<digest>.png`). The returned path must be relative and free of `..`
segments. If the strategy reads `context.media_type` to pick the path, the same `MediaContext`
must be supplied on `get`/`exists` so the blob can be found again.

## Further reading

- [StepPersistence](../step_persistence/) -- the capability that uses these stores today
- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)

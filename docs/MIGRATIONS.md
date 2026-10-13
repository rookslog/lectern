# Bundle migrations

## Manifest `1.0.0` and `1.1.0` compatibility

New bundles declare `1.1.0`. This minor version only adds optional transcript
language/ASR backend metadata, external source metadata, and stage warnings, so
`1.0.0` bundles remain readable without migration. Readers accept exactly these
two versions, not later same-major versions such as `1.2.0`.

`lectern migrate BUNDLE` on a valid `1.0.0` bundle is a validated no-op: it
preserves the version stamp and evidence and reports `already_current` in JSON,
with the message "already readable; 1.0.0 is a subset of 1.1.0" on stderr.
Valid `1.1.0` bundles also receive a validated no-op. Keeping the established
`0.1.0` to `1.0.0` target preserves its backup, staging, and recovery contract;
there is no new migration or evidence rewrite for optional fields.

## Manifest `0.1.0` to `1.0.0`

Run `lectern migrate BUNDLE` while no other Lectern process is writing that
bundle. Lectern validates the complete source bundle, builds and validates a
sibling target, keeps the migrated bundle at `BUNDLE`, and retains the original
as `BUNDLE.v0.1.0.bak`. It does not delete the backup.

The major version changes because a local manifest's `source.ref` now denotes
`sha256:<digest>` rather than a filesystem path. The source digest and byte size
come from the bundle's existing `source.json`; media and transcript content are
not sent anywhere and are not regenerated.

The command refuses malformed or hash-invalid bundles, symlinks, unknown
versions, and unrelated occupied backup/staging roles before overwriting them. Its
JSON and error output do not echo filesystem paths. Re-running it on a valid
`1.0.0` bundle is a checked no-op.

Legacy JSON extension fields that are not declared by Lectern's artifact
models are refused rather than silently discarded. The original `0.1.0`
bundle remains unchanged so an operator can inspect or export that extension
data before retrying with a supported artifact shape.

Every transcript segment must contain at least one non-whitespace character,
using Python's Unicode whitespace definition. Migration refuses empty or
whitespace-only segments and preserves the original bundle and unrelated sibling
directories on refusal. Recovery may discard and rebuild obsolete staging that
Lectern identifies as its own, even if the retry is refused. It does not trim
text, remove or renumber segments, or invent
replacement evidence. Valid segment text and the segment artifact's bytes are
preserved exactly, including surrounding whitespace and internal line breaks.

A successful migration or already-current no-op requires manifest declarations
for both `source.json` and the transcript segments selected by that document.
Every declaration of either artifact must match its actual bytes and SHA-256
digest. Missing declarations are refused; migration does not invent them.
Declared alternate segment paths remain supported. Library readiness, indexing,
and new citations use this same selected-evidence requirement. Existing-anchor
diagnostics can still report changed evidence without minting a new citation.

Migration transforms supported machine-location fields; it is not arbitrary
content anonymization. User-authored transcript text and source labels are
preserved even when they contain POSIX- or Windows-style paths.

Migration validates target models more strictly than the compatibility reader.
It can refuse integral floating-point representations such as `12.0` for an
integer field even when `Manifest.load` accepts `12.0` as the equivalent
value `12`.
This also applies to validation of an already-current no-op; refusal preserves
the original evidence rather than silently rewriting its numeric representation.

To roll back while Lectern is stopped, move the `1.0.0` directory aside and
rename `BUNDLE.v0.1.0.bak` to the original bundle name. Do not merge files from
the two schema versions.

Transcript metadata must agree with source evidence: `source_media.sha256` must
match `source.json`'s original-media SHA, and a supplied non-null `source_media.bytes`
must match its byte size (including an explicit zero). Omitted or null size remains
valid. Independently, `normalized_audio` must name a contained regular file whose
SHA and size match its metadata; original and normalized media identities can
differ. These checks follow the selected metadata path and the named normalized
artifact, without imposing a fixed filename or new declaration membership.
Contradictions are refused, not repaired. The generated `metadata.artifacts` map
need not equal the source document's current selected-artifact pointers. These
checks establish consistency of retained claims, not historical authenticity or
transcript faithfulness; the existing source/backup/staging recovery rules apply.

Migration result output redacts path-bearing bundle IDs using the existing path
projection. This display value need not be reversible: the internal result ID,
on-disk manifest identity and retained backup identity remain unchanged. Ordinary
bundle IDs are displayed unchanged.


Publication and recovery require atomic no-replace directory rename support:
macOS `renameatx_np` with `RENAME_EXCL`, or Linux `renameat2` with
`RENAME_NOREPLACE`, on a filesystem supporting the flag. If the platform, native
interface, kernel or filesystem lacks that capability, migration refuses without
falling back to an overwriting rename. Already-current validation does not load
this publication interface. Preparation may already have created owned staging.

A destination that appears after the absence check is preserved, even when it is
an empty directory. If source publication and restoration are blocked by an
occupied source role, the original backup and marked staging remain available;
migration does not remove the conflicting role to restore the source. Follow the
existing recovery procedure after resolving that unrelated entry. Keep other
Lectern writers quiescent as required above. This does not protect against hostile
parent/source replacement or promise network-filesystem or power-loss durability.

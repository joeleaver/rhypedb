use thiserror::Error;

pub type EngineResult<T> = Result<T, EngineError>;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("storage error: {0}")]
    Storage(#[from] rhypedb_storage::Error),

    #[error("schema error: {0}")]
    Schema(#[from] rhypedb_schema::SchemaError),

    #[error("catalog error: {0}")]
    Catalog(#[from] CatalogError),

    #[error("type not found: {0}")]
    TypeNotFound(String),

    #[error("object not found: {type_name}:{object_id}")]
    ObjectNotFound { type_name: String, object_id: u64 },

    #[error("field not found: '{field}' on type '{type_name}'")]
    FieldNotFound { type_name: String, field: String },

    #[error("type mismatch for field '{field}': expected {expected}, got {got}")]
    TypeMismatch {
        field: String,
        expected: String,
        got: String,
    },

    #[error("unique constraint violated: {type_name}.{field} = {value}")]
    UniqueViolation {
        type_name: String,
        field: String,
        value: String,
    },

    #[error(
        "delete denied: {type_name}:{object_id} is referenced by {referencing_type}.{referencing_field}"
    )]
    DeleteDenied {
        type_name: String,
        object_id: u64,
        referencing_type: String,
        referencing_field: String,
    },

    #[error("write conflict")]
    WriteConflict,

    /// The named type was retired (tombstoned) via a previous
    /// schema-shrink open. Its on-disk data is unreachable through
    /// every public API. The error carries enough metadata that the
    /// operator can confirm "yes, this is the one we dropped on date
    /// X" without consulting a separate audit log.
    #[error(
        "type {name:?} (id={id}) was retired at {retired_at_unix_ms} ms — its data is no longer accessible; update your schema or restore from a backup taken before retirement"
    )]
    TypeRetired {
        name: String,
        id: u64,
        retired_at_unix_ms: u64,
    },

    /// The named field on a still-live parent type was retired via a
    /// previous schema-shrink open. Reads against objects of the parent
    /// type still succeed but strip this field from the returned
    /// FieldMap; APIs that NAME the field error with this variant.
    #[error(
        "field {type_name:?}.{field:?} (id={field_id}) was retired at {retired_at_unix_ms} ms — reads via this name are no longer valid; the parent type's objects are still accessible but the field is stripped from results"
    )]
    FieldRetired {
        type_name: String,
        field: String,
        field_id: u64,
        retired_at_unix_ms: u64,
    },

    /// The named relation was retired. Traversal through it returns
    /// an empty edge set; APIs that NAME the relation (e.g. `link`,
    /// `unlink`) error with this variant.
    #[error(
        "relation {type_name:?}.{relation:?} (id={relation_id}) was retired at {retired_at_unix_ms} ms — traversal yields no edges; the parent type's objects are still accessible"
    )]
    RelationRetired {
        type_name: String,
        relation: String,
        relation_id: u64,
        retired_at_unix_ms: u64,
    },

    /// A method was called on a `Database` handle that has been
    /// migrated away — one of the `_consuming` migrate variants
    /// returned a fresh `Arc<Database>` and marked this one as
    /// dead. The caller should be using the new handle. The error
    /// is a guardrail against stale-cache reads (the old schema /
    /// field_ids / indexed_fields are no longer valid for the
    /// underlying storage shape).
    #[error("database handle has been migrated away — switch to the Arc returned by the migrate verb")]
    DatabaseMigratedAway,

    /// A write was rejected because its target type is under an active
    /// chunked field-type migration (shadow-field card 1). Card 1 quiesces
    /// the migrating type — no concurrent writes to its objects — until the
    /// migration completes or is cancelled. Card 2 lifts this by carrying
    /// live writes through a double-write hook. Fires for create / update /
    /// delete, and for a cascade-delete that reaches the migrating type
    /// from a delete issued against another type.
    #[error(
        "type {type_name} is quiesced by in-flight migration plan {plan_id}; writes are rejected until it completes or is cancelled"
    )]
    MigrationTypeQuiesced { type_name: String, plan_id: u64 },

    /// `create_field_type_migration` was asked for a converter `(name,
    /// version)` that is not in this `Database`'s converter registry. Fail
    /// fast at create rather than persist a plan that can never run; the
    /// operator must `register_converter` first. (After a restart a plan
    /// whose converter is missing parks `AwaitingConverter` instead — at
    /// create time we have no plan to park.)
    #[error("no converter registered as ({name}, v{version}); call register_converter before creating the migration")]
    ConverterNotRegistered { name: String, version: u32 },

    /// On open, an in-flight migration plan was found but the schema passed
    /// to `open` still declares the field at its OLD (source) type. Driving
    /// the migration would flip the catalog to the target kind while this
    /// handle validates writes against the source kind — silent corruption.
    /// Reopen with the schema where the field has the TARGET type to let
    /// auto-resume finish the cutover.
    #[error(
        "migration plan {plan_id} is resuming a change to kind {expected} but the open schema still declares the field as kind {found}; reopen with the target schema"
    )]
    MigrationResumeSchemaMismatch {
        plan_id: u64,
        expected: &'static str,
        found: &'static str,
    },
}

/// Failures specific to the persisted schema catalog (see
/// `crates/rhypedb-engine/src/catalog.rs`). Each variant maps to a
/// distinct operator action — distinguishing "your binary is too old"
/// from "your schema dropped a type" from "the catalog is corrupted"
/// matters because the responses are completely different.
#[derive(Debug, Error)]
pub enum CatalogError {
    /// The on-disk catalog declares a format version newer than this
    /// binary understands. Refuse cleanly so a downgraded binary can't
    /// silently truncate forward-compat state. Upgrade the binary.
    #[error(
        "catalog format version {got} is not supported by this binary (max supported = {max_supported}); upgrade the rhypedb binary"
    )]
    UnsupportedFormat { got: u64, max_supported: u64 },

    /// A specific catalog row uses a record-format byte newer than this
    /// binary knows. Distinct from `UnsupportedFormat` because the
    /// catalog-wide format may match while individual rows carry future
    /// semantics (phases 2+).
    #[error("catalog row {row} uses record format version {got}; max supported = {max_supported}")]
    UnsupportedRecordFormat {
        row: String,
        got: u8,
        max_supported: u8,
    },

    /// Value-kind discriminant (byte 1 of every catalog value) does not
    /// match any kind this binary knows.
    #[error("catalog value at {key_debug} has unknown value-kind tag 0x{tag:02x}")]
    UnknownValueKind { key_debug: String, tag: u8 },

    /// Value-kind discriminant didn't match the expected kind for the
    /// catalog row we were reading. Indicates external tampering — e.g.
    /// a counter row was overwritten with an id-entry envelope.
    #[error(
        "catalog value at {key_debug} has wrong kind: expected {expected}, got tag 0x{got_tag:02x}"
    )]
    WrongValueKind {
        key_debug: String,
        expected: &'static str,
        got_tag: u8,
    },

    /// Catalog row payload was shorter than the format requires.
    #[error("catalog value at {key_debug} is truncated (len={len}, expected at least {min})")]
    Truncated {
        key_debug: String,
        len: usize,
        min: usize,
    },

    /// Two TLV records with the same tag inside one id-entry body.
    #[error("catalog row {key_debug} has duplicate TLV tag 0x{tag:02x}")]
    DuplicateTlv { key_debug: String, tag: u8 },

    /// A required TLV tag was absent from an id-entry body.
    #[error("catalog row {key_debug} missing required TLV tag 0x{tag:02x}")]
    MissingRequiredTlv { key_debug: String, tag: u8 },

    /// A required top-level catalog key was absent on a catalog that
    /// otherwise appeared to be initialized (e.g. `c:I:` present but
    /// `c:N:T` missing).
    #[error("required catalog key missing: {key_debug}")]
    MissingRequiredKey { key_debug: String },

    /// A catalog key's payload (the bytes after the subtype tag) was
    /// malformed — e.g. a `c:E:` key with no NUL separator.
    #[error("catalog key has malformed payload: {key_debug}")]
    MalformedKey { key_debug: String },

    /// The catalog header keys (`c:F:` and `c:I:`) disagree about
    /// whether the catalog is initialized. Phase 1 recovers by clearing
    /// the catalog and re-running deterministic backfill; this error is
    /// only surfaced if the recovery itself fails.
    #[error(
        "catalog header keys are inconsistent (c:F: present={format_present}, c:I: present={initialized_present}) after a recovery attempt; manual inspection required"
    )]
    PartialCatalog {
        format_present: bool,
        initialized_present: bool,
    },

    /// The new schema removes catalog entries but the caller didn't
    /// pass `OpenOptions::allow_schema_shrink(true)`. The hint surfaces
    /// the opt-in path in the error itself.
    #[error(
        "schema would drop catalog entries (types={dropped_types:?}, fields={dropped_fields:?}, relations={dropped_rels:?}). To retire these entries, set OpenOptions::allow_schema_shrink(true); retirement is one-way and cannot be reversed"
    )]
    SchemaShrinkRequiresOptIn {
        dropped_types: Vec<String>,
        dropped_fields: Vec<String>,
        dropped_rels: Vec<String>,
    },

    /// The schema names an entry whose name is already tombstoned in
    /// the catalog. Re-binding the name would either resurrect the
    /// retired data (if we reused the old ID) or strand it (if we
    /// allocated a fresh ID). Refuse loudly. The operator must pick a
    /// different name or restore from a backup taken before retirement.
    #[error(
        "{kind} name {name:?} is tombstoned in the catalog (retired id={retired_id}); re-binding a retired name is forbidden — pick a different name or restore from a backup taken before retirement"
    )]
    NameReuseOfRetiredEntry {
        kind: &'static str,
        name: String,
        retired_id: u64,
    },

    /// A catalog row carries `status = Tombstoned` but the on-disk
    /// `c:F:` is v1 (which never wrote tombstones). Indicates manual
    /// tampering or a binary that bumped a row format without bumping
    /// the catalog format. Refuse to open.
    #[error(
        "{row_kind} catalog row id={row_id} is tombstoned but the catalog format is v1 — catalog is internally inconsistent"
    )]
    TombstoneOnV1Catalog {
        row_kind: &'static str,
        row_id: u64,
    },

    /// A catalog row carries an unknown tombstone-status byte. The
    /// decoder refuses rather than guessing whether the row is live or
    /// retired — getting that decision wrong silently corrupts queries.
    #[error("catalog row {row} has unknown tombstone-status byte 0x{status:02x}")]
    UnknownTombstoneStatus { row: String, status: u8 },

    /// A catalog row carries an unknown retirement-reason byte.
    #[error("catalog row {row} has unknown retirement-reason byte 0x{reason:02x}")]
    UnknownRetireReason { row: String, reason: u8 },

    /// A migration plan record (`c:P:<id>`) carries an unknown status
    /// byte. The decoder refuses rather than guessing a lifecycle state.
    #[error("migration plan {row} has unknown status byte 0x{status:02x}")]
    UnknownMigrationStatus { row: String, status: u8 },

    /// A migration plan record (`c:P:<id>`) is structurally malformed
    /// (e.g. a non-utf8 string TLV). Distinct from `Truncated` so the
    /// operator can tell a torn write from a corrupt field value.
    #[error("migration plan {row} is malformed: {reason}")]
    MalformedMigrationPlan { row: String, reason: &'static str },

    /// An offline single-commit field-type change was refused because a
    /// chunked migration plan is already in flight on the same field
    /// (shadow-field card 1). Running both would put two independent
    /// flippers + a foreign converter on the same field — corruption.
    /// Cancel or finish the plan first.
    #[error(
        "field {qualified} has an active migration plan {plan_id}; finish or cancel it before an offline change_field_type"
    )]
    MigrationFieldHasActivePlan { qualified: String, plan_id: u64 },

    /// A catalog row's `previous_names` TLV has `count = 0`. An empty
    /// chain should be omitted entirely; storing one with count zero
    /// indicates external tampering or a writer bug.
    #[error("catalog row {row} has empty rename chain (count = 0); absent chain should be omitted")]
    EmptyRenameChain { row: String },

    /// A catalog row's `previous_names` TLV declares more entries than
    /// the per-row cap.
    #[error("catalog row {row} has rename chain count {count} which exceeds the per-row cap of {cap}")]
    RenameChainOverCap { row: String, count: usize, cap: usize },

    /// `Database::migrate` plan's source name doesn't exist in the
    /// catalog (or refers to a tombstoned entry).
    #[error("rename source {kind} {name:?} is not a live entry in the catalog — old name must refer to a live, un-retired entry")]
    RenameSourceNotFound { kind: &'static str, name: String },

    /// The source entry the operator named has been tombstoned.
    /// Retired entries cannot be renamed.
    #[error("rename source {kind} {name:?} (id={retired_id}) was retired at {retired_at_ms} ms — retired entries cannot be renamed")]
    RenameSourceRetired {
        kind: &'static str,
        name: String,
        retired_id: u64,
        retired_at_ms: u64,
    },

    /// The new name already names a live catalog entry.
    #[error("rename target {kind} {name:?} already exists in the catalog (id={existing_id}); pick a different new name or rename the existing entry first")]
    RenameTargetCollision {
        kind: &'static str,
        name: String,
        existing_id: u64,
    },

    /// The new name is tombstoned in the catalog. Re-binding a retired
    /// name is forbidden (would either resurrect retired data via the
    /// old ID or strand it via a fresh ID).
    #[error("rename target {kind} {name:?} is tombstoned (id={retired_id}); re-binding a retired name is forbidden by the catalog")]
    RenameTargetIsRetired {
        kind: &'static str,
        name: String,
        retired_id: u64,
    },

    /// `old == new` — no-op rename rejected up front rather than
    /// silently committed.
    #[error("rename {kind} {name:?}: old and new names are equal; remove the verb from the plan")]
    RenameNoOp { kind: &'static str, name: String },

    /// Rename would push the per-row rename history beyond
    /// `MAX_RENAME_HISTORY`. Forces the operator to make a deliberate
    /// decision rather than silently dropping audit data.
    #[error("rename target {kind} {name:?} would exceed the per-row rename history cap of {cap} entries")]
    RenameHistoryCapExceeded {
        kind: &'static str,
        name: String,
        cap: usize,
    },

    /// A catalog row's `type_change_history` TLV has count = 0.
    #[error("catalog row {row} has empty type-change chain (count = 0); absent chain should be omitted")]
    EmptyTypeChangeChain { row: String },

    /// A catalog row's `type_change_history` TLV declares more entries
    /// than the per-row cap.
    #[error("catalog row {row} has type-change chain count {count} which exceeds the per-row cap of {cap}")]
    TypeChangeChainOverCap { row: String, count: usize, cap: usize },

    /// `Database::change_field_type` named a field whose qualified name
    /// is not present in the catalog (or has been tombstoned).
    #[error("change_field_type source {qualified:?} is not a live field in the catalog")]
    FieldTypeChangeSourceNotFound { qualified: String },

    /// The named field is tombstoned. Retired fields cannot be migrated.
    #[error("change_field_type source {qualified:?} (id={field_id}) is tombstoned at {retired_at_ms} ms; retired fields cannot be migrated")]
    FieldTypeChangeSourceRetired {
        qualified: String,
        field_id: u64,
        retired_at_ms: u64,
    },

    /// The current and target kinds are the same. No-op rejected
    /// up-front rather than scanning every object for nothing.
    #[error("change_field_type {qualified:?}: source and target kind are the same; remove the verb")]
    FieldTypeChangeNoOp { qualified: String },

    /// The target kind has no writable `Value` representation (DateTime /
    /// Json), so a converter could never produce a matching value and the
    /// migration would fail every row. Refused up front.
    #[error("change_field_type {qualified:?}: target kind {kind} has no writable Value representation; pick a representable scalar")]
    FieldTypeChangeUnrepresentableTarget {
        qualified: String,
        kind: &'static str,
    },

    /// Card 4/5 phase 1 only supports migrating plain scalar fields.
    /// `@indexed`, `@unique`, `@vectorize` are deferred to a follow-on
    /// card because each requires additional on-disk rewriting
    /// (index rebuild, uniqueness re-check, embedding pipeline).
    #[error("change_field_type {qualified:?}: field carries the {directive:?} directive which is not supported by the phase-1 verb; the {planned_phase} card will lift this restriction")]
    FieldTypeChangeDirectiveUnsupported {
        qualified: String,
        directive: &'static str,
        planned_phase: &'static str,
    },

    /// The verb only handles scalar field-type changes.
    #[error("change_field_type {qualified:?}: only scalar fields can be migrated (current kind = {current_kind}, requested kind = {requested_kind})")]
    FieldTypeChangeNonScalar {
        qualified: String,
        current_kind: &'static str,
        requested_kind: &'static str,
    },

    /// The operator-provided converter returned an error for one of
    /// the rows. The whole migration aborts; no catalog mutation has
    /// been committed.
    #[error("change_field_type {qualified:?}: converter failed on object id={object_id}: {reason}")]
    FieldTypeChangeConverterFailed {
        qualified: String,
        object_id: u64,
        reason: String,
    },

    /// The converter returned a Value whose kind doesn't match the
    /// requested target kind.
    #[error("change_field_type {qualified:?}: converter returned a {got_kind} value for object id={object_id} but target kind is {want_kind}")]
    FieldTypeChangeConverterReturnedWrongKind {
        qualified: String,
        object_id: u64,
        got_kind: &'static str,
        want_kind: &'static str,
    },

    /// The audit chain is already at the per-row cap.
    #[error("change_field_type {qualified:?} would exceed the per-row type-change history cap of {cap} entries")]
    FieldTypeChangeHistoryCapExceeded { qualified: String, cap: usize },

    /// A chunked migration found an object row whose stored value for the
    /// migrating field is neither the plan's source kind (convertible) nor
    /// its target kind (already converted — idempotently skipped). The
    /// on-disk data disagrees with the catalog; the migration parks `Failed`
    /// (quiesce held) rather than guess. Card 4 introduces a per-row
    /// ErrorPolicy (quarantine/skip); card 1 fails the whole plan.
    #[error("migration plan {plan_id}: object id={object_id} has unexpected kind {got_kind} for the migrating field (expected source kind {expected_kind})")]
    MigrationRowUnexpectedKind {
        plan_id: u64,
        object_id: u64,
        got_kind: &'static str,
        expected_kind: &'static str,
    },

    /// Card 3/5 phase 2 of rename_field refuses fields carrying
    /// `@indexed`, `@unique`, or `@vectorize`. Each requires follow-on
    /// rewriting that's deliberately deferred (covering-index values
    /// would be stale, uniqueness re-check has to know whether the
    /// rename collides with an existing record under the new name in
    /// the abstract namespace, the vectorize source ref has to update).
    /// Refuse up-front rather than ship a half-correct rewrite.
    #[error("rename_field {qualified:?}: field carries the {directive:?} directive which is not supported by the phase-2 verb; lift this restriction in a follow-on card")]
    RenameFieldDirectiveUnsupported {
        qualified: String,
        directive: &'static str,
    },

    /// Relation fields aren't supported by rename_field phase 2 — every
    /// reverse-edge cover blob from this source type embeds the field
    /// by name, and an in-flight rewrite of those covers is deferred to
    /// the same follow-on card as the directive lift.
    #[error("rename_field {qualified:?}: relation fields cannot be renamed by the phase-2 verb; lift this restriction in a follow-on card")]
    RenameFieldRelationUnsupported { qualified: String },

    /// The supplied migration list has fewer entries than the catalog
    /// reports already-applied. Indicates that the binary was downgraded
    /// or the operator's source-controlled migration list lost entries.
    #[error("migration list has {code_count} entries but catalog reports {catalog_count} applied — DB is ahead of code; do not run with this binary")]
    MigrationListShorterThanApplied {
        code_count: u64,
        catalog_count: u64,
    },

    /// At ordinal `ordinal`, the supplied migration's name differs from
    /// the name recorded in the catalog at that ordinal. Indicates the
    /// operator reordered or renamed an already-applied migration —
    /// silent acceptance would let the same ordinal mean different
    /// things in different deploys. Refuse loudly.
    #[error("migration at ordinal {ordinal} is named {code_name:?} in the code but {catalog_name:?} in the catalog — applied migrations cannot be renamed or reordered")]
    MigrationNameMismatch {
        ordinal: u64,
        code_name: String,
        catalog_name: String,
    },

    /// A migration's closure returned an error from one of its verbs.
    #[error("migration #{ordinal} {name:?} failed: {reason}")]
    MigrationVerbFailed {
        ordinal: u64,
        name: String,
        reason: String,
    },

    /// A field changed shape between catalog and schema — either a
    /// scalar swapped types (Int → String) or scalar↔relation flipped.
    /// On-disk values cannot be reinterpreted under the new kind without
    /// the per-row re-encode pass that card 4/5 will provide.
    #[error(
        "field {qualified} changed kind from {was} to {now}; on-disk values cannot be reinterpreted under the new kind (card 4/5 will add a migration path)"
    )]
    FieldKindChanged {
        qualified: String,
        was: &'static str,
        now: &'static str,
    },

    /// A monotonic counter loaded from the catalog is smaller than or
    /// equal to the largest ID already assigned in the catalog. This is
    /// a tampering indicator (external tooling rolled back the counter)
    /// and would cause silent ID reuse on the next additive allocation.
    #[error(
        "counter c:N:{kind} ({counter}) is not greater than the largest allocated {kind} id ({max_allocated}); catalog is internally inconsistent"
    )]
    CounterBelowAllocatedId {
        kind: &'static str,
        counter: u64,
        max_allocated: u64,
    },

    /// Two catalog rows of the same family carry the same numeric ID.
    /// Internal-consistency violation; engine refuses to open rather
    /// than silently picking a winner.
    #[error("duplicate {kind} id {id} on catalog entries {first} and {second}")]
    DuplicateId {
        kind: &'static str,
        id: u64,
        first: String,
        second: String,
    },

    /// Counter overflowed u64 during allocation. Practically unreachable
    /// (would require 2^64 schema additions) but better than `unwrap`.
    #[error("counter c:N:{kind} overflowed u64")]
    CounterOverflow { kind: &'static str },

    /// Concurrent catalog write conflicted with another opener and
    /// exhausted the bounded internal retry budget. Surface as a hard
    /// error; the caller may retry `Database::open`.
    #[error(
        "concurrent catalog initialization or reconcile detected (write conflict exhausted retry budget); retry Database::open"
    )]
    ConcurrentInit,

    /// An identifier (type name, field name) contains a byte the
    /// catalog encoder reserves (`\x00` or `:`). Should be prevented by
    /// the schema validator; surfaces here as a defense-in-depth check.
    #[error("identifier {name:?} contains reserved byte 0x{byte:02x}")]
    ReservedByteInIdentifier { name: String, byte: u8 },

    /// An identifier contains `__` anywhere — reserved for engine-
    /// internal cover-blob and shadow-field sidecar keys
    /// (`<field>__cover`, `<field>__cover_v`, `<field>__shadow`,
    /// `__cover_<field_id>`, etc.). Schema validation and the rename
    /// verbs both refuse user identifiers in this namespace.
    #[error("identifier {name:?} contains reserved `__` substring (engine sidecar namespace)")]
    ReservedDoubleUnderscoreInIdentifier { name: String },
}

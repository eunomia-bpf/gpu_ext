# Transition-validator plan review

## Round 1: BLOCK

Independent review blocked revision 1 before implementation for seven concrete
reasons:

1. Native interleave levels are `LOW=0`, `MEDIUM=1`, and `HIGH=2`, so using
   zero for both LOW and “no request” made the contract impossible. Direct BPF
   context writes could also bypass setter conflict tracking.
2. The stale check did not name an immutable pre-callback identity/phase and a
   separately reconstructed live identity/phase immediately before commit.
3. The UVM prefetch coordinate system, widening/overflow order, and iterator
   return semantics were not frozen. The production iterator currently mutates
   the same region while its action is ignored.
4. PMM validation lacked an O(1), driver-owned source-list state updated at
   every native used, unused, free, eviction, and lazy transition under the
   list lock. `list_empty()` cannot distinguish membership.
5. No native `uvm_va_space_t` retain API was identified. A safe token design
   would have to define issuance, non-reuse/generation, synchronized lookup,
   teardown invalidation, and a real object-lifetime acquisition; otherwise the
   class must be declared `PARTIAL`.
6. Host-only pure tests cannot establish kernel list-lock or teardown behavior.
   The plan needed to name the exact production header/object used by tests and
   kernel-native integration coverage.
7. The proposed live rejection preflight conflicted with the ban on sending an
   invalid region to the live driver. Live rejection must use an invalid action
   paired with a legal empty region; range/order/overflow cases stay offline.

No implementation started after this review.

## Revision 2 response

Revision 2 makes the following binding changes:

- presence bits replace numeric sentinels, direct scheduler context writes are
  rejected, and explicit interleave LOW=0 is tested;
- the task-init policy call moves after the native MEDIUM initialization that
  currently overwrites its interleave output, so accepted policy values are
  committed from a stable-default point through native setters;
- immutable `expected` and separately reconstructed `current` snapshots are
  compared before native setters;
- all BPF-visible prefetch regions are absolute half-open VA-block coordinates,
  the native-relative translation order is explicit and widened, and iterator
  actions are consumed rather than ignored;
- each root chunk gains a lock-protected source-list state, native generation,
  and last-policy record; every known native transition site is enumerated, and
  raw destination pointers are replaced by typed list requests;
- the raw VA-space-handle kfunc is removed from registration and deferred
  migration is declared `PARTIAL` instead of inventing a lifetime primitive;
- host tests include the exact production header, while PMM membership and
  scheduler access control receive named kernel-native tests; and
- the only live invalid policy value is an invalid action paired with legal
  `(0, 0)`; malformed regions and arithmetic boundaries remain offline.

Revision 2 remains blocked from implementation until the independent reviewer
approves these semantics.

## Round 2: BLOCK

Independent review accepted the scheduler relocation, LOW=0 presence,
two-snapshot comparison, coordinate/action semantics, deferred-lifetime
`PARTIAL`, and live-preflight ceiling, but found four remaining defects:

1. Persisting the last policy reorder across callbacks would reject later legal
   adaptive HEAD/TAIL changes and could misclassify a request after other nodes
   moved.
2. Searching only `root_chunk->chunk.list` missed root aliases such as
   `free_next_available_root_chunk()`'s `result`.
3. The existing narrow region-setter parameters would truncate an invalid BPF
   endpoint before the proposed validator could inspect it.
4. A C selftest compiled with the verifier source would not prove actual BPF
   load admission; the plan also mislabeled the live invalid action as a
   scheduler action.

No implementation started after this review.

## Revision 3 response

- PMM decisions are now callback-local: setters only populate a hidden local
  record, repeats/conflicts are scoped to one invocation, and the driver commits
  at most once under the still-held lock. Only source membership/generation is
  persistent, and later callbacks always re-evaluate live state.
- Every list mutation that can alias a root must use one state/generation-aware
  helper. The plan explicitly names `chunk`/`result` aliases and
  `free_next_available_root_chunk()`.
- Prefetch requested endpoints and setter arguments are `u64`; validation sees
  original values and narrows only after checks. The matrix adds above-type,
  above-block, and maximum-integer cases.
- Five real BPF load fixtures provide exactly two expected verifier rejections
  and three admissions without attach. Their run shares one safely loadable
  stack preflight with the kernel-native PMM test.
- The live invalid action is correctly labeled UVM prefetch action.

Revision 3 remains blocked from implementation until the independent reviewer
approves these semantics.

## Round 3: BLOCK

Independent review confirmed every Round 2 item was closed, then found two
remaining PMM execution gaps:

1. The plan did not define how `gpu_block_access`'s raw action interacts with a
   recorded reorder request. The existing caller can perform a native tail move
   after the callback, which could overwrite an applied policy request. The
   passive BYPASS row and invalid/conflict/stale fallbacks were also undefined.
2. The kfunc required a `decision_ctx`, but the plan had not explicitly changed
   the struct_ops callback ABI to pass that context to BPF or tested that ABI
   with the real verifier.

No implementation started after this review.

## Revision 4 response

- `gpu_block_access` now has a complete action×request table. Native LRU runs
  only for no-request+DEFAULT; no-request+BYPASS is passive; a legal request
  commits once and suppresses native movement; invalid, conflict, and stale
  rows commit nothing and preserve callback-entry position.
- `gpu_block_activate` is specified separately because the native used-list
  move precedes its callback. The raw action remains routing-irrelevant, a
  request may reorder once, and every rejection preserves the post-native
  callback-entry state.
- Both callback signatures now carry a protected invocation-local
  `decision_ctx`; production callbacks, wrappers, CFI/BTF declarations, public
  headers, and in-tree policies must change together.
- The real-load matrix now has seven fixtures: the new PMM hidden-write fixture
  must fail and a PMM reorder-setter using the new ABI must pass, for exact
  totals of three rejections and four admissions.

Revision 4 remains blocked from implementation until the independent reviewer
approves these semantics.

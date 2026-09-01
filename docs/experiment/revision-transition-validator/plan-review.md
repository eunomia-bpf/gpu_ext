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

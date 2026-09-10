#!/usr/bin/env bash
# Prepare a deployable native XSched source tree from a Git revision.
#
# Usage:
#   prepare_native_source.sh <source-git-checkout> <git-revision> \
#       <new-output-directory> [<live-source-tree>]
#
# - <source-git-checkout>: path to a checkout or worktree of the XSched
#   upstream repo (e.g. the workloads deps/xsched clone). The checkout
#   is only READ through git plumbing (`git archive`, `git ls-tree`,
#   `git rev-parse`); its working tree, dirty or clean, is never used
#   as content and never modified. Worktrees (.git being a file) work.
#   Unrelated dirty edits in the checkout are NOT borrowed: content
#   comes from the archived revision and the pinned submodule commits.
# - <git-revision>: commitish the chain is pinned to. Known good:
#   f49289f0220931df78de948ed841ecbaf960a919 (deps/xsched HEAD at
#   publish time; see level2/native/README.md).
# - <new-output-directory>: must not exist, or must be an EMPTY
#   directory. Nothing is ever deleted or overwritten: the extracted
#   upstream tar, every patch log, and every submodule tar stay under
#   <output>/logs/.
# - [<live-source-tree>]: optional existing prepared tree (the
#   successful build's source). A plain per-file cmp report is printed
#   at the end. The report is informational and does not gate the exit
#   status; the run fails only on real preparation failures (bad input,
#   failed patch application, reject/backup leftovers).
#
# Submodules: `git archive` of the parent repo emits EMPTY directories
# for gitlinks, so for every gitlink recorded at <git-revision> the
# matching submodule repo (searched at the same path inside the source
# checkout; nested gitlinks recurse) is archived at its PINNED commit
# and extracted into the output tree. No submodule working-tree files
# are copied and nothing is fetched; if a pinned commit is not present
# in a local submodule repo the run fails and names the required
# initialization instead.
#
# CPU patch application only; no build, no GPU, no driver interaction.

set -euo pipefail

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "usage: $0 <source-git-checkout> <revision> <new-output-dir> [<live-source-tree>]" >&2
    exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
LEVEL2="$(dirname "$HERE")"

# resolve paths before any cd so relative CLI arguments work
SRC="$(realpath -e "$1")"
REV="$2"
OUT="$(realpath -m "$3")"
LIVE=""
if [ $# -eq 4 ]; then
    [ -e "$4" ] || { echo "live source tree not found: $4" >&2; exit 1; }
    LIVE="$(realpath -e "$4")"
fi

# supports normal checkouts and worktrees (.git a file)
git -C "$SRC" rev-parse --git-dir >/dev/null 2>&1 \
    || { echo "not a git checkout or worktree: $SRC" >&2; exit 1; }

if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "output directory exists and is not empty: $OUT (kept as is; choose a new directory)" >&2
    exit 1
fi
mkdir -p "$OUT/logs"

SM120_PATCH="$LEVEL2/xsched-level2-sm120.patch"
PATCH_CHAIN=(
    "$SM120_PATCH"                                           # 1 sm_120 Level-2 port
    "$HERE/xsched-native-diag700-instrument.patch"           # 2 native-700 probes
    "$HERE/xsched-native-resume-allocation-padding.patch"    # 3 resume-alloc padding
    "$HERE/xsched-native-original-entry-control.patch"       # 4 entry-point control
    "$HERE/xsched-native-window-args-relay.patch"            # 5 window-args relay
    "$HERE/xsched-native-launch-error-propagation.patch"     # 6 fail-fast launch
    "$HERE/xsched-native-meta-extend.patch"                  # 7 meta extend + marshal
    "$HERE/xsched-native-level2-source-sync.patch"           # 8 level2-era sync
    "$HERE/xsched-native-async-xqueue-audit.patch"           # 9 audit counters
)
CHAIN_FINAL=(
    "platforms/cuda/shim/include/xsched/cuda/shim/window_meta_extend.h"
    "platforms/cuda/shim/src/window_meta_extend.cpp"
    "platforms/cuda/shim/src/intercept.cpp"
    "platforms/cuda/shim/src/shim.cpp"
    "platforms/cuda/hal/src/common/cuda_command.cpp"
    "platforms/cuda/hal/src/level2/instrument.cpp"
    "platforms/cuda/hal/src/level2/cuda_queue.cpp"
    "preempt/include/xsched/preempt/xqueue/async_xqueue.h"
    "preempt/src/xqueue/async_xqueue.cpp"
)

for p in "${PATCH_CHAIN[@]}"; do
    [ -f "$p" ] || { echo "missing patch: $p" >&2; exit 1; }
done

echo "== extracting git revision $REV from $SRC (archive, read-only)"
git -C "$SRC" archive --format=tar "$REV" > "$OUT/logs/upstream-$REV.tar"
tar -xf "$OUT/logs/upstream-$REV.tar" -C "$OUT"
echo "revision $REV extracted into $OUT"

# ---- submodule content at the pinned commits -------------------------
# git archive leaves gitlinks as empty dirs; populate every recorded
# gitlink from its own repo's pinned commit (recursive for nested
# gitlinks; e.g. 3rdparty/ipc carries 3rdparty/boost-ipc and
# 3rdparty/gtest). The pinned commits are read with git ls-tree of the
# PARENT revision; the CONTENT comes from git archive of the submodule
# repos - never from a working tree, live directory, or fetch.
populate() { # $1 = mount point relative to OUT/SRC ("" for root), $2 = commit;
             # recursively archives the repo at $SRC/$mount for every gitlink
             # recorded under $commit in that repo's tree (recursive ls-tree,
             # so nested gitlinks like ipc's 3rdparty/boost-ipc are found).
    local mount="$1"
    local commit="$2"
    local repo="$SRC/$mount"
    local line path pin sub
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        pin="${line%% *}"
        path="${line#* }"
        sub="$SRC/$mount/$path"
        if ! git -C "$sub" rev-parse -q --verify "${pin}^{commit}" >/dev/null 2>&1; then
            echo "submodule '$mount/$path' pinned at $pin has no object in $sub" >&2
            echo "initialize it first (e.g.: git -C '$SRC' submodule update --init '$mount/$path')" >&2
            echo "then re-run this script; nothing was deleted" >&2
            exit 1
        fi
        echo "== submodule $mount/$path @ $pin (archive, read-only)"
        git -C "$sub" archive --format=tar "$pin" > "$OUT/logs/sub-${mount//\//_}${mount:+_}${path//\//_}-$pin.tar"
        mkdir -p "$OUT/$mount/$path"
        tar -xf "$OUT/logs/sub-${mount//\//_}${mount:+_}${path//\//_}-$pin.tar" -C "$OUT/$mount/$path"
        populate "$mount/$path" "$pin"
    done < <(git -C "$repo" ls-tree -r "$commit" | awk '$1 == "160000" {print $3, $4}')
}
populate "" "$REV"

# ---- patch chain ------------------------------------------------------
cd "$OUT"

for p in "${PATCH_CHAIN[@]}"; do
    name="$(basename "$p")"
    log="$OUT/logs/$name.log"
    echo "== applying $name (log: logs/$name.log)"
    patch -p1 --batch --no-backup-if-mismatch < "$p" >> "$log" 2>&1
    if grep -q fuzz "$log"; then
        echo "   (fuzz recorded in the log; the chain's final file bytes are the check)"
    fi
    leftovers="$(find . \( -name '*.rej' -o -name '*.orig' \) -print -quit)"
    if [ -n "$leftovers" ]; then
        echo "reject/backup files left by $name: $leftovers; see logs/" >&2
        exit 1
    fi
done

echo "== chain applied clean (all logs kept in $OUT/logs/)"

if [ -n "$LIVE" ]; then
    echo "== cmp report against $LIVE (informational, not a gate)"
    for f in "${CHAIN_FINAL[@]}"; do
        if cmp -s "$f" "$LIVE/$f"; then
            echo "MATCH   $f"
        else
            echo "DIFF    $f"
        fi
    done
fi
echo "done: $OUT"

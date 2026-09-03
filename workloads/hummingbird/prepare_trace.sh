#!/usr/bin/env bash
# Reuse the frozen GPreempt DISB frontend without editing its source or binaries.
set -euo pipefail
hb_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
hb_upstream="$hb_root/../gpreempt/deps/upstream"
hb_copy="$hb_root/build/disb-src"
if [[ ! -d "$hb_copy" ]]; then
    mkdir -p "$hb_root/build"
    cp -a -- "$hb_upstream/third_party/disb" "$hb_copy"
    # The copy is build input, not another Git checkout.
    if [[ -f "$hb_copy/.git" ]]; then
        mv -- "$hb_copy/.git" "$hb_root/build/disb-source-git-pointer.txt"
    fi
    patch --batch --forward -p3 -d "$hb_copy" --dry-run < "$hb_root/trace-fifo.patch"
    patch --batch --forward -p3 -d "$hb_copy" < "$hb_root/trace-fifo.patch"
else
    patch --batch --reverse -p3 -d "$hb_copy" --dry-run < "$hb_root/trace-fifo.patch"
fi
taskset -c "${HB_CPUSET:-16-17}" cmake -S "$hb_copy" -B "$hb_root/build/disb-trace" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_CXX_STANDARD=17 \
    "-DCMAKE_CXX_FLAGS=-DGPREEMPT_LOAD_STUDY -I$hb_root -I$hb_root/../gpreempt -I$hb_upstream/third_party/jsoncpp/include"
taskset -c "${HB_CPUSET:-16-17}" cmake --build "$hb_root/build/disb-trace" --target disb -j2

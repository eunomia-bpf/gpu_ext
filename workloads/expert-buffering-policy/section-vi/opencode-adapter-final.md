

# Expert Buffering §VI — concrete minimal patch proposal (FineMoE source copy)

All snippets below are insertions into the **independent FineMoE copy** and use only the already-implemented `eb_adapter::State` API from `adapter_state.h`. Nothing invents a second state model. I treat `policy.h`/`eb_context` as opaque snapshot values validated only through `State::Validate/Evicted`. Boundaries of what is *not* in the excerpts are listed at the end.

## Global invariants this patch must not violate

- Execution stays serial, increasing `expert_idx`, whole-expert, top-k gated. Arms change only *residency*, never loop order or gating. `K=16` fixed per `(GPU0, layer)` is a port choice, not a re-derivation of the original cross-layer undocumented buffer.
- Lock order (existing, unchanged): `AcquireTensor` holds `Node::mutex` → `StartExec` takes `unified_mutex_`/`exec_mutex_`; worker releases both before `RemoveCachedSparseNode`/`CompleteDemand`, which then takes `Node::mutex` and publishes ready. **New `eb_mutex_` ordering rule:** `eb_mutex_` is taken by the inference thread (Begin/End) while holding **no** Node lock, and by the worker (victim commit) while taking Node locks **only via `try_lock`** (bounded). `Admitted` is called by the worker **after** `Node::mutex` is released. Therefore there is never a blocking Node lock under `eb_mutex_`, and never `Node::mutex → eb_mutex_`. No cycle.
- Single GPU0 copy worker + single inference thread is the supported mode; a second concurrent Begin before End is rejected (State already forbids overlapping active layers).
- The pybind calls touch only C++/C (State dlopen/uBPF) — they never call back into Python; `EbAdmit` runs on the worker with no GIL.

---

## 1. `modeling_qwen2_moe.py` — top-k counts, Begin/End, equal prediction disable

Replace the current prediction block + expert loop (excerpt lines 930–991) with the guarded version. Counts come from **this batch's** one-hot mask (positive CURRENT counts), not trajectories.

```python
        # --- EB Section VI: disable trajectory/embed prediction equally in all arms ---
        # (set self.eb_arm to None / "predict" on the prediction baseline only)
        eb_active = getattr(self, "eb_arm", None) is not None
        if not eb_active:
            for i, seq_id in enumerate(self.seq_id_list):
                self.expert_tracer.update_entry(
                    seq_id=seq_id, expert_list=expert_index[i], layer_idx=self.layer_id,
                    hidden_states=hidden_states[i * sequence_length:(i + 1) * sequence_length],
                    expert_probs=expert_probs[i * sequence_length:(i + 1) * sequence_length])
            input_trajs = torch.stack(
                [torch.stack([it["probs"][:self.layer_id + 1] for it in
                              self.expert_tracer.trace[s].iters[-1:]], dim=0)
                 for s in self.seq_id_list], dim=0
            ).to(dtype=torch.float, device=self.device, non_blocking=True)
            assert not (input_trajs == 0).all(dim=-1).any()
            for seq_id, trajs in zip(self.seq_id_list, input_trajs):
                self.expert_map_matcher.traj_prefetch(seq_id, trajs)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype, device=hidden_states.device)
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # --- Begin BEFORE the first expert fetch, using CURRENT batch counts ---
        if eb_active:
            counts = expert_mask.sum(dim=(1, 2)).to(torch.int32).cpu().tolist()  # len == num_experts
            # representative tensor id per expert, in expert-index order (== cohort slot order)
            tensor_ids = [self.expert_tensor_map[(self.layer_id, e)]
                          for e in range(self.num_experts)]
            self._eb_epoch = self.archer_engine.eb_begin(
                self.layer_id, self.device.index if self.device.index is not None else 0,
                tensor_ids, counts)
            # NOTE: eb_begin runs under the handle metadata lock only, no Node locks here.

        for expert_idx in range(self.num_experts):          # serial, increasing ID, unchanged
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:                          # zero-row: no demand load / fake hit
                continue
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state).to(
                routing_weights.device) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # --- End AFTER the expert loop (resident-release still owned by ReleaseTensor) ---
        if eb_active:
            self.archer_engine.eb_end(self.layer_id, self._eb_epoch)
            self._eb_epoch = None
```

The `update_embed`/`embed_prefetch` block (excerpt lines 1002–1007) is wrapped the same way with `if not eb_active:` so prediction is disabled identically across Fifo/Native/Bpf; only the baseline arm keeps it.

---

## 2. `model_offload.py` — engine configure + layer metadata; pybind methods

In `__enter__` (after `self.moe_layers.append(module)`, excerpt line ~1104) configure the buffer once, resolving expert tensor maps. `K=16` is a port parameter, not inferred from the model.

```python
                self.expert_layer_metadata = []
                for module in self.expert_layer_modules:
                    rep_ids = [self.expert_tensor_map[(module.layer_id, e)]
                               for e in range(module.num_experts)]
                    self.expert_layer_metadata.append(rep_ids)

                if os.environ.get("EB_ARM", "") in ("fifo", "native", "bpf"):
                    self.archer_engine.eb_configure(
                        os.environ["EB_ARM"],
                        int(os.environ.get("EB_K", "16")),          # fixed per-(GPU0,layer) port choice
                        os.environ.get("EB_NATIVE_LIB", ""),
                        os.environ.get("EB_BYTECODE", ""))
                    # representative ids resolve to COMPLETE topology nodes inside C++ (whole-expert)
                    for rep_ids in self.expert_layer_metadata:
                        self.archer_engine.eb_check_nodes_resolve(rep_ids)  # throws if any id misses a node
```

`py_archer_prefetch.cpp` — add to `prefetch_handle` bindings (excerpt near line 1181):

```cpp
        .def("eb_configure", &ArcherPrefetchHandle::EbConfigure,
             py::call_guard<py::gil_scoped_release>())
        .def("eb_begin", &ArcherPrefetchHandle::EbBegin,
             py::call_guard<py::gil_scoped_release>())   // State/dlopen/BPF never re-enter Python
        .def("eb_end", &ArcherPrefetchHandle::EbEnd,
             py::call_guard<py::gil_scoped_release>())
        .def("eb_check_nodes_resolve", &ArcherPrefetchHandle::EbCheckNodesResolve)
        .def("eb_stats", &ArcherPrefetchHandle::EbStats);
```

GIL note: `eb_configure/begin/end` release the GIL because the bodies only enter C++ `State` + the dlopen'd native/BPF C entrypoints (`native_`, `bpf_`, `jit_calls_`) — there is no Python callback, so the direct C ABI call does not reacquire the GIL and cannot deadlock the worker.

---

## 3. `task_scheduler.cpp` / handle — real victim commit, bounded failure, epoch + residency + exec eligibility under locks

Add to `ArcherPrefetchHandle` (new members + methods in `archer_prefetch_handle.{h,cpp}`):

```cpp
// archer_prefetch_handle.h members
    std::unique_ptr<eb_adapter::State> eb_state_;
    std::mutex eb_mutex_;                       // metadata lock; NEVER a blocking Node lock under it
// file-scope, for the worker (lifetime-surviving indirection)
    std::atomic<ArcherPrefetchHandle*> g_eb_handle{nullptr};
```

Begin/End are inference-thread, metadata-only (no Node lock held), and Begin does the bounded "no live exec / real residency" cross-check the contract demands:

```cpp
std::uint64_t ArcherPrefetchHandle::EbBegin(
    std::uint32_t layer, std::uint32_t device,
    const std::vector<std::uint32_t>& rep_tensor_ids,
    const std::vector<std::uint32_t>& counts)
{
    std::vector<eb_adapter::NodeId> nodes; nodes.reserve(rep_tensor_ids.size());
    for (std::uint32_t tid : rep_tensor_ids)
        nodes.push_back(kTopologyHandle->GetNodeFromTensorID(tid)->id);   // structural, no Node lock

    std::lock_guard<std::mutex> lk(eb_mutex_);                             // metadata only
    for (eb_adapter::NodeId nid : nodes) {                                 // bounded residency/exec check
        NodePtr n = kTopologyHandle->GetNodeFromCorrID(                    // resolve id->node helper
            (std::size_t)layer << 32);                                     // placeholder; see boundary B1
        if (!n) continue;
        if (!n->mutex.try_lock()) {                                        // live exec on this cohort node
            n->mutex.unlock();
            throw std::runtime_error("eb_begin: live execution on cohort node");  // reject, no overlap
        }
        n->mutex.unlock();                                                 // never a blocking lock here
    }
    return eb_state_->Begin(layer, device, nodes, counts);                 // caller-serialized
}

void ArcherPrefetchHandle::EbEnd(std::uint32_t layer, std::uint64_t epoch) {
    std::lock_guard<std::mutex> lk(eb_mutex_);
    eb_state_->End(layer, epoch);                                          // requires no live exec (Begin-checked)
}
```

Worker-side victim commit is inserted **inside** `GPUThreadFunc`, replacing the bare `RemoveCachedSparseNode(...)` recheck block (excerpt lines 456–472), so the BPF/native victim is chosen and applied here — not preselected in C nor driven by Python priorities. The global strict byte-budget check in `RemoveCachedSparseNode` is kept and runs in addition to cohort-K:

```cpp
        if (!node->device.is_cuda() && task->dst_device.is_cuda()) {
            bool cohort_ok = true;
            ArcherPrefetchHandle* h = g_eb_handle.load();                  // arm active?
            eb_adapter::State* s = h ? h->EbStatePtr() : nullptr;
            if (s && node->is_sparse) {
                std::lock_guard<std::mutex> lk(h->EbMutex());              // metadata first
                auto [layer, incoming] = s->Locate(node->id);              // current cohort slot
                eb_adapter::eb_u64 epoch = s->ActiveEpoch(layer, incoming);
                if (epoch == 0) cohort_ok = false;                        // epoch already rolled -> bounded fail
                const auto& coh = s->Get(layer);
                std::vector<bool> eligible(coh.nodes.size(), false);
                for (std::size_t i = 0; i < coh.nodes.size(); ++i) {
                    if (coh.nodes[i] == node->id) continue;               // do not evict the incoming node
                    NodePtr c = ResolveNode(coh.nodes[i]);
                    if (!c || !c->device.is_cuda()) continue;             // must be actually resident
                    if (!c->mutex.try_lock()) continue;                    // executing / mid-move -> ineligible
                    bool running = false;
                    { std::lock_guard<std::mutex> el(exec_mutex_);
                      running = exec_queue_.count(coh.nodes[i]) > 0; }     // exec eligibility revalidated
                    if (running) { c->mutex.unlock(); continue; }
                    eligible[i] = true; c->mutex.unlock();                // probe only; commit re-locks below
                }
                if (epoch && cohort_ok) {
                    auto snap = s->Decide(layer, incoming, eligible);      // native/BPF/FIFO choose the victim
                    s->Validate(snap);                                     // snapshot still matches cohort state
                    NodePtr v = ResolveNode(snap.victim_node /*boundary B2*/);
                    if (v && v->mutex.try_lock()) {                        // re-lock the CHOSEN victim
                        bool still_res = v->device.is_cuda() && v->device_memory_ptr;
                        bool still_free;
                        { std::lock_guard<std::mutex> el(exec_mutex_);
                          still_free = exec_queue_.count(v->id) == 0; }
                        bool epoch_live = s->ActiveEpoch(layer, incoming) == epoch;
                        if (still_res && still_free && epoch_live) {
                            v->SetDevice(v->default_host);                 // real eviction under its lock
                            v->incache_visit_count = 0; v->prob = 0;
                            s->Evicted(snap);
                            finemoe_revision::Ledger().Count("eb_cohort_evictions");
                        } else cohort_ok = false;
                        v->mutex.unlock();
                    } else cohort_ok = false;                             // victim locked -> bounded fail
                }
            }
            if (!node->is_sparse) RemoveCachedDenseNode(node);
            bool budget_ok = RemoveCachedSparseNode(node, gpu_id, task->priority > 0);  // global byte budget KEPT
            if (!cohort_ok || !budget_ok) {
                if (task->priority > 0) {                                  // speculative: bounded skip, no overflow
                    finemoe_revision::Ledger().Count(task ? "prefetch_eviction_skip" : "");
                    finemoe_revision::Ledger().Count("eb_admission_bounded_fail");
                    FinishTask(); continue;
                }
                std::fprintf(stderr, "FINEMOE_EB_BUDGET_ERROR node=%zu bytes=%ld\n",
                             node->id, node->byte_size);
                std::abort();                                              // demand: no silent overflow / locked reuse
            }
        }
```

`Admitted` is **not** called here and not under any Node lock; it happens only after the physical copy in point 4. `EbStatePtr()`/`EbMutex()` are trivial accessors. `ResolveNode(id)` walks the topology id→Node (structural). The victim is committed by the worker, epoch/whole-node-residency/exec eligibility all revalidated under `exec_mutex_` and the victim's own lock; only `try_lock` is used while `eb_mutex_` is held.

---

## 4. `SetNodeDevice` / `Node::SetDevice` — stamp admission only after synchronous whole-expert H2D

`Node::SetDevice` already performs a synchronous copy (`cudaMemcpy` for the on-demand `stream==nullptr` path, else `cudaMemcpyAsync`+`cudaStreamSynchronize`), `abort()`s on any copy error or null allocation, and never reallocates over budget — so there is **no silent fallback and no unsafe overflow allocation**. We only add success propagation so the worker stamps admission after it, with `Node::mutex` released.

Add `bool eb_h2d_done = false;` to `Task` (`task_scheduler.h`), then in `ArcherTaskPool::SetNodeDevice` (excerpt lines 502–541):

```cpp
void ArcherTaskPool::SetNodeDevice(const TaskPtr& task)
{
    auto node = task->node;
    if (!task->on_demand) { if (!node->mutex.try_lock()) return; }
    if (node->device.type() == task->dst_device.type()) {
        if (!task->on_demand) node->mutex.unlock();
        return;
    }
    const bool was_cuda = node->device.is_cuda();
    auto start_time = MCIROSECONDS_SINCE_EPOCH;
    node->SetDevice(task->dst_device, task->on_demand, nullptr, task->priority > 0); // synchronous H2D
    auto end_time = MCIROSECONDS_SINCE_EPOCH;
    // Stamp only if a real host->cuda whole-expert copy landed (device flipped to cuda).
    task->eb_h2d_done = task->dst_device.is_cuda() && !was_cuda && node->device.is_cuda();
    node->io_state = NODE_STATE_CACHED;
    if (task->priority > 0 && task->dst_device.is_cuda()) { /* ...existing prefetch bookkeeping unchanged... */ }
    if (!task->on_demand) node->mutex.unlock();                            // speculative releases here
}
```

In `GPUThreadFunc`, after the copy is published — on-demand via `CompleteDemand` (which holds `Node::mutex`, does the copy, then sets `state=0`/`notify` and **releases**), speculative after `SetNodeDevice` returns with the lock already released — stamp the cohort admission:

```cpp
        if (task->on_demand) {
            finemoe_revision::CompleteDemand(node->mutex, node->cv, node->state,
                                             [&] { SetNodeDevice(task); });
            // Node::mutex released above; admission stamp is now lock-order-safe.
        } else {
            SetNodeDevice(task);
        }
        if (task->eb_h2d_done) {
            ArcherPrefetchHandle* h = g_eb_handle.load();
            if (h) h->EbAdmit(node->id, node->corr_id & 0xFFFFFFFF);       // metadata-only, no Node lock
        }
        FinishTask();
```

```cpp
void ArcherPrefetchHandle::EbAdmit(std::uint64_t node_id, std::uint32_t layer) {
    std::lock_guard<std::mutex> lk(eb_mutex_);                             // metadata only, no Node lock
    eb_adapter::eb_u64 epoch = eb_state_->ActiveEpoch(layer, eb_state_->Locate(node_id).second);
    if (epoch) { eb_state_->CanAdmit(layer, epoch, eb_state_->Locate(node_id).second);
                 eb_state_->Admitted(layer, epoch, eb_state_->Locate(node_id).second); }
    else finemoe_revision::Ledger().Count("eb_admit_epoch_stale");        // bounded, never a silent bookkeeping lie
}
```

`Admitted` is reached strictly after `cudaStreamSynchronize` inside `SetDevice` succeeded, so cohort metadata reflects real residency. `SynchronizeNodeCompute` on `ReleaseTensor` (excerpt 630–644) is untouched, preserving acquire/release compute sync so a speculative copy never races live compute on a released node.

---

## Lifetime / teardown

`EbConfigure` sets `g_eb_handle = this` after `State` is built; the destructor calls `kTaskPool->DrainForSnapshot()` **first**, then `g_eb_handle = nullptr`, then destroys `eb_state_` (which closes the dlopen'd `library_`/`jit_`). Because the worker reads `g_eb_handle` atomically and the drain guarantees no in-flight task, a teardown can never dereference a freed `State`; the worker never enters pybind, so no GIL is reacquired during this window.

## Boundaries (details not present in the attached excerpts)

- **B1** `ArcherTopologyHandle` exposes `GetNodeFromTensorID` and `GetNodeBodyFromCorrID` (both used in the excerpts) but no public id→`NodePtr` lookup; `ResolveNode`/`GetNodeFromCorrID` must add one (linear scan of `GetSparseNodes()` is acceptable for the cohort). I did not assume its signature.
- **B2** The exact public field on `eb_context` carrying the chosen victim node id is not shown (State treats it as opaque). The commit path reads it as `snap.victim_node`; if `policy.h` instead exposes only a cohort slot index, resolve `coh.nodes[idx]`.
- **B3** `ArcherConfig`/`offload_path` fields and `Ledger::Count/Start/Complete/Evict/DemandUse` signatures are taken from call-site usage in the excerpts only.
- **B4** Whether the on-demand H2D in `SetDevice` uses `cudaMemcpy` (blocking) or the `stream`-arg `cudaStreamSynchronize` path is fixed at the `SetNodeDevice(...nullptr...)` call (excerpt line 523): here `stream==nullptr`, so `cudaMemcpy` is already synchronous; no extra barrier is added.

No build or GPU run has been performed — this is a source proposal only, and none of the residency/eviction guarantees above are claimed as live-verified until built and exercised on GPU0 with `EB_ARM=native` and `EB_ARM=bpf`.



s$ sudo BPFTRACE_MAX_BPF_PROGS=2000 BPFTRACE_MAX_PROBES=2000 bpftrace /home/yunwei37/workspace/gpu/xpu-perf/tools/bpftrace-script/test_uvm_1.bt 
Attaching 385 probes...
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_hal_wfi_membar, skipping.
Tracing uvm_hal_*, uvm_test_*, uvm_perf_* functions...
^C
=== uvm_hal_*/uvm_test_*/uvm_perf_* Function Calls ===
@calls[kprobe:uvm_hal_turing_disable_replayable_faults]: 1307
@calls[kprobe:uvm_hal_turing_clear_replayable_faults]: 1314
@calls[kprobe:uvm_hal_pascal_enable_replayable_faults]: 1475
@calls[kprobe:uvm_hal_blackwell_host_tlb_invalidate_all]: 15939
@calls[kprobe:uvm_hal_volta_replay_faults]: 16383
@calls[kprobe:uvm_hal_volta_fault_buffer_read_put]: 17172
@calls[kprobe:uvm_hal_volta_fault_buffer_write_get]: 32587
@calls[kprobe:uvm_hal_maxwell_host_noop]: 775361
@calls[kprobe:uvm_hal_downgrade_membar_type]: 917526
@calls[kprobe:uvm_perf_prefetch_get_hint_va_block]: 1258174
@calls[kprobe:uvm_perf_thrashing_get_hint]: 1700807
@calls[kprobe:uvm_perf_thrashing_get_thrashing_pages]: 2406354
@calls[kprobe:uvm_hal_hopper_host_semaphore_acquire]: 2608727
@calls[kprobe:uvm_hal_pascal_host_membar_gpu]: 2854609
@calls[kprobe:uvm_hal_maxwell_host_wait_for_idle]: 3940721
@calls[kprobe:uvm_hal_volta_fault_buffer_parse_replayable_entry]: 4118725
@calls[kprobe:uvm_hal_blackwell_host_tlb_invalidate_va]: 4772914
@calls[kprobe:uvm_hal_volta_host_write_gpu_put]: 4885061
@calls[kprobe:uvm_hal_mmu_mode_hopper]: 5117558
@calls[kprobe:uvm_hal_hopper_fault_buffer_get_ve_id]: 6471864
@calls[kprobe:uvm_hal_blackwell_mmu_client_id_to_utlb_id]: 6768598
@calls[kprobe:uvm_hal_blackwell_fault_buffer_get_mmu_engine_type]: 6791955
@calls[kprobe:uvm_hal_blackwell_fault_buffer_get_fault_type]: 6796454
@calls[kprobe:uvm_hal_pascal_fault_buffer_entry_is_valid]: 7495247
@calls[kprobe:uvm_hal_pascal_fault_buffer_entry_clear_valid]: 7514268
@calls[kprobe:uvm_hal_hopper_host_set_gpfifo_entry]: 7782878
@calls[kprobe:uvm_hal_hopper_ce_semaphore_release]: 7898557
@calls[kprobe:uvm_perf_event_notify]: 8352985
@calls[kprobe:uvm_hal_volta_ce_memcopy]: 10878018
@calls[kprobe:uvm_hal_hopper_ce_offset_in_out]: 16654081
@calls[kprobe:uvm_hal_hopper_ce_memcopy_copy_type]: 16892843
@calls[kprobe:uvm_hal_ce_memcopy_patch_src_stub]: 17109966
@calls[kprobe:uvm_hal_ampere_ce_phys_mode]: 17439213
@calls[kprobe:uvm_hal_hopper_ce_offset_out]: 33222877
@calls[kprobe:uvm_hal_hopper_ce_memset_8]: 33347556
@calls[kprobe:uvm_perf_prefetch_bitmap_tree_iter_get_range]: 36231600
@calls[kprobe:uvm_hal_ampere_ce_plc_mode_c7b5]: 59840895



yunwei37@lab:~/workspace/gpu/xpu-perf/tools$ sudo BPFTRACE_MAX_BPF_PROGS=2000 BPFTRACE_MAX_PROBES=2000 bpftrace /home/yunwei37/workspace/gpu/xpu-perf/tools/bpftrace-script/test_uvm_2.bt 
Attaching 312 probes...
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_push_get_and_reset_membar_flag, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_gpu_address_copy.isra.0, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_va_space_mm_or_current_retain_lock, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_va_block_next_page_in_mask.isra.0, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_va_block_first_page_in_mask, skipping.
Tracing uvm_va_*/uvm_gpu_*/uvm_channel_*/uvm_push_* functions...
^C
=== uvm_va_*/uvm_gpu_*/uvm_channel_*/uvm_push_* Function Calls ===
@calls[kprobe:uvm_push_inline_data_add]: 169
@calls[kprobe:uvm_va_block_revoke_prot_mask]: 1556
@calls[kprobe:uvm_va_block_revoke_prot]: 1560
@calls[kprobe:uvm_va_block_first_subregion_in_mask]: 35440
@calls[kprobe:uvm_va_block_next_subregion_in_mask.isra.0]: 57133
@calls[kprobe:uvm_push_inline_data_get]: 131109
@calls[kprobe:uvm_push_inline_data_end]: 132405
@calls[kprobe:uvm_channel_update_progress_with_max]: 164892
@calls[kprobe:uvm_va_space_mm_retain]: 338789
@calls[kprobe:uvm_va_space_mm_release]: 340340
@calls[kprobe:uvm_channel_check_errors]: 347657
@calls[kprobe:uvm_va_space_has_access_counter_migrations]: 350453
@calls[kprobe:uvm_channel_get_status]: 384715
@calls[kprobe:uvm_channel_tracking_semaphore_get_gpu_va_in_channel]: 450240
@calls[kprobe:uvm_va_block_evict_chunks]: 617499
@calls[kprobe:uvm_va_block_context_free]: 641382
@calls[kprobe:uvm_va_block_context_alloc]: 643708
@calls[kprobe:uvm_va_block_context_init]: 656126
@calls[kprobe:uvm_gpu_chunk_get_gpu]: 1184878
@calls[kprobe:uvm_push_end]: 1234312
@calls[kprobe:uvm_va_block_cpu_set_resident_all_chunks]: 1258012
@calls[kprobe:uvm_channel_reserve_type]: 1395144
@calls[kprobe:uvm_channel_pool_uses_mutex]: 1472400
@calls[kprobe:uvm_va_block_service_finish]: 1655872
@calls[kprobe:uvm_va_space_iter_next]: 1737594
@calls[kprobe:uvm_va_space_iter_gmmu_mappable_first]: 1745698
@calls[kprobe:uvm_channel_end_push]: 1828627
@calls[kprobe:uvm_va_block_map]: 1841929
@calls[kprobe:uvm_gpu_va_space_get]: 1844106
@calls[kprobe:uvm_channel_begin_push]: 1856005
@calls[kprobe:uvm_va_block_cpu_clear_resident_all_chunks]: 2091720
@calls[kprobe:uvm_va_block_service_locked]: 2151472
@calls[kprobe:uvm_va_range_block_index]: 2152454
@calls[kprobe:uvm_va_block_add_mappings_after_migration]: 2161473
@calls[kprobe:uvm_va_range_block_create]: 2163413
@calls[kprobe:uvm_va_block_service_copy]: 2171074
@calls[kprobe:uvm_va_block_resident_mask_get]: 2223217
@calls[kprobe:uvm_va_block_unmap]: 2370032
@calls[kprobe:uvm_va_block_make_resident_finish]: 2420387
@calls[kprobe:uvm_va_block_gpu_big_page_size]: 2505193
@calls[kprobe:uvm_va_block_retry_init]: 2597444
@calls[kprobe:uvm_va_block_retry_deinit]: 2607012
@calls[kprobe:uvm_gpu_tracking_semaphore_update_completed_value]: 2694553
@calls[kprobe:uvm_va_block_big_page_index]: 2727189
@calls[kprobe:uvm_va_block_big_page_region_subset]: 2775605
@calls[kprobe:uvm_gpu_semaphore_get_gpu_va]: 2802023
@calls[kprobe:uvm_va_block_select_residency]: 2817815
@calls[kprobe:uvm_va_block_page_get_closest_resident]: 2986603
@calls[kprobe:uvm_gpu_semaphore_get_cpu_va]: 2991689
@calls[kprobe:uvm_va_block_make_resident_copy]: 2997632
@calls[kprobe:uvm_va_block_find_create_in_range]: 3065107
@calls[kprobe:uvm_channel_is_value_completed]: 3255403
@calls[kprobe:uvm_gpu_tracking_semaphore_is_value_completed]: 3599104
@calls[kprobe:uvm_va_block_page_is_gpu_authorized]: 4257435
@calls[kprobe:uvm_va_block_unmap_mask]: 5085077
@calls[kprobe:uvm_va_block_check_logical_permissions]: 5116493
@calls[kprobe:uvm_gpu_address_is_peer]: 5214640
@calls[kprobe:uvm_va_policy_get_region]: 7888427
@calls[kprobe:uvm_va_block_gpu_chunk_index_range]: 9981655
@calls[kprobe:uvm_va_space_can_read_duplicate]: 24540697
@calls[kprobe:uvm_va_block_get_gpu_va_space]: 26986146
@calls[kprobe:uvm_va_block_get_va_space]: 29612071
@calls[kprobe:uvm_va_policy_is_read_duplicate]: 46803779
@calls[kprobe:uvm_va_policy_preferred_location_equal]: 117195648



yunwei37@lab:~/workspace/gpu/xpu-perf/tools$ sudo BPFTRACE_MAX_BPF_PROGS=2000 BPFTRACE_MAX_PROBES=2000 bpftrace /home/yunwei37/workspace/gpu/xpu-perf/tools/bpftrace-script/test_uvm_3.bt 
Attaching 177 probes...
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_page_mask_zero, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_page_mask_test, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_page_mask_set, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_page_mask_region_weight, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_page_mask_region_fill, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_page_mask_region_empty, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_page_mask_copy, skipping.
Tracing uvm_pmm_*/uvm_hmm_*/uvm_page_*/uvm_range_*/uvm_tracker_* functions...
^C
=== uvm_pmm_*/uvm_hmm_*/uvm_page_*/uvm_range_*/uvm_tracker_* Function Calls ===
@calls[kprobe:uvm_page_mask_region_clear]: 14
@calls[kprobe:uvm_tracker_wait_for_other_gpus]: 137150
@calls[kprobe:uvm_tracker_wait]: 144300
@calls[kprobe:uvm_page_tree_write_pde]: 248369
@calls[kprobe:uvm_pmm_gpu_unpin_allocated]: 420221
@calls[kprobe:uvm_pmm_gpu_mark_root_chunk_unused]: 443335
@calls[kprobe:uvm_pmm_gpu_mark_chunk_evicted]: 445581
@calls[kprobe:uvm_tracker_entry_gpu]: 489873
@calls[kprobe:uvm_page_mask_init_from_big_ptes]: 697004
@calls[kprobe:uvm_pmm_gpu_alloc_user]: 930995
@calls[kprobe:uvm_range_group_range_migratability_iter_first]: 1264017
@calls[kprobe:uvm_tracker_add_push_safe]: 1508914
@calls[kprobe:uvm_page_mask_region_full]: 1554741
@calls[kprobe:uvm_range_group_range_iter_first]: 1612479
@calls[kprobe:uvm_pmm_gpu_mark_root_chunk_used]: 1876643
@calls[kprobe:uvm_hmm_must_use_sysmem]: 1967986
@calls[kprobe:uvm_range_tree_iter_first]: 3536334
@calls[kprobe:uvm_tracker_get_entries]: 3923767
@calls[kprobe:uvm_tracker_reserve]: 4392987
@calls[kprobe:uvm_page_mask_region_clear_outside]: 4507708
@calls[kprobe:uvm_tracker_add_tracker_safe]: 4800542
@calls[kprobe:uvm_tracker_add_tracker]: 4849651
@calls[kprobe:uvm_tracker_remove_completed]: 5066964
@calls[kprobe:uvm_tracker_deinit]: 5081676
@calls[kprobe:uvm_tracker_add_entry]: 5369948



yunwei37@lab:~/workspace/gpu/xpu-perf/tools$ sudo BPFTRACE_MAX_BPF_PROGS=2000 BPFTRACE_MAX_PROBES=2000 bpftrace /home/yunwei37/workspace/gpu/xpu-perf/tools/bpftrace-script/test_uvm_4.bt 
Attaching 295 probes...
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_uuid_eq, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_uuid_copy, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_processor_mask_zero, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_processor_mask_test, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_processor_mask_set, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_processor_mask_copy, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_processor_mask_clear, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_pte_batch_single_write_ptes, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_parent_processor_mask_test, skipping.
cannot attach kprobe, Invalid argument
WARNING: could not attach probe kprobe:uvm_mem_alloc_sysmem_and_map_cpu_kernel.constprop.0, skipping.
Tracing remaining uvm_* functions...
^C
=== Remaining uvm_* Function Calls ===
@calls[kprobe:uvm_thread_context_wrapper_is_used]: 503
@calls[kprobe:uvm_thread_context_remove]: 525
@calls[kprobe:uvm_thread_context_add]: 569
@calls[kprobe:uvm_parent_gpu_service_replayable_faults]: 871
@calls[kprobe:uvm_parent_gpu_replayable_faults_pending]: 878
@calls[kprobe:uvm_parent_gpu_replayable_faults_isr_unlock]: 881
@calls[kprobe:uvm_parent_gpu_replayable_faults_intr_disable]: 891
@calls[kprobe:uvm_parent_gpu_non_replayable_faults_pending]: 911
@calls[kprobe:uvm_parent_gpu_get_by_uuid_locked]: 930
@calls[kprobe:uvm_parent_gpu_kref_put]: 1828
@calls[kprobe:uvm_ats_invalidate_tlbs]: 9270
@calls[kprobe:uvm_rb_tree_find]: 10499
@calls[kprobe:uvm_parent_gpu_find_first_valid_gpu]: 12245
@calls[kprobe:uvm_parent_gpu_fault_entry_to_va_space]: 12307
@calls[kprobe:uvm_tools_broadcast_replay]: 17543
@calls[kprobe:uvm_service_block_context_free]: 184714
@calls[kprobe:uvm_service_block_context_alloc]: 188916
@calls[kprobe:uvm_pushbuffer_mark_completed]: 1039296
@calls[kprobe:uvm_pushbuffer_get_offset_for_push]: 1125175
@calls[kprobe:uvm_pushbuffer_end_push]: 1177513
@calls[kprobe:uvm_pushbuffer_begin_push]: 1233150
@calls[kprobe:uvm_pushbuffer_get_gpu_va_for_push]: 1274729
@calls[kprobe:uvm_pte_batch_end]: 1439993
@calls[kprobe:uvm_pte_batch_begin]: 1459219
@calls[kprobe:uvm_tlb_batch_end]: 1494684
@calls[kprobe:uvm_tlb_batch_begin]: 1503089
@calls[kprobe:uvm_pte_batch_write_pte]: 1525105
@calls[kprobe:uvm_processor_has_memory]: 1710451
@calls[kprobe:uvm_tlb_batch_invalidate]: 1769442
@calls[kprobe:uvm_tools_record_block_migration_begin]: 2464332
@calls[kprobe:uvm_processor_get_name]: 2467760
@calls[kprobe:uvm_processor_mask_range_fill.constprop.0]: 2510886
@calls[kprobe:uvm_tools_record_read_duplicate_invalidate]: 2970824
@calls[kprobe:uvm_parent_gpu_canonical_address]: 3210106
@calls[kprobe:uvm_rm_mem_get_gpu_va]: 3424361
@calls[kprobe:uvm_spin_loop]: 3973030
@calls[kprobe:uvm_processor_mask_find_closest_id]: 4828198
@calls[kprobe:uvm_processor_mask_cache_free]: 9623406
@calls[kprobe:uvm_processor_mask_cache_alloc]: 9756988
@calls[kprobe:uvm_rm_mem_get_cpu_va]: 9955406
@calls[kprobe:uvm_pte_batch_clear_ptes]: 10766984



yunwei37@lab:~/workspace/gpu/xpu-perf/tools$ 

// SPDX-License-Identifier: GPL-2.0
//
// sm_120 compiler-declared extended-parameter meta extend (Level-2 native
// actuator port). See window_meta_extend.cpp for the mechanism and scope.

#pragma once

#include <cstddef>

namespace xsched::cuda
{

struct XgExtendedImage
{
    const void *ptr;  // pointer to load: original image or owned patched copy
    bool owned;       // true when ptr is an owned copy created by this module
};

/// @brief Opt-in (XG_NATIVE_META_EXTEND) module-image extension: raises the
/// declared sm_120 parameter extent of eligible kernels to cover the fused
/// parameter window. With XG_NATIVE_META_KPARAM also set, the KPARAM_INFO
/// ordinal table is grown so the declared launch-parameter extent covers
/// the relay blob (safe only for blob-form launches). Returns the original
/// image untouched when disabled, unidentifiable, or on any anomaly (the
/// loader never fails because of us).
/// @param image the module image passed to cuModuleLoadData and friends
/// @return an XgExtendedImage; the returned ptr is what the driver load call
///         must receive. If owned, it is an owned copy whose storage starts
///         16 bytes before ptr; free it with XgFreeExtendedImage(ptr) - the
///         SAME pointer - after the load call returns. A copy may also be a
///         retargeted runtime fatbin wrapper (__fatBinC_Wrapper_t) whose
///         data field points at a patched nested container in the same
///         block; freeing the wrapper pointer covers both.
XgExtendedImage XgMetaExtendImage(const void *image);

/// @brief Frees an owned copy whose ptr was returned by XgMetaExtendImage
///        (pass exactly ext.ptr). Pointers without the module's ownership
///        header are reported and deliberately left untouched; the header
///        is read before validation, so this is NOT safe for arbitrary
///        foreign pointers.
void XgFreeExtendedImage(void *p);

} // namespace xsched::cuda

/// @brief Original per-ordinal parameter layout, registered by the extender
///        for kernels whose EIATTR_KPARAM table was grown under
///        XG_NATIVE_META_KPARAM (keyed by the kernel's mangled name, i.e.
///        the string cuFuncGetName reports). The HAL host-side marshaling
///        (constructor deep copy and window-relay adoption) uses these
///        ORIGINAL sizes/offsets so the app's own arguments keep their
///        compiled layout; grown sizes only reach the driver's launch-side
///        blob validation.
/// @param kernel_name mangled kernel name (may be nullptr: no lookup)
/// @param ordinals/offsets/sizes caller buffers of at least max_entries
///        elements (may be nullptr to only query the entry count)
/// @param max_entries capacity of the caller buffers
/// @param out_entries receives the registered entry count (not nullptr)
/// @return 1 when a registered layout exists for kernel_name, else 0
extern "C" __attribute__((visibility("default"))) int XgGetRelayOriginalParams(const char *kernel_name,
                                        unsigned int *ordinals,
                                        unsigned int *offsets,
                                        unsigned int *sizes,
                                        size_t max_entries,
                                        size_t *out_entries);

/// @brief Name-free identification of a grown kernel's ORIGINAL layout
///        from its LOADED layout. The meta extend keeps the KPARAM count
///        and every ordinal offset unchanged and inflates exactly one
///        ordinal's size so its blob-relative end equals 0x1520; a
///        registered entry matches when count/offsets agree and every
///        registered size is <= the loaded size, with zero diffs (no-op)
///        or exactly one strictly-grown ordinal ending at 0x1520.
///        Value-identical duplicate registrations are equivalent;
///        differing candidates are ambiguous and refused (fall back).
///        This closes the lookup without depending on the exact string
///        cuFuncGetName reports.
/// @param count loaded param count (0 skipped)
/// @param offsets/sizes loaded per-ordinal offsets and sizes (not nullptr)
/// @param ordinals/out_offsets/out_sizes caller buffers (may be nullptr
///        to only query the entry count)
/// @param max_entries capacity of the caller buffers
/// @param out_entries receives the entry count (not nullptr)
/// @return 1 when a registered layout matches the loaded shape, else 0
extern "C" __attribute__((visibility("default"))) int XgFindRelayOriginalParams(unsigned int count,
                                        const unsigned int *offsets,
                                        const unsigned int *sizes,
                                        unsigned int *ordinals,
                                        unsigned int *out_offsets,
                                        unsigned int *out_sizes,
                                        size_t max_entries,
                                        size_t *out_entries);

/// @brief Cheap gate: 1 when at least one original layout is registered
///        (i.e. the KPARAM knob grew at least one kernel in this
///        process), else 0. Lets zero-cost passthrough corridors skip
///        the param-info enumeration entirely when the knob is off.
extern "C" __attribute__((visibility("default"))) int XgHasRelayLayouts(void);

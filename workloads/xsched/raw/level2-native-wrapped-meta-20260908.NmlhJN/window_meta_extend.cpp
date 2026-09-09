// SPDX-License-Identifier: GPL-2.0
//
// sm_120 compiler-declared extended-parameter meta extend (Level-2 native
// actuator port, additive and unintegrated until tested by the build owner).
//
// Opt-in with the XG_NATIVE_META_EXTEND environment variable (presence
// based). When enabled, every CUDA module image that passes through the
// module-load entry points of the shim is copied and, inside its sm_120
// cubins, every kernel whose declared constant-bank parameter region starts
// at the sm_120 parameter base 0x380 with a smaller extent gets that extent
// raised to 0x1520, which covers the fused guardian/resume parameter window
// at c[0x0][0x1880..0x189c] (blob-relative 0x1500..0x151c, see
// CudaKernelCommand::AdoptWindowRelayExtra and the 28-byte XG_ARGS_BYTES
// block in level2/xsched_guardian_abi.h).
//
// Module images arrive either bare (raw fatbin container 0xba55ed50 or ELF
// cubin) or wrapped in the CUDA runtime fatbin wrapper (fatbinary_section.h
// __fatBinC_Wrapper_t: u32 magic 0x466243b1, u32 version 1, data -> the
// real container, filename_or_fatbins). Static cudart hands the wrapper to
// cuLibraryLoadData and the driver follows wrapper->data (root gdb
// specimen). For a wrapped image the extender owns one block holding a
// wrapper copy whose data field is retargeted at the patched nested
// container copy inside the same block, so the driver receives a wrapper
// identical to the original except for the owned data pointer.
//
// Two compiler-written nvinfo records describe that extent and both are
// raised together (cuobjdump -elf equivalence, nvcc 12.9 specimen):
//   EIATTR_CBANK_PARAM_SIZE (id 0x19, fmt 3): 16-bit whole bank size
//       (0x20 for the specimen kernel) lives at record offset +2.
//   EIATTR_PARAM_CBANK     (id 0x0a, fmt 4): payload
//       {u32 kind 0xa/0x9, u16 param_start, u16 param_size}; the param
//       size is the 16-bit field at payload offset +6.
// The KPARAM_INFO records (id 0x17) that describe the app's own arguments
// are never touched, so per-argument ordinals/offsets/sizes and hence the
// app launch parameters keep their compiled layout; the window args are
// extra, undeclared-by-KPARAM bank bytes inside the enlarged extent. All
// other records (KPARAM_INFO, EXIT_INSTR_OFFSETS 0x1c, SW_WAR 0x36, the
// .nv.merc mirrors, module-wide .nv.info) are left byte-identical.
//
// Limitation kept explicit: the current implementation does not grow the
// bank DATA image (.nv.constant0, 0x3a0 bytes for the specimen); growing it
// would require inserting bytes and rewriting every later section/table
// offset. Whether the driver sizes the c[0x0] bank from the EIATTR records
// above (making the data section irrelevant) is unproven and is resolved by
// the actual run.
//
// This is a CPU-side byte patch of the module image before the real driver
// loads it (cuModuleLoadData copies the image, so the mutated buffer can be
// freed right after the call). All failures pass the original image through
// unchanged: the extender never aborts the loading process.
//
// Ownership: the patched copy is a raw malloc block with a 16-byte header
// [uint64 magic 0x474d495458455847 ("XGEXTIMG") | uint64 data size] placed
// in front of the image bytes. The handed-out pointer is the image data
// pointer itself; XgFreeExtendedImage steps back over the header, validates
// the magic, and frees the whole block. For a wrapped image the block also
// holds the 24-byte wrapper copy in a 32-byte slot in front of the
// container copy; freeing the handed-out wrapper pointer frees the whole
// block. Pointers that do not carry the header are reported and
// deliberately NOT freed (leak beats crash).

#include "xsched/cuda/shim/window_meta_extend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "xsched/utils/log.h"

namespace xsched::cuda {

namespace {

constexpr uint16_t kMetaParamStart = 0x380;  // sm_120 parameter-region base
constexpr uint16_t kMetaParamSize  = 0x1520; // covers window args rel 0x1500 + 0x1c, 16-aligned

constexpr uint32_t kFatbinMagic = 0xba55ed50u;
constexpr uint32_t kElfMagic    = 0x464c457fu; // "\x7f" "ELF" little-endian

// CUDA runtime fatbin wrapper (fatbinary_section.h __fatBinC_Wrapper_t):
// {u32 magic 0x466243b1 (FATBINC_MAGIC), u32 version (FATBINC_VERSION 1),
// const unsigned long long *data -> the real fatbin container,
// void *filename_or_fatbins}. Static cudart hands this wrapper to the
// module-load entry and the driver follows wrapper->data (root gdb
// specimen: cuLibraryLoadData called from libcudart_static with the
// wrapper at rdi). The owned block reserves 32 bytes for the 24-byte
// wrapper copy (8 zero pad) so the nested container copy behind it stays
// 16-byte aligned; the wrapper copy's data field is retargeted at it.
constexpr uint32_t kFatbinWrapperMagic = 0x466243b1u; // FATBINC_MAGIC
constexpr uint32_t kFatbinWrapperVer1  = 1u;          // FATBINC_VERSION
constexpr size_t   kFatbinWrapSize     = 24;          // sizeof(__fatBinC_Wrapper_t), lp64
constexpr size_t   kFatbinWrapCopy     = 32;          // incl. alignment pad

constexpr uint8_t kNvInfoFmt4 = 0x04;         // {id, paylen, payload...}
constexpr uint8_t kNvInfoFmt3 = 0x03;         // {id, u16 value}  (HVAL)
constexpr uint8_t kNvInfoFmt2 = 0x02;         // {id, u16 value}  (BVAL)
constexpr uint8_t kNvInfoParamCBank   = 0x0a; // EIATTR_PARAM_CBANK (SVAL)
constexpr uint8_t kNvInfoCBankPrmSize = 0x19; // EIATTR_CBANK_PARAM_SIZE (HVAL)

// paylen no larger than a .nv.info record can sanity-carry; anything above
// is treated as a parse anomaly rather than trusted (prevents an unwalkable
// record from crashing the section scan).
constexpr uint64_t kNvInfoMaxPaylen = 0x8000;

constexpr uint64_t kRectMagic   = 0x474d495458455847ull; // "XGEXTIMG"
constexpr size_t   kRectHdrSize = 16;                    // magic u64 + size u64

constexpr size_t kImageMaxBytes = (size_t)1 << 27; // sanity bound: 128 MiB

bool meta_extend_enabled()
{
    static const bool enabled = std::getenv("XG_NATIVE_META_EXTEND") != nullptr;
    return enabled;
}

uint64_t le(const char *p, unsigned n)
{
    uint64_t v = 0;
    for (unsigned i = 0; i < n; ++i)
        v |= (uint64_t)(uint8_t)p[i] << (8 * i);
    return v;
}

struct Range
{
    size_t off;
    size_t size;
};

// ELF64 little-endian only (CUDA cubins are always ELF64 LE).
bool elf64le(const char *b, size_t size)
{
    return size >= 64 && le(b, 4) == kElfMagic && b[4] == 2 && b[5] == 1;
}

// Whole-file extent a CUDA loader must map: the greater of the ELF header,
// the section table, the program header table, and every section body. nvcc
// cubins place the program header table after the section table (readelf on
// the 11136-byte specimen: section headers end at 10856, program headers at
// 10856..11136), so the section-table end alone truncates real images.
bool cubin_whole_size(const char *b, size_t size, size_t *whole)
{
    if (!elf64le(b, size)) return false;
    uint64_t shoff = le(b + 0x28, 8);
    uint16_t shent = (uint16_t)le(b + 0x3a, 2);
    uint16_t shnum = (uint16_t)le(b + 0x3c, 2);
    uint64_t phoff = le(b + 0x20, 8);
    uint16_t phent = (uint16_t)le(b + 0x36, 2);
    uint16_t phnum = (uint16_t)le(b + 0x38, 2);
    uint64_t end = 64;
    if (shoff != 0 && shnum != 0 && shent >= 0x40)
        end = std::max<uint64_t>(end, shoff + (uint64_t)shnum * shent);
    if (phoff != 0 && phnum != 0 && phent != 0)
        end = std::max<uint64_t>(end, phoff + (uint64_t)phnum * phent);
    // section bodies referenced by in-bounds headers
    if (shoff != 0 && shnum != 0 && shent >= 0x40 &&
        shoff <= size && shnum <= (size - shoff) / shent) {
        for (uint16_t i = 0; i < shnum; ++i) {
            const char *h = b + shoff + (uint64_t)i * shent;
            uint64_t s_off  = le(h + 0x18, 8);
            uint64_t s_size = le(h + 0x20, 8);
            if (s_off <= size && s_size <= size - s_off)
                end = std::max<uint64_t>(end, s_off + s_size);
        }
    }
    *whole = (size_t)end;
    return true;
}

// Whole-container extent of a fatbin. Measured nvcc 12.9 layout: container
// header 0x10 bytes, then kind-2 entries; the fileSize field (0x2be0) counts
// after the container header, entry extent is entryHdrSize + paddedPayload,
// and the whole image is hdrsz + fileSize (0x2bf0). A legacy container whose
// fileSize already includes the header is accepted with a warning.
bool fatbin_whole_size(const char *b, size_t size, size_t *whole,
                       bool *legacy)
{
    uint16_t hdrsz = (uint16_t)le(b + 6, 2);
    uint64_t fsize = le(b + 8, 8);
    if (hdrsz >= 0x10 && fsize >= hdrsz &&
        hdrsz + fsize <= size && hdrsz + fsize >= (uint64_t)hdrsz * 2) {
        *whole = (size_t)(hdrsz + fsize);
        *legacy = false;
        return true;
    }
    if (hdrsz >= 0x10 && fsize >= hdrsz && fsize <= size) {
        XWARN("meta-extend: fatbin fileSize 0x%llx spans past the "
              "header-relative extent (image 0x%llx); using legacy extent",
              (unsigned long long)fsize, (unsigned long long)size);
        *whole = (size_t)fsize;
        *legacy = true;
        return true;
    }
    XWARN("meta-extend: fatbin extents (hdr 0x%x file 0x%llx image 0x%llx) "
          "inconsistent; leaving the image unpatched",
          (unsigned)hdrsz, (unsigned long long)fsize,
          (unsigned long long)size);
    return false;
}

// Collect per-kernel nvinfo sections: names beginning ".nv.info.", i.e. one
// section per kernel. The module-wide ".nv.info" and the ".nv.merc.*"
// mirrors carry no PARAM_CBANK records and are deliberately left out.
bool elf_kernel_info_sections(const char *base, size_t size,
                              std::vector<Range> *nvinfo_out)
{
    if (!elf64le(base, size)) return false;
    uint64_t shoff = le(base + 0x28, 8);
    uint16_t shentsize = (uint16_t)le(base + 0x3a, 2);
    uint16_t shnum = (uint16_t)le(base + 0x3c, 2);
    uint16_t shstrndx = (uint16_t)le(base + 0x3e, 2);
    if (shoff == 0 || shnum == 0 || shentsize < 0x40 || shstrndx >= shnum)
        return false;
    if (shoff > size || shnum > (size - shoff) / shentsize)
        return false;
    const char *shstr_hdr = base + shoff + (uint64_t)shstrndx * shentsize;
    uint64_t str_off = le(shstr_hdr + 0x18, 8);
    uint64_t str_size = le(shstr_hdr + 0x20, 8);
    if (str_off > size || str_size > size - str_off)
        return false;
    const char *shstr = base + str_off;
    size_t str_left = (size_t)(size - str_off);
    for (uint16_t i = 0; i < shnum; ++i) {
        const char *h = base + shoff + (uint64_t)i * shentsize;
        uint64_t s_off = le(h + 0x18, 8);
        uint64_t s_size = le(h + 0x20, 8);
        if (s_off > size || s_size > size - s_off) return false;
        uint64_t n_ofs = le(h, 4);
        if (n_ofs >= str_left) continue;
        const char *name = shstr + n_ofs;
        size_t name_left = str_left - (size_t)n_ofs;
        if (name_left < 10) continue; // ".nv.info." + NUL minimum
        if (std::memcmp(name, ".nv.info.", 9) != 0) continue;
        char buf[256];
        size_t n = 0;
        while (n + 1 < sizeof(buf) && name[n] != '\0') buf[n] = name[n], ++n;
        buf[n] = '\0';
        Range r{(size_t)s_off, (size_t)s_size};
        nvinfo_out->push_back(r);
        if (s_size >= 12) {
            XDEBG("meta-extend: scan %s (0x%llx B)", buf,
                  (unsigned long long)s_size);
        }
    }
    return true;
}

// Patch the parameter-extent nvinfo records of one kernel section.
//
// Pass A (read-only) finds EIATTR_PARAM_CBANK and decides eligibility from
// its param_start/param_size. Pass B raises EIATTR_CBANK_PARAM_SIZE (the
// record precedes PARAM_CBANK in compiler layout) and PARAM_CBANK itself.
//
// Returns: -1 parse anomaly, 0 no eligible bank, 1 patched, 2 already
// covering the window.
int extend_kernel_info_section(const char *tag, char *base,
                               size_t /*sec_walk_span*/, size_t sec_off,
                               size_t sec_size, bool *wrote)
{
    if (sec_size < 4) return 0;
    const char *sec = base + sec_off;

    // pass A: find PARAM_CBANK, read param region
    size_t cbank_rec = sec_size; // offset of the PARAM_CBANK record
    bool cbank_present = false;
    uint16_t pstart = 0, psize = 0;
    size_t o = 0;
    while (o + 4 <= sec_size) {
        uint8_t fmt = (uint8_t)sec[o];
        uint8_t id = (uint8_t)sec[o + 1];
        if (fmt != kNvInfoFmt4 && fmt != kNvInfoFmt3 && fmt != kNvInfoFmt2)
            return -1;
        uint64_t paylen = (fmt == kNvInfoFmt4) ? (uint64_t)le(sec + o + 2, 2) : 0;
        uint64_t rec = (fmt == kNvInfoFmt4) ? 4 + paylen : 4;
        if (rec > kNvInfoMaxPaylen || o + rec > sec_size) return -1;
        if (fmt == kNvInfoFmt4 && id == kNvInfoParamCBank && paylen == 8) {
            cbank_present = true;
            cbank_rec = o;
            pstart = (uint16_t)le(sec + o + 8, 2);
            psize  = (uint16_t)le(sec + o + 10, 2);
        }
        o += (size_t)rec;
    }
    if (o != sec_size) {
        // trailing halves of records: tolerated on a read-only pass only
        // when nothing to patch; otherwise flag the anomaly
        return cbank_present ? -1 : 0;
    }
    if (!cbank_present || pstart != kMetaParamStart) return 0;
    if (psize >= kMetaParamSize) {
        // already extended on a previous load of the same image; make the
        // paired CBANK_PARAM_SIZE record consistent before reporting
        bool aligned = true;
        o = 0;
        while (o + 4 <= sec_size) {
            uint8_t fmt = (uint8_t)sec[o];
            if (fmt == kNvInfoFmt3 &&
                (uint8_t)sec[o + 1] == kNvInfoCBankPrmSize) {
                uint16_t v = (uint16_t)le(sec + o + 2, 2);
                if (v < kMetaParamSize) {
                    if (tag[0] != '\0' && v != 0) {
                        XWARN("meta-extend: %s CBANK_PARAM_SIZE 0x%x "
                              "unextended beside PARAM_CBANK 0x%x",
                              tag, (unsigned)v, (unsigned)psize);
                    }
                    aligned = false;
                    break;
                }
            }
            o += 4;
        }
        return aligned ? 2 : -1;
    }

    // pass B: raise CBANK_PARAM_SIZE (id 0x19, u16 at record offset +2)
    char *buf = base + sec_off;
    o = 0;
    int patched = 0;
    while (o + 4 <= sec_size) {
        uint8_t fmt = buf[o];
        uint8_t id = buf[o + 1];
        if (fmt == kNvInfoFmt4) {
            uint64_t paylen = (uint64_t)le(buf + o + 2, 2);
            if (o + 4 + paylen > sec_size) break; // pass A already validated
            if (id == kNvInfoParamCBank && paylen == 8 && o == cbank_rec) {
                char *field = buf + o + 10;
                field[0] = (char)(kMetaParamSize & 0xff);
                field[1] = (char)((kMetaParamSize >> 8) & 0xff);
                XINFO("meta-extend: %s PARAM_CBANK param_size 0x%x -> 0x%x",
                      tag, (unsigned)psize, (unsigned)kMetaParamSize);
                ++patched;
            }
            o += (size_t)(4 + paylen);
        } else {
            if (id == kNvInfoCBankPrmSize && patched >= 0) {
                uint16_t v = (uint16_t)le(buf + o + 2, 2);
                if (v != kMetaParamSize) {
                    buf[o + 2] = (char)(kMetaParamSize & 0xff);
                    buf[o + 3] = (char)((kMetaParamSize >> 8) & 0xff);
                    XINFO("meta-extend: %s CBANK_PARAM_SIZE 0x%x -> 0x%x",
                          tag, (unsigned)v, (unsigned)kMetaParamSize);
                    ++patched;
                }
            }
            o += 4;
        }
    }
    if (patched == 0) return -1; // pass A said eligible but nothing was patched
    *wrote = true;
    XINFO("meta-extend: %s declared c[0x0] param region [0x%x, 0x%x]",
          tag, (unsigned)kMetaParamStart,
          (unsigned)(kMetaParamStart + kMetaParamSize));
    return 1;
}

// Patch all eligible kernels of one cubin ELF in place. Returns false on
// structural anomalies (caller then uses the unpatched copy as-is; nothing
// aborts the load).
bool extend_cubin(const char *what, char *base, size_t size, int *patched)
{
    std::vector<Range> sections;
    if (!elf_kernel_info_sections(base, size, &sections)) {
        XDEBG("meta-extend: %s is not a supported ELF64 cubin", what);
        return false;
    }
    char tag[128];
    int idx = 0;
    bool wrote = false;
    for (const Range &r : sections) {
        std::snprintf(tag, sizeof(tag), "%s[%d]", what, idx++);
        int rc = extend_kernel_info_section(tag, base, size, r.off, r.size,
                                            &wrote);
        if (rc < 0) {
            XWARN("meta-extend: %s nvinfo parse anomaly; kernel left "
                  "unpatched", tag);
        }
    }
    *patched = wrote ? 1 : 0;
    return true;
}

// Copy-and-patch the whole image held in data[0..size). Returns the number
// of kernels for which a record was rewritten (only used for the log line).
void meta_extend_image(char *data, size_t size)
{
    if (le(data, 4) != kFatbinMagic) {
        int kernels = 0;
        extend_cubin("cubin", data, size, &kernels);
        if (kernels == 0) {
            XDEBG("meta-extend: cubin untouched (no eligible kernel)");
        }
        return;
    }

    size_t whole = 0;
    bool legacy = false;
    // header fields were validated during the whole-size computation in
    // XgMetaExtendImage; re-derive the walk bound from the copy itself
    if (!fatbin_whole_size(data, size, &whole, &legacy) || whole > size ||
        whole >= kImageMaxBytes) {
        XWARN("meta-extend: fatbin bound re-check failed; loading unpatched");
        return;
    }
    uint16_t hdrsz = (uint16_t)le(data + 6, 2);
    size_t off = hdrsz;
    int kernels = 0;
    while (off + 0x20 <= whole) {
        const char *e = data + off;
        uint16_t kind = (uint16_t)le(e, 2);
        uint32_t ehdr = (uint32_t)le(e + 4, 4);
        uint64_t padded = le(e + 8, 8);
        if (ehdr < 0x28 || off + ehdr + padded > whole) {
            XWARN("meta-extend: fatbin entry beyond container (off 0x%llx); "
                  "leaving the rest unpatched", (unsigned long long)off);
            break;
        }
        if (kind == 2) {
            char *elf = const_cast<char *>(e) + ehdr;
            if (std::memcmp(elf, "\x7f" "ELF", 4) == 0)
                extend_cubin("fatbin-cubin", elf, (size_t)padded, &kernels);
        }
        off += (size_t)ehdr + (size_t)padded;
    }
    if (kernels == 0) {
        XDEBG("meta-extend: fatbin had no eligible kernel");
    }
}

} // namespace

XgExtendedImage XgMetaExtendImage(const void *image)
{
    XgExtendedImage out{image, false};
    if (image == nullptr) return out;
    if (!meta_extend_enabled()) return out;

    const char *b = (const char *)image;
    size_t size = 0;
    bool wrapped = false; // b is the runtime fatbin wrapper; payload at nest
    const char *nest = nullptr;
    if (le(b, 4) == kFatbinMagic) {
        size_t whole = 0;
        bool legacy = false;
        if (!fatbin_whole_size(b, kImageMaxBytes, &whole, &legacy)) {
            // the first probe runs unbounded; verify against a real read
            // only after the caller's buffer size is unknown: use the
            // second stage below
        }
        (void)whole;
        // Container extent must be derived from the copy below; a bare
        // header probe cannot bound the caller's allocation, so size the
        // buffer from the fatbin header itself and let the staging path
        // validate.
        uint16_t hdrsz = (uint16_t)le(b + 6, 2);
        uint64_t fsize = le(b + 8, 8);
        if (hdrsz < 0x10 || fsize < hdrsz ||
            fsize + hdrsz > kImageMaxBytes) {
            XWARN("meta-extend: unusable fatbin header (hdr 0x%x size "
                  "0x%llx); loading unpatched", (unsigned)hdrsz,
                  (unsigned long long)fsize);
            return out;
        }
        size = (size_t)(fsize + hdrsz);
    } else if (le(b, 4) == kElfMagic) {
        if (!cubin_whole_size(b, kImageMaxBytes, &size)) {
            XWARN("meta-extend: unusable cubin header; loading unpatched");
            return out;
        }
    } else if (le(b, 4) == kFatbinWrapperMagic &&
               le(b + 4, 4) == kFatbinWrapperVer1) {
        // the runtime fatbin wrapper: own a retargeted wrapper copy plus the
        // patched nested container, in one block (see kFatbinWrapCopy)
        uint64_t dptr = le(b + 8, 8);
        if (dptr == 0) {
            XWARN("meta-extend: fatbin wrapper without data; loading "
                  "unpatched");
            return out;
        }
        const char *d = (const char *)(uintptr_t)dptr;
        if (le(d, 4) == kFatbinMagic) {
            uint16_t hdrsz = (uint16_t)le(d + 6, 2);
            uint64_t fsize = le(d + 8, 8);
            if (hdrsz < 0x10 || fsize < hdrsz ||
                fsize + hdrsz > kImageMaxBytes) {
                XWARN("meta-extend: unusable wrapped fatbin header (hdr "
                      "0x%x size 0x%llx); loading unpatched", (unsigned)hdrsz,
                      (unsigned long long)fsize);
                return out;
            }
            size = (size_t)(fsize + hdrsz);
        } else if (le(d, 4) == kElfMagic) {
            if (!cubin_whole_size(d, kImageMaxBytes, &size)) {
                XWARN("meta-extend: unusable wrapped cubin header; loading "
                      "unpatched");
                return out;
            }
        } else {
            XDEBG("meta-extend: wrapper payload is neither fatbin nor "
                  "cubin; unchanged");
            return out;
        }
        wrapped = true;
        nest = d;
    } else {
        return out; // PTX text or other unknown container: unchanged
    }
    if (size < 64 || size > kImageMaxBytes) {
        XWARN("meta-extend: image extent 0x%llx rejected; loading unpatched",
              (unsigned long long)size);
        return out;
    }

    size_t wrap = wrapped ? kFatbinWrapCopy : 0;
    try {
        void *base = nullptr;
        // header-checked allocation: [kRectMagic | copy size | image bytes];
        // the wrapper case prepends a 32-byte retargeted wrapper copy
        // (24 bytes per fatbinary_section.h plus 8 zero pad) in front of
        // the nested container copy, so one free covers both
        void *block = std::malloc(kRectHdrSize + wrap + size);
        base = block;
        if (base == nullptr) {
            XWARN("meta-extend: image copy allocation failed; loading "
                  "unpatched");
            return out;
        }
        *(uint64_t *)base = kRectMagic;
        *((uint64_t *)base + 1) = (uint64_t)(wrap + size);
        char *data = (char *)base + kRectHdrSize;
        if (wrapped) {
            std::memcpy(data, b, kFatbinWrapSize);
            std::memset(data + kFatbinWrapSize, 0, wrap - kFatbinWrapSize);
            // retarget the copy's data field at the nested fatbin copy
            *(char **)(data + 8) = data + wrap;
            std::memcpy(data + wrap, nest, size);
            meta_extend_image(data + wrap, size);
            XINFO("meta-extend: owned fatbin-wrapper copy 0x%llx + 0x%llx "
                  "bytes (source wrapper 0x%llx)",
                  (unsigned long long)wrap, (unsigned long long)size,
                  (unsigned long long)(uintptr_t)image);
        } else {
            std::memcpy(data, b, size);
            meta_extend_image(data, size);
            XINFO("meta-extend: owned copy 0x%llx bytes "
                  "(source image 0x%llx)", (unsigned long long)size,
                  (unsigned long long)(uintptr_t)image);
        }
        out.ptr = data;
        out.owned = true;
    } catch (...) {
        XWARN("meta-extend: image copy failed; loading unpatched");
        if (out.owned)
            XgFreeExtendedImage(const_cast<void *>(out.ptr));
        return out;
    }
    return out;
}

void XgFreeExtendedImage(void *p)
{
    if (p == nullptr) return;
    char *hdr = (char *)p - kRectHdrSize;
    uint64_t magic = le(hdr, 8);
    if (magic != kRectMagic) {
        // not an owned copy: never free pointers we did not hand out
        XWARN("meta-extend: pointer 0x%llx is not an owned image copy; "
              "NOT freed", (unsigned long long)(uintptr_t)p);
        return;
    }
    std::free(hdr);
}

} // namespace xsched::cuda

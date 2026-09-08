// SPDX-License-Identifier: GPL-2.0
//
// CPU-side sm_120 guardian adapter tool. Derives the LDC constant-bank
// offset encoding from the GENERATED probe cubin (raw .text instruction
// words plus nvdisasm text), re-encodes the parameter-region LDC consumers
// of the two guardian stub cubins onto the upstream debugger parameter
// region, verifies each patched cubin by writing it out and re-running
// nvdisasm against it, asserts the exact consumed-offset multisets and a
// register-indirect transfer instruction in the restore artifact, and only
// then emits the C arrays that the vendored arch/sm120.cpp includes.
//
// usage:
//   xg_ldc_patcher PROBE.cubin CHECK.cubin RESTORE.cubin NVDISASM OUT.h
//
// Every step fails loudly instead of emitting a blob derived from an
// assumption.
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

uint64_t le(const char *p, unsigned n)
{
    uint64_t v = 0;
    for (unsigned i = 0; i < n; ++i)
        v |= (uint64_t)(uint8_t)p[i] << (8 * i);
    return v;
}

struct Section {
    std::string name;
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct Elf {
    std::vector<char> bytes;
    std::vector<Section> sections;

    explicit Elf(const char *path) { load(path); }
    Elf() = default;

    void load(const char *path)
    {
        FILE *f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "ldc_patcher: cannot open %s: %s\n", path,
                    strerror(errno));
            exit(1);
        }
        fseek(f, 0, SEEK_END);
        long total = ftell(f);
        fseek(f, 0, SEEK_SET);
        bytes.resize((size_t)total);
        if (fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) exit(1);
        fclose(f);
        if (bytes.size() < 64 || memcmp(bytes.data(), "\x7f" "ELF", 4) != 0) {
            fprintf(stderr, "ldc_patcher: %s is not an ELF file\n", path);
            exit(1);
        }
        if (bytes[4] != 2 || bytes[5] != 1) { // little-endian ELF64
            fprintf(stderr, "ldc_patcher: %s is not little-endian ELF64\n", path);
            exit(1);
        }
        uint64_t shoff = le(&bytes[0x28], 8);
        uint16_t shentsize = (uint16_t)le(&bytes[0x3a], 2);
        uint16_t shnum = (uint16_t)le(&bytes[0x3c], 2);
        uint16_t shstrndx = (uint16_t)le(&bytes[0x3e], 2);
        if (shoff == 0 || shnum == 0 || shstrndx >= shnum ||
            shoff + (uint64_t)shnum * shentsize > bytes.size()) {
            fprintf(stderr, "ldc_patcher: %s has a thin section table\n", path);
            exit(1);
        }
        const char *h = &bytes[shoff + (uint64_t)shstrndx * shentsize];
        uint64_t str_off = le(h + 0x18, 8);
        uint64_t str_size = le(h + 0x20, 8);
        if (str_off + str_size > bytes.size()) {
            fprintf(stderr, "ldc_patcher: %s string table out of range\n", path);
            exit(1);
        }
        const char *shstr = &bytes[str_off];
        for (uint16_t i = 0; i < shnum; ++i) {
            const char *hi = &bytes[shoff + (uint64_t)i * shentsize];
            uint32_t name_off = (uint32_t)le(hi + 0, 4);
            uint64_t s_off = le(hi + 0x18, 8);
            uint64_t s_size = le(hi + 0x20, 8);
            if (s_off + s_size > bytes.size()) {
                fprintf(stderr, "ldc_patcher: %s section %u out of range\n", path, i);
                exit(1);
            }
            Section s;
            s.name = shstr + name_off;
            s.offset = s_off;
            s.size = s_size;
            sections.push_back(s);
        }
    }

    const Section *find(const std::string &name) const
    {
        for (const Section &s : sections)
            if (s.name == name) return &s;
        return nullptr;
    }
};

struct Sample {
    uint64_t pc;     // byte offset inside the section
    uint64_t offset; // c[0x0][offset]
    uint64_t word[2];
    std::string form; // opcode family: "LDC" scalar/vector, "LDCU" uniform
    unsigned width;   // decoded bytes consumed: 4 (no/.32) or 8 (.64)
};

// Extract "/*NNNN*/ ... OP ..., c[0x0][0xYYYY] ;" pc/offset pairs and keep
// the decoded instruction form and consumed width. On nvcc 12.9 sm_120 the
// observed parameter-consumer forms are "LDC" (scalar bank loads, also with
// a ".64" suffix for 8-byte consumers) and "LDCU" (uniform loads, also with
// ".64"); the legacy MOV fallback of other toolchains stays accepted and is
// annotated with its raw token and width 4. Decimal and hexadecimal /*_*/
// pc forms both occur across nvdisasm releases; any misread pc is caught by
// the patched-artifact round trip.
bool parse_ldc(const std::string &line, Sample *s)
{
    size_t at = line.find("LDC");
    if (at == std::string::npos) {
        at = line.find("MOV");
        if (at == std::string::npos) return false;
    }
    // Resolve the opcode token; a predicate prefix such as "@!P0" is not
    // alphanumeric, so the token boundaries are found by walking over
    // alphanumeric characters around the LDC substring.
    size_t tok = at;
    while (tok > 0 &&
           isalnum(static_cast<unsigned char>(line[tok - 1])) != 0)
        --tok;
    size_t pos = tok;
    while (pos < line.size() &&
           isalnum(static_cast<unsigned char>(line[pos])) != 0)
        ++pos;
    std::string op = line.substr(tok, pos - tok);
    if (op == "LDC") {
        s->form = "LDC";
    } else if (op == "LDCU" || op == "ULDC") {
        s->form = "LDCU";
    } else {
        s->form = op;
    }
    s->width = 4;
    if (s->form == "LDC" || s->form == "LDCU") {
        // Width suffix: ".64" consumes 8 bytes, ".32" or no suffix consume
        // 4 bytes; any other suffix is not a supported parameter consumer.
        if (pos < line.size() && line[pos] == '.') {
            ++pos;
            unsigned long long suffix = 0;
            bool any = false;
            while (pos < line.size() &&
                   isdigit(static_cast<unsigned char>(line[pos])) != 0) {
                suffix = suffix * 10 +
                         (unsigned long long)(line[pos] - '0');
                ++pos;
                any = true;
            }
            if (!any) return false;
            if (suffix == 64) {
                s->width = 8;
            } else if (suffix != 32) {
                return false;
            }
        }
    }
    size_t bank = line.find("c[0x0][0x", at);
    if (bank == std::string::npos) return false;
    size_t pc0 = line.find("/*");
    if (pc0 == std::string::npos) return false;
    size_t digits = pc0 + 2;
    while (digits < line.size() && line[digits] != '*') digits++;
    if (digits >= line.size()) return false;
    const std::string num = line.substr(pc0 + 2, digits - pc0 - 2);
    if (num.empty()) return false;
    s->pc = std::strtoull(num.c_str(), NULL, 16);
    s->offset = std::strtoull(line.c_str() + bank + 9, NULL, 16);
    return true;
}

std::string nvdisasm_text(const char *nvdisasm, const char *cubin)
{
    std::string cmd = std::string("\"") + nvdisasm + "\" -c \"" + cubin +
                      "\" 2>/dev/null";
    FILE *p = popen(cmd.c_str(), "r");
    if (!p) {
        fprintf(stderr, "ldc_patcher: cannot run nvdisasm\n");
        exit(1);
    }
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), p)) > 0) out.append(buf, n);
    int status = pclose(p);
    if (status != 0) {
        fprintf(stderr, "ldc_patcher: nvdisasm failed on %s\n", cubin);
        exit(1);
    }
    return out;
}

std::string nvdisasm_full(const char *nvdisasm, const char *cubin)
{
    std::string cmd = std::string("\"") + nvdisasm + "\" \"" + cubin +
                      "\" 2>/dev/null";
    FILE *p = popen(cmd.c_str(), "r");
    if (!p) {
        fprintf(stderr, "ldc_patcher: cannot run nvdisasm\n");
        exit(1);
    }
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), p)) > 0) out.append(buf, n);
    int status = pclose(p);
    if (status != 0) {
        fprintf(stderr, "ldc_patcher: nvdisasm failed on %s\n", cubin);
        exit(1);
    }
    return out;
}

// EIATTR_PARAM_CBANK in the full nvdisasm dump records the c[0x0] window
// holding this kernel's parameters; the textual record is:
//             .word   index@(.nv.constant0.<kernel>)
//             .short  0x0380                 <- parameter-region start
//             .short  0x0018                 <- parameter-region size
bool parse_param_cbank(const std::string &text, const std::string &kernel,
                       uint64_t *start, uint64_t *size)
{
    const std::string needle = "index@(.nv.constant0." + kernel + ")";
    size_t at = text.find(needle);
    if (at == std::string::npos) return false;
    size_t line_end = text.find('\n', at);
    unsigned found = 0;
    uint64_t values[2] = {0, 0};
    while (found < 2 && line_end != std::string::npos) {
        size_t begin = line_end + 1;
        line_end = text.find('\n', begin);
        std::string line = text.substr(
            begin,
            line_end == std::string::npos ? std::string::npos
                                          : line_end - begin);
        if (line.find("//----- nvinfo") != std::string::npos) break;
        size_t sh = line.find(".short");
        if (sh == std::string::npos) continue;
        size_t hex = line.find("0x", sh);
        if (hex == std::string::npos) continue;
        char *end = NULL;
        unsigned long long v = strtoull(line.c_str() + hex, &end, 16);
        if (end == line.c_str() + hex) continue;
        values[found++] = v;
    }
    if (found != 2) return false;
    *start = values[0];
    *size = values[1];
    return true;
}

std::vector<Sample> collect_samples(const std::string &disasm, const Elf &elf,
                                    const Section &text)
{
    std::vector<Sample> samples;
    size_t begin = 0;
    while (begin < disasm.size()) {
        size_t eol = disasm.find('\n', begin);
        std::string line = disasm.substr(
            begin, eol == std::string::npos ? std::string::npos : eol - begin);
        begin = eol == std::string::npos ? disasm.size() : eol + 1;
        Sample s;
        if (!parse_ldc(line, &s)) continue;
        if (s.pc + 16 > text.size) {
            fprintf(stderr, "ldc_patcher: LDC pc %llu outside %s (%llu B)\n",
                    (unsigned long long)s.pc, text.name.c_str(),
                    (unsigned long long)text.size);
            exit(1);
        }
        const char *p = &elf.bytes[text.offset + s.pc];
        s.word[0] = le(p, 8);
        s.word[1] = le(p + 8, 8);
        samples.push_back(s);
    }
    return samples;
}

struct Encoding {
    int half;  // which 8-byte half of the 16-byte SASS instruction
    int shift; // bit shift of the offset field inside that half
    int width; // field width
};

std::vector<Encoding> solve(const std::vector<Sample> &samples)
{
    std::vector<Encoding> candidates;
    for (int half = 0; half < 2; ++half) {
        for (int shift = 0; shift <= 52; shift += 2) {
            for (int width : {12, 14, 16, 20, 24, 32}) {
                if (shift + width > 64) continue;
                uint64_t mask = ((1ULL << width) - 1ULL) << shift;
                bool ok = true;
                for (const Sample &s : samples) {
                    if (((s.word[half] & mask) >> shift) != s.offset) {
                        ok = false;
                        break;
                    }
                }
                if (ok) candidates.push_back(Encoding{half, shift, width});
            }
        }
    }
    return candidates;
}

bool multiset_equals(const std::vector<Sample> &samples,
                     const std::vector<uint64_t> &expected)
{
    if (samples.size() != expected.size()) return false;
    std::vector<bool> hit(expected.size(), false);
    for (const Sample &s : samples) {
        bool found = false;
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!hit[i] && expected[i] == s.offset) {
                hit[i] = true;
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr,
                "usage: %s PROBE.cubin CHECK.cubin RESTORE.cubin NVDISASM OUT.h\n",
                argv[0]);
        return 2;
    }
    const char *probe_path = argv[1];
    const char *check_path = argv[2];
    const char *restore_path = argv[3];
    const char *nvdisasm_bin = argv[4];
    const char *out_header = argv[5];
    const uint64_t dbg = 0x1880ULL; // upstream debugger-parameter region

    // ---- 1. probe: derive candidate encodings from the artifact ----
    Elf probe(probe_path);
    const Section *probe_text = probe.find(".text.xg_ldc_probe");
    if (!probe_text) {
        fprintf(stderr, "ldc_patcher: probe cubin misses .text.xg_ldc_probe\n");
        return 1;
    }
    std::vector<Sample> probe_samples =
        collect_samples(nvdisasm_text(nvdisasm_bin, probe_path), probe,
                        *probe_text);
    uint64_t param_base = ~0ULL;
    for (const Sample &s : probe_samples)
        if (s.offset < param_base) param_base = s.offset;
    if (probe_samples.size() != 6 || param_base == ~0ULL) {
        fprintf(stderr, "ldc_patcher: probe exposes %zu LDC samples (need 6)\n",
                probe_samples.size());
        return 1;
    }
    if (param_base > dbg) {
        fprintf(stderr,
                "ldc_patcher: parameter base 0x%llx beyond the debugger "
                "region 0x%llx\n",
                (unsigned long long)param_base, (unsigned long long)dbg);
        return 1;
    }
    {
        // The probe must contain no LDC consumer outside one parameter set
        // of three u64 parameters (six consecutive 32-bit halves).
        std::vector<uint64_t> expect;
        for (uint64_t k = 0; k < 6; ++k) expect.push_back(param_base + 4 * k);
        if (!multiset_equals(probe_samples, expect)) {
            fprintf(stderr,
                    "ldc_patcher: probe LDC offsets are not the three consecutive "
                    "u64 parameters at base 0x%llx; artifact polluted\n",
                    (unsigned long long)param_base);
            return 1;
        }
    }
    std::vector<Encoding> candidates = solve(probe_samples);
    if (candidates.empty()) {
        fprintf(stderr,
                "ldc_patcher: no contiguous (half, shift, width) LDC encoding "
                "reproduces the %zu probe samples at base 0x%llx; sm_120 encoding "
                "not derivable from artifact\n",
                probe_samples.size(), (unsigned long long)param_base);
        return 1;
    }

    // ---- 2. patch + verify on the real artifacts ----
    const uint64_t delta = dbg - param_base;
    Elf check(check_path), restore(restore_path);
    const Section *check_text = check.find(".text.check_preempt_port");
    const Section *restore_text = restore.find(".text.restore_exec_port");
    if (!check_text || !restore_text) {
        fprintf(stderr, "ldc_patcher: stub cubins miss their .text sections\n");
        return 1;
    }
    const std::vector<uint64_t> check_expect = {dbg + 0, dbg + 4, dbg + 16,
                                                dbg + 20};
    const std::vector<uint64_t> restore_expect = {dbg + 0, dbg + 4, dbg + 8,
                                                  dbg + 12};
    std::vector<Sample> check_src =
        collect_samples(nvdisasm_text(nvdisasm_bin, check_path), check,
                        *check_text);
    std::vector<Sample> restore_src =
        collect_samples(nvdisasm_text(nvdisasm_bin, restore_path), restore,
                        *restore_text);
    {
        // Original consumed set must sit inside the derived parameter base.
        bool ok = true;
        for (const Sample &s : check_src)
            if (s.offset < param_base ||
                s.offset >= param_base + 28)
                ok = false;
        for (const Sample &s : restore_src)
            if (s.offset < param_base ||
                s.offset >= param_base + 28)
                ok = false;
        if (check_src.size() != 4 || restore_src.size() != 4 || !ok) {
            fprintf(stderr,
                    "ldc_patcher: unexpected stub LDC consumers (check=%zu, "
                    "restore=%zu, expected 4 parameter loads each within the "
                    "28-byte actuator region)\n",
                    check_src.size(), restore_src.size());
            return 1;
        }
    }

    const char *tmp_check = "/tmp/opencode/xg_check_patched.cubin";
    const char *tmp_restore = "/tmp/opencode/xg_restore_patched.cubin";
    if (system("mkdir -p /tmp/opencode") != 0) return 1;

    bool verified = false;
    for (const Encoding &enc : candidates) {
        const uint64_t mask = ((1ULL << enc.width) - 1ULL) << enc.shift;
        // Patch CHECK.
        Elf patched_check(check_path);
        unsigned patched = 0;
        for (const Sample &s : check_src) {
            uint64_t new_off = s.offset + delta;
            if (((new_off << enc.shift) >> enc.shift) != new_off) {
                patched = 0;
                break;
            }
            char *dst = &patched_check.bytes[check_text->offset + s.pc + 8 * enc.half];
            uint64_t word = le(dst, 8);
            word &= ~mask;
            word |= new_off << enc.shift;
            for (unsigned i = 0; i < 8; ++i)
                dst[i] = (char)((word >> (8 * i)) & 0xff);
            patched++;
        }
        if (patched != 4) continue;
        FILE *t = fopen(tmp_check, "wb");
        if (!t) { fprintf(stderr, "ldc_patcher: cannot write %s\n", tmp_check); return 1; }
        if (fwrite(patched_check.bytes.data(), 1, patched_check.bytes.size(), t)
            != patched_check.bytes.size()) exit(1);
        fclose(t);
        // Re-disassemble the patched artifact.
        std::string dis = nvdisasm_text(nvdisasm_bin, tmp_check);
        Elf verified_elf(tmp_check);
        const Section *vt = verified_elf.find(".text.check_preempt_port");
        if (!vt) continue;
        std::vector<Sample> got =
            collect_samples(dis, verified_elf, *vt);
        if (!multiset_equals(got, check_expect)) continue;

        // The same encoding must hold for restore.
        Elf patched_restore(restore_path);
        patched = 0;
        for (const Sample &s : restore_src) {
            uint64_t new_off = s.offset + delta;
            if (((new_off << enc.shift) >> enc.shift) != new_off) {
                patched = 0;
                break;
            }
            char *dst = &patched_restore.bytes[restore_text->offset + s.pc + 8 * enc.half];
            uint64_t word = le(dst, 8);
            word &= ~mask;
            word |= new_off << enc.shift;
            for (unsigned i = 0; i < 8; ++i)
                dst[i] = (char)((word >> (8 * i)) & 0xff);
            patched++;
        }
        if (patched != 4) continue;
        t = fopen(tmp_restore, "wb");
        if (!t) { fprintf(stderr, "ldc_patcher: cannot write %s\n", tmp_restore); return 1; }
        if (fwrite(patched_restore.bytes.data(), 1, patched_restore.bytes.size(), t)
            != patched_restore.bytes.size()) exit(1);
        fclose(t);
        std::string rdis = nvdisasm_text(nvdisasm_bin, tmp_restore);
        Elf verified_restore(tmp_restore);
        const Section *rt = verified_restore.find(".text.restore_exec_port");
        if (!rt) continue;
        std::vector<Sample> rgot =
            collect_samples(rdis, verified_restore, *rt);
        if (!multiset_equals(rgot, restore_expect)) continue;

        // ---- restore artifact must contain a register-indirect CALL ----
        bool indirect = false;
        {
            size_t begin = 0;
            while (begin < rdis.size() && !indirect) {
                size_t eol = rdis.find('\n', begin);
                size_t end = eol == std::string::npos ? rdis.size() : eol;
                std::string line = rdis.substr(begin, end - begin);
                begin = eol == std::string::npos ? rdis.size() : eol + 1;
                size_t at = line.find("CALL");
                if (at == std::string::npos) continue;
                if (line.find("[R", at) != std::string::npos) indirect = true;
            }
        }
        if (!indirect) {
            fprintf(stderr,
                    "ldc_patcher: restore artifact shows no register-indirect CALL; "
                    "the entry-point transfer instruction differs from the ported "
                    "assumption. Boundary: transfer encoding must be re-derived "
                    "from the actual artifact before vendoring\n");
            return 1;
        }

        // Encoding verified end to end on real artifacts.
        fprintf(stderr,
                "ldc_patcher: verified encoding half=%d shift=%d width=%d "
                "param_base=0x%llx delta=0x%llx\n",
                enc.half, enc.shift, enc.width, (unsigned long long)param_base,
                (unsigned long long)delta);
        // ---- 3. emit ----
        FILE *out = fopen(out_header, "w");
        if (!out) {
            fprintf(stderr, "ldc_patcher: cannot write %s\n", out_header);
            return 1;
        }
        fprintf(out,
                "/* Generated by level2/native/ldc_patcher.cpp from the generated "
                "sm_120\n * guardian stub cubins (encoding half=%d shift=%d width=%d, "
                "parameter base\n * 0x%llx re-encoded onto the debugger region 0x1880).\n"
                " * Do not edit by hand. */\n"
                "#ifndef XG_SM120_GUARDIAN_ARRAYS_H\n"
                "#define XG_SM120_GUARDIAN_ARRAYS_H\n\n",
                enc.half, enc.shift, enc.width, (unsigned long long)param_base);
        struct Named {
            const char *name;
            const Section *text;
            const std::vector<Sample> *src;
        };
        Named named[2] = {
            {"check_preempt", check_text, &check_src},
            {"restore_exec", restore_text, &restore_src},
        };
        for (int b = 0; b < 2; ++b) {
            Elf &art = b == 0 ? patched_check : patched_restore;
            fprintf(out, "static const unsigned long long xg_sm120_%s[] = {\n",
                    named[b].name);
            const Section *sec = named[b].text;
            for (uint64_t off = 0; off + 16 <= sec->size; off += 16) {
                uint64_t w0 = le(&art.bytes[sec->offset + off], 8);
                uint64_t w1 = le(&art.bytes[sec->offset + off + 8], 8);
                fprintf(out, "    0x%016llxULL, 0x%016llxULL,\n",
                        (unsigned long long)w0, (unsigned long long)w1);
            }
            fprintf(out, "};\n");
        }
        fprintf(out, "\n#endif /* XG_SM120_GUARDIAN_ARRAYS_H */\n");
        fclose(out);
        fprintf(stderr, "ldc_patcher: emitted %s\n", out_header);
        verified = true;
        break;
    }
    if (!verified) {
        fprintf(stderr,
                "ldc_patcher: no encoding candidate passed patched-artifact "
                "verification; refusing to emit sm_120 arrays\n");
        return 1;
    }
    return 0;
}

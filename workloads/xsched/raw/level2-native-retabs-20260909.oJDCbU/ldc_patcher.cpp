// SPDX-License-Identifier: GPL-2.0
//
// CPU-side sm_120 guardian extractor. Consumes the actual generated stub
// cubins of the padded-ABI specimens (native/check_preempt_port.cu,
// native/restore_exec_port.cu), verifies the real parameter placement
// against the upstream debugger-parameter window that
// platforms/cuda/hal/src/level2/instrument.cpp fills at launch with
// cuXtraSetDebuggerParams (28 bytes at c[0x0][0x1880]: preempt buffer,
// entry point, kernel index, killable flag), verifies the conditional-exit
// retention of the guardian and the ABI call transfer of the resume stub
// directly on the actual instruction words, rewrites the resume transfer
// tail in the emitted blob to the runtime-proven sm_70/sm_86 return-pair
// form (the compiled register-target call has never been run through on
// sm_120), and only then emits the raw
// SASS streams for platforms/cuda/hal/src/arch/sm120.cpp (GuardianSM120).
//
// usage:
//   xg_ldc_patcher CHECK.cubin RESTORE.cubin NVDISASM OUT.h
//
// The leading unused 0x1500-byte by-value pad parameter (device of proof:
// native/probe.cu) makes the existing compiler emit the guardian parameter
// consumers directly on c[0x0][0x1880/0x1890] and the resume consumers on
// c[0x0][0x1880/0x1888], so this tool performs no LDC immediate
// re-encoding and no pad coverage solving. The probe cubin is therefore
// no longer an input; it stays the historical device of proof only. Every
// step fails loudly instead of emitting a stream derived from an
// assumption.
#include <algorithm>
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
            fprintf(stderr, "ldc_patcher: %s is not little-endian ELF64\n",
                    path);
            exit(1);
        }
        uint64_t shoff = le(&bytes[0x28], 8);
        uint16_t shentsize = (uint16_t)le(&bytes[0x3a], 2);
        uint16_t shnum = (uint16_t)le(&bytes[0x3c], 2);
        uint16_t shstrndx = (uint16_t)le(&bytes[0x3e], 2);
        if (shoff == 0 || shnum == 0 || shstrndx >= shnum ||
            shoff + (uint64_t)shnum * shentsize > bytes.size()) {
            fprintf(stderr, "ldc_patcher: %s has a thin section table\n",
                    path);
            exit(1);
        }
        const char *h = &bytes[shoff + (uint64_t)shstrndx * shentsize];
        uint64_t str_off = le(h + 0x18, 8);
        uint64_t str_size = le(h + 0x20, 8);
        if (str_off + str_size > bytes.size()) {
            fprintf(stderr, "ldc_patcher: %s string table out of range\n",
                    path);
            exit(1);
        }
        const char *shstr = &bytes[str_off];
        for (uint16_t i = 0; i < shnum; ++i) {
            const char *hi = &bytes[shoff + (uint64_t)i * shentsize];
            uint32_t name_off = (uint32_t)le(hi + 0, 4);
            uint64_t s_off = le(hi + 0x18, 8);
            uint64_t s_size = le(hi + 0x20, 8);
            if (s_off + s_size > bytes.size()) {
                fprintf(stderr, "ldc_patcher: %s section %u out of range\n",
                        path, i);
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
    uint64_t pc;      // byte offset inside the section
    uint64_t offset;  // c[0x0][offset]
    uint64_t word[2]; // raw instruction words at pc
    std::string form; // opcode family: "LDC", "LDCU", or other token
    unsigned width;   // decoded bytes consumed: 4 (no/.32) or 8 (.64)
};

// Extract "/*NNNN*/ ... OP ..., c[0x0][0xYYYY] ;" pc/offset pairs and keep
// the decoded instruction form and consumed width. Observed parameter
// consumer forms are "LDC" and "LDCU" (also with ".64"); the c[0x4] marker
// load of the build-only exit-retention store carries no c[0x0] reference
// and falls through this parser. Decimal and hexadecimal /*_*/ pc forms
// both occur across nvdisasm releases; a misread pc is caught by the
// per-artifact instruction assertions below.
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

// Raw EIATTR records of one kernel's .nv.info.<kernel> section, decoded
// straight from the cubin bytes: EIATTR_PARAM_CBANK (id 0x0a) gives the
// c[0x0] parameter window, EIATTR_EXIT_INSTR_OFFSETS (id 0x1c) the EXIT
// offsets inside .text, EIATTR_KPARAM_INFO_V2 (id 0x45) the parameter
// layout (ordinal, region offset, size). Record formats this tool does
// not need (fmt 0x02 and 0x03) are skipped; unexpected formats fail
// loudly.
struct NvInfo {
    struct Param { uint64_t ord, off, size; };
    uint64_t param_start = 0;
    uint64_t param_size = 0;
    uint64_t exits[4];
    unsigned exit_n = 0;
    Param kparam[8];
    unsigned kparam_n = 0;
};

bool parse_nvinfo(const Elf &elf, const std::string &kernel, NvInfo *ni)
{
    const Section *s = elf.find(".nv.info." + kernel);
    if (!s) return false;
    const char *b = &elf.bytes[s->offset];
    uint64_t o = 0;
    while (o < s->size) {
        if (o + 4 > s->size) return false;
        unsigned fmt = (unsigned char)b[o];
        unsigned id = (unsigned char)b[o + 1];
        if (fmt == 0x04) {
            uint64_t paylen = le(b + o + 2, 2);
            if (o + 4 + paylen > s->size) return false;
            const char *p = b + o + 4;
            if (id == 0x1c && paylen % 4 == 0) {
                for (uint64_t k = 0; k < paylen; k += 4) {
                    if (ni->exit_n >= 4) return false;
                    ni->exits[ni->exit_n++] = le(p + k, 4);
                }
            } else if (id == 0x0a && paylen == 8) {
                ni->param_start = le(p + 4, 2);
                ni->param_size = le(p + 6, 2);
            } else if (id == 0x45 && paylen == 12) {
                if (ni->kparam_n >= 8) return false;
                uint64_t packed = le(p + 4, 4);
                ni->kparam[ni->kparam_n++] =
                    NvInfo::Param{packed & 0xffff, packed >> 16,
                                  le(p + 8, 2)};
            }
            o += 4 + paylen;
        } else if (fmt == 0x03 || fmt == 0x02) {
            o += 4;
        } else {
            fprintf(stderr,
                    "ldc_patcher: unexpected nvinfo format 0x%x (id 0x%x) in "
                    ".nv.info.%s\n", fmt, id, kernel.c_str());
            return false;
        }
    }
    return true;
}

const NvInfo::Param *kparam_find(const NvInfo &ni, uint64_t ord)
{
    for (unsigned i = 0; i < ni.kparam_n; ++i)
        if (ni.kparam[i].ord == ord) return &ni.kparam[i];
    return nullptr;
}

// One 16-byte SASS instruction as carried in the cubin text: the 8-byte
// encoding word followed by the 8-byte scheduling/control word. This
// layout is what the HAL consumers copy into instruction memory.
struct Instr { uint64_t enc, ctl; };

bool instr_at(const Elf &elf, const Section &text, uint64_t pc, Instr *in)
{
    if (pc % 16 != 0 || pc + 16 > text.size) return false;
    const char *p = &elf.bytes[text.offset + pc];
    in->enc = le(p, 8);
    in->ctl = le(p + 8, 8);
    return true;
}

// The HAL path copies the extracted bytes raw into instruction memory; a
// load-time text relocation would silently go unresolved. The actual stub
// cubins carry no main .rela.text entries, and this must stay true.
bool rela_clear(const Elf &elf, const std::string &kernel)
{
    const Section *s = elf.find(".rela.text." + kernel);
    return !s || s->size == 0;
}

// In-window parameter-consumer samples of an artifact: skips fixed ABI
// metadata reads below the window, fails loudly on consumers above it and
// on unsupported opcode forms.
static std::vector<Sample> window_samples(const std::string &dis,
                                          const Elf &elf,
                                          const Section &text, uint64_t start,
                                          uint64_t size, const char *which)
{
    std::vector<Sample> inwin;
    for (const Sample &s : collect_samples(dis, elf, text)) {
        if (s.offset < start) continue;
        if (s.offset >= start + size) {
            fprintf(stderr,
                    "ldc_patcher: %s consumer at 0x%llx lies above its "
                    "parameter window [0x%llx, 0x%llx)\n", which,
                    (unsigned long long)s.offset, (unsigned long long)start,
                    (unsigned long long)(start + size));
            exit(1);
        }
        if (s.form != "LDC" && s.form != "LDCU") {
            fprintf(stderr,
                    "ldc_patcher: unsupported %s consumer form '%s' at "
                    "c[0x0][0x%llx]\n", which, s.form.c_str(),
                    (unsigned long long)s.offset);
            exit(1);
        }
        inwin.push_back(s);
    }
    return inwin;
}

struct Range {
    uint64_t begin;
    uint64_t end;
    bool operator<(const Range &o) const { return begin < o.begin; }
};

static bool ranges_equal(const std::vector<Range> &a,
                         const std::vector<Range> &b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].begin != b[i].begin || a[i].end != b[i].end) return false;
    return true;
}

// Width-aware consumed-byte union of the samples (each sample consumes
// s.width bytes at s.offset). Offsets stay absolute c[0x0] positions, so
// the union is anchored at 0 rather than at the window start.
static std::vector<Range> covered_union(const std::vector<Sample> &win,
                                        uint64_t start)
{
    std::vector<Range> rs;
    for (const Sample &s : win)
        rs.push_back(Range{s.offset - start, s.offset - start + s.width});
    std::sort(rs.begin(), rs.end());
    std::vector<Range> out;
    for (const Range &r : rs) {
        if (!out.empty() && out.back().end >= r.begin)
            out.back().end = std::max(out.back().end, r.end);
        else
            out.push_back(r);
    }
    return out;
}

static void dump_ranges(const char *which, const std::vector<Range> &rs)
{
    fprintf(stderr, "ldc_patcher: %s consumed union:", which);
    for (const Range &r : rs)
        fprintf(stderr, " [0x%llx,0x%llx)", (unsigned long long)r.begin,
                (unsigned long long)r.end);
    fprintf(stderr, "\n");
}

// The extracted blob: every 16-byte instruction word pair of the text in
// [0, end), exactly as the HAL consumer memcpys it into instruction
// memory.
static std::vector<Instr> blob_of(const Elf &elf, const Section &text,
                                  uint64_t end)
{
    std::vector<Instr> blob;
    for (uint64_t pc = 0; pc < end; pc += 16) {
        Instr in;
        if (!instr_at(elf, text, pc, &in)) {
            fprintf(stderr, "ldc_patcher: text ends before prefix at 0x%llx\n",
                    (unsigned long long)pc);
            exit(1);
        }
        blob.push_back(in);
    }
    return blob;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: %s CHECK.cubin RESTORE.cubin NVDISASM OUT.h\n",
                argv[0]);
        return 2;
    }
    const char *check_path = argv[1];
    const char *restore_path = argv[2];
    const char *nvdisasm_bin = argv[3];
    const char *out_header = argv[4];

    // Upstream debugger-parameter ABI: InstrumentContext::Launch fills
    // args_buf (28 bytes) which cuXtraSetDebuggerParams places at
    // c[0x0][0x1880]: +0 preempt buffer, +8 entry point, +16 kernel index,
    // +24 killable flag.
    const uint64_t dbg = 0x1880ULL;

    Elf check(check_path), restore(restore_path);
    const Section *check_text = check.find(".text.check_preempt_port");
    const Section *restore_text = restore.find(".text.restore_exec_port");
    if (!check_text || !restore_text) {
        fprintf(stderr, "ldc_patcher: stub cubins miss their .text sections\n");
        return 1;
    }
    if (check_text->size % 16 != 0 || restore_text->size % 16 != 0) {
        fprintf(stderr,
                "ldc_patcher: stub .text sizes are not 16-byte streams\n");
        return 1;
    }
    if (!rela_clear(check, "check_preempt_port") ||
        !rela_clear(restore, "restore_exec_port")) {
        fprintf(stderr,
                "ldc_patcher: main .rela.text is not empty; the HAL copies "
                "these bytes raw into instruction memory and would never "
                "resolve a load-time relocation\n");
        return 1;
    }

    NvInfo cn, rn;
    if (!parse_nvinfo(check, "check_preempt_port", &cn) ||
        !parse_nvinfo(restore, "restore_exec_port", &rn)) {
        fprintf(stderr, "ldc_patcher: cannot parse the stub nvinfo records\n");
        return 1;
    }

    // Padded ABI: the 0x1500 pad parameter sits at the parameter-region
    // start and the consumed parameters land in the debugger window above
    // it. The exact extents pin the compiled specimen; a rebuild that
    // changes the layout fails here instead of emitting a wrong blob.
    if (cn.param_start != 0x380 || cn.param_size != 0x1518 ||
        rn.param_start != 0x380 || rn.param_size != 0x1510) {
        fprintf(stderr,
                "ldc_patcher: padded parameter window changed: check "
                "0x%llx/0x%llx restore 0x%llx/0x%llx (want 0x380/0x1518, "
                "0x380/0x1510)\n",
                (unsigned long long)cn.param_start,
                (unsigned long long)cn.param_size,
                (unsigned long long)rn.param_start,
                (unsigned long long)rn.param_size);
        return 1;
    }
    const NvInfo::Param *cp0 = kparam_find(cn, 0);
    const NvInfo::Param *rp0 = kparam_find(rn, 0);
    if (!cp0 || !rp0 || cp0->off != 0 || cp0->size != 0x1500 ||
        rp0->off != 0 || rp0->size != 0x1500) {
        fprintf(stderr, "ldc_patcher: 0x1500 pad parameter is missing\n");
        return 1;
    }
    struct Want { uint64_t ord, slot; };
    // check_preempt_port keeps the reserved entry slot (dbg+8) unread;
    // restore_exec_port consumes the entry slot at dbg+8.
    const Want check_want[] = {{1, dbg}, {2, dbg + 8}, {3, dbg + 16}};
    const Want restore_want[] = {{1, dbg}, {2, dbg + 8}};
    struct WantChecker {
        static bool one(const NvInfo &ni, const char *which,
                        const Want *want, unsigned n)
        {
            for (unsigned i = 0; i < n; ++i) {
                const NvInfo::Param *p = kparam_find(ni, want[i].ord);
                if (!p || ni.param_start + p->off != want[i].slot ||
                    p->size != 8) {
                    fprintf(stderr,
                            "ldc_patcher: %s parameter ordinal %llu does not "
                            "land at debugger slot 0x%llx\n", which,
                            (unsigned long long)want[i].ord,
                            (unsigned long long)want[i].slot);
                    return false;
                }
            }
            if (ni.kparam_n != n + 1) {
                fprintf(stderr,
                        "ldc_patcher: %s has %u KPARAM records, want pad + %u "
                        "window parameters\n", which, ni.kparam_n, n);
                return false;
            }
            return true;
        }
    };
    if (!WantChecker::one(cn, "check_preempt_port", check_want, 3) ||
        !WantChecker::one(rn, "restore_exec_port", restore_want, 2))
        return 1;

    // Width-aware consumed unions of the debugger-window loads must match
    // the compiled prototypes exactly.
    std::string check_dis = nvdisasm_text(nvdisasm_bin, check_path);
    std::string restore_dis = nvdisasm_text(nvdisasm_bin, restore_path);
    std::vector<Sample> check_in = window_samples(
        check_dis, check, *check_text, cn.param_start, cn.param_size,
        "check_preempt_port");
    std::vector<Sample> restore_in = window_samples(
        restore_dis, restore, *restore_text, rn.param_start, rn.param_size,
        "restore_exec_port");
    const std::vector<Range> check_policy = {
        Range{dbg, dbg + 8}, Range{dbg + 16, dbg + 24}};
    const std::vector<Range> restore_policy = {Range{dbg, dbg + 16}};
    std::vector<Range> check_covered = covered_union(check_in, 0);
    std::vector<Range> restore_covered = covered_union(restore_in, 0);
    if (!ranges_equal(check_covered, check_policy)) {
        fprintf(stderr, "ldc_patcher: check_preempt_port consumer set has "
                        "changed against the pinned prototype\n");
        dump_ranges("check_preempt_port", check_covered);
        return 1;
    }
    if (!ranges_equal(restore_covered, restore_policy)) {
        fprintf(stderr, "ldc_patcher: restore_exec_port consumer set has "
                        "changed against the pinned prototype\n");
        dump_ranges("restore_exec_port", restore_covered);
        return 1;
    }

    // ---- guardian prefix: ends at the retained conditional exit ----
    // EIATTR_EXIT_INSTR_OFFSETS lists exactly the cooperative @P0 EXIT
    // (exits[0]) and the final standalone-kernel EXIT (exits[1]); the
    // build-only marker store sits between them and is excluded by cutting
    // at exits[0]. If the retention marker were removed from
    // check_preempt_port.cu, nvcc would merge both paths into one
    // unconditional EXIT and this fails loudly because the retained
    // predicated exit disappears.
    if (cn.exit_n != 2 || cn.exits[0] > cn.exits[1]) {
        fprintf(stderr, "ldc_patcher: check_preempt_port has %u exit offsets "
                        "(want exactly 2, predicated one first)\n", cn.exit_n);
        return 1;
    }
    Instr c_exit, c_final;
    if (!instr_at(check, *check_text, cn.exits[0], &c_exit) ||
        !instr_at(check, *check_text, cn.exits[1], &c_final)) {
        fprintf(stderr,
                "ldc_patcher: check exit offsets 0x%llx/0x%llx exceed the "
                "text\n", (unsigned long long)cn.exits[0],
                (unsigned long long)cn.exits[1]);
        return 1;
    }
    if (cn.exits[0] == 0 || (c_exit.enc & 0xffff) != 0x094d ||
        (c_final.enc & 0xffff) != 0x794d) {
        fprintf(stderr,
                "ldc_patcher: check does not end in a retained predicated "
                "EXIT (@P0 EXIT 0x...094d at 0x%llx, final 0x...794d at "
                "0x%llx); observed 0x%llx/0x%llx\n",
                (unsigned long long)cn.exits[0],
                (unsigned long long)cn.exits[1],
                (unsigned long long)(c_exit.enc & 0xffff),
                (unsigned long long)(c_final.enc & 0xffff));
        return 1;
    }
    const uint64_t check_end = cn.exits[0] + 16;
    const std::vector<Instr> check_blob =
        blob_of(check, *check_text, check_end);

    // ---- resume prefix: everything before the final EXIT ----
    // restore_exec_port: @!P0 EXIT at exits[0], barrier, entry load from
    // c[0x0][0x1888] feeding R2, the RPC return-site preload (R20 = call
    // end offset 0x150, R21 = 0) and the compiler-emitted register-target
    // call CALL.REL.NOINC R2 0xfffffffc, then the final EXIT at exits[1].
    // The three-word tail (entry load, RPC preload, call) is verified
    // directly on the actual instruction words below and then REWRITTEN
    // in the emitted blob: the full-VA-in-register REL-call transfer has
    // never been run through on sm_120, while both runtime-proven
    // upstream resume stubs (sm_70 and sm_86) transfer with an identical
    // instruction tail instead: load the R20/R21 return pair from the
    // debugger window entry slot (c[0x0][0x1888]/[0x188c]) and
    // RET.ABS.NODEC R20 into the instrumented image, which ends with the
    // original kernel EXIT, so nothing ever returns into the copied
    // prefix.
    if (rn.exit_n != 2 || rn.exits[0] > rn.exits[1]) {
        fprintf(stderr, "ldc_patcher: restore_exec_port has %u exit offsets "
                        "(want exactly 2)\n", rn.exit_n);
        return 1;
    }
    Instr r_exit, r_final;
    if (!instr_at(restore, *restore_text, rn.exits[0], &r_exit) ||
        !instr_at(restore, *restore_text, rn.exits[1], &r_final)) {
        fprintf(stderr,
                "ldc_patcher: restore exit offsets 0x%llx/0x%llx exceed the "
                "text\n", (unsigned long long)rn.exits[0],
                (unsigned long long)rn.exits[1]);
        return 1;
    }
    if (rn.exits[0] == 0 || (r_exit.enc & 0xffff) != 0x894d ||
        (r_final.enc & 0xffff) != 0x794d) {
        fprintf(stderr,
                "ldc_patcher: restore @!P0 EXIT (0x...894d) or final EXIT "
                "(0x...794d) form changed; observed 0x%llx/0x%llx\n",
                (unsigned long long)(r_exit.enc & 0xffff),
                (unsigned long long)(r_final.enc & 0xffff));
        return 1;
    }
    const uint64_t restore_end = rn.exits[1];
    if (restore_end < 48) {
        fprintf(stderr, "ldc_patcher: restore prefix 0x%llu too short for "
                        "RPC preload + call\n",
                (unsigned long long)restore_end);
        return 1;
    }
    const uint64_t call_pc = restore_end - 16;
    Instr r_call, r_r20, r_r21;
    if (!instr_at(restore, *restore_text, call_pc, &r_call) ||
        !instr_at(restore, *restore_text, call_pc - 0x10, &r_r20) ||
        !instr_at(restore, *restore_text, call_pc - 0x20, &r_r21)) {
        fprintf(stderr, "ldc_patcher: restore text truncated before the call "
                        "at 0x%llx\n", (unsigned long long)call_pc);
        return 1;
    }
    if ((r_call.enc & 0xffff) != 0x7344 ||
        (r_call.enc >> 32) != 0xfffffffcULL) {
        fprintf(stderr,
                "ldc_patcher: expected CALL.REL.NOINC R2 0xfffffffc at "
                "0x%llx, observed encoding 0x%llx\n",
                (unsigned long long)call_pc, (unsigned long long)r_call.enc);
        return 1;
    }
    if ((r_r20.enc & 0xffff) != 0x7802 ||
        (r_r20.enc >> 32) != restore_end ||
        ((r_r20.enc >> 16) & 0xff) != 0x14) {
        fprintf(stderr,
                "ldc_patcher: expected MOV R20, 0x%llx (call end, return "
                "site) before the restore call; observed encoding 0x%llx\n",
                (unsigned long long)restore_end,
                (unsigned long long)r_r20.enc);
        return 1;
    }
    if ((r_r21.enc & 0xffff) != 0x7431 || (r_r21.enc >> 32) != 0) {
        fprintf(stderr,
                "ldc_patcher: expected R21 return-site high half "
                "(0x...7431 with imm 0) before the restore call; observed "
                "encoding 0x%llx\n", (unsigned long long)r_r21.enc);
        return 1;
    }
    bool entry_load = false;
    for (const Sample &s : restore_in)
        if (s.offset == dbg + 8 && s.pc + 16 <= restore_end) entry_load = true;
    if (!entry_load) {
        fprintf(stderr,
                "ldc_patcher: restore does not load the instrumented entry "
                "point c[0x0][0x%llx] inside the final prefix\n",
                (unsigned long long)(dbg + 8));
        return 1;
    }
    std::vector<Instr> restore_blob =
        blob_of(restore, *restore_text, restore_end);
    // Tail replacement over the about-to-be-emitted blob (the cubin keeps
    // the compiled call form so the verification above stays meaningful).
    // The last three slots go from HFMA2 R21=0 / MOV R20 0x150 /
    // CALL.REL.NOINC R2 to the sm_70/sm_86-proven transfer, copied
    // verbatim from both upstream arrays. RET.ABS.NODEC consumes the
    // R20/R21 pair as the full absolute transfer target without touching
    // the call-depth counter, matching a launch-level entry that never
    // nests; the target is the entry-point slot value already written by
    // the launch side into the same window word the dead LDC.64 R2 above
    // reads.
    restore_blob[restore_blob.size() - 3] =
        Instr{0x00062300ff157b82ULL, 0x000fc00000000800ULL}; // LDC R21, c[0x0][0x188c]
    restore_blob[restore_blob.size() - 2] =
        Instr{0x00062200ff147b82ULL, 0x000fc00000000800ULL}; // LDC R20, c[0x0][0x1888]
    restore_blob[restore_blob.size() - 1] =
        Instr{0x0000000014007950ULL, 0x001fea0003e00000ULL}; // RET.ABS.NODEC R20 0x0

    // ---- emit the HAL-consumed arrays header ----
    FILE *h = fopen(out_header, "w");
    if (!h) {
        fprintf(stderr, "ldc_patcher: cannot open %s: %s\n", out_header,
                strerror(errno));
        return 1;
    }
    fprintf(h,
            "// SPDX-License-Identifier: GPL-2.0\n"
            "//\n"
            "// Generated by native/ldc_patcher.cpp from the pinned sm_120 "
            "stub cubins.\n"
            "// Do not edit by hand.\n"
            "//\n"
            "// Each array carries the raw 16-byte SASS instructions of the "
            "extracted\n"
            "// stream: two unsigned long long per instruction (the "
            "instruction\n"
            "// encoding followed by its scheduling/control word), "
            "byte-identical to\n"
            "// the cubin .text. Both arrays are consumed by memcpy into "
            "instruction\n"
            "// memory:\n"
            "//  - xg_sm120_check_preempt is the leading prefix of the "
            "instrumented\n"
            "//    kernel (GuardianSM120::GetGuardianInstructions) and falls "
            "through\n"
            "//    into the original kernel text appended behind it;\n"
            "//  - xg_sm120_restore_exec is the resume entry "
            "(GetResumeInstructions)\n"
            "//    and transfers with RET.ABS.NODEC R20 - the return pair "
            "R20/R21\n"
            "//    loaded from the debugger window entry slot "
            "c[0x0][0x1888]/\n"
            "//    [0x188c], the same mechanism as the runtime-proven "
            "sm_70 and\n"
            "//    sm_86 resume streams - into the instrumented entry "
            "point taken\n"
            "//    from that slot. The compiled register-target call tail "
            "was\n"
            "//    verified on the actual words and then replaced; the "
            "sm_120 call\n"
            "//    form is not runtime-proven.\n"
            "#ifndef XG_SM120_GUARDIAN_ARRAYS_H\n"
            "#define XG_SM120_GUARDIAN_ARRAYS_H\n\n");
    fprintf(h,
            "#define XG_SM120_CHECK_PREEMPT_BYTES 0x%llxu\n"
            "#define XG_SM120_RESTORE_EXEC_BYTES 0x%llxu\n\n",
            (unsigned long long)check_end,
            (unsigned long long)restore_end);
    fprintf(h, "static const unsigned long long xg_sm120_check_preempt[] = "
               "{\n");
    for (const Instr &in : check_blob)
        fprintf(h, "    0x%016llxULL, 0x%016llxULL,\n",
                (unsigned long long)in.enc, (unsigned long long)in.ctl);
    fprintf(h, "};\n\n");
    fprintf(h, "static const unsigned long long xg_sm120_restore_exec[] = "
               "{\n");
    for (const Instr &in : restore_blob)
        fprintf(h, "    0x%016llxULL, 0x%016llxULL,\n",
                (unsigned long long)in.enc, (unsigned long long)in.ctl);
    fprintf(h, "};\n\n");
    fprintf(h,
            "static_assert(sizeof(xg_sm120_check_preempt) == "
            "XG_SM120_CHECK_PREEMPT_BYTES, \"check stream size\");\n"
            "static_assert(sizeof(xg_sm120_restore_exec) == "
            "XG_SM120_RESTORE_EXEC_BYTES, \"restore stream size\");\n\n"
            "#endif\n");
    int werr = ferror(h);
    if (fclose(h) != 0 || werr) {
        fprintf(stderr, "ldc_patcher: writing %s failed\n", out_header);
        return 1;
    }

    fprintf(stderr,
            "ldc_patcher: emitted %s: guardian prefix %llu B ending at the "
            "retained @P0 EXIT (0x%llx; marker store and final EXIT 0x%llx "
            "excluded), resume prefix %llu B ending at the rewritten "
            "RET.ABS.NODEC R20 transfer (0x%llx; final EXIT excluded; the "
            "compiled CALL.REL.NOINC R2 tail was verified then replaced)\n",
            out_header, (unsigned long long)check_end,
            (unsigned long long)cn.exits[0],
            (unsigned long long)cn.exits[1],
            (unsigned long long)restore_end,
            (unsigned long long)call_pc);
    return 0;
}

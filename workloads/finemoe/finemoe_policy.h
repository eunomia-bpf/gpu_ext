/* FineMoE Eq. 6–8 selector. CPU/host-eBPF ABI; no CUDA pointers. */
#ifndef FINEMOE_POLICY_H
#define FINEMOE_POLICY_H
typedef unsigned int fm_u32;
typedef unsigned long long fm_u64;
#define FM_MAX_EXPERTS 60
struct fm_input {
    fm_u32 count, top_k, threshold_bits, reserved;
    fm_u32 probability_bits[FM_MAX_EXPERTS];
};
struct fm_output {
    fm_u64 mask, cumulative_bits;
    fm_u32 selected, status;
};
struct fm_context { struct fm_input input; struct fm_output output; };

/* Nonnegative IEEE binary32 -> binary64, exactly, including subnormals. */
static __attribute__((always_inline)) inline fm_u64 fm_promote(fm_u32 bits)
{
    fm_u32 exponent = bits >> 23, fraction = bits & 0x7fffffU;
    if (!exponent) {
        if (!fraction) return 0;
        fm_u32 shifted = fraction, distance = 0;
        for (int i = 0; i < 23 && shifted < 0x800000U; ++i) {
            shifted <<= 1;
            ++distance;
        }
        return ((fm_u64)(897U - distance) << 52) |
               ((fm_u64)(shifted & 0x7fffffU) << 29);
    }
    return ((fm_u64)(exponent + 896U) << 52) | ((fm_u64)fraction << 29);
}

/* Positive, normal binary64 addition, round-to-nearest/ties-to-even.
 * Promoted binary32 inputs and sums <= 60 cannot under/overflow binary64.
 * BPF has no FP instructions; guard/round/sticky bits preserve this arithmetic.
 */
static __attribute__((always_inline)) inline fm_u64 fm_add(fm_u64 a, fm_u64 b)
{
#ifdef FM_BPF
    if (!a) return b;
    if (!b) return a;
    if (a < b) { fm_u64 tmp = a; a = b; b = tmp; }
    fm_u32 exponent = (fm_u32)(a >> 52);
    fm_u32 shift = exponent - (fm_u32)(b >> 52);
    fm_u64 x = ((a & 0xfffffffffffffULL) | 0x10000000000000ULL) << 3;
    fm_u64 y = ((b & 0xfffffffffffffULL) | 0x10000000000000ULL) << 3;
    if (shift >= 64) y = 1;
    else if (shift) y = (y >> shift) | ((y << (64 - shift)) != 0);
    fm_u64 sum = x + y;
    if (sum & (1ULL << 56)) {
        sum = (sum >> 1) | (sum & 1);
        ++exponent;
    }
    fm_u64 rounded = (sum + 3 + ((sum >> 3) & 1)) >> 3;
    if (rounded == (1ULL << 53)) { rounded >>= 1; ++exponent; }
    return ((fm_u64)exponent << 52) | (rounded & 0xfffffffffffffULL);
#else
    union { fm_u64 bits; double value; } x = {a}, y = {b}, sum;
    sum.value = x.value + y.value;
    return sum.bits;
#endif
}

static __attribute__((always_inline)) inline fm_u32 fm_select(struct fm_context *ctx)
{
    struct fm_input *in = &ctx->input;
    struct fm_output *out = &ctx->output;
    fm_u32 ids[FM_MAX_EXPERTS];
    out->mask = out->cumulative_bits = 0;
    out->selected = 0;
    out->status = 1;
    if (!in->count || in->count > FM_MAX_EXPERTS || !in->top_k ||
        in->top_k > in->count || in->reserved || in->threshold_bits > 0x3f800000U)
        return 1;
    fm_u32 positive = 0;
    for (fm_u32 i = 0; i < FM_MAX_EXPERTS; ++i) {
        if (i >= in->count) break;
        fm_u32 p = in->probability_bits[i];
        if (p > 0x3f800000U) return 1; /* Includes negative, NaN, Inf. */
        positive |= p;
        ids[i] = i;
    }
    out->status = 0;
    if (!positive) return 0; /* Inactive layer / empty historical map. */
    for (fm_u32 i = 1; i < FM_MAX_EXPERTS; ++i) {
        if (i >= in->count) break;
        fm_u32 key = ids[i], j = i;
        for (fm_u32 k = 0; k < FM_MAX_EXPERTS && j; ++k) {
            if (in->probability_bits[ids[j - 1]] >= in->probability_bits[key]) break;
            ids[j] = ids[j - 1];
            --j;
        }
        ids[j] = key;
    }
    fm_u64 target = fm_promote(in->threshold_bits), cumulative = 0;
    for (fm_u32 i = 0; i < FM_MAX_EXPERTS; ++i) {
        if (i >= in->count) break;
        fm_u32 id = ids[i];
        cumulative = fm_add(cumulative, fm_promote(in->probability_bits[id]));
        out->mask |= 1ULL << id;
        out->selected = i + 1;
        if (out->selected >= in->top_k && cumulative >= target) break;
    }
    out->cumulative_bits = cumulative;
    return 0;
}
#endif

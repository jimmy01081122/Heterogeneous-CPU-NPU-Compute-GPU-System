/*
 * prof.h - lightweight per-operator profiler for the llama2.c runtimes.
 *
 * Enabled with -DPROFILE. When disabled every macro compiles to nothing, so
 * the timed and untimed builds share the exact same arithmetic.
 *
 *   PROF_BEGIN(OP_X);  ...work...  PROF_END(OP_X);
 *   prof_report(stderr, n_tokens);           // human readable table
 *   prof_dump_csv("prof.csv", n_tokens);     // machine readable (tools/)
 */
#ifndef HETERO_PROF_H
#define HETERO_PROF_H

#include <stdio.h>
#include <stdint.h>
#include <time.h>

/* Operator taxonomy, chosen to match the CPU / NPU / GPU mapping study. */
typedef enum {
    OP_EMBED = 0,     /* token embedding lookup                 (CPU)     */
    OP_RMSNORM,       /* attention / ffn / final rmsnorm        (GPU cand) */
    OP_QUANTIZE,      /* activation fp32 -> int8 group quant    (GPU cand) */
    OP_MM_QKV,        /* Wq, Wk, Wv projections                 (NPU cand) */
    OP_ROPE,          /* rotary positional encoding             (GPU cand) */
    OP_KVCACHE,       /* write k,v into cache                   (DMA)      */
    OP_ATTN,          /* QK^T, softmax, *V over the KV cache    (GPU cand) */
    OP_MM_O,          /* Wo projection                          (NPU cand) */
    OP_RESIDUAL,      /* residual adds                          (GPU cand) */
    OP_MM_FFN13,      /* W1, W3 projections                     (NPU cand) */
    OP_SWIGLU,        /* silu(w1 x) * w3 x                      (GPU cand) */
    OP_MM_FFN2,       /* W2 projection                          (NPU cand) */
    OP_MM_CLS,        /* classifier, vocab x dim                (NPU cand) */
    OP_SAMPLE,        /* argmax / softmax / top-p sampling      (CPU/GPU)  */
    OP_COUNT
} ProfOp;

static const char *const PROF_OP_NAME[OP_COUNT] = {
    "embed", "rmsnorm", "quantize", "mm_qkv", "rope", "kvcache", "attention",
    "mm_o", "residual", "mm_ffn13", "swiglu", "mm_ffn2", "mm_cls", "sample",
};

#ifdef PROFILE

typedef struct {
    uint64_t ns[OP_COUNT];
    uint64_t calls[OP_COUNT];
    uint64_t t0[OP_COUNT];
} ProfState;

static ProfState g_prof;

static inline uint64_t prof_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

#define PROF_BEGIN(op) do { g_prof.t0[op] = prof_now_ns(); } while (0)
#define PROF_END(op)   do { g_prof.ns[op] += prof_now_ns() - g_prof.t0[op]; \
                            g_prof.calls[op]++; } while (0)

static void prof_reset(void) {
    for (int i = 0; i < OP_COUNT; i++) { g_prof.ns[i] = 0; g_prof.calls[i] = 0; }
}

static void prof_report(FILE *f, int n_tokens) {
    uint64_t total = 0;
    for (int i = 0; i < OP_COUNT; i++) total += g_prof.ns[i];
    if (n_tokens <= 0) n_tokens = 1;
    fprintf(f, "\n%-10s %10s %12s %12s %7s\n", "op", "calls", "total_ms", "us/token", "share");
    for (int i = 0; i < OP_COUNT; i++) {
        if (!g_prof.calls[i]) continue;
        fprintf(f, "%-10s %10llu %12.3f %12.2f %6.2f%%\n", PROF_OP_NAME[i],
                (unsigned long long)g_prof.calls[i], g_prof.ns[i] / 1e6,
                g_prof.ns[i] / 1e3 / n_tokens, total ? 100.0 * g_prof.ns[i] / total : 0.0);
    }
    fprintf(f, "%-10s %10s %12.3f %12.2f\n", "TOTAL", "", total / 1e6, total / 1e3 / n_tokens);
}

static void prof_dump_csv(const char *path, int n_tokens) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "prof: cannot open %s\n", path); return; }
    fprintf(f, "op,calls,total_ns,tokens\n");
    for (int i = 0; i < OP_COUNT; i++)
        fprintf(f, "%s,%llu,%llu,%d\n", PROF_OP_NAME[i], (unsigned long long)g_prof.calls[i],
                (unsigned long long)g_prof.ns[i], n_tokens);
    fclose(f);
}

#else /* !PROFILE */

#define PROF_BEGIN(op) do { } while (0)
#define PROF_END(op)   do { } while (0)
static inline void prof_reset(void) {}
static inline void prof_report(FILE *f, int n) { (void)f; (void)n; }
static inline void prof_dump_csv(const char *p, int n) { (void)p; (void)n; }

#endif /* PROFILE */
#endif /* HETERO_PROF_H */

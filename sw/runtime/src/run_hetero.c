/* SPDX-License-Identifier: MIT
 * run_hetero.c - Unified Heterogeneous CPU-NPU-Compute GPU Inference Runtime
 *                for Tiny Llama-2 (stories260K / stories15M, Q8_0 quantized).
 *
 * Supports the 4 Q8_0 experiment configurations from method.md Section 6.3:
 *   -E B : Experiment B - CPU-only INT8 baseline
 *   -E C : Experiment C - CPU + NPU (INT8 GEMV offload + activation reuse + 8-row tiling)
 *   -E D : Experiment D - CPU + Compute GPU (RMSNorm, RoPE, Softmax, SwiGLU, Residual offload)
 *   -E E : Experiment E - Full Heterogeneous CPU + NPU + Compute GPU (all offloads active;
 *                         optional -F 1 offloads Q8_0 group dequant epilogue to Compute GPU)
 *
 * Backend selection (-b <rtl|sim|hw> or $HETERO_BACKEND):
 *   rtl : In-process Verilated SystemVerilog RTL (npu_top.sv + gpu_top.sv)
 *   sim : Cycle-accurate C behavioral / GPU 32-bit ISA simulator
 *   hw  : PYNQ-Z2 /dev/mem + Xilinx AXI DMA + CMA physical buffers
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include "prof.h"
#include "../../driver/accel.h"
#include "../../gpu_kernels/gpu_kernels_bin.h"

#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// ----------------------------------------------------------------------------
// Globals & Heterogeneous Configuration
int GS = 0; // group size global for quantization of the weights

static char        g_exp          = 'E';   // 'B', 'C', 'D', 'E'
static const char *g_backend      = "rtl"; // "rtl", "sim", "hw"
static int         g_tile_rows    = 8;     // 1 (row-major) or 8 (8x8 MAC array tiled)
static int         g_use_npu      = 1;
static int         g_use_gpu      = 1;
static int         g_gpu_epilogue = 0;     // 1 = run Q8_0 dequant epilogue on Compute GPU

#define MAX_NPU_CHUNK_ROWS 1024

static inline uint32_t f2u_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    int8_t* q;       // quantized values (row-major in checkpoint)
    float*  s;       // scaling factors
    int8_t* q_tiled; // NPU DMA buffer (tiled by g_tile_rows per MAX_NPU_CHUNK_ROWS block)
} QuantizedTensor;

typedef struct {
    QuantizedTensor *q_tokens;
    float* token_embedding_table;

    float* rms_att_weight;
    float* rms_ffn_weight;

    QuantizedTensor *wq;
    QuantizedTensor *wk;
    QuantizedTensor *wv;
    QuantizedTensor *wo;

    QuantizedTensor *w1;
    QuantizedTensor *w2;
    QuantizedTensor *w3;

    float* rms_final_weight;
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    QuantizedTensor xq;
    QuantizedTensor hq;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
    int32_t *psum_buf; // [MAX_NPU_CHUNK_ROWS * max_groups]
    float   *rope_fcr; // [dim / 2]
    float   *rope_fci; // [dim / 2]
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int max_n  = (p->hidden_dim > p->dim) ? p->hidden_dim : p->dim;
    int max_ng = (GS > 0) ? (max_n / GS) : max_n;

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim, sizeof(float)), .q_tiled = NULL };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim, sizeof(float)), .q_tiled = NULL };
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->psum_buf = calloc((size_t)MAX_NPU_CHUNK_ROWS * (size_t)max_ng, sizeof(int32_t));
    s->rope_fcr = calloc(p->dim / 2, sizeof(float));
    s->rope_fci = calloc(p->dim / 2, sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache || !s->psum_buf || !s->rope_fcr || !s->rope_fci) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
    free(s->psum_buf);
    free(s->rope_fcr);
    free(s->rope_fci);
}

// ----------------------------------------------------------------------------
// Quantization & NPU Weight Preparation

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0f;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;
        for (int i = 0; i < GS; i++) {
            float quant_value = (scale != 0.0f) ? (x[group * GS + i] / scale) : 0.0f;
            int8_t quantized = (int8_t) round(quant_value);
            qx->q[group * GS + i] = quantized;
        }
    }
}

static void pack_tensor_for_npu(QuantizedTensor *qt, int layers, int d, int n) {
    if (!g_use_npu) return;
    for (int l = 0; l < layers; l++) {
        size_t bytes = npu_packed_size(d, n);
        qt[l].q_tiled = (int8_t *)accel_alloc(bytes);
        if (!qt[l].q_tiled) {
            fprintf(stderr, "accel_alloc(%zu) failed\n", bytes);
            exit(EXIT_FAILURE);
        }
        /* Pack in chunks of MAX_NPU_CHUNK_ROWS so chunked NPU dispatch aligns */
        for (int r0 = 0; r0 < d; r0 += MAX_NPU_CHUNK_ROWS) {
            int r_chunk = (d - r0 < MAX_NPU_CHUNK_ROWS) ? (d - r0) : MAX_NPU_CHUNK_ROWS;
            const int8_t *src = qt[l].q + (size_t)r0 * (size_t)n;
            int8_t       *dst = qt[l].q_tiled + (size_t)r0 * (size_t)n;
            npu_pack_tiled(src, r_chunk, n, GS, g_tile_rows, dst);
        }
    }
}

static void free_packed_tensor(QuantizedTensor *qt, int layers) {
    if (!qt) return;
    for (int l = 0; l < layers; l++) {
        if (qt[l].q_tiled) {
            accel_free(qt[l].q_tiled);
            qt[l].q_tiled = NULL;
        }
    }
}

QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = calloc(n, sizeof(QuantizedTensor));
    for (int i = 0; i < n; i++) {
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
        res[i].q_tiled = NULL;
    }
    *ptr = p;
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    int kv_dim    = (p->dim * p->n_kv_heads) / p->n_heads;

    float* fptr = (float*) ptr;
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    ptr = (void*)fptr;
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * kv_dim);
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * kv_dim);
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);

    if (g_use_npu) {
        pack_tensor_for_npu(w->wq,   p->n_layers, p->dim,        p->dim);
        pack_tensor_for_npu(w->wk,   p->n_layers, kv_dim,        p->dim);
        pack_tensor_for_npu(w->wv,   p->n_layers, kv_dim,        p->dim);
        pack_tensor_for_npu(w->wo,   p->n_layers, p->dim,        p->dim);
        pack_tensor_for_npu(w->w1,   p->n_layers, p->hidden_dim, p->dim);
        pack_tensor_for_npu(w->w2,   p->n_layers, p->dim,        p->hidden_dim);
        pack_tensor_for_npu(w->w3,   p->n_layers, p->hidden_dim, p->dim);
        pack_tensor_for_npu(w->wcls, 1,           p->vocab_size, p->dim);
    }
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); exit(EXIT_FAILURE); }
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 2) { fprintf(stderr, "Bad version %d, need version 2\n", version); exit(EXIT_FAILURE); }
    int header_size = 256;
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    uint8_t shared_classifier;
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    int group_size;
    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    GS = group_size;
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    void* weights_ptr = ((char*)*data) + header_size;
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    if (g_use_npu) {
        free_packed_tensor(t->weights.wq,   t->config.n_layers);
        free_packed_tensor(t->weights.wk,   t->config.n_layers);
        free_packed_tensor(t->weights.wv,   t->config.n_layers);
        free_packed_tensor(t->weights.wo,   t->config.n_layers);
        free_packed_tensor(t->weights.w1,   t->config.n_layers);
        free_packed_tensor(t->weights.w2,   t->config.n_layers);
        free_packed_tensor(t->weights.w3,   t->config.n_layers);
        free_packed_tensor(t->weights.wcls, 1);
    }
    if (t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// CPU / NPU / Compute GPU Operator Implementations

void rmsnorm_cpu(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    PROF_BEGIN(OP_RMSNORM);
    if (g_use_gpu) {
        gpu_write_spm(GPU_SPM_A, x, size);
        gpu_write_spm(GPU_SPM_B, weight, size);
        gpu_run(GPU_PC_RMSNORM, (uint32_t)size, f2u_bits(1.0f / (float)size), f2u_bits(1e-5f));
        gpu_read_spm(GPU_SPM_C, o, size);
    } else {
        rmsnorm_cpu(o, x, weight, size);
    }
    PROF_END(OP_RMSNORM);
}

void softmax_cpu(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void softmax_hetero(float* x, int size) {
    if (g_use_gpu && size <= 1024) {
        gpu_write_spm(GPU_SPM_A, x, size);
        gpu_run(GPU_PC_SOFTMAX, (uint32_t)size, 0, 0);
        gpu_read_spm(GPU_SPM_A, x, size);
    } else {
        softmax_cpu(x, size);
    }
}

void matmul_cpu(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        for (int j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }
        xout[i] = val;
    }
}

/* Executes W(d, n) @ x on the NPU (assuming x is already loaded into npu_act_buf)
 * and performs the Q8_0 group scale dequantization epilogue on either CPU or GPU. */
static void matmul_npu_reuse(float *xout, QuantizedTensor *x, QuantizedTensor *w,
                             int n, int d, int32_t *psum_buf) {
    int ng = n / GS;
    const int8_t *w_base = w->q_tiled ? w->q_tiled : w->q;

    for (int r0 = 0; r0 < d; r0 += MAX_NPU_CHUNK_ROWS) {
        int r_chunk = (d - r0 < MAX_NPU_CHUNK_ROWS) ? (d - r0) : MAX_NPU_CHUNK_ROWS;
        const int8_t *w_ptr = w_base + (size_t)r0 * (size_t)n;
        const float  *ws_base = w->s + (size_t)r0 * (size_t)ng;

        npu_gemv_psum(w_ptr, r_chunk, n, GS, g_tile_rows, psum_buf);

        if (g_use_gpu && g_gpu_epilogue) {
            /* Offload Q8_0 dequantization epilogue to Compute GPU */
            float   zero_buf[MAX_NPU_CHUNK_ROWS];
            int32_t psum_col[MAX_NPU_CHUNK_ROWS];
            float   ws_col[MAX_NPU_CHUNK_ROWS];
            memset(zero_buf, 0, (size_t)r_chunk * sizeof(float));
            gpu_write_spm(GPU_SPM_C, zero_buf, r_chunk);
            for (int g = 0; g < ng; g++) {
                for (int r = 0; r < r_chunk; r++) {
                    psum_col[r] = psum_buf[r * ng + g];
                    ws_col[r]   = ws_base[r * ng + g];
                }
                gpu_write_spm(GPU_SPM_A, psum_col, r_chunk);
                gpu_write_spm(GPU_SPM_B, ws_col,   r_chunk);
                gpu_run(GPU_PC_Q80_EPILOGUE, (uint32_t)r_chunk, f2u_bits(x->s[g]), 0);
            }
            gpu_read_spm(GPU_SPM_C, xout + r0, r_chunk);
        } else {
            for (int r = 0; r < r_chunk; r++) {
                float acc = 0.0f;
                const int32_t *ps = psum_buf + r * ng;
                const float   *ws = ws_base  + r * ng;
                for (int g = 0; g < ng; g++) {
                    acc += (float)ps[g] * ws[g] * x->s[g];
                }
                xout[r0 + r] = acc;
            }
        }
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // 1. Token embedding copy
    PROF_BEGIN(OP_EMBED);
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));
    PROF_END(OP_EMBED);

    // Precompute RoPE cos/sin tables once per position when GPU is used
    if (g_use_gpu) {
        for (int k = 0; k < dim / 2; k++) {
            int head_dim = (2 * k) % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val  = pos * freq;
            s->rope_fcr[k] = cosf(val);
            s->rope_fci[k] = sinf(val);
        }
    }

    // Forward all layers
    for (int l = 0; l < p->n_layers; l++) {

        // Attention RMSNorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // Quantize xb -> xq
        PROF_BEGIN(OP_QUANTIZE);
        quantize(&s->xq, s->xb, dim);
        PROF_END(OP_QUANTIZE);

        // QKV Projections (load xq ONCE into NPU activation buffer, reuse across Wq, Wk, Wv)
        PROF_BEGIN(OP_MM_QKV);
        if (g_use_npu) {
            npu_load_x(s->xq.q, dim);
            matmul_npu_reuse(s->q, &s->xq, w->wq + l, dim, dim,    s->psum_buf);
            matmul_npu_reuse(s->k, &s->xq, w->wk + l, dim, kv_dim, s->psum_buf);
            matmul_npu_reuse(s->v, &s->xq, w->wv + l, dim, kv_dim, s->psum_buf);
        } else {
            matmul_cpu(s->q, &s->xq, w->wq + l, dim, dim);
            matmul_cpu(s->k, &s->xq, w->wk + l, dim, kv_dim);
            matmul_cpu(s->v, &s->xq, w->wv + l, dim, kv_dim);
        }
        PROF_END(OP_MM_QKV);

        // RoPE relative positional encoding
        PROF_BEGIN(OP_ROPE);
        if (g_use_gpu) {
            gpu_write_spm(GPU_SPM_B, s->rope_fcr, dim / 2);
            gpu_write_spm(GPU_SPM_C, s->rope_fci, dim / 2);
            gpu_write_spm(GPU_SPM_A, s->q, dim);
            gpu_run(GPU_PC_ROPE, (uint32_t)(dim / 2), 0, 0);
            gpu_read_spm(GPU_SPM_A, s->q, dim);
            gpu_write_spm(GPU_SPM_A, s->k, kv_dim);
            gpu_run(GPU_PC_ROPE, (uint32_t)(kv_dim / 2), 0, 0);
            gpu_read_spm(GPU_SPM_A, s->k, kv_dim);
        } else {
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    float* vec = v == 0 ? s->q : s->k;
                    float v0 = vec[i];
                    float v1 = vec[i+1];
                    vec[i]   = v0 * fcr - v1 * fci;
                    vec[i+1] = v0 * fci + v1 * fcr;
                }
            }
        }
        PROF_END(OP_ROPE);

        // Save key,value at this time step (pos) to KV cache
        PROF_BEGIN(OP_KVCACHE);
        int loff = l * p->seq_len * kv_dim;
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));
        PROF_END(OP_KVCACHE);

        // Multihead attention (with Softmax offloaded to Compute GPU in D / E)
        PROF_BEGIN(OP_ATTN);
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax_hetero(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        PROF_END(OP_ATTN);

        // Output projection Wo
        PROF_BEGIN(OP_QUANTIZE);
        quantize(&s->xq, s->xb, dim);
        PROF_END(OP_QUANTIZE);
        PROF_BEGIN(OP_MM_O);
        if (g_use_npu) {
            npu_load_x(s->xq.q, dim);
            matmul_npu_reuse(s->xb2, &s->xq, w->wo + l, dim, dim, s->psum_buf);
        } else {
            matmul_cpu(s->xb2, &s->xq, w->wo + l, dim, dim);
        }
        PROF_END(OP_MM_O);

        // Residual connection x += xb2
        PROF_BEGIN(OP_RESIDUAL);
        if (g_use_gpu) {
            gpu_write_spm(GPU_SPM_A, x, dim);
            gpu_write_spm(GPU_SPM_B, s->xb2, dim);
            gpu_run(GPU_PC_RESIDUAL, (uint32_t)dim, 0, 0);
            gpu_read_spm(GPU_SPM_A, x, dim);
        } else {
            for (int i = 0; i < dim; i++) {
                x[i] += s->xb2[i];
            }
        }
        PROF_END(OP_RESIDUAL);

        // FFN RMSNorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // W1 and W3 FFN projections (load xq ONCE into NPU activation buffer, reuse across W1 & W3)
        PROF_BEGIN(OP_QUANTIZE);
        quantize(&s->xq, s->xb, dim);
        PROF_END(OP_QUANTIZE);
        PROF_BEGIN(OP_MM_FFN13);
        if (g_use_npu) {
            npu_load_x(s->xq.q, dim);
            matmul_npu_reuse(s->hb,  &s->xq, w->w1 + l, dim, hidden_dim, s->psum_buf);
            matmul_npu_reuse(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim, s->psum_buf);
        } else {
            matmul_cpu(s->hb,  &s->xq, w->w1 + l, dim, hidden_dim);
            matmul_cpu(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
        }
        PROF_END(OP_MM_FFN13);

        // SwiGLU non-linearity
        PROF_BEGIN(OP_SWIGLU);
        if (g_use_gpu) {
            gpu_write_spm(GPU_SPM_A, s->hb,  hidden_dim);
            gpu_write_spm(GPU_SPM_B, s->hb2, hidden_dim);
            gpu_run(GPU_PC_SWIGLU, (uint32_t)hidden_dim, 0, 0);
            gpu_read_spm(GPU_SPM_A, s->hb, hidden_dim);
        } else {
            for (int i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                val *= (1.0f / (1.0f + expf(-val)));
                val *= s->hb2[i];
                s->hb[i] = val;
            }
        }
        PROF_END(OP_SWIGLU);

        // W2 FFN projection
        PROF_BEGIN(OP_QUANTIZE);
        quantize(&s->hq, s->hb, hidden_dim);
        PROF_END(OP_QUANTIZE);
        PROF_BEGIN(OP_MM_FFN2);
        if (g_use_npu) {
            npu_load_x(s->hq.q, hidden_dim);
            matmul_npu_reuse(s->xb, &s->hq, w->w2 + l, hidden_dim, dim, s->psum_buf);
        } else {
            matmul_cpu(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
        }
        PROF_END(OP_MM_FFN2);

        // Residual connection x += xb
        PROF_BEGIN(OP_RESIDUAL);
        if (g_use_gpu) {
            gpu_write_spm(GPU_SPM_A, x, dim);
            gpu_write_spm(GPU_SPM_B, s->xb, dim);
            gpu_run(GPU_PC_RESIDUAL, (uint32_t)dim, 0, 0);
            gpu_read_spm(GPU_SPM_A, x, dim);
        } else {
            for (int i = 0; i < dim; i++) {
                x[i] += s->xb[i];
            }
        }
        PROF_END(OP_RESIDUAL);
    }

    // Final RMSNorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // Classifier into logits
    PROF_BEGIN(OP_QUANTIZE);
    quantize(&s->xq, x, dim);
    PROF_END(OP_QUANTIZE);
    PROF_BEGIN(OP_MM_CLS);
    if (g_use_npu) {
        npu_load_x(s->xq.q, dim);
        matmul_npu_reuse(s->logits, &s->xq, w->wcls, dim, p->vocab_size, s->psum_buf);
    } else {
        matmul_cpu(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    }
    PROF_END(OP_MM_CLS);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL || piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < (int)str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) {
            break;
        }
        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax_cpu(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// Generation loop & Hardware Co-Design Reporting

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

static void print_accel_report(int n_tokens) {
    if (!g_use_npu && !g_use_gpu) return;
    AccelStats *st = accel_stats();
    int tok = n_tokens > 0 ? n_tokens : 1;
    double npu_ms_100mhz = (double)st->cycles[DEV_NPU] / 100000.0;
    double gpu_ms_100mhz = (double)st->cycles[DEV_GPU] / 100000.0;

    fprintf(stderr, "\n=== Heterogeneous Accelerator Report (Exp %c | backend=%s | tile_rows=%d | %d tokens) ===\n",
            g_exp, g_backend, g_tile_rows, n_tokens);
    fprintf(stderr, "  %-8s  %12s  %12s  %10s  %12s  %12s  %12s\n",
            "Device", "HW Cycles", "Cycles/Tok", "Calls", "DMA In (KB)", "DMA Out (KB)", "Est@100MHz");
    if (g_use_npu) {
        fprintf(stderr, "  %-8s  %12llu  %12llu  %10llu  %12.2f  %12.2f  %9.3f ms\n",
                "NPU",
                (unsigned long long)st->cycles[DEV_NPU],
                (unsigned long long)(st->cycles[DEV_NPU] / (uint64_t)tok),
                (unsigned long long)st->calls[DEV_NPU],
                (double)st->bytes_in[DEV_NPU] / 1024.0,
                (double)st->bytes_out[DEV_NPU] / 1024.0,
                npu_ms_100mhz);
    }
    if (g_use_gpu) {
        fprintf(stderr, "  %-8s  %12llu  %12llu  %10llu  %12.2f  %12.2f  %9.3f ms\n",
                "GPU",
                (unsigned long long)st->cycles[DEV_GPU],
                (unsigned long long)(st->cycles[DEV_GPU] / (uint64_t)tok),
                (unsigned long long)st->calls[DEV_GPU],
                (double)st->bytes_in[DEV_GPU] / 1024.0,
                (double)st->bytes_out[DEV_GPU] / 1024.0,
                gpu_ms_100mhz);
    }
    fprintf(stderr, "=========================================================================================\n");
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            PROF_BEGIN(OP_SAMPLE);
            next = sample(sampler, logits);
            PROF_END(OP_SAMPLE);
        }
        pos++;

        if (next == 1) { break; }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    prof_report(stderr, pos);
    if (getenv("HETERO_PROF_CSV")) { prof_dump_csv(getenv("HETERO_PROF_CSV"), pos); }
    print_accel_report(pos);

    free(prompt_tokens);
}

void error_usage() {
    fprintf(stderr, "Usage:   run_hetero <checkpoint> [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -E <B|C|D|E> Experiment mode (B=CPU, C=CPU+NPU, D=CPU+GPU, E=CPU+NPU+GPU; default E)\n");
    fprintf(stderr, "  -b <backend> Backend: rtl | sim | hw (default rtl)\n");
    fprintf(stderr, "  -R <int>     NPU tile rows: 1 or 8 (default 8)\n");
    fprintf(stderr, "  -F <0|1>     Fused NPU->GPU Q8_0 dequant epilogue (default 0)\n");
    fprintf(stderr, "  -t <float>   temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>   p value in top-p sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>     random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>     number of steps to run for, default 256\n");
    fprintf(stderr, "  -i <string>  input prompt\n");
    fprintf(stderr, "  -z <string>  optional path to custom tokenizer\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;

    if (getenv("HETERO_EXP"))       g_exp          = (char)toupper((unsigned char)getenv("HETERO_EXP")[0]);
    if (getenv("HETERO_BACKEND"))   g_backend      = getenv("HETERO_BACKEND");
    if (getenv("HETERO_TILE_ROWS")) g_tile_rows    = atoi(getenv("HETERO_TILE_ROWS"));
    if (getenv("HETERO_FUSED_EPI")) g_gpu_epilogue = atoi(getenv("HETERO_FUSED_EPI"));

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-' || strlen(argv[i]) != 2) { error_usage(); }
        char flag = argv[i][1];
        if      (flag == 't') { temperature = atof(argv[i + 1]); }
        else if (flag == 'p') { topp = atof(argv[i + 1]); }
        else if (flag == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (flag == 'n') { steps = atoi(argv[i + 1]); }
        else if (flag == 'i') { prompt = argv[i + 1]; }
        else if (flag == 'z') { tokenizer_path = argv[i + 1]; }
        else if (flag == 'E') { g_exp = (char)toupper((unsigned char)argv[i + 1][0]); }
        else if (flag == 'b') { g_backend = argv[i + 1]; }
        else if (flag == 'R') { g_tile_rows = atoi(argv[i + 1]); }
        else if (flag == 'F') { g_gpu_epilogue = atoi(argv[i + 1]); }
        else { error_usage(); }
    }

    if (g_exp == 'B')      { g_use_npu = 0; g_use_gpu = 0; }
    else if (g_exp == 'C') { g_use_npu = 1; g_use_gpu = 0; }
    else if (g_exp == 'D') { g_use_npu = 0; g_use_gpu = 1; }
    else if (g_exp == 'E') { g_use_npu = 1; g_use_gpu = 1; }
    else { fprintf(stderr, "Unknown experiment '%c' (expected B, C, D, or E)\n", g_exp); error_usage(); }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0f) temperature = 0.0f;
    if (topp < 0.0f || 1.0f < topp) topp = 0.9f;
    if (steps < 0) steps = 0;

    if (g_use_npu || g_use_gpu) {
        if (accel_init(g_backend) != 0) {
            return EXIT_FAILURE;
        }
        if (g_use_gpu) {
            gpu_load_image(gpu_image, GPU_IMAGE_WORDS);
        }
    }

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    generate(&transformer, &tokenizer, &sampler, prompt, steps);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    if (g_use_npu || g_use_gpu) {
        accel_close();
    }
    return 0;
}

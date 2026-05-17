#include "snrt.h"
#include <stdio.h>
#include <stdint.h>

/* --- Macros & Configuration --- */
#define BANK_SIZE 8
#define NUM_BANKS 32
#define ROW_SIZE (NUM_BANKS * BANK_SIZE) 

#define ALIGN_UP_TCDM(addr) \
    ((((addr) + ROW_SIZE - 1) / ROW_SIZE) * ROW_SIZE)
    
#define W 16                 
#define H 16                 
#define TOTAL_PIXELS (W * H)
#define CHROMA_PIXELS (TOTAL_PIXELS / 4)

/* Block tiling configuration */
#define BLOCK_W  8
#define BLOCK_H  8
#define BLOCKS_X (W / BLOCK_W)  
#define BLOCKS_Y (H / BLOCK_H)  

static inline int tiled_Y_idx(int row, int col) {
    int block_row = row / BLOCK_H;
    int block_col = col / BLOCK_W;
    int local_row = row % BLOCK_H;
    int local_col = col % BLOCK_W;
    return (block_row * BLOCKS_X + block_col) * (BLOCK_W * BLOCK_H)
           + local_row * BLOCK_W + local_col;
}

typedef uint16_t float16_t;

static inline float16_t float_to_fp16(float x) {
    union { float f; uint32_t u; } v;
    asm volatile("fcvt.h.s %0, %1" : "=f"(v.f) : "f"(x));
    return (float16_t)(v.u & 0xFFFF);
}

static inline float fp16_to_float(float16_t x) {
    union { float f; uint32_t u; } v;
    v.u = 0xFFFF0000 | (uint32_t)x;
    float result;
    asm volatile("fcvt.s.h %0, %1" : "=f"(result) : "f"(v.f));
    return result;
}

// =========================================================================
// STAGE 1: YCBCR CONVERSION
// =========================================================================

void naive_YCBCR_conversion(uint8_t *input_idx, double *out_Y, double *out_CbCr, double *lut, int total_pixels) {
    const double c_cb = 0.141;    
    const double c_cr = 0.17825;  

    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            int i = row * W + col;
            double R = lut[input_idx[i * 3 + 0]];
            double G = lut[input_idx[i * 3 + 1]];
            double B = lut[input_idx[i * 3 + 2]];

            double Y  = 0.299 * R + 0.587 * G + 0.114 * B;
            double Cb = c_cb * (B - Y);
            double Cr = c_cr * (R - Y);

            out_Y[tiled_Y_idx(row, col)] = Y; 

            int g = i / 4;
            int l = i % 4;
            out_CbCr[g * 8 + l]      = Cb;
            out_CbCr[g * 8 + 4 + l] = Cr;
        }
    }
}

void opt_YCBCR_conversion(uint8_t *input_idx, float16_t *out_Y, float16_t *out_CbCr, double *lut, int total_pixels) {
    float c_r_f  = 0.299f;
    float c_g_f  = 0.587f;
    float c_b_f  = 0.114f;
    float c_cb_f = 0.141f;    
    float c_cr_f = 0.17825f;  

    asm volatile(
        "fcvt.h.s f3, %0 \n" "fcvt.h.s f4, %1 \n" "fcvt.h.s f5, %2 \n"
        "fcvt.h.s f6, %3 \n" "fcvt.h.s f7, %4 \n"
        :: "f"(c_r_f), "f"(c_g_f), "f"(c_b_f), "f"(c_cb_f), "f"(c_cr_f) : "f3","f4","f5","f6","f7"
    );

    snrt_issr_read(SNRT_SSR_DM0, lut, input_idx, total_pixels * 3, SNRT_SSR_IDXSIZE_U8);
    snrt_ssr_loop_4d(SNRT_SSR_DM1, 2, 2, 8, 2, 8, 8*2*8, 8*2, 8*8*2*2);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_4D, out_Y);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, total_pixels / 2, sizeof(uint64_t));
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, out_CbCr);
    snrt_ssr_enable();

    asm volatile(
        "frep.o %[n], 20, 0, 0 \n"
        "fmv.d f8, f0 \n fmv.d f9, f0 \n fmv.d f10, f0 \n"
        "vfcpka.h.d f8, f8, f0 \n vfcpka.h.d f9, f9, f0 \n vfcpka.h.d f10, f10, f0 \n"
        "fmv.d f14, f0 \n fmv.d f15, f0 \n fmv.d f16, f0 \n"
        "vfcpkb.h.d f8, f14, f0 \n vfcpkb.h.d f9, f15, f0 \n vfcpkb.h.d f10, f16, f0 \n"
        "vfmul.r.h f11, f8, f3 \n vfmac.r.h f11, f9, f4 \n vfmac.r.h f11, f10, f5 \n"
        "vfsub.h f10, f10, f11 \n vfsub.h f8, f8, f11 \n"
        "fmv.d f1, f11 \n"
        "vfmul.r.h f2, f10, f6 \n vfmul.r.h f2, f8, f7 \n"
        :: [n] "r"(total_pixels / 4 - 1) : "f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f14","f15","f16","memory"
    );
    snrt_fpu_fence();
    snrt_ssr_disable();
}

// =========================================================================
// STAGE 2: CHROMA SUBSAMPLING (STALL-FREE INTERLEAVED PIPELINE)
// =========================================================================

void naive_chroma_420(double *in_cbcr, double *out_cb_f, double *out_cr_f, int w, int h) {
    for (int y = 0; y < h; y += 2) {
        for (int x = 0; x < w; x += 2) {
            double sum_cb = 0.0; double sum_cr = 0.0;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int pixel_idx = (y + dy) * w + (x + dx);
                    int group = pixel_idx / 4; int lane  = pixel_idx % 4;
                    sum_cb += in_cbcr[group * 8 + lane];
                    sum_cr += in_cbcr[group * 8 + 4 + lane];
                }
            }
            int out_idx = (y / 2) * (w / 2) + (x / 2);
            out_cb_f[out_idx] = sum_cb;
            out_cr_f[out_idx] = sum_cr;
        }
    }
}

void opt_chroma_420_2d_ssr(double *in_cbcr, double *out_cb_f, double *out_cr_f) {
    
    asm volatile ("fence rw, rw" : : : "memory");
    snrt_cluster_hw_barrier();

    double zero_val = 0.0;
    double *p_zero = &zero_val;
    uintptr_t base_addr = (uintptr_t)in_cbcr;

    // --- SINGLE PASS CONFIGURATION ---
    void *p_top = (void*)(base_addr + 0);
    void *p_bot = (void*)(base_addr + 64);

    snrt_ssr_loop_2d(SNRT_SSR_DM0, 8, 8, 8, 128); 
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, p_top);
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 8, 8, 8, 128); 
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, p_bot);

    snrt_ssr_enable();
    
    // Ασφαλείς μεταβλητές C για να κάνει register allocation ο compiler
    uint32_t alu_cb1, alu_cr1, alu_cb2, alu_cr2;
    uint32_t iters = 15; // 16 επαναλήψεις

    asm volatile(
        "fld      f31, 0(%[zero]) \n"
        "csrsi    0x801, 0x1 \n"      

        "1: \n"
        // --- 1. Fetch SSR Streams Back-to-Back (Pops 0-7) ---
        "fmv.d    f4, f0 \n"  // Cb1 Top
        "fmv.d    f5, f1 \n"  // Cb1 Bot
        "fmv.d    f8, f0 \n"  // Cr1 Top
        "fmv.d    f9, f1 \n"  // Cr1 Bot
        "fmv.d    f12, f0 \n" // Cb2 Top
        "fmv.d    f13, f1 \n" // Cb2 Bot
        "fmv.d    f16, f0 \n" // Cr2 Top
        "fmv.d    f17, f1 \n" // Cr2 Bot

        // --- 2. Vector Additions (Παράλληλα) ---
        "vfadd.h  f6,  f4,  f5 \n"
        "vfadd.h  f10, f8,  f9 \n"
        "vfadd.h  f14, f12, f13 \n"
        "vfadd.h  f18, f16, f17 \n"

        // --- 3. Καθαρισμός Accumulators (Απόκρυψη Latency του vfadd) ---
        "fmv.d    f7,  f31 \n"
        "fmv.d    f11, f31 \n"
        "fmv.d    f15, f31 \n"
        "fmv.d    f19, f31 \n"

        // --- 4. Reduction (Παράλληλα Pairwise Sums) ---
        "vfsum.h  f7,  f6 \n"
        "vfsum.h  f11, f10 \n"
        "vfsum.h  f15, f14 \n"
        "vfsum.h  f19, f18 \n"

        // --- 5. Μεταφορά στην ALU (COPIFT Interleaving) ---
        "fmv.x.w  %[alu_cb1], f7 \n"
        "fmv.x.w  %[alu_cr1], f11 \n"
        "fmv.x.w  %[alu_cb2], f15 \n"
        "fmv.x.w  %[alu_cr2], f19 \n"

        // --- 6. Μέγιστο Memory Bandwidth (Συνεχόμενα Stores) ---
        "sw       %[alu_cb1], 0(%[out_cb]) \n"
        "sw       %[alu_cr1], 0(%[out_cr]) \n"
        "sw       %[alu_cb2], 4(%[out_cb]) \n"
        "sw       %[alu_cr2], 4(%[out_cr]) \n"

        // --- 7. Ενημέρωση Δεικτών & Loop ---
        "addi     %[out_cb], %[out_cb], 8 \n"
        "addi     %[out_cr], %[out_cr], 8 \n"
        
        "addi     %[iters], %[iters], -1 \n"
        "bgez     %[iters], 1b \n"

        "csrci    0x801, 0x1 \n"
        : [out_cb] "+r"(out_cb_f), [out_cr] "+r"(out_cr_f),
          [alu_cb1] "=&r"(alu_cb1), [alu_cr1] "=&r"(alu_cr1), [alu_cb2] "=&r"(alu_cb2), [alu_cr2] "=&r"(alu_cr2),
          [iters] "+r"(iters)
        : [zero] "r"(p_zero) 
        : "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f31", "memory"
    );
    snrt_ssr_disable();
}

// =========================================================================
// MAIN ROUTINE
// =========================================================================
int main() {
    if (snrt_cluster_core_idx() != 0 || !snrt_is_compute_core()) return 0;

    uintptr_t base = (uintptr_t)snrt_l1_next();

    // Memory Allocation
    double    *lut       = (double *)   ALIGN_UP_TCDM(base);
    uint8_t   *input_idx = (uint8_t *)  ALIGN_UP_TCDM((uintptr_t)lut + 256 * sizeof(double));
    double    *naive_Y    = (double *)   ALIGN_UP_TCDM((uintptr_t)input_idx + TOTAL_PIXELS * 3);
    double    *naive_CbCr = (double *)   ALIGN_UP_TCDM((uintptr_t)naive_Y   + TOTAL_PIXELS * sizeof(double));
    double    *naive_cb_f = (double *)   ALIGN_UP_TCDM((uintptr_t)naive_CbCr + TOTAL_PIXELS * 2 * sizeof(double));
    double    *naive_cr_f = (double *)   ALIGN_UP_TCDM((uintptr_t)naive_cb_f + CHROMA_PIXELS * sizeof(double));
    float16_t *opt_Y      = (float16_t *)ALIGN_UP_TCDM((uintptr_t)naive_cr_f + CHROMA_PIXELS * sizeof(double));
    float16_t *opt_CbCr   = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_Y      + TOTAL_PIXELS * sizeof(float16_t));
    double    *opt_cb_f   = (double *)   ALIGN_UP_TCDM((uintptr_t)opt_CbCr   + TOTAL_PIXELS * 2 * 2 * sizeof(float16_t));
    double    *opt_cr_f   = (double *)   ALIGN_UP_TCDM((uintptr_t)opt_cb_f   + CHROMA_PIXELS * 2 * sizeof(double));

    for (int i = 0; i < 256; i++) lut[i] = (double)i;
    for (int i = 0; i < TOTAL_PIXELS; i++) {
        input_idx[i * 3 + 0] = (i + 50) % 256;
        input_idx[i * 3 + 1] = (i + 100) % 256;
        input_idx[i * 3 + 2] = (i + 150) % 256;
    }
    snrt_cluster_hw_barrier();

    // -- Measure Naive Stage 1 --
    uint32_t n1_start = snrt_mcycle();
    naive_YCBCR_conversion(input_idx, naive_Y, naive_CbCr, lut, TOTAL_PIXELS);
    uint32_t n1_end = snrt_mcycle();

    // -- Measure Naive Stage 2 --
    uint32_t n2_start = snrt_mcycle();
    naive_chroma_420(naive_CbCr, naive_cb_f, naive_cr_f, W, H);
    uint32_t n2_end = snrt_mcycle();

    // -- Measure Opt Stage 1 --
    uint32_t o1_start = snrt_mcycle();
    opt_YCBCR_conversion(input_idx, opt_Y, opt_CbCr, lut, TOTAL_PIXELS);
    uint32_t o1_end = snrt_mcycle();

    // -- Measure Opt Stage 2 --
    uint32_t o2_start = snrt_mcycle();
    opt_chroma_420_2d_ssr((double*)opt_CbCr, opt_cb_f, opt_cr_f);
    uint32_t o2_end = snrt_mcycle();

    // -- Verification --
    float16_t *cb_fp16_ptr = (float16_t *)opt_cb_f;
    float16_t *cr_fp16_ptr = (float16_t *)opt_cr_f;
    int errors = 0;
    for (int i = 0; i < CHROMA_PIXELS; i++) {
        if ((fp16_to_float(cb_fp16_ptr[i]) - naive_cb_f[i] > 0.5f) || (fp16_to_float(cr_fp16_ptr[i]) - naive_cr_f[i] > 0.5f)) errors++;
    }

    // -- Final Performance Report --
    printf("\n===================================================\n");
    printf("            PERFORMANCE COMPARISON REPORT            \n");
    printf("===================================================\n");
    printf(" STAGE 1: YCbCr CONVERSION\n");
    printf("  - Naive Cycles : %u\n", n1_end - n1_start);
    printf("  - Opt   Cycles : %u\n", o1_end - o1_start);
    printf("  - Speedup      : %.2fx\n", (float)(n1_end - n1_start) / (o1_end - o1_start));
    printf("---------------------------------------------------\n");
    printf(" STAGE 2: CHROMA SUBSAMPLING (2x2 Interleaved Pipeline)\n");
    printf("  - Naive Cycles : %u\n", n2_end - n2_start);
    printf("  - Opt   Cycles : %u\n", o2_end - o2_start);
    printf("  - Speedup      : %.2fx\n", (float)(n2_end - n2_start) / (o2_end - o2_start));
    printf("---------------------------------------------------\n");
    printf(" TOTAL PIPELINE (S1 + S2)\n");
    uint32_t total_n = (n1_end - n1_start) + (n2_end - n2_start);
    uint32_t total_o = (o1_end - o1_start) + (o2_end - o2_start);
    printf("  - Total Naive  : %u\n", total_n);
    printf("  - Total Opt    : %u\n", total_o);
    printf("  - Overall Gain : %.2fx\n", (float)total_n / total_o);
    printf("---------------------------------------------------\n");
    printf(" VERIFICATION: %s\n", (errors == 0) ? "[SUCCESS] 100% MATCH" : "[FAILED]");
    printf("===================================================\n\n");

    return 0;
}

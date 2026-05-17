#include "snrt.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* --- Macros & Configuration --- */
#define BANK_SIZE 8
#define NUM_BANKS 32
#define ROW_SIZE (NUM_BANKS * BANK_SIZE)

#define ALIGN_UP_TCDM(addr) \
    ((((addr) + ROW_SIZE - 1) / ROW_SIZE) * ROW_SIZE)

#define W 8
#define H 8
#define TOTAL_PIXELS (W * H)

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

/* --- Σταθερός Πίνακας DCT (FP16) --- */
const uint64_t dct_m[16] __attribute__((aligned(64))) = {
    0x35a835a835a835a8, 0x35a835a835a835a8, // Σειρά 0
    0x2e3f34e93aa73be2, 0xbbe2baa7b4e9ae3f, // Σειρά 1
    0xbaabb21632163b64, 0x3b643216b216baab, // Σειρά 2
    0xb8e3bc53b2403aa7, 0xbaa732403c5338e3, // Σειρά 3
    0x35a8b5a8b5a835a8, 0x35a8b5a8b5a835a8, // Σειρά 4
    0x3aa73240bbe234e9, 0xb4e93be2b240baa7, // Σειρά 5
    0xb2163696b6963216, 0x3216b6963696b216, // Σειρά 6
    0xbbe23b64baa72e3f, 0xae3f3aa7bb643be2  // Σειρά 7
};

/* --- Πίνακας DCT για το Pass 2 (Broadcasted: 4x FP16 ανά 64-bit word) --- */
const uint64_t dct_m_bcast[64] __attribute__((aligned(64))) = {
    0x35a835a835a835a8, 0x35a835a835a835a8, 0x35a835a835a835a8, 0x35a835a835a835a8,
    0x35a835a835a835a8, 0x35a835a835a835a8, 0x35a835a835a835a8, 0x35a835a835a835a8,
    0x3be23be23be23be2, 0x3aa73aa73aa73aa7, 0x34e934e934e934e9, 0x2e3f2e3f2e3f2e3f,
    0xae3fae3fae3fae3f, 0xb4e9b4e9b4e9b4e9, 0xbaa7baa7baa7baa7, 0xbbe2bbe2bbe2bbe2,
    0x3b643b643b643b64, 0x3216321632163216, 0xb216b216b216b216, 0xbaabbaabbaabbaab,
    0xbaabbaabbaabbaab, 0xb216b216b216b216, 0x3216321632163216, 0x3b643b643b643b64,
    0x3aa73aa73aa73aa7, 0xb240b240b240b240, 0xbc53bc53bc53bc53, 0xb8e3b8e3b8e3b8e3,
    0x38e338e338e338e3, 0x3c533c533c533c53, 0x3240324032403240, 0xbaa7baa7baa7baa7,
    0x35a835a835a835a8, 0xb5a8b5a8b5a8b5a8, 0xb5a8b5a8b5a8b5a8, 0x35a835a835a835a8,
    0x35a835a835a835a8, 0xb5a8b5a8b5a8b5a8, 0xb5a8b5a8b5a8b5a8, 0x35a835a835a835a8,
    0x34e934e934e934e9, 0xbbe2bbe2bbe2bbe2, 0x3240324032403240, 0x3aa73aa73aa73aa7,
    0xbaa7baa7baa7baa7, 0xb240b240b240b240, 0x3be23be23be23be2, 0xb4e9b4e9b4e9b4e9,
    0x3216321632163216, 0xb696b696b696b696, 0x3696369636963696, 0xb216b216b216b216,
    0xb216b216b216b216, 0x3696369636963696, 0xb696b696b696b696, 0x3216321632163216,
    0x2e3f2e3f2e3f2e3f, 0xbaa7baa7baa7baa7, 0x3b643b643b643b64, 0xbbe2bbe2bbe2bbe2,
    0x3be23be23be23be2, 0xbb64bb64bb64bb64, 0x3aa73aa73aa73aa7, 0xae3fae3fae3fae3f
};

// ========================================================================
// STAGE 1: PASS 1 (ROWS)
// ========================================================================
void simd_dct_pass1(float16_t *in, float16_t *out, float16_t *tmp, const float16_t *dct_m_param) {
    float16_t off16 = float_to_fp16(-128.0f);
    uint64_t off_vec __attribute__((aligned(8))) =
        ((uint64_t)off16 << 48) | ((uint64_t)off16 << 32) | ((uint64_t)off16 << 16) | off16;
    snrt_ssr_loop_1d(SNRT_SSR_DM0, 16, 8);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)in);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 16, 8);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)out);
    snrt_ssr_enable();
    asm volatile(
        "fld     f10, 0(%0)     \n"
        "frep.o  %[n], 1, 0, 0 \n"
        "vfadd.h f1, f0, f10   \n"
        :: "r"(&off_vec), [n]"r"(15)
        : "f1", "f0", "f10", "memory"
    );
    snrt_fpu_fence();
    snrt_ssr_disable();

    snrt_ssr_loop_3d(SNRT_SSR_DM0, 2, 8, 8, 8, 0, 16);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_3D, (void*)out);
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 16, 8, 8, 0);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, (void*)dct_m_param);
    
    snrt_ssr_enable();
    double zero_val = 0.0;
    double *p_zero = &zero_val;
    uint32_t alu_f0, alu_f1, alu_f2, alu_f3;
    uint32_t alu_idx = 16;

    asm volatile(
        "fld     f31, 0(%[zero]) \n"  
        "csrsi   0x801, 0x1      \n"  
        "1:                      \n"
        "vfmul.h  f4,  f0, f1 \n" "vfmul.h  f5,  f0, f1 \n"  
        "vfmul.h  f8,  f0, f1 \n" "vfmul.h  f9,  f0, f1 \n"  
        "vfmul.h  f12, f0, f1 \n" "vfmul.h  f13, f0, f1 \n"  
        "vfmul.h  f16, f0, f1 \n" "vfmul.h  f17, f0, f1 \n"  
        "vfadd.h  f5,  f5,  f4 \n" "vfadd.h  f9,  f9,  f8 \n"  
        "vfadd.h  f13, f13, f12 \n" "vfadd.h  f17, f17, f16 \n"  
        "fmv.d    f4,  f31 \n"  "fmv.d    f8,  f31 \n"  
        "fmv.d    f12, f31 \n"  "fmv.d    f16, f31 \n"
        "vfsum.h  f4,  f5 \n" "vfsum.h  f8,  f9 \n"  
        "vfsum.h  f12, f13 \n" "vfsum.h  f16, f17 \n"  
        "fmv.d    f5,  f31 \n" "fmv.d    f9,  f31 \n"
        "fmv.d    f13, f31 \n" "fmv.d    f17, f31 \n"
        "vfsum.h  f5,  f4 \n" "vfsum.h  f9,  f8 \n"  
        "vfsum.h  f13, f12 \n" "vfsum.h  f17, f16 \n"  
        "fmv.x.w  %[alu_f0], f5 \n" "fmv.x.w  %[alu_f1], f9 \n"
        "fmv.x.w  %[alu_f2], f13 \n" "fmv.x.w  %[alu_f3], f17 \n"
        "sh       %[alu_f0], 0(%[tmp]) \n" "sh       %[alu_f1], 2(%[tmp]) \n"
        "sh       %[alu_f2], 4(%[tmp]) \n" "sh       %[alu_f3], 6(%[tmp]) \n"
        "addi     %[tmp], %[tmp], 8 \n"
        "addi     %[alu_idx], %[alu_idx], -1 \n"
        "bnez     %[alu_idx], 1b \n"
        "csrci   0x801, 0x1 \n"  
        : [tmp] "+r"(tmp), [alu_idx] "+r"(alu_idx), [alu_f0] "=&r"(alu_f0), [alu_f1] "=&r"(alu_f1), [alu_f2] "=&r"(alu_f2), [alu_f3] "=&r"(alu_f3)
        : [zero] "r"(p_zero) : "f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f31","memory"
    );
    snrt_ssr_disable();
}

// ========================================================================
// STAGE 2: PASS 2 (COLUMNS) - VECTOR REDUCTION TREE
// ========================================================================
void simd_dct_pass2(float16_t *in, float16_t *out, uint64_t *mat_bcast) {
    // Η μαγεία του Hardware: Διαβάζει τα 64 στοιχεία του mat_bcast, 
    // αλλά επαναλαμβάνει το ΚΑΘΕΝΑ 2 φορές αυτόματα (stride 0)! -> 128 reads
    snrt_ssr_loop_2d(SNRT_SSR_DM0, 2, 64, 0, 8);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, mat_bcast);
    
    // DM1: Pass 1 Output (Επανάληψη των 16 λέξεων για 8 φορές)
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 16, 8, 8, 0); 
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, in);
    
    // DM2: Pass 2 Output (16 λέξεις σειριακά)
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 16, 8);
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, out);

    snrt_ssr_enable();

    uint32_t iters = 8; // 8 γραμμές τελικού αποτελέσματος
    asm volatile(
        "1: \n"
        // --- Vector Multiplications (Back-to-back Pops) ---
        "vfmul.h f4, f0, f1 \n" "vfmul.h f5, f0, f1 \n" // k=0
        "vfmul.h f6, f0, f1 \n" "vfmul.h f7, f0, f1 \n" // k=1
        "vfmul.h f8, f0, f1 \n" "vfmul.h f9, f0, f1 \n" // k=2
        "vfmul.h f10, f0, f1 \n" "vfmul.h f11, f0, f1 \n" // k=3
        "vfmul.h f12, f0, f1 \n" "vfmul.h f13, f0, f1 \n" // k=4
        "vfmul.h f14, f0, f1 \n" "vfmul.h f15, f0, f1 \n" // k=5
        "vfmul.h f16, f0, f1 \n" "vfmul.h f17, f0, f1 \n" // k=6
        "vfmul.h f18, f0, f1 \n" "vfmul.h f19, f0, f1 \n" // k=7

        // --- Vector Additions Level 1 ---
        "vfadd.h f4, f4, f6 \n"  "vfadd.h f5, f5, f7 \n"
        "vfadd.h f8, f8, f10 \n" "vfadd.h f9, f9, f11 \n"
        "vfadd.h f12, f12, f14 \n" "vfadd.h f13, f13, f15 \n"
        "vfadd.h f16, f16, f18 \n" "vfadd.h f17, f17, f19 \n"

        // --- Vector Additions Level 2 ---
        "vfadd.h f4, f4, f8 \n"   "vfadd.h f5, f5, f9 \n"
        "vfadd.h f12, f12, f16 \n" "vfadd.h f13, f13, f17 \n"

        // --- Vector Additions Level 3 (Τελικό Άθροισμα) ---
        "vfadd.h f4, f4, f12 \n" "vfadd.h f5, f5, f13 \n"

        // --- Stream Out στον DM2 ---
        "fmv.d f2, f4 \n" "fmv.d f2, f5 \n"

        "addi %[iters], %[iters], -1 \n"
        "bnez %[iters], 1b \n"
        : [iters] "+r"(iters)
        :
        : "f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","memory"
    );

    snrt_ssr_disable();
}

// --- Naive Reference Functions ---
void naive_dct_pass1(float16_t *in, double *out, const float16_t *dct_m_fp16) {
    double s[8][8], tmp_mat[8][8];
    for (int i = 0; i < TOTAL_PIXELS; i++) {
        s[i / 8][i % 8] = (double)fp16_to_float(in[i]) - 128.0;
    }
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            double sum = 0;
            for (int k = 0; k < 8; k++) {
                double weight = (double)fp16_to_float(dct_m_fp16[c * 8 + k]);
                sum += s[r][k] * weight;
            }
            tmp_mat[r][c] = sum;
        }
    }
    for (int i = 0; i < 64; i++) out[i] = tmp_mat[i/8][i%8];
}

void naive_dct_pass2(double *pass1_out, double *final_out, const float16_t *dct_m_fp16) {
    double tmp_mat[8][8];
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            double sum = 0;
            for (int k = 0; k < 8; k++) {
                double weight = (double)fp16_to_float(dct_m_fp16[r * 8 + k]);
                sum += pass1_out[k * 8 + c] * weight;
            }
            tmp_mat[r][c] = sum;
        }
    }
    for (int i = 0; i < 64; i++) final_out[i] = tmp_mat[i/8][i%8];
}

// =========================================================================
// MAIN ROUTINE
// =========================================================================
int main() {
    // EARLY EXIT ΓΙΑ CORES 1-7
    if (snrt_is_compute_core() && snrt_cluster_core_idx() != 0) {
        return 0;
    }

    uint32_t core_id = snrt_cluster_core_idx();
    uintptr_t base = (uintptr_t)snrt_l1_next();
    
    // Memory allocation
    double    *dct_in     = (double *)   ALIGN_UP_TCDM(base);
    double    *naive_p1   = (double *)   ALIGN_UP_TCDM((uintptr_t)dct_in   + TOTAL_PIXELS * sizeof(double));
    double    *naive_p2   = (double *)   ALIGN_UP_TCDM((uintptr_t)naive_p1 + TOTAL_PIXELS * sizeof(double));
    float16_t *in_h       = (float16_t *)ALIGN_UP_TCDM((uintptr_t)naive_p2 + TOTAL_PIXELS * sizeof(double));
    float16_t *out_p1_h   = (float16_t *)ALIGN_UP_TCDM((uintptr_t)in_h     + TOTAL_PIXELS * sizeof(float16_t));
    float16_t *out_p2_h   = (float16_t *)ALIGN_UP_TCDM((uintptr_t)out_p1_h + TOTAL_PIXELS * sizeof(float16_t));
    float16_t *tmp_h      = (float16_t *)ALIGN_UP_TCDM((uintptr_t)out_p2_h + TOTAL_PIXELS * sizeof(float16_t));
    
    float16_t *mat_h      = (float16_t *)ALIGN_UP_TCDM((uintptr_t)tmp_h    + TOTAL_PIXELS * sizeof(float16_t));
    uint64_t  *mat_bcast  = (uint64_t *) ALIGN_UP_TCDM((uintptr_t)mat_h    + 16 * sizeof(uint64_t));

    // Φόρτωση και των δύο πινάκων μέσω DMA!
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(mat_h, (void*)dct_m, 16 * sizeof(uint64_t)); 
        snrt_dma_start_1d(mat_bcast, (void*)dct_m_bcast, 64 * sizeof(uint64_t));
        snrt_dma_wait_all();
    }
    
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core() && core_id == 0) {
        // Αρχικοποίηση δεδομένων
        for (int i = 0; i < TOTAL_PIXELS; i++) {
            dct_in[i] = (double)((i*13 + (i/8)*27) % 256);
            in_h[i] = float_to_fp16(dct_in[i]);
        }

        // --- Naive Εκτέλεση ---
        uint32_t start_naive = snrt_mcycle();
        naive_dct_pass1(in_h, naive_p1, mat_h);
        naive_dct_pass2(naive_p1, naive_p2, mat_h);
        uint32_t end_naive = snrt_mcycle();

        // --- Optimized Εκτέλεση ---
        uint32_t start_opt = snrt_mcycle();
        simd_dct_pass1(in_h, out_p1_h, tmp_h, mat_h);
        simd_dct_pass2(tmp_h, out_p2_h, mat_bcast);
        uint32_t end_opt = snrt_mcycle();

        // --- Verification ---
        int err_p1 = 0, err_p2 = 0;
        for (int i = 0; i < TOTAL_PIXELS; i++) {
            double diff_p1 = (double)fp16_to_float(tmp_h[i]) - naive_p1[i];
            if (diff_p1 > 0.5 || diff_p1 < -0.5 || diff_p1 != diff_p1) err_p1++;
            
            double diff_p2 = (double)fp16_to_float(out_p2_h[i]) - naive_p2[i];
            if (diff_p2 > 1.0 || diff_p2 < -1.0 || diff_p2 != diff_p2) err_p2++;
        }

        printf("\n===================================================\n");
        printf("             PERFORMANCE COMPARISON REPORT            \n");
        printf("===================================================\n");
        printf(" 2D DCT FULL PIPELINE (Pass 1 + Pass 2)\n");
        printf("  - Naive Cycles : %u\n", end_naive - start_naive);
        printf("  - Opt   Cycles : %u\n", end_opt   - start_opt);
        printf("  - Speedup      : %.2fx\n", (float)(end_naive - start_naive) / (end_opt - start_opt));
        printf("---------------------------------------------------\n");
        printf(" VERIFICATION PASS 1: %s\n", (err_p1 == 0) ? "[SUCCESS] 100% MATCH" : "[FAILED]");
        printf(" VERIFICATION PASS 2: %s\n", (err_p2 == 0) ? "[SUCCESS] 100% MATCH" : "[FAILED]");
        if (err_p1 || err_p2) printf(" Errors -> P1: %d, P2: %d\n", err_p1, err_p2);
        printf("===================================================\n\n");

        printf("--- First 8 Final Output Values (Pass 2 - Row 0) ---\n");
        printf(" Index | Naive (double) | Opt (FP16 Packed)  \n");
        printf("---------------------------------------\n");
        for (int i = 0; i < 8; i++) {
            printf(" %5d | %14.3f | %12.3f\n", i, naive_p2[i], fp16_to_float(out_p2_h[i]));
        }
        printf("---------------------------------------\n\n");
    }

    return 0;
}

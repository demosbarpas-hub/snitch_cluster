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

/* --- Standard Tables (Stored in L2) --- */
const double quant_recip_chr_l2[64] __attribute__((aligned(64))) = {
    0.058824, 0.055556, 0.041667, 0.021277, 0.010101, 0.010101, 0.010101, 0.010101,
    0.055556, 0.047619, 0.038462, 0.015152, 0.010101, 0.010101, 0.010101, 0.010101,
    0.041667, 0.038462, 0.017857, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.021277, 0.015152, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101,
    0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101, 0.010101
};

// =========================================================================
// NAIVE FP16 QUANTIZATION
// =========================================================================
void naive_quant_fp16(float16_t *in, float16_t *out, const float16_t *recip) {
    for (int i = 0; i < 64; i++) {
        float val = fp16_to_float(in[i]) * fp16_to_float(recip[i]);
        out[i] = float_to_fp16(nearbyintf(val));
    }
}

// =========================================================================
// OPTIMIZED FP16 QUANTIZATION (4-Way Interleaved Vectorized)
// =========================================================================
void opt_quant_fp16(float16_t *in, float16_t *out, const float16_t *recip) {
    
    // Το magic value (1536.0)
    float16_t magic_val = float_to_fp16(1536.0f);
    
    // Η ΣΩΤΗΡΙΑ ΑΛΛΑΓΗ: Το __attribute__((aligned(8))) σώζει από το Misaligned Error!
    uint64_t magic_vec __attribute__((aligned(8))) = 
        ((uint64_t)magic_val << 48) | ((uint64_t)magic_val << 32) | 
        ((uint64_t)magic_val << 16) | magic_val;

    // SSR Configuration
    snrt_ssr_loop_1d(SNRT_SSR_DM0, 16, 8);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)in);
    
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 16, 8);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)recip);
    
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 16, 8);
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, (void*)out);

    snrt_ssr_enable();
    
    asm volatile(
        // Ασφαλές, aligned load!
        "fld     f10, 0(%[magic]) \n"    

        "frep.o  %[cnt], 12, 0, 0 \n"    
        
        // 1. Vector Multiplications
        "vfmul.h f4, f0, f1 \n"          
        "vfmul.h f5, f0, f1 \n"          
        "vfmul.h f6, f0, f1 \n"          
        "vfmul.h f7, f0, f1 \n"          
        
        // 2. Magic Rounding Add
        "vfadd.h f8, f4, f10 \n"         
        "vfadd.h f9, f5, f10 \n"         
        "vfadd.h f12, f6, f10 \n"        
        "vfadd.h f13, f7, f10 \n"        
        
        // 3. Magic Rounding Sub -> Write to DM2 (f2)
        "vfsub.h f2, f8, f10 \n"         
        "vfsub.h f2, f9, f10 \n"         
        "vfsub.h f2, f12, f10 \n"        
        "vfsub.h f2, f13, f10 \n"        
        : 
        : [cnt] "r"(3), [magic] "r"(&magic_vec)
        : "f0", "f1", "f2", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f12", "f13", "memory" 
    );

    snrt_fpu_fence();
    snrt_ssr_disable();
}

// =========================================================================
// MAIN ROUTINE
// =========================================================================
int main() {
    uint32_t core_id = snrt_cluster_core_idx();
    uintptr_t base = ALIGN_UP_TCDM((uintptr_t)snrt_l1_next());

    // --- TCDM Memory Allocation ---
    float16_t *q_in        = (float16_t *)base; base += 64 * sizeof(float16_t);
    float16_t *q_out_n     = (float16_t *)base; base += 64 * sizeof(float16_t);
    float16_t *q_out_o     = (float16_t *)base; base += 64 * sizeof(float16_t);
    float16_t *recip_fp16  = (float16_t *)base; base += 64 * sizeof(float16_t);
    
    // DMA Target buffer
    double    *recip_f64   = (double *)base;    base += 64 * sizeof(double);

    // --- DMA TRANSFER ---
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(recip_f64, (void*)quant_recip_chr_l2, 64 * sizeof(double));
        snrt_dma_wait_all();
    }
    
    snrt_cluster_hw_barrier();
    if (snrt_is_compute_core() && core_id != 0) return 0; // Early exit Cores 1-7

    if (snrt_is_compute_core() && core_id == 0) {
        
        // 1. Μετατροπή του DMA'd πίνακα (Double -> FP16)
        for(int i = 0; i < 64; i++) {
            recip_fp16[i] = float_to_fp16((float)recip_f64[i]);
        }

        // 2. Δημιουργία δεδομένων
        for (int i = 0; i < 64; i++) {
            float val = (float)((i * 17 + 123) % 1024) - 512.0f; 
            q_in[i] = float_to_fp16(val);
        }

        // --- NAIVE EXECUTION ---
        uint32_t s_n = snrt_mcycle();
        naive_quant_fp16(q_in, q_out_n, recip_fp16);
        uint32_t e_n = snrt_mcycle();

        // --- OPTIMIZED EXECUTION ---
        uint32_t s_o = snrt_mcycle();
        opt_quant_fp16(q_in, q_out_o, recip_fp16);
        uint32_t e_o = snrt_mcycle();

        // --- VERIFICATION ---
        int errors = 0;
        for (int i = 0; i < 64; i++) {
            float diff = fp16_to_float(q_out_o[i]) - fp16_to_float(q_out_n[i]);
            if (diff > 0.5f || diff < -0.5f) errors++;
        }

        printf("\n===================================================\n");
        printf("           FP16 QUANTIZATION PERFORMANCE           \n");
        printf("===================================================\n");
        printf("  - Naive Cycles : %u\n", e_n - s_n);
        printf("  - Opt   Cycles : %u\n", e_o - s_o);
        printf("  - Speedup      : %.2fx\n", (float)(e_n - s_n) / (e_o - s_o));
        printf("---------------------------------------------------\n");
        printf("  VERIFICATION   : %s\n", (errors == 0) ? "[SUCCESS] 100% MATCH" : "[FAILED]");
        if(errors) printf("  Errors Found   : %d\n", errors);
        printf("===================================================\n\n");

        printf("--- First 8 Quantized Values ---\n");
        printf(" Index | Input (FP16) | Opt (Int) | Naive (Int)\n");
        printf("-----------------------------------------------\n");
        for (int i = 0; i < 8; i++) {
            printf(" %5d | %12.3f | %9.1f | %9.1f\n", 
                   i, fp16_to_float(q_in[i]), 
                   fp16_to_float(q_out_o[i]), 
                   fp16_to_float(q_out_n[i]));
        }
    }

    snrt_cluster_hw_barrier();
    return 0;
}
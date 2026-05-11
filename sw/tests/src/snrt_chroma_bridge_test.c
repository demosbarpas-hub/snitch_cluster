#include "snrt.h"
#include <stdio.h>
#include <stdint.h>

/* --- Macros --- */
#define ALIGN_UP_TCDM(addr) ((((addr) + 63) / 64) * 64)
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

int main() {
    if (snrt_cluster_core_idx() != 0 || !snrt_is_compute_core()) return 0;

    // Πίνακες για το τεστ
    float16_t src_top[8] __attribute__((aligned(16)));
    float16_t src_bot[8] __attribute__((aligned(16)));
    float16_t dst[4]     __attribute__((aligned(16))) = {0,0,0,0};

    for (int i = 0; i < 8; i++) {
        src_top[i] = float_to_fp16(10.0f);
        src_bot[i] = float_to_fp16(20.0f);
    }

    // Βοηθητικοί δείκτες για την assembly
    float16_t *p_top = src_top;
    float16_t *p_bot = src_bot;
    float16_t *p_dst = dst;

    printf("\n--- Start Stable 1D Chroma Test (Fix Constraints) ---\n");

    float scale = 0.25f;
    int n_iters = 1; // 2 επαναλήψεις (0 και 1)
    int alu_count = n_iters;

    asm volatile("fcvt.h.s f3, %0" :: "f"(scale) : "f3");

    asm volatile(
        "csrsi 0x801, 0x1 \n"      
        "mv    t6, x0 \n"          

        // FPU Loop (Hardware) - 6 εντολές
        "frep.o %[n_fpu], 6, 0, 0 \n"  
        "fld       f8,  0(%[in_t]) \n" 
        "fld       f9,  0(%[in_b]) \n" 
        "vfadd.h   f10, f8, f9 \n"     
        "vfsum.h   f16, f10 \n"        
        "vfmul.r.h f16, f16, f3 \n"    
        "fmv.x.w   t6, f16 \n"         

        // ALU Loop (Software)
        "1: \n"
        "mv        t0, t6 \n"          
        "sw        t0, 0(%[out]) \n"   
        "addi      %[in_t], %[in_t], 8 \n"
        "addi      %[in_b], %[in_b], 8 \n"
        "addi      %[out],  %[out],  4 \n"
        "addi      %[alu_c], %[alu_c], -1 \n"
        "bgez      %[alu_c], 1b \n"

        "csrci 0x801, 0x1 \n"          
        
        // --- ΔΙΟΡΘΩΜΕΝΑ CONSTRAINTS ---
        : [out] "+r"(p_dst), [alu_c] "+r"(alu_count), [in_t] "+r"(p_top), [in_b] "+r"(p_bot)
        : [n_fpu] "r"(n_iters)
        : "f3", "f8", "f9", "f10", "f16", "t0", "t6", "memory"
    );

    snrt_fpu_fence();

    printf("Result [0]: %.1f\n", fp16_to_float(dst[0]));
    printf("Result [1]: %.1f\n", fp16_to_float(dst[1]));

    if (fp16_to_float(dst[0]) == 15.0f) {
        printf("\n[SUCCESS] Math and Bridge are perfect!\n");
    } else {
        printf("\n[FAILED] Value: %.1f\n", fp16_to_float(dst[0]));
    }

    return 0;
}
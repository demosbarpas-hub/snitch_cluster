#include "snrt.h"
#include "printf.h"
#include <stdint.h>

#define LENGTH 4

int main() {
#if defined(SNRT_SUPPORTS_FREP) && defined(SNRT_SUPPORTS_SSR)
    if (snrt_is_dm_core()) return 0;
    if (snrt_cluster_core_idx() != 0) return 0;

    double *input = (double *)snrt_l1_alloc_compute_core_local(
        LENGTH * sizeof(double), sizeof(double));

    input[0] = 10.0;
    input[1] = 20.0;
    input[2] = 30.0;
    input[3] = 40.0;

    double output[4] = {0.0, 0.0, 0.0, 0.0};

    snrt_ssr_loop_1d(SNRT_SSR_DM0, LENGTH, sizeof(double));
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, input);
    snrt_ssr_enable();

    // n_outer = 2 επαναλήψεις (τιμή 1)
    // n_inner = 1 επανάληψη (τιμή 0)
    asm volatile(
        // Εξωτερικό loop: 2 iterations, n_inst=2 (τα δύο fmv.d), stagger max 1 στο rd
        "frep.o %[no], 2, 1, 0b0001 \n"
        "frep.o %[ni], 1, 0, 0 \n"      // Εσωτερικό 1
        "fmv.d  f14, f0 \n"             // 1η FP εντολή
        "frep.o %[ni], 1, 0, 0 \n"      // Εσωτερικό 2
        "fmv.d  f16, f0 \n"             // 2η FP εντολή
        :
        : [no] "r"(2 - 1), [ni] "r"(1 - 1)
        : "f0", "f14", "f15", "f16", "f17", "memory"
    );

    snrt_fpu_fence();
    snrt_ssr_disable();

    // Διάβασμα αποτελεσμάτων (f14, f15, f16, f17)
    asm volatile(
        "fsd f14, 0(%[out]) \n"
        "fsd f15, 8(%[out]) \n"
        "fsd f16, 16(%[out]) \n"
        "fsd f17, 24(%[out]) \n"
        :
        : [out] "r"(output)
        : "memory"
    );

    printf("\n--- Nested SSR + Staggered FREP test ---\n");
    printf("Input:    %.1f %.1f %.1f %.1f\n", input[0], input[1], input[2], input[3]);
    
    // Ανάλυση ροής:
    // Iter 0: f14 = input[0] (10.0), f16 = input[1] (20.0)
    // Iter 1: f15 = input[2] (30.0), f17 = input[3] (40.0) -> λόγω outer stagger
    printf("Output:\n");
    printf("  f14 = %.1f (expected 10.0)\n", output[0]);
    printf("  f16 = %.1f (expected 20.0)\n", output[2]);
    printf("  f15 = %.1f (expected 30.0)\n", output[1]);
    printf("  f17 = %.1f (expected 40.0)\n", output[3]);

    int errors = 0;
    if (output[0] != 10.0) errors++;
    if (output[2] != 20.0) errors++;
    if (output[1] != 30.0) errors++;
    if (output[3] != 40.0) errors++;

    if (errors == 0) {
        printf("\nSUCCESS! Nested Staggered FREP works correctly.\n");
    } else {
        printf("\nFAILED! %d errors.\n", errors);
    }

    return errors;
#endif
    return 0;
}
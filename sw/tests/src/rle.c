#include "snrt.h"
#include <stdint.h>
#include <stdio.h>

/* --- Configuration --- */
#define ROW_SIZE 256
#define ALIGN_UP_TCDM(addr) ((((addr) + ROW_SIZE - 1) / ROW_SIZE) * ROW_SIZE)

/* --- Tables & Data --- */
// Standard JPEG Zig-Zag order (Stored in slow L2 Memory)
const uint8_t zigzag_table_l2[64] __attribute__((aligned(64))) = {
    0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

/* --- The Zig-Zag & RLE Kernel --- */
void process_zigzag_rle(double *q_data, int16_t *out_rle, uint8_t *l1_zigzag_indices) {
    uint32_t run = 0;
    uint32_t out_p = 0;

    // SSR: indexed read of q_data via zigzag indices
    snrt_issr_read(SNRT_SSR_DM0, q_data, l1_zigzag_indices, 64, SNRT_SSR_IDXSIZE_U8);
    snrt_ssr_enable();

    // Pre-feed one value to unblock frep (same pattern as copift_queues.c)
    int32_t val;
    asm volatile(
        "mv        t6, x0          \n"  // clear t6
        "csrsi     copift, 0x1     \n"  // enable queues
        // Kick off FPU loop — it will push 64 values into f2i queue
        "frep.o    %[n], 1, 0, 0   \n"
        "fcvt.w.d  t6, ft0         \n"  // FPU: read SSR, convert, push to f2i
        // Read first value from f2i to unblock
        "mv        %[val], t6      \n"
        : [val] "=r"(val)
        : [n] "r"(63)
        : "t6", "ft0", "memory"
    );

    // Process first value
    if (val == 0) run++;
    else { out_rle[out_p++] = (int16_t)run; out_rle[out_p++] = (int16_t)val; run = 0; }

    // Integer loop reads remaining 63 values from f2i queue
    // FPU is pushing them concurrently — no fence needed
    for (int i = 1; i < 64; i++) {
        asm volatile(
            "mv %[val], t6 \n"  // pop from f2i queue
            : [val] "=r"(val) : : "t6"
        );
        if (val == 0) run++;
        else { out_rle[out_p++] = (int16_t)run; out_rle[out_p++] = (int16_t)val; run = 0; }
    }

    asm volatile("csrci copift, 0x1 \n" ::: "memory");  // disable queues
    if (run > 0) { out_rle[out_p++] = 0; out_rle[out_p++] = 0; }

    snrt_fpu_fence();
    snrt_ssr_disable();
}

/* --- Main Entry Point --- */
int main() {
    uint32_t core_id = snrt_cluster_core_idx();
    uintptr_t base = (uintptr_t)snrt_l1_next();

    // 1. Allocate aligned TCDM memory blocks in L1
    double  *q_in      = (double *) ALIGN_UP_TCDM(base);
    int16_t *rle_out   = (int16_t *)ALIGN_UP_TCDM((uintptr_t)q_in + 64 * sizeof(double));
    uint8_t *l1_zigzag = (uint8_t *)ALIGN_UP_TCDM((uintptr_t)rle_out + 128 * sizeof(int16_t));

    // 2. DMA Transfer: Move indices from L2 to L1
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l1_zigzag, (void*)zigzag_table_l2, 64 * sizeof(uint8_t));
        snrt_dma_wait_all();
    }

    // Barrier: Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // 3. Compute Core Task
    if (snrt_is_compute_core() && core_id == 0) {
        
        // A realistic post-quantization JPEG block
        const double real_jpeg_block[64] = {
            15.0,  0.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            -2.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
             0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
        };

        // Initialize L1 memory
        for (int i = 0; i < 64; i++) {
            q_in[i] = real_jpeg_block[i];
        }

        // Benchmark
        uint32_t start = snrt_mcycle();
        process_zigzag_rle(q_in, rle_out, l1_zigzag);
        uint32_t end = snrt_mcycle();

        // Verification Printout
        printf("\n--- RLE VERIFICATION (REAL DATA) ---\n");
        printf("Zig-Zag Stream: ");
        
        for (int i = 0; i < 64; i++) {
            printf("%0.f ", q_in[zigzag_table_l2[i]]);
        }
        
        printf("\n\nEncoded (Run, Level) Pairs:\n");
        for (int i = 0; i < 128; i += 2) {
            printf("(%d, %d) ", rle_out[i], rle_out[i+1]);
            // Stop printing at the End of Block marker
            if ((rle_out[i] == 0 && rle_out[i+1] == 0 && i > 0) || i >= 126) break;
        }
        printf("\n------------------------\n");
        printf("RLE Processed in %u cycles\n", end - start);
    }

    snrt_cluster_hw_barrier();
    return 0;
}
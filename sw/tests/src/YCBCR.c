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

/* Block tiling: image divided into (W/8)x(H/8) blocks of 8x8 pixels.
 * Blocks stored left-to-right, top-to-bottom (row-major over blocks).
 * Within each block, pixels stored row-major.
 *
 * For 16x16: 2x2 grid of 8x8 blocks → 4 blocks of 64 fp16 each.
 * Memory order: [block(0,0)][block(0,1)][block(1,0)][block(1,1)]
 */
#define BLOCK_W  8
#define BLOCK_H  8
#define BLOCKS_X (W / BLOCK_W)  /* 2 */
#define BLOCKS_Y (H / BLOCK_H)  /* 2 */

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

/* --------------------------------------------------------------------------
 * NAIVE reference — mirrors the opt kernel exactly:
 *
 *  1. Y layout   : tiled 8x8 blocks (see tiled_Y_idx above).
 *
 *  2. Cb/Cr scale: coefficients pre-divided by 4 to match opt.
 *       c_cb = 0.564 / 4 = 0.141
 *       c_cr = 0.713 / 4 = 0.17825
 *
 *  3. CbCr layout: SSR DM2 is a 1D stream with stride=8 bytes (sizeof uint64_t).
 *     Each write to f2 pushes one 64-bit word = 4×fp16.
 *     The kernel has TWO vfmul writes to f2 per 4-pixel iteration:
 *       push 0: vfmul f2, f6, f10  →  [Cb_p0, Cb_p1, Cb_p2, Cb_p3]
 *       push 1: vfmul f2, f7, f8   →  [Cr_p0, Cr_p1, Cr_p2, Cr_p3]
 *     So for group g = i/4, lane l = i%4:
 *       out_CbCr[g*8 + l]     = Cb for pixel g*4+l
 *       out_CbCr[g*8 + 4 + l] = Cr for pixel g*4+l
 * -------------------------------------------------------------------------- */
void naive_YCBCR_conversion(uint8_t *input_idx, double *out_Y, double *out_CbCr,
                             double *lut, int total_pixels) {
    const double c_cb = 0.141;    /* 0.564 / 4 */
    const double c_cr = 0.17825;  /* 0.713 / 4 */

    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            int i = row * W + col;

            double R = lut[input_idx[i * 3 + 0]];
            double G = lut[input_idx[i * 3 + 1]];
            double B = lut[input_idx[i * 3 + 2]];

            double Y  = 0.299 * R + 0.587 * G + 0.114 * B;
            double Cb = c_cb * (B - Y);
            double Cr = c_cr * (R - Y);

            /* Y → tiled */
            out_Y[tiled_Y_idx(row, col)] = Y;

            /* CbCr → planar-per-4: [Cb×4][Cr×4] per group of 4 pixels */
            int g = i / 4;
            int l = i % 4;
            out_CbCr[g * 8 + l]     = Cb;
            out_CbCr[g * 8 + 4 + l] = Cr;
        }
    }
}

/* --------------------------------------------------------------------------
 * OPT: FP16 vector version
 * -------------------------------------------------------------------------- */
void opt_YCBCR_conversion(uint8_t *input_idx, float16_t *out_Y, float16_t *out_CbCr,
                           double *lut, int total_pixels) {
    float c_r_f  = 0.299f;
    float c_g_f  = 0.587f;
    float c_b_f  = 0.114f;
    float c_cb_f = 0.141f;    /* 0.564 / 4 */
    float c_cr_f = 0.17825f;  /* 0.713 / 4 */

    // --- ΒΕΛΤΙΣΤΟΠΟΙΗΣΗ 1: Setup σε 5 εντολές ---
    // Το fcvt.h.s μετατρέπει τον FP32 σε FP16 και τον αφήνει στο Lane 0.
    asm volatile(
        "fcvt.h.s f3, %0 \n"
        "fcvt.h.s f4, %1 \n"
        "fcvt.h.s f5, %2 \n"
        "fcvt.h.s f6, %3 \n"
        "fcvt.h.s f7, %4 \n"
        :
        : "f"(c_r_f), "f"(c_g_f), "f"(c_b_f), "f"(c_cb_f), "f"(c_cr_f)
        : "f3", "f4", "f5", "f6", "f7"
    );

    snrt_issr_read(SNRT_SSR_DM0, lut, input_idx, total_pixels * 3, SNRT_SSR_IDXSIZE_U8);
    snrt_ssr_loop_4d(SNRT_SSR_DM1, 2, 2, 8, 2,
                     8, 8*2*8, 8*2, 8*8*2*2);//Κάνω γραμμή όλα τα Υ ανά block 8x8 έτοιμα
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_4D, out_Y);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, total_pixels / 2, sizeof(uint64_t));
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, out_CbCr);
    snrt_ssr_enable();

    asm volatile(
        // --- ΒΕΛΤΙΣΤΟΠΟΙΗΣΗ 2: Το loop μειώθηκε στις 20 εντολές ---
        "frep.o %[n], 20, 0, 0 \n"

        // [FETCH R1, G1, B1]
        "fmv.d f8,  f0 \n"   
        "fmv.d f9,  f0 \n"   
        "fmv.d f10, f0 \n"   
        
        "vfcpka.h.d f8,  f8,  f0 \n"
        "vfcpka.h.d f9,  f9,  f0 \n"
        "vfcpka.h.d f10, f10, f0 \n"
        
        // [FETCH R3, G3, B3]
        "fmv.d f14, f0 \n"   
        "fmv.d f15, f0 \n"   
        "fmv.d f16, f0 \n"   
        
        "vfcpkb.h.d f8,  f14, f0 \n"
        "vfcpkb.h.d f9,  f15, f0 \n"
        "vfcpkb.h.d f10, f16, f0 \n"

        // --- ΒΕΛΤΙΣΤΟΠΟΙΗΣΗ 3: Γραμμικός υπολογισμός Y με .r ---
        "vfmul.r.h  f11, f8,  f3 \n"  /* f11 = R * c_r */
        "vfmac.r.h  f11, f9,  f4 \n"  /* f11 += G * c_g */
        "vfmac.r.h  f11, f10, f5 \n"  /* f11 += B * c_b  (Τελικό Y) */

        // [Υπολογισμός Differences]
        "vfsub.h    f10, f10, f11 \n" /* B - Y */
        "vfsub.h    f8,  f8,  f11 \n" /* R - Y */

        "fmv.d      f1,  f11 \n"      /* push Y (4×fp16) to SSR DM1 */

        // --- ΒΕΛΤΙΣΤΟΠΟΙΗΣΗ 4: Χρήση .r για τα Cb / Cr ---
        "vfmul.r.h  f2,  f10, f6 \n"  /* push Cb (4×fp16) to SSR DM2 */
        "vfmul.r.h  f2,  f8,  f7 \n"  /* push Cr (4×fp16) to SSR DM2 */
        :
        : [n] "r"(total_pixels / 4 - 1)
        : "f0", "f1", "f2",
          "f3", "f4", "f5", "f6", "f7",
          "f8", "f9", "f10", "f11",
          /* Ο f12 αφαιρέθηκε από τα clobbers γιατί γλιτώσαμε τη χρήση του! */
          "f13", "f14", "f15", "f16", "f17", "f18", "f19",
          "memory"
    );

    snrt_fpu_fence();
    snrt_ssr_disable();
}

/* --------------------------------------------------------------------------
 * main
 * -------------------------------------------------------------------------- */
int main() {
    if (snrt_cluster_core_idx() != 0 || !snrt_is_compute_core()) return 0;

    uintptr_t base = (uintptr_t)snrt_l1_next();

    double    *lut       = (double *)   ALIGN_UP_TCDM(base);
    uint8_t   *input_idx = (uint8_t *)  ALIGN_UP_TCDM((uintptr_t)lut + 256 * sizeof(double));

    double    *naive_Y    = (double *)  ALIGN_UP_TCDM((uintptr_t)input_idx + TOTAL_PIXELS * 3);
    double    *naive_CbCr = (double *)  ALIGN_UP_TCDM((uintptr_t)naive_Y   + TOTAL_PIXELS * sizeof(double));

    float16_t *opt_Y    = (float16_t *)ALIGN_UP_TCDM((uintptr_t)naive_CbCr + TOTAL_PIXELS * 2 * sizeof(double));
    float16_t *opt_CbCr = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_Y      + TOTAL_PIXELS * sizeof(float16_t));

    for (int i = 0; i < 256; i++) lut[i] = (double)i;
    for (int i = 0; i < TOTAL_PIXELS * 3; i++) input_idx[i] = (uint8_t)(i % 256);

    snrt_cluster_hw_barrier();

    uint32_t start_naive_cc = snrt_mcycle();
    naive_YCBCR_conversion(input_idx, naive_Y, naive_CbCr, lut, TOTAL_PIXELS);
    uint32_t end_naive_cc = snrt_mcycle();

    uint32_t start_opt_cc = snrt_mcycle();
    opt_YCBCR_conversion(input_idx, opt_Y, opt_CbCr, lut, TOTAL_PIXELS);
    uint32_t end_opt_cc = snrt_mcycle();

    printf("\n--- STAGE 1: Planar Color Conversion ---\n");
    printf("Naive Cycles : %u\n", end_naive_cc - start_naive_cc);
    printf("Opt Cycles   : %u\n", end_opt_cc   - start_opt_cc);
    printf("Speedup      : %.2fx\n",
           (double)(end_naive_cc - start_naive_cc) / (end_opt_cc - start_opt_cc));

    /* --- Y: print tiled blocks, opt vs naive --- */
    printf("\n--- Y layout: opt (fp16) vs naive (double), per 8x8 block ---\n");
    for (int blk = 0; blk < BLOCKS_X * BLOCKS_Y; blk++) {
        int brow = blk / BLOCKS_X;
        int bcol = blk % BLOCKS_X;
        printf("Block (%d,%d) [rows %d-%d, cols %d-%d]:\n",
               brow, bcol,
               brow*BLOCK_H, brow*BLOCK_H+BLOCK_H-1,
               bcol*BLOCK_W, bcol*BLOCK_W+BLOCK_W-1);
        for (int lr = 0; lr < BLOCK_H; lr++) {
            printf("  row%2d opt: ", brow*BLOCK_H + lr);
            for (int lc = 0; lc < BLOCK_W; lc++)
                printf("%6.2f ", fp16_to_float(opt_Y[blk*BLOCK_W*BLOCK_H + lr*BLOCK_W + lc]));
            printf("\n        ref: ");
            for (int lc = 0; lc < BLOCK_W; lc++)
                printf("%6.2f ", naive_Y[blk*BLOCK_W*BLOCK_H + lr*BLOCK_W + lc]);
            printf("\n");
        }
    }

    /* --- CbCr: print first 4 groups of 4 pixels, opt vs naive ---
     * Layout: out_CbCr[g*8 + 0..3] = Cb for pixels g*4..g*4+3
     *         out_CbCr[g*8 + 4..7] = Cr for pixels g*4..g*4+3
     */
    printf("\n--- CbCr layout (planar-per-4): opt vs naive ---\n");
    for (int g = 0; g < 4; g++) {
        printf("Group %d (pixels %d-%d):\n", g, g*4, g*4+3);
        printf("  opt   Cb: %6.3f %6.3f %6.3f %6.3f  "
               "Cr: %6.3f %6.3f %6.3f %6.3f\n",
            fp16_to_float(opt_CbCr[g*8+0]), fp16_to_float(opt_CbCr[g*8+1]),
            fp16_to_float(opt_CbCr[g*8+2]), fp16_to_float(opt_CbCr[g*8+3]),
            fp16_to_float(opt_CbCr[g*8+4]), fp16_to_float(opt_CbCr[g*8+5]),
            fp16_to_float(opt_CbCr[g*8+6]), fp16_to_float(opt_CbCr[g*8+7]));
        printf("  naive Cb: %6.3f %6.3f %6.3f %6.3f  "
               "Cr: %6.3f %6.3f %6.3f %6.3f\n",
            naive_CbCr[g*8+0], naive_CbCr[g*8+1],
            naive_CbCr[g*8+2], naive_CbCr[g*8+3],
            naive_CbCr[g*8+4], naive_CbCr[g*8+5],
            naive_CbCr[g*8+6], naive_CbCr[g*8+7]);
    }

    /* --- Verification ---
     * Both arrays now share the same layout, so we compare flat.
     * Y    : tiled, TOTAL_PIXELS entries
     * CbCr : planar-per-4, TOTAL_PIXELS*2 entries (Cb half then Cr half per group)
     */
    int errors_Y    = 0;
    int errors_CbCr = 0;

    for (int i = 0; i < TOTAL_PIXELS; i++) {
        double diff = (double)fp16_to_float(opt_Y[i]) - naive_Y[i];
        if (diff > 0.5 || diff < -0.5) errors_Y++;
    }

    for (int i = 0; i < TOTAL_PIXELS * 2; i++) {
        double diff = (double)fp16_to_float(opt_CbCr[i]) - naive_CbCr[i];
        if (diff > 0.5 || diff < -0.5) errors_CbCr++;
    }

    printf("\n--- Verification ---\n");
    printf("Y    : %s\n", errors_Y    == 0 ? "SUCCESS" : "FAILED");
    printf("CbCr : %s\n", errors_CbCr == 0 ? "SUCCESS" : "FAILED");
    if (errors_Y)    printf("  Y errors   : %d\n", errors_Y);
    if (errors_CbCr) printf("  CbCr errors: %d\n", errors_CbCr);
    printf("\n");

    return 0;
}
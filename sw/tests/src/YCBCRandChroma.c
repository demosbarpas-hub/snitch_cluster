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

// NAIVE Stage 1
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
            out_CbCr[g * 8 + l]     = Cb;
            out_CbCr[g * 8 + 4 + l] = Cr;
        }
    }
}

// NAIVE Stage 2
void naive_chroma_420(double *in_cbcr, double *out_cb_f, double *out_cr_f, int w, int h) {
    for (int y = 0; y < h; y += 2) {
        for (int x = 0; x < w; x += 2) {
            double sum_cb = 0, sum_cr = 0;
            int group = (y * w + x) / 4;
            int lane  = (x % 4); 
            sum_cb += in_cbcr[group * 8 + lane] + in_cbcr[group * 8 + lane + 1];
            sum_cb += in_cbcr[(group + w/4) * 8 + lane] + in_cbcr[(group + w/4) * 8 + lane + 1];
            sum_cr += in_cbcr[group * 8 + 4 + lane] + in_cbcr[group * 8 + 4 + lane + 1];
            sum_cr += in_cbcr[(group + w/4) * 8 + 4 + lane] + in_cbcr[(group + w/4) * 8 + 4 + lane + 1];
            int out_idx = (y / 2) * (w / 2) + (x / 2);
            out_cb_f[out_idx] = sum_cb;
            out_cr_f[out_idx] = sum_cr;
        }
    }
}

// OPT Stage 1
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

/* --------------------------------------------------------------------------
 * OPT Stage 2: Chroma Subsampling 4:2:0 (Perfectly Paced FIFO)
 * -------------------------------------------------------------------------- */
// OPT Stage 2: Chroma Subsampling (The Ultimate Hardware-Aligned Fix)
// OPT Stage 2: Chroma Subsampling (Perfectly Synced FIFO)
void opt_chroma_420_2d_ssr(double *in_cbcr, double *out_cb_f, double *out_cr_f) {
    
    // Η FPU τρέχει 64 φορές (32 Cb + 32 Cr)
    int n_fpu = 63; 
    
    // Ο ALU τρέχει 32 φορές (Κάθε iter κάνει 2 pops: Cb και Cr)
    int n_alu = 31; 
    
    double *p_in_top = in_cbcr;
    double *p_in_bot = in_cbcr + 8; // Row 1 είναι 64 bytes μετά
    double *p_cb = out_cb_f;
    double *p_cr = out_cr_f;

    // --- SSR CONFIGURATION ---
    snrt_ssr_loop_2d(SNRT_SSR_DM0, 8, 8, 8, 128); 
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, p_in_top);
    
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 8, 8, 8, 128); 
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, p_in_bot);

    snrt_ssr_enable();

    double zero_val = 0.0;
    double *p_zero = &zero_val;

    asm volatile(
        "fld       f31, 0(%[zero]) \n" // Ασφαλής φόρτωση του 0.0
        
        "csrsi     0x801, 0x1 \n"      // 1. Ενεργοποιούμε το FIFO
        // [ΑΦΑΙΡΕΘΗΚΕ ΤΟ mv t6, x0 ΓΙΑΤΙ ΜΟΛΥΝΕ ΤΟ QUEUE]

        // FPU Loop: ΑΚΡΙΒΩΣ 6 ΕΝΤΟΛΕΣ
        "frep.o    %[n_fpu], 6, 0, 0 \n"  
        "fmv.d     f8,  f0 \n"        // Pop Cb/Cr από Top
        "fmv.d     f9,  f1 \n"        // Pop Cb/Cr από Bot
        "vfadd.h   f10, f8,  f9 \n"   // Κάθετο Sum
        "fmv.d     f16, f31 \n"       // Clean Accumulator
        "vfsum.h   f16, f10 \n"       // Οριζόντιο Sum
        "fmv.x.w   t6, f16 \n"        // Push τα 32 καθαρά bits (2 pixels)

        // ALU Loop: Απευθείας Stores για να σπάσουμε το Forwarding
        "1: \n"
        "sw        t6, 0(%[cb]) \n"   // Pop 1: Γράφει 2 Cb pixels ΑΠΕΥΘΕΙΑΣ από το t6
        "addi      %[cb], %[cb], 4 \n"// Advance 4 bytes
        
        "sw        t6, 0(%[cr]) \n"   // Pop 2: Γράφει 2 Cr pixels ΑΠΕΥΘΕΙΑΣ από το t6
        "addi      %[cr], %[cr], 4 \n"// Advance 4 bytes
        
        "addi      %[alu_c], %[alu_c], -1 \n"
        "bgez      %[alu_c], 1b \n"

        "csrci     0x801, 0x1 \n"     // Απενεργοποιούμε το FIFO
        
        : [cb] "+r"(p_cb), [cr] "+r"(p_cr), [alu_c] "+r"(n_alu)
        : [n_fpu] "r"(n_fpu), [zero] "r"(p_zero)
        : "f8", "f9", "f10", "f16", "f31", "t6", "memory"
    );
    
    snrt_fpu_fence();
    snrt_ssr_disable();
}

int main() {
    if (snrt_cluster_core_idx() != 0 || !snrt_is_compute_core()) return 0;
    uintptr_t base = (uintptr_t)snrt_l1_next();

    double    *lut       = (double *)   ALIGN_UP_TCDM(base);
    uint8_t   *input_idx = (uint8_t *)  ALIGN_UP_TCDM((uintptr_t)lut + 256 * sizeof(double));
    double    *naive_Y    = (double *)   ALIGN_UP_TCDM((uintptr_t)input_idx + TOTAL_PIXELS * 3);
    double    *naive_CbCr = (double *)   ALIGN_UP_TCDM((uintptr_t)naive_Y   + TOTAL_PIXELS * 8);
    double    *naive_cb_f = (double *)   ALIGN_UP_TCDM((uintptr_t)naive_CbCr + TOTAL_PIXELS * 16);
    double    *naive_cr_f = (double *)   ALIGN_UP_TCDM((uintptr_t)naive_cb_f + CHROMA_PIXELS * 8);

    float16_t *opt_Y      = (float16_t *)ALIGN_UP_TCDM((uintptr_t)naive_cr_f + CHROMA_PIXELS * 8);
    float16_t *opt_CbCr   = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_Y      + TOTAL_PIXELS * 2);
    float16_t *opt_cb_f   = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_CbCr   + TOTAL_PIXELS * 4);
    float16_t *opt_cr_f   = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_cb_f   + CHROMA_PIXELS * 2);

    for (int i = 0; i < 256; i++) lut[i] = (double)i;
    for (int i = 0; i < TOTAL_PIXELS * 3; i++) input_idx[i] = (uint8_t)(i % 256);

    snrt_cluster_hw_barrier();

    opt_YCBCR_conversion(input_idx, opt_Y, opt_CbCr, lut, TOTAL_PIXELS);
    naive_YCBCR_conversion(input_idx, naive_Y, naive_CbCr, lut, TOTAL_PIXELS);

    naive_chroma_420(naive_CbCr, naive_cb_f, naive_cr_f, W, H);
    opt_chroma_420_2d_ssr((double*)opt_CbCr, (double*)opt_cb_f, (double*)opt_cr_f);

    // Verification
    int errors_Y = 0, errors_Chroma = 0;
    for (int i = 0; i < TOTAL_PIXELS; i++) {
        if (fp16_to_float(opt_Y[i]) - naive_Y[i] > 0.5 || fp16_to_float(opt_Y[i]) - naive_Y[i] < -0.5) errors_Y++;
    }
    for (int i = 0; i < CHROMA_PIXELS; i++) {
        if (fp16_to_float(opt_cb_f[i]) - naive_cb_f[i] > 0.5 || fp16_to_float(opt_cb_f[i]) - naive_cb_f[i] < -0.5) errors_Chroma++;
        if (fp16_to_float(opt_cr_f[i]) - naive_cr_f[i] > 0.5 || fp16_to_float(opt_cr_f[i]) - naive_cr_f[i] < -0.5) errors_Chroma++;
    }

    printf("\n--- Final Verification ---\n");
    printf("Y (Tiled) : %s\n", errors_Y == 0 ? "SUCCESS" : "FAILED");
    printf("Chroma    : %s (Errors: %d)\n", errors_Chroma == 0 ? "SUCCESS" : "FAILED", errors_Chroma);

    // Detailed Table Print
    printf("\n=========================================================================\n");
    printf("                  DETAILED CHROMA VERIFICATION TABLE                     \n");
    printf("=========================================================================\n");
    printf("Idx (Y, X) | Cb Opt   | Cb Ref   | Cr Opt   | Cr Ref   | Status\n");
    printf("-------------------------------------------------------------------------\n");
    
    for (int i = 0; i < CHROMA_PIXELS; i++) {
        int r = i / 8; 
        int c = i % 8;

        float cb_opt_val = fp16_to_float(opt_cb_f[i]);
        float cb_ref_val = naive_cb_f[i];
        float cr_opt_val = fp16_to_float(opt_cr_f[i]);
        float cr_ref_val = naive_cr_f[i];

        int cb_err = (cb_opt_val - cb_ref_val > 0.5f || cb_opt_val - cb_ref_val < -0.5f);
        int cr_err = (cr_opt_val - cr_ref_val > 0.5f || cr_opt_val - cr_ref_val < -0.5f);

        if (cb_err || cr_err) {
            printf("%3d (%d, %d) | %8.3f | %8.3f | %8.3f | %8.3f | <--- [ERROR] %s%s\n",
                   i, r, c, cb_opt_val, cb_ref_val, cr_opt_val, cr_ref_val,
                   cb_err ? "Cb " : "", cr_err ? "Cr" : "");
        } else {
            printf("%3d (%d, %d) | %8.3f | %8.3f | %8.3f | %8.3f | OK\n",
                   i, r, c, cb_opt_val, cb_ref_val, cr_opt_val, cr_ref_val);
        }
    }
    printf("=========================================================================\n");

    return 0;
}

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

// =========================================================================
// Η Πειραματική 4D SSR Συνάρτησή σου (Διορθωμένη για να κάνει compile)
// =========================================================================
// =========================================================================
// THE ULTIMATE KERNEL: Two-Pass 2D SSR (Respects Depth-1 FIFO)
// =========================================================================
void opt_chroma_420_2d_ssr(double *in_cbcr, double *out_cb_f, double *out_cr_f) {
    
    // Ασφαλής μηδενισμός Accumulator
    double zero_val = 0.0;
    double *p_zero = &zero_val;

    int n_iters_cb = 31; 
    int n_iters_cr = 31;

    // ==========================================
    // PASS 1: Cb ONLY (1 Push / 1 Pop)
    // ==========================================
    // Row 0 ξεκινάει στο 0. Row 1 ξεκινάει 64 bytes μετά (8 doubles).
    double *p_in_top_cb = in_cbcr;
    double *p_in_bot_cb = in_cbcr + 8; 
    double *p_cb = out_cb_f;

    // b0=4 chunks. s0=16 bytes (Ρουφάει 8 bytes Cb, πηδάει 8 bytes Cr!)
    // b1=8 pairs of rows. s1=128 bytes (Πηδάει στο επόμενο ζεύγος γραμμών)
    snrt_ssr_loop_2d(SNRT_SSR_DM0, 4, 8, 16, 128); 
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, p_in_top_cb);
    
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 4, 8, 16, 128); 
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, p_in_bot_cb);

    snrt_ssr_enable();

    asm volatile(
        "fld       f31, 0(%[zero]) \n"
        "csrsi     0x801, 0x1 \n"      // Ενεργοποιούμε το FIFO
        
        // FPU Loop: Υπολογίζει και σπρώχνει ΜΟΝΟ Cb
        "frep.o    %[n], 5, 0, 0 \n"  
        "fmv.d     f8,  f0 \n"        // Pop Cb top
        "fmv.d     f9,  f1 \n"        // Pop Cb bottom
        "vfadd.h   f10, f8,  f9 \n"   // Add
        "fmv.d     f16, f31 \n"       // Clean Acc
        "vfsum.h   f16, f10 \n"       // Sum
        "fmv.x.w   t6, f16 \n"        // <--- ΑΚΡΙΒΩΣ 1 PUSH ΣΤΟ FIFO!
        
        // ALU Loop
        "1: \n"
        "mv        t0, t6 \n"         // <--- ΑΚΡΙΒΩΣ 1 POP ΑΠΟ ΤΟ FIFO!
        "sw        t0, 0(%[cb]) \n"   // Safe store 32-bits (2 pixels)
        "addi      %[cb], %[cb], 4 \n"
        "addi      %[alu_c], %[alu_c], -1 \n"
        "bgez      %[alu_c], 1b \n"

        "csrci     0x801, 0x1 \n"     // Κλείνουμε το FIFO
        
        : [cb] "+r"(p_cb), [alu_c] "+r"(n_iters_cb)
        : [n] "r"(31), [zero] "r"(p_zero)
        : "f8", "f9", "f10", "f16", "f31", "t0", "t6", "memory"
    );
    snrt_fpu_fence();
    snrt_ssr_disable();

    // ==========================================
    // PASS 2: Cr ONLY (1 Push / 1 Pop)
    // ==========================================
    // Το Cr βρίσκεται ακριβώς 1 double (8 bytes) δίπλα στο Cb!
    double *p_in_top_cr = in_cbcr + 1; 
    double *p_in_bot_cr = in_cbcr + 9; 
    double *p_cr = out_cr_f;

    snrt_ssr_loop_2d(SNRT_SSR_DM0, 4, 8, 16, 128); 
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, p_in_top_cr);
    
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 4, 8, 16, 128); 
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, p_in_bot_cr);

    snrt_ssr_enable();

    asm volatile(
        "fld       f31, 0(%[zero]) \n"
        "csrsi     0x801, 0x1 \n"
        
        // FPU Loop: Υπολογίζει και σπρώχνει ΜΟΝΟ Cr
        "frep.o    %[n], 5, 0, 0 \n"  
        "fmv.d     f8,  f0 \n"        
        "fmv.d     f9,  f1 \n"        
        "vfadd.h   f10, f8,  f9 \n"   
        "fmv.d     f16, f31 \n"       
        "vfsum.h   f16, f10 \n"       
        "fmv.x.w   t6, f16 \n"        // <--- ΑΚΡΙΒΩΣ 1 PUSH ΣΤΟ FIFO!
        
        // ALU Loop
        "1: \n"
        "mv        t0, t6 \n"         // <--- ΑΚΡΙΒΩΣ 1 POP ΑΠΟ ΤΟ FIFO!
        "sw        t0, 0(%[cr]) \n"   // Safe store 32-bits (2 pixels)
        "addi      %[cr], %[cr], 4 \n"
        "addi      %[alu_c], %[alu_c], -1 \n"
        "bgez      %[alu_c], 1b \n"

        "csrci     0x801, 0x1 \n"
        
        : [cr] "+r"(p_cr), [alu_c] "+r"(n_iters_cr)
        : [n] "r"(31), [zero] "r"(p_zero)
        : "f8", "f9", "f10", "f16", "f31", "t0", "t6", "memory"
    );
    snrt_fpu_fence();
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
    
    // Οι δείκτες εξόδου για την 4D ρουτίνα σου
    double    *opt_cb_f   = (double *)   ALIGN_UP_TCDM((uintptr_t)opt_CbCr   + TOTAL_PIXELS * 2 * 2 * sizeof(float16_t));
    double    *opt_cr_f   = (double *)   ALIGN_UP_TCDM((uintptr_t)opt_cb_f   + CHROMA_PIXELS * 2 * sizeof(double));

    for (int i = 0; i < 256; i++) lut[i] = (double)i;
    for (int i = 0; i < TOTAL_PIXELS * 3; i++) input_idx[i] = (uint8_t)(i % 256);

    snrt_cluster_hw_barrier();

    // -- Run References --
    naive_YCBCR_conversion(input_idx, naive_Y, naive_CbCr, lut, TOTAL_PIXELS);
    naive_chroma_420(naive_CbCr, naive_cb_f, naive_cr_f, W, H);
    
    // ΠΡΕΠΕΙ να τρέξουμε το Stage 1 Opt για να γεμίσουμε τον πίνακα opt_CbCr!
    opt_YCBCR_conversion(input_idx, opt_Y, opt_CbCr, lut, TOTAL_PIXELS);

    // -- Η ΔΙΚΗ ΣΟΥ ΣΥΝΑΡΤΗΣΗ --
    uint32_t start_cycles = snrt_mcycle();
    opt_chroma_420_2d_ssr((double*)opt_CbCr, opt_cb_f, opt_cr_f);
    uint32_t end_cycles = snrt_mcycle();

    // -- Verification --
    float16_t *cb_fp16_ptr = (float16_t *)opt_cb_f;
    float16_t *cr_fp16_ptr = (float16_t *)opt_cr_f;
    int errors_Cb = 0, errors_Cr = 0;

    printf("\n===================================================\n");
    printf("              CHROMA 4D TEST ERROR LOG             \n");
    printf("===================================================\n");

    for (int i = 0; i < CHROMA_PIXELS; i++) {
        float opt_cb_val = fp16_to_float(cb_fp16_ptr[i]);
        float opt_cr_val = fp16_to_float(cr_fp16_ptr[i]);

        int cb_err = (opt_cb_val - naive_cb_f[i] > 0.5f || opt_cb_val - naive_cb_f[i] < -0.5f);
        int cr_err = (opt_cr_val - naive_cr_f[i] > 0.5f || opt_cr_val - naive_cr_f[i] < -0.5f);

        if (cb_err || cr_err) {
            printf("Mismatch at Idx %d:\n", i);
            if (cb_err) {
                printf("  -> Cb : Opt = %8.3f | Ref = %8.3f\n", opt_cb_val, naive_cb_f[i]);
                errors_Cb++;
            }
            if (cr_err) {
                printf("  -> Cr : Opt = %8.3f | Ref = %8.3f\n", opt_cr_val, naive_cr_f[i]);
                errors_Cr++;
            }
        }
    }

    if (errors_Cb == 0 && errors_Cr == 0) {
        printf("  [SUCCESS] 4D SSR WORKS! 100%% Match.\n");
    }

    printf("\n[PERFORMANCE]\n");
    printf("  - Cycles       : %u\n", end_cycles - start_cycles);
    printf("  - Errors Cb    : %d / %d\n", errors_Cb, CHROMA_PIXELS);
    printf("  - Errors Cr    : %d / %d\n", errors_Cr, CHROMA_PIXELS);
    printf("===================================================\n\n");

    return 0;
}

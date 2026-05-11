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

// ========================================================================
// 1. NAIVE C IMPLEMENTATIONS
// ========================================================================

/**
 * Μετατροπή RGB σε YCbCr (Planar) χρησιμοποιώντας standard C loops.
 */
void naive_YCBCR_conversion(uint8_t *input_idx, double *out_Y, double *out_CbCr, double *lut, int total_pixels) {
    for (int i = 0; i < total_pixels; i++) {
        double R = lut[input_idx[i * 3 + 0]];
        double G = lut[input_idx[i * 3 + 1]];
        double B = lut[input_idx[i * 3 + 2]];

        double Y  = 0.299 * R + 0.587 * G + 0.114 * B;
        double Cb = 0.564 * (B - Y);
        double Cr = 0.713 * (R - Y);

        out_Y[i] = Y;
        out_CbCr[i * 2 + 0] = Cb;
        out_CbCr[i * 2 + 1] = Cr;
    }
}

/**
 * Chroma Subsampling 4:2:0 χρησιμοποιώντας μέσο όρο 2x2 παραθύρου.
 */
void naive_chroma_420(double *in_cbcr, double *out_cbcr_final, int w, int h) {
    for (int y = 0; y < h; y += 2) {
        for (int x = 0; x < w; x += 2) {
            double sum_cb = 0, sum_cr = 0;
            
            // Μέσος όρος 2x2 τετραγώνου
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int in_idx = ((y + dy) * w + (x + dx)) * 2;
                    sum_cb += in_cbcr[in_idx + 0];
                    sum_cr += in_cbcr[in_idx + 1];
                }
            }
            
            int out_idx = ((y / 2) * (w / 2) + (x / 2)) * 2;
            out_cbcr_final[out_idx + 0] = sum_cb * 0.25;
            out_cbcr_final[out_idx + 1] = sum_cr * 0.25;
        }
    }
}

// ========================================================================
// 2. OPTIMIZED ASSEMBLY (Dual Output Channels: DM1 & DM2)
// ========================================================================

/**
 * Βελτιστοποιημένη μετατροπή YCbCr με χρήση SSR και frep.
 */
void opt_YCBCR_conversion(uint8_t *input_idx, double *out_Y, double *out_CbCr, double *lut, int total_pixels) {
    register double r_c_r  asm("ft3") = 0.299;
    register double r_c_g  asm("ft4") = 0.587;
    register double r_c_b  asm("ft5") = 0.114;
    register double r_c_cb asm("ft6") = 0.564;
    register double r_c_cr asm("ft7") = 0.713;

    snrt_issr_read(SNRT_SSR_DM0, lut, input_idx, total_pixels * 3, SNRT_SSR_IDXSIZE_U8);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, total_pixels, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, out_Y);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, total_pixels * 2, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, out_CbCr);
    snrt_ssr_enable();

    asm volatile(
        "frep.o %[n], 22, 0, 0            \n"

        // Load R1,G1,B1,R2,G2,B2 in stream order
        "fmv.d ft8,  ft0                  \n"  // R1
        "fmv.d ft9,  ft0                  \n"  // G1
        "fmv.d ft10, ft0                  \n"  // B1
        "fmv.d ft11, ft0                  \n"  // R2
        "fmv.d f8,   ft0                  \n"  // G2
        "fmv.d f9,   ft0                  \n"  // B2

        "fmul.d f10, %[c_r], ft8          \n"  // Y1 = c_r*R1
        "fmul.d f11, %[c_r], ft11         \n"  // Y2 = c_r*R2

        "fmadd.d f10, %[c_g], ft9,  f10   \n"  // Y1 += c_g*G1
        "fmadd.d f11, %[c_g], f8,   f11   \n"  // Y2 += c_g*G2

        "fmadd.d f10, %[c_b], ft10, f10   \n"  // Y1 done
        "fmadd.d f11, %[c_b], f9,   f11   \n"  // Y2 done

        "fmv.d ft1, f10                   \n"  // Y1 -> out_Y
        "fmv.d ft1, f11                   \n"  // Y2 -> out_Y

        "fsub.d ft10, ft10, f10           \n"  // B1-Y1
        "fsub.d f9,   f9,   f11           \n"  // B2-Y2
        "fsub.d ft8,  ft8,  f10           \n"  // R1-Y1
        "fsub.d ft11, ft11, f11           \n"  // R2-Y2

        "fmul.d ft2, %[c_cb], ft10        \n"  // Cb1
        "fmul.d ft2, %[c_cr], ft8         \n"  // Cr1
        "fmul.d ft2, %[c_cb], f9          \n"  // Cb2
        "fmul.d ft2, %[c_cr], ft11        \n"  // Cr2

        :
        : [n]    "r" (total_pixels / 2 - 1),
          [c_r]  "f" (r_c_r),  [c_g] "f" (r_c_g), [c_b] "f" (r_c_b),
          [c_cb] "f" (r_c_cb), [c_cr] "f" (r_c_cr)
        : "ft0", "ft1", "ft2",
          "ft8", "ft9", "ft10", "ft11",
          "f8",  "f9",  "f10", "f11",
          "memory"
    );

    snrt_fpu_fence();
    snrt_ssr_disable();

}

/**
 * Βελτιστοποιημένο 2D Chroma Subsampling με SSR 2D loops.
 */
void opt_chroma_420_2d_ssr(double *in_cbcr, double *out_cbcr_final) {
    register double quarter asm("ft11") = 0.25;   

    snrt_ssr_loop_4d(SNRT_SSR_DM0, 4,2,8,8,8,32*8,32,32*8*2);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, in_cbcr);

    // snrt_ssr_loop_2d(SNRT_SSR_DM2, b0, b1, s0, s1);
    // snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, &in_cbcr[W * 2]);

    snrt_ssr_loop_1d(SNRT_SSR_DM1, (W/2) * (H/2) * 2, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, out_cbcr_final);

    snrt_cluster_hw_barrier();
    snrt_ssr_enable();

#ifdef SNRT_SUPPORTS_FREP
    asm volatile(
        "frep.o %[total_blocks], 16, 0, 0 \n"  

        // --- STAGE 1: FETCH ---
        "fmv.d  ft8,  ft0 \n"         // TL_Cb 
        "fmv.d  ft9,  ft0 \n"         // TL_Cr 
        "fmv.d  ft3,  ft0 \n"         // TR_Cb 
        "fmv.d  ft10, ft0 \n"         // TR_Cr 
        "fmv.d  ft4,  ft0 \n"         // BL_Cb 
        "fmv.d  ft7,  ft0 \n"         // BL_Cr 
        "fmv.d  ft5,  ft0 \n"         // BR_Cb 
        "fmv.d  ft6,  ft0 \n"         // BR_Cr 

        // --- STAGE 2: TREE REDUCTION (No Stalls) ---
        "fadd.d ft8, ft8, ft3 \n"     // Top_Cb = TL_Cb + TR_Cb
        "fadd.d ft9, ft9, ft10 \n"    // Top_Cr = TL_Cr + TR_Cr
        
        "fadd.d ft4, ft4, ft5 \n"     // Bot_Cb = BL_Cb + BR_Cb
        "fadd.d ft7, ft7, ft6 \n"     // Bot_Cr = BL_Cr + BR_Cr
        
        "fadd.d ft8, ft8, ft4 \n"     // sum_Cb = Top_Cb + Bot_Cb
        "fadd.d ft9, ft9, ft7 \n"     // sum_Cr = Top_Cr + Bot_Cr

        // --- STAGE 3: WRITE OUT ---
        "fmul.d ft1, ft8, %[q] \n"    // Write Cb_avg 
        "fmul.d ft1, ft9, %[q] \n"    // Write Cr_avg 

        : 
        : [total_blocks] "r" ((W/2 * H/2) - 1), [q] "f" (quarter)
        : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "memory"
    );
#endif

    snrt_fpu_fence();
    snrt_ssr_disable();
}

// ========================================================================
// 3. MAIN (BENCHMARKING & VERIFICATION)
// ========================================================================

int main() {
    // Εκτέλεση μόνο από τον Core 0 του Cluster
    if (snrt_cluster_core_idx() != 0 || !snrt_is_compute_core()) {
        return 0;
    }

    // --- Δυναμική Δέσμευση & Ευθυγράμμιση στην L1 TCDM ---
    uintptr_t base = (uintptr_t)snrt_l1_next();

    double *lut = (double *)ALIGN_UP_TCDM(base);
    uint8_t *input_idx = (uint8_t *)ALIGN_UP_TCDM((uintptr_t)lut + 256 * sizeof(double));
    
    // Χώροι Μνήμης για Naive Implementation
    double *naive_Y = (double *)ALIGN_UP_TCDM((uintptr_t)input_idx + TOTAL_PIXELS * 3);
    double *naive_CbCr = (double *)ALIGN_UP_TCDM((uintptr_t)naive_Y + TOTAL_PIXELS * sizeof(double));
    double *naive_chroma_final = (double *)ALIGN_UP_TCDM((uintptr_t)naive_CbCr + TOTAL_PIXELS * 2 * sizeof(double));
    
    // Χώροι Μνήμης για Optimized Implementation
    double *opt_Y = (double *)ALIGN_UP_TCDM((uintptr_t)naive_chroma_final + (TOTAL_PIXELS / 4) * 2 * sizeof(double));
    double *opt_CbCr = (double *)ALIGN_UP_TCDM((uintptr_t)opt_Y + TOTAL_PIXELS * sizeof(double));
    double *opt_chroma_final = (double *)ALIGN_UP_TCDM((uintptr_t)opt_CbCr + TOTAL_PIXELS * 2 * sizeof(double));
	
    // --- Setup Αρχικών Δεδομένων ---
    for (int i = 0; i < 256; i++) lut[i] = (double)i;
    for (int i = 0; i < TOTAL_PIXELS * 3; i++) input_idx[i] = (uint8_t)(i % 256);

    snrt_cluster_hw_barrier();

    // --- BENCHMARK 1: COLOR CONVERSION (RGB -> Planar YCbCr) ---
    uint32_t start_naive_cc = snrt_mcycle();
    naive_YCBCR_conversion(input_idx, naive_Y, naive_CbCr, lut, TOTAL_PIXELS);
    uint32_t end_naive_cc = snrt_mcycle();

	uint32_t start_opt_cc = snrt_mcycle();
	opt_YCBCR_conversion(input_idx, opt_Y, opt_CbCr, lut,  TOTAL_PIXELS);
	uint32_t end_opt_cc = snrt_mcycle();

    // --- BENCHMARK 2: CHROMA SUBSAMPLING (4:2:0) ---
    uint32_t start_naive_sub = snrt_mcycle();
    naive_chroma_420(naive_CbCr, naive_chroma_final, W, H);
    uint32_t end_naive_sub = snrt_mcycle();

    uint32_t start_opt_sub = snrt_mcycle();
    opt_chroma_420_2d_ssr(opt_CbCr, opt_chroma_final);
    uint32_t end_opt_sub = snrt_mcycle();

    // --- ΕΚΤΥΠΩΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ---
    printf("\n--- STAGE 1: Planar Color Conversion ---\n");
    printf("Naive Cycles : %u\n", end_naive_cc - start_naive_cc);
    printf("Opt Cycles   : %u\n", end_opt_cc - start_opt_cc);
    printf("Speedup      : %.2fx\n", (double)(end_naive_cc - start_naive_cc) / (end_opt_cc - start_opt_cc));

    printf("\n--- STAGE 2: 4:2:0 Chroma Subsampling ---\n");
    printf("Naive Cycles : %u\n", end_naive_sub - start_naive_sub);
    printf("Opt Cycles   : %u\n", end_opt_sub - start_opt_sub);
    printf("Speedup      : %.2fx\n", (double)(end_naive_sub - start_naive_sub) / (end_opt_sub - start_opt_sub));

    // --- ΕΠΑΛΗΘΕΥΣΗ (VERIFICATION) ---
    int errors_cc = 0;
    for (int i = 0; i < TOTAL_PIXELS; i++) {
        if ((opt_Y[i] - naive_Y[i]) > 1e-5 || (opt_Y[i] - naive_Y[i]) < -1e-5) errors_cc++;
    }
    
    int errors_sub = 0;
    for (int i = 0; i < (TOTAL_PIXELS / 4) * 2; i++) {
        if ((opt_chroma_final[i] - naive_chroma_final[i]) > 1e-5 || (opt_chroma_final[i] - naive_chroma_final[i]) < -1e-5) {
            errors_sub++;
        }
    }

    printf("\n--- Verification ---\n");
    if (errors_cc == 0 && errors_sub == 0) {
        printf("SUCCESS! Pipeline matches Naive perfectly.\n\n");
    } else {
        printf("FAILED! Errors in CC: %d, Errors in Sub: %d\n\n", errors_cc, errors_sub);
    }

    return 0;
}
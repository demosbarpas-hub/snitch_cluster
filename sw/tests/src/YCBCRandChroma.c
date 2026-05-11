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

// NAIVE: αρχική double έκδοση (γρήγορη, ακριβής)
void naive_YCBCR_conversion(uint8_t *input_idx, double *out_Y, double *out_CbCr, double *lut, int total_pixels) {
    for (int i = 0; i < total_pixels; i++) {
        double R = lut[input_idx[i * 3 + 0]];
        double G = lut[input_idx[i * 3 + 1]];
        double B = lut[input_idx[i * 3 + 2]];

        double Y  = 0.299 * R + 0.587 * G + 0.114 * B;
        double Cb = 0.564 * (B - Y);
        double Cr = 0.713 * (R - Y);

        out_Y[i]            = Y;
        out_CbCr[i * 2 + 0] = Cb;
        out_CbCr[i * 2 + 1] = Cr;
    }
}
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
// OPT: FP16 vector έκδοση
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
void opt_chroma_420_2d_ssr(float16_t *in_cbcr, float16_t *out_cb_f, float16_t *out_cr_f) {
    // n_iters = 31 γιατί έχουμε 32 ζεύγη (Cb + Cr) για 16x16 εικόνα
    int n_fpu = 63;   // 64 συνολικά pushes στο FIFO (32 Cb + 32 Cr)
    int n_alu = 31;   // 32 επαναλήψεις στο ALU (κάθε μία κάνει pop 1 Cb και 1 Cr)
    
    float scale = 0.25f;
    asm volatile("fcvt.h.s f3, %0" :: "f"(scale) : "f3");

    // --- CONFIG SSR ---
    // DM0: Top Row. s0=8 (Cb->Cr), s1=16 (Group->Group), s2=128 (Skip 1 row)
    snrt_ssr_loop_3d(SNRT_SSR_DM0, 2, 2, 8, 8, 16, 128); 
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_3D, in_cbcr);

    // DM1: Bottom Row. Ίδια strides, αλλά ξεκινάει 64 bytes μετά
    snrt_ssr_loop_3d(SNRT_SSR_DM1, 2, 2, 8, 8, 16, 128); 
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_3D, &in_cbcr[32]); // 32 float16 = 64 bytes

    snrt_ssr_enable();

    // --- THE FINAL KERNEL ---
    asm volatile(
        "csrsi 0x801, 0x1 \n"      
        "mv    t6, x0 \n"          

        // FPU Loop: Επεξεργάζεται 4 pixels τη φορά (64-bit chunks)
        "frep.o %[n_fpu], 6, 0, 0 \n"  
        "fmv.d     f8,  f0 \n"        // Pop Top (Cb ή Cr)
        "fmv.d     f9,  f1 \n"        // Pop Bottom (Cb ή Cr)
        "vfadd.h   f10, f8, f9 \n"    // Vertical Sum
        "vfsum.h   f16, f10 \n"       // Horizontal Sum
        "vfmul.r.h f16, f16, f3 \n"    // Average
        "fmv.x.w   t6, f16 \n"         // Push 32-bit (2x Avg) to FIFO

        // ALU Loop: Ξεχωρίζει Cb από Cr
        "1: \n"
        "mv        t0, t6 \n"          // Pop Cb result
        "sw        t0, 0(%[cb]) \n"    // Store 2x Cb
        "mv        t1, t6 \n"          // Pop Cr result
        "sw        t1, 0(%[cr]) \n"    // Store 2x Cr
        "addi      %[cb], %[cb], 4 \n"
        "addi      %[cr], %[cr], 4 \n"
        "addi      %[alu_c], %[alu_c], -1 \n"
        "bgez      %[alu_c], 1b \n"

        "csrci 0x801, 0x1 \n"          
        
        : [cb] "+r"(out_cb_f), [cr] "+r"(out_cr_f), [alu_c] "+r"(n_alu)
        : [n_fpu] "r"(n_fpu)
        : "f3", "f8", "f9", "f10", "f16", "t0", "t1", "t6", "memory"
    );

    snrt_fpu_fence();
    snrt_ssr_disable();
}
int main() {
    if (snrt_cluster_core_idx() != 0 || !snrt_is_compute_core()) return 0;

    uintptr_t base = (uintptr_t)snrt_l1_next();

    double    *lut       = (double *)   ALIGN_UP_TCDM(base);
    uint8_t   *input_idx = (uint8_t *)  ALIGN_UP_TCDM((uintptr_t)lut + 256 * sizeof(double));

    // Naive: double outputs
    double    *naive_Y    = (double *)   ALIGN_UP_TCDM((uintptr_t)input_idx + TOTAL_PIXELS * 3);
    double    *naive_CbCr = (double *)   ALIGN_UP_TCDM((uintptr_t)naive_Y   + TOTAL_PIXELS * sizeof(double));

    // Opt: FP16 outputs
    float16_t *opt_Y      = (float16_t *)ALIGN_UP_TCDM((uintptr_t)naive_CbCr + TOTAL_PIXELS * 2 * sizeof(double));
    float16_t *opt_CbCr   = (float16_t *)ALIGN_UP_TCDM((uintptr_t)opt_Y      + TOTAL_PIXELS * sizeof(float16_t));

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
    printf("Speedup      : %.2fx\n", (double)(end_naive_cc - start_naive_cc) / (end_opt_cc - start_opt_cc));

    printf("\n--- Opt: packed FP16 layout ανά iteration (4 pixels/iter) ---\n");
    for (int iter = 0; iter < 3; iter++) {
        printf("Iter %d:\n", iter);
        printf("  Y  packed = {%.3f, %.3f, %.3f, %.3f}\n",
            fp16_to_float(opt_Y[iter*4 + 0]),
            fp16_to_float(opt_Y[iter*4 + 1]),
            fp16_to_float(opt_Y[iter*4 + 2]),
            fp16_to_float(opt_Y[iter*4 + 3]));
        printf("  Cb packed = {%.3f, %.3f, %.3f, %.3f}\n",
            fp16_to_float(opt_CbCr[iter*8 + 0]),
            fp16_to_float(opt_CbCr[iter*8 + 1]),
            fp16_to_float(opt_CbCr[iter*8 + 2]),
            fp16_to_float(opt_CbCr[iter*8 + 3]));
        printf("  Cr packed = {%.3f, %.3f, %.3f, %.3f}\n",
            fp16_to_float(opt_CbCr[iter*8 + 4]),
            fp16_to_float(opt_CbCr[iter*8 + 5]),
            fp16_to_float(opt_CbCr[iter*8 + 6]),
            fp16_to_float(opt_CbCr[iter*8 + 7]));
    }

    printf("\n--- Naive: ανά pixel για σύγκριση ---\n");
    for (int i = 0; i < 8; i++) {
        printf("Pixel %d: Y=%.3f  Cb=%.3f  Cr=%.3f\n",
            i,
            naive_Y[i],
            naive_CbCr[i*2 + 0],
            naive_CbCr[i*2 + 1]);
    }
    printf("\n");

    // --- Verification: μετατροπή FP16 → double για σύγκριση ---
    int errors_cc = 0;
    for (int i = 0; i < TOTAL_PIXELS; i++) {
        double opt_y_as_double = (double)fp16_to_float(opt_Y[i]);
        double diff = opt_y_as_double - naive_Y[i];
        if (diff > 0.5 || diff < -0.5) errors_cc++;
    }

    printf("\n--- Verification ---\n");
    if (errors_cc == 0)
        printf("SUCCESS! Pipeline matches Naive (within FP16 tolerance).\n\n");
    else
        printf("FAILED! Errors in CC: %d\n", errors_cc);

    return 0;
}
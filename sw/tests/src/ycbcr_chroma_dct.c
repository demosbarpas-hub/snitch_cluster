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

/* =========================================================================
   ΣΤΑΘΕΡΟΙ ΠΙΝΑΚΕΣ DCT
========================================================================= */
const uint64_t dct_m[16] __attribute__((aligned(64))) = {
    0x35a835a835a835a8, 0x35a835a835a835a8, 0x2e3f34e93aa73be2, 0xbbe2baa7b4e9ae3f, 
    0xbaabb21632163b64, 0x3b643216b216baab, 0xb8e3bc53b2403aa7, 0xbaa732403c5338e3, 
    0x35a8b5a8b5a835a8, 0x35a8b5a8b5a835a8, 0x3aa73240bbe234e9, 0xb4e93be2b240baa7, 
    0xb2163696b6963216, 0x3216b6963696b216, 0xbbe23b64baa72e3f, 0xae3f3aa7bb643be2  
};

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

// =========================================================================
// STAGE 1: YCBCR CONVERSION
// =========================================================================
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
            out_CbCr[g * 8 + l]      = Cb;
            out_CbCr[g * 8 + 4 + l] = Cr;
        }
    }
}

void opt_YCBCR_conversion(uint8_t *input_idx, float16_t *out_Y, float16_t *out_CbCr, double *lut, int total_pixels) {
    float c_r_f  = 0.299f;  float c_g_f  = 0.587f;  float c_b_f  = 0.114f;
    float c_cb_f = 0.141f;  float c_cr_f = 0.17825f;  

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

        // FETCH R1, G1, B1 (Lower halves)
        "fmv.d f8,  f0 \n"
        "fmv.d f9,  f0 \n"
        "fmv.d f10, f0 \n"
        "vfcpka.h.d f8,  f8,  f0 \n"
        "vfcpka.h.d f9,  f9,  f0 \n"
        "vfcpka.h.d f10, f10, f0 \n"

        // FETCH R2, G2, B2 (Upper halves prep)
        "fmv.d f14, f0 \n"
        "fmv.d f15, f0 \n"
        "fmv.d f16, f0 \n"

        // --- STALL-FREE INTERLEAVING ---
        "vfcpkb.h.d f8,  f14, f0 \n"  // Το f8 (R) είναι έτοιμο!
        "vfmul.r.h  f11, f8,  f3 \n"  // START MATH: Υπολογισμός R * c_r

        "vfcpkb.h.d f9,  f15, f0 \n"  // Το f9 (G) είναι έτοιμο!
        "vfmac.r.h  f11, f9,  f4 \n"  // MATH: Προσθήκη G * c_g (ενώ η FPU δούλευε το R)

        "vfcpkb.h.d f10, f16, f0 \n"  // Το f10 (B) είναι έτοιμο!
        "vfmac.r.h  f11, f10, f5 \n"  // MATH: Προσθήκη B * c_b (Το Y ολοκληρώθηκε)

        // Write Y & Compute Differences
        "fmv.d      f1,  f11 \n"      // Αποστολή Y στον SSR (f1)
        "vfsub.h    f10, f10, f11 \n" // B - Y
        "vfsub.h    f8,  f8,  f11 \n" // R - Y

        // Final Cb/Cr writes
        "vfmul.r.h  f2,  f10, f6 \n"  // Αποστολή Cb στον SSR (f2)
        "vfmul.r.h  f2,  f8,  f7 \n"  // Αποστολή Cr στον SSR (f2)
        :
        : [n] "r"(total_pixels / 4 - 1)
        : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
          "f8", "f9", "f10", "f11", "f14", "f15", "f16", "memory"
    );
    snrt_fpu_fence();
    snrt_ssr_disable();
}

// =========================================================================
// STAGE 2: CHROMA SUBSAMPLING
// =========================================================================
void naive_chroma_420(double *in_cbcr, double *out_cb_f, double *out_cr_f, int w, int h) {
    for (int y = 0; y < h; y += 2) {
        for (int x = 0; x < w; x += 2) {
            double sum_cb = 0.0; double sum_cr = 0.0;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int pixel_idx = (y + dy) * w + (x + dx);
                    int group = pixel_idx / 4; int lane  = pixel_idx % 4;
                    sum_cb += in_cbcr[group * 8 + lane];
                    sum_cr += in_cbcr[group * 8 + 4 + lane];
                }
            }
            int out_idx = (y / 2) * (w / 2) + (x / 2);
            out_cb_f[out_idx] = sum_cb;
            out_cr_f[out_idx] = sum_cr;
        }
    }
}

void opt_chroma_420_2d_ssr(double *in_cbcr, float16_t *out_cb_f, float16_t *out_cr_f) {
    double zero_val = 0.0;
    double *p_zero = &zero_val;
    uintptr_t base_addr = (uintptr_t)in_cbcr;

    void *p_top = (void*)(base_addr + 0);
    void *p_bot = (void*)(base_addr + 64);

    snrt_ssr_loop_2d(SNRT_SSR_DM0, 8, 8, 8, 128); 
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, p_top);
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 8, 8, 8, 128); 
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, p_bot);
    snrt_ssr_enable();
    
    uint32_t alu_cb1, alu_cr1, alu_cb2, alu_cr2;
    uint32_t iters = 15;

    asm volatile(
        "fld      f31, 0(%[zero]) \n"
        "csrsi    0x801, 0x1 \n"      
        "1: \n"
        "fmv.d    f4, f0 \n"  "fmv.d    f5, f1 \n"  
        "fmv.d    f8, f0 \n"  "fmv.d    f9, f1 \n"  
        "fmv.d    f12, f0 \n" "fmv.d    f13, f1 \n" 
        "fmv.d    f16, f0 \n" "fmv.d    f17, f1 \n" 
        "vfadd.h  f6,  f4,  f5 \n" "vfadd.h  f10, f8,  f9 \n"
        "vfadd.h  f14, f12, f13 \n" "vfadd.h  f18, f16, f17 \n"
        "fmv.d    f7,  f31 \n" "fmv.d    f11, f31 \n"
        "fmv.d    f15, f31 \n" "fmv.d    f19, f31 \n"
        "vfsum.h  f7,  f6 \n" "vfsum.h  f11, f10 \n"
        "vfsum.h  f15, f14 \n" "vfsum.h  f19, f18 \n"
        "fmv.x.w  %[alu_cb1], f7 \n" "fmv.x.w  %[alu_cr1], f11 \n"
        "fmv.x.w  %[alu_cb2], f15 \n" "fmv.x.w  %[alu_cr2], f19 \n"
        "sw       %[alu_cb1], 0(%[out_cb]) \n" "sw       %[alu_cb2], 4(%[out_cb]) \n"
        "sw       %[alu_cr1], 0(%[out_cr]) \n" "sw       %[alu_cr2], 4(%[out_cr]) \n"
        "addi     %[out_cb], %[out_cb], 8 \n"
        "addi     %[out_cr], %[out_cr], 8 \n"
        "addi     %[iters], %[iters], -1 \n"
        "bgez     %[iters], 1b \n"
        "csrci    0x801, 0x1 \n"
        : [out_cb] "+r"(out_cb_f), [out_cr] "+r"(out_cr_f),
          [alu_cb1] "=&r"(alu_cb1), [alu_cr1] "=&r"(alu_cr1), [alu_cb2] "=&r"(alu_cb2), [alu_cr2] "=&r"(alu_cr2),
          [iters] "+r"(iters)
        : [zero] "r"(p_zero) 
        : "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f31", "memory"
    );
    snrt_ssr_disable();
}

// =========================================================================
// STAGE 3: 2D DCT
// =========================================================================
void naive_dct_pass1(double *in, double *out, const float16_t *dct_m_fp16) {
    static double s[8][8], tmp_mat[8][8];
    for (int i = 0; i < 64; i++) s[i/8][i%8] = in[i] - 128.0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            double sum = 0;
            for (int k = 0; k < 8; k++) sum += s[r][k] * (double)fp16_to_float(dct_m_fp16[c*8+k]);
            tmp_mat[r][c] = sum;
        }
    }
    for (int i = 0; i < 64; i++) out[i] = tmp_mat[i/8][i%8];
}

void naive_dct_pass2(double *pass1_out, double *final_out, const float16_t *dct_m_fp16) {
    static double tmp_mat[8][8];
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            double sum = 0;
            for (int k = 0; k < 8; k++) sum += pass1_out[k*8+c] * (double)fp16_to_float(dct_m_fp16[r*8+k]);
            tmp_mat[r][c] = sum;
        }
    }
    for (int i = 0; i < 64; i++) final_out[i] = tmp_mat[i/8][i%8];
}

void simd_dct_pass1(float16_t *in, float16_t *out, float16_t *tmp, const float16_t *dct_m_param) {
    float16_t off16 = float_to_fp16(-128.0f);
    uint64_t off_vec __attribute__((aligned(8))) =
        ((uint64_t)off16 << 48) | ((uint64_t)off16 << 32) | ((uint64_t)off16 << 16) | off16;
    
    // Level Shift
    snrt_ssr_loop_1d(SNRT_SSR_DM0, 16, 8);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)in);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 16, 8);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)out);
    snrt_ssr_enable();
    asm volatile(
        "fld     f10, 0(%0) \n" "frep.o  %[n], 1, 0, 0 \n" "vfadd.h f1, f0, f10 \n"
        :: "r"(&off_vec), [n]"r"(15) : "f1", "f0", "f10", "memory"
    );
    snrt_fpu_fence(); snrt_ssr_disable();

    // Multiply & Reduce
    snrt_ssr_loop_3d(SNRT_SSR_DM0, 2, 8, 8, 8, 0, 16);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_3D, (void*)out);
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 16, 8, 8, 0);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, (void*)dct_m_param);
    snrt_ssr_enable();
    
    double zero_val = 0.0; double *p_zero = &zero_val;
    uint32_t alu_f0, alu_f1, alu_f2, alu_f3; uint32_t alu_idx = 16;
    asm volatile(
        "fld     f31, 0(%[zero]) \n"  "csrsi   0x801, 0x1 \n"  
        "1: \n"
        "vfmul.h  f4, f0, f1 \n" "vfmul.h  f5, f0, f1 \n"  
        "vfmul.h  f8, f0, f1 \n" "vfmul.h  f9, f0, f1 \n"  
        "vfmul.h  f12, f0, f1 \n" "vfmul.h  f13, f0, f1 \n"  
        "vfmul.h  f16, f0, f1 \n" "vfmul.h  f17, f0, f1 \n"  
        "vfadd.h  f5, f5, f4 \n" "vfadd.h  f9, f9, f8 \n"  
        "vfadd.h  f13, f13, f12 \n" "vfadd.h  f17, f17, f16 \n"  
        "fmv.d    f4, f31 \n"  "fmv.d    f8, f31 \n"  
        "fmv.d    f12, f31 \n"  "fmv.d    f16, f31 \n"
        "vfsum.h  f4, f5 \n" "vfsum.h  f8, f9 \n"  
        "vfsum.h  f12, f13 \n" "vfsum.h  f16, f17 \n"  
        "fmv.d    f5, f31 \n" "fmv.d    f9, f31 \n"
        "fmv.d    f13, f31 \n" "fmv.d    f17, f31 \n"
        "vfsum.h  f5, f4 \n" "vfsum.h  f9, f8 \n"  
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

void simd_dct_pass2(float16_t *in, float16_t *out, uint64_t *mat_bcast) {
    snrt_ssr_loop_2d(SNRT_SSR_DM0, 2, 64, 0, 8);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, mat_bcast);
    
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 16, 8, 8, 0); 
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, in);
    
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 16, 8);
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, out);
    snrt_ssr_enable();

    uint32_t iters = 8;
    asm volatile(
        "1: \n"
        "vfmul.h f4, f0, f1 \n" "vfmul.h f5, f0, f1 \n" "vfmul.h f6, f0, f1 \n" "vfmul.h f7, f0, f1 \n"
        "vfmul.h f8, f0, f1 \n" "vfmul.h f9, f0, f1 \n" "vfmul.h f10, f0, f1 \n" "vfmul.h f11, f0, f1 \n"
        "vfmul.h f12, f0, f1 \n" "vfmul.h f13, f0, f1 \n" "vfmul.h f14, f0, f1 \n" "vfmul.h f15, f0, f1 \n"
        "vfmul.h f16, f0, f1 \n" "vfmul.h f17, f0, f1 \n" "vfmul.h f18, f0, f1 \n" "vfmul.h f19, f0, f1 \n"

        "vfadd.h f4, f4, f6 \n" "vfadd.h f5, f5, f7 \n" "vfadd.h f8, f8, f10 \n" "vfadd.h f9, f9, f11 \n"
        "vfadd.h f12, f12, f14 \n" "vfadd.h f13, f13, f15 \n" "vfadd.h f16, f16, f18 \n" "vfadd.h f17, f17, f19 \n"

        "vfadd.h f4, f4, f8 \n" "vfadd.h f5, f5, f9 \n" "vfadd.h f12, f12, f16 \n" "vfadd.h f13, f13, f17 \n"
        "vfadd.h f4, f4, f12 \n" "vfadd.h f5, f5, f13 \n"

        "fmv.d f2, f4 \n" "fmv.d f2, f5 \n"
        "addi %[iters], %[iters], -1 \n" "bnez %[iters], 1b \n"
        : [iters] "+r"(iters) :: "f4","f5","f6","f7","f8","f9","f10","f11","f12","f13","f14","f15","f16","f17","f18","f19","memory"
    );
    snrt_ssr_disable();
}

// =========================================================================
// MAIN ROUTINE (FULL YCBCR -> CHROMA -> DCT PIPELINE)
// =========================================================================
int main() {
    uint32_t core_id = snrt_cluster_core_idx();
    uintptr_t base = ALIGN_UP_TCDM((uintptr_t)snrt_l1_next());

    // --- ΜΝΗΜΗ ΕΙΣΟΔΟΥ/ΕΞΟΔΟΥ ΓΙΑ YCbCr & CHROMA ---
    double    *lut         = (double *)   base; base += 256 * sizeof(double);
    uint8_t   *input_idx   = (uint8_t *)  base; base += TOTAL_PIXELS * 3 * sizeof(uint8_t);
    
    double    *naive_Y     = (double *)   base; base += TOTAL_PIXELS * sizeof(double);
    
    // Η ΜΑΓΙΚΗ ΔΙΟΡΘΩΣΗ (* 2) ΓΙΑ ΝΑ ΣΤΑΜΑΤΗΣΕΙ ΤΟ MEMORY BLEEDING
    double    *naive_CbCr  = (double *)   base; base += TOTAL_PIXELS * 2 * sizeof(double);
    double    *naive_cb_f  = (double *)   base; base += CHROMA_PIXELS * sizeof(double);
    double    *naive_cr_f  = (double *)   base; base += CHROMA_PIXELS * sizeof(double);
    
    float16_t *opt_Y       = (float16_t *)base; base += TOTAL_PIXELS * sizeof(float16_t);
    
    // Η ΜΑΓΙΚΗ ΔΙΟΡΘΩΣΗ (* 2) ΓΙΑ ΝΑ ΣΤΑΜΑΤΗΣΕΙ ΤΟ MEMORY BLEEDING
    float16_t *opt_CbCr    = (float16_t *)base; base += TOTAL_PIXELS * 2 * sizeof(float16_t);
    float16_t *opt_cb_f    = (float16_t *)base; base += CHROMA_PIXELS * sizeof(float16_t);
    float16_t *opt_cr_f    = (float16_t *)base; base += CHROMA_PIXELS * sizeof(float16_t);

    // --- ΜΝΗΜΗ ΓΙΑ DCT (6 ΜΠΛΟΚ: 4xY, 1xCb, 1xCr) ---
    double    *naive_tmp_p1= (double *)   base; base += 64 * sizeof(double); 
    double    *naive_dct_Y = (double *)   base; base += TOTAL_PIXELS * sizeof(double);
    double    *naive_dct_Cb= (double *)   base; base += CHROMA_PIXELS * sizeof(double);
    double    *naive_dct_Cr= (double *)   base; base += CHROMA_PIXELS * sizeof(double);

    float16_t *opt_shift_out=(float16_t *)base; base += 64 * sizeof(float16_t); 
    float16_t *opt_tmp_p1  = (float16_t *)base; base += 64 * sizeof(float16_t); 
    
    float16_t *opt_dct_Y   = (float16_t *)base; base += TOTAL_PIXELS * sizeof(float16_t);
    float16_t *opt_dct_Cb  = (float16_t *)base; base += CHROMA_PIXELS * sizeof(float16_t);
    float16_t *opt_dct_Cr  = (float16_t *)base; base += CHROMA_PIXELS * sizeof(float16_t);

    float16_t *mat_h       = (float16_t *)base; base += 16 * sizeof(uint64_t);
    uint64_t  *mat_bcast   = (uint64_t *) base; base += 64 * sizeof(uint64_t);

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(mat_h, (void*)dct_m, 16 * sizeof(uint64_t)); 
        snrt_dma_start_1d(mat_bcast, (void*)dct_m_bcast, 64 * sizeof(uint64_t));
        snrt_dma_wait_all();
    }
    
    snrt_cluster_hw_barrier();
    if (snrt_is_compute_core() && core_id != 0) return 0; // EARLY EXIT

    if (snrt_is_compute_core() && core_id == 0) {
        for (int i = 0; i < 256; i++) lut[i] = (double)i;
        for (int i = 0; i < TOTAL_PIXELS; i++) {
            input_idx[i * 3 + 0] = (i + 50) % 256;
            input_idx[i * 3 + 1] = (i + 100) % 256;
            input_idx[i * 3 + 2] = (i + 150) % 256;
        }

        uint32_t n1_start = snrt_mcycle();
        naive_YCBCR_conversion(input_idx, naive_Y, naive_CbCr, lut, TOTAL_PIXELS);
        uint32_t n1_end = snrt_mcycle();

        uint32_t n2_start = snrt_mcycle();
        naive_chroma_420(naive_CbCr, naive_cb_f, naive_cr_f, W, H);
        uint32_t n2_end = snrt_mcycle();

        uint32_t n3_start = snrt_mcycle();
        for(int b = 0; b < 4; b++) {
            naive_dct_pass1(&naive_Y[b*64], naive_tmp_p1, mat_h);
            naive_dct_pass2(naive_tmp_p1, &naive_dct_Y[b*64], mat_h);
        }
        naive_dct_pass1(naive_cb_f, naive_tmp_p1, mat_h);
        naive_dct_pass2(naive_tmp_p1, naive_dct_Cb, mat_h);
        naive_dct_pass1(naive_cr_f, naive_tmp_p1, mat_h);
        naive_dct_pass2(naive_tmp_p1, naive_dct_Cr, mat_h);
        uint32_t n3_end = snrt_mcycle();


        uint32_t o1_start = snrt_mcycle();
        opt_YCBCR_conversion(input_idx, opt_Y, opt_CbCr, lut, TOTAL_PIXELS);
        uint32_t o1_end = snrt_mcycle();

        uint32_t o2_start = snrt_mcycle();
        opt_chroma_420_2d_ssr((double*)opt_CbCr, opt_cb_f, opt_cr_f);
        uint32_t o2_end = snrt_mcycle();

        uint32_t o3_start = snrt_mcycle();
        for(int b = 0; b < 4; b++) {
            simd_dct_pass1(&opt_Y[b*64], opt_shift_out, opt_tmp_p1, mat_h);
            simd_dct_pass2(opt_tmp_p1, &opt_dct_Y[b*64], mat_bcast);
        }
        simd_dct_pass1(opt_cb_f, opt_shift_out, opt_tmp_p1, mat_h);
        simd_dct_pass2(opt_tmp_p1, opt_dct_Cb, mat_bcast);
        simd_dct_pass1(opt_cr_f, opt_shift_out, opt_tmp_p1, mat_h);
        simd_dct_pass2(opt_tmp_p1, opt_dct_Cr, mat_bcast);
        uint32_t o3_end = snrt_mcycle();

        // ΕΛΑΦΡΩΣ ΧΑΛΑΡΩΜΕΝΑ ΟΡΙΑ ΛΟΓΩ FP16
        int err_s2 = 0, err_dct = 0;
        int max_prints = 15; 
        
        printf("\n--- STAGE 2: CHROMA ERRORS (Diff > 1.5) ---\n");
        for (int i = 0; i < CHROMA_PIXELS; i++) {
            double v_opt_cb = fp16_to_float(opt_cb_f[i]);
            double d_cb = v_opt_cb - naive_cb_f[i];
            if (d_cb > 1.5 || d_cb < -1.5) {
                if (err_s2 < max_prints) printf(" Cb[%3d]: Naive = %8.3f | Opt = %8.3f | Diff = %6.3f\n", i, naive_cb_f[i], v_opt_cb, d_cb);
                err_s2++;
            }
            double v_opt_cr = fp16_to_float(opt_cr_f[i]);
            double d_cr = v_opt_cr - naive_cr_f[i];
            if (d_cr > 1.5 || d_cr < -1.5) {
                if (err_s2 < max_prints) printf(" Cr[%3d]: Naive = %8.3f | Opt = %8.3f | Diff = %6.3f\n", i, naive_cr_f[i], v_opt_cr, d_cr);
                err_s2++;
            }
        }
        if (err_s2 == 0) printf(" None!\n");
        else if (err_s2 > max_prints) printf(" ... and %d more errors in Stage 2.\n", err_s2 - max_prints);

        printf("\n--- STAGE 3: DCT ERRORS (Diff > 3.0) ---\n");
        for (int i = 0; i < TOTAL_PIXELS; i++) {
            double v_opt = fp16_to_float(opt_dct_Y[i]);
            double diff = v_opt - naive_dct_Y[i];
            if (diff > 3.0 || diff < -3.0 || diff != diff) {
                if (err_dct < max_prints) printf(" Y [%3d]: Naive = %8.3f | Opt = %8.3f | Diff = %6.3f\n", i, naive_dct_Y[i], v_opt, diff);
                err_dct++;
            }
        }
        for (int i = 0; i < CHROMA_PIXELS; i++) { 
            double v_opt = fp16_to_float(opt_dct_Cb[i]);
            double diff = v_opt - naive_dct_Cb[i];
            if (diff > 3.0 || diff < -3.0 || diff != diff) {
                if (err_dct < max_prints) printf(" Cb[%3d]: Naive = %8.3f | Opt = %8.3f | Diff = %6.3f\n", i, naive_dct_Cb[i], v_opt, diff);
                err_dct++;
            }
        }
        for (int i = 0; i < CHROMA_PIXELS; i++) {
            double v_opt = fp16_to_float(opt_dct_Cr[i]);
            double diff = v_opt - naive_dct_Cr[i];
            if (diff > 3.0 || diff < -3.0 || diff != diff) {
                if (err_dct < max_prints) printf(" Cr[%3d]: Naive = %8.3f | Opt = %8.3f | Diff = %6.3f\n", i, naive_dct_Cr[i], v_opt, diff);
                err_dct++;
            }
        }
        if (err_dct == 0) printf(" None!\n");
        else if (err_dct > max_prints) printf(" ... and %d more errors in Stage 3.\n", err_dct - max_prints);


        printf("\n===================================================\n");
        printf("         FULL IMAGE PROCESSING PIPELINE REPORT         \n");
        printf("===================================================\n");
        printf(" STAGE 1: YCbCr CONVERSION\n");
        printf("  - Naive Cycles : %u\n", n1_end - n1_start);
        printf("  - Opt   Cycles : %u\n", o1_end - o1_start);
        printf("  - Speedup      : %.2fx\n", (float)(n1_end - n1_start) / (o1_end - o1_start));
        printf("---------------------------------------------------\n");
        printf(" STAGE 2: CHROMA SUBSAMPLING (2x2 Interleaved)\n");
        printf("  - Naive Cycles : %u\n", n2_end - n2_start);
        printf("  - Opt   Cycles : %u\n", o2_end - o2_start);
        printf("  - Speedup      : %.2fx\n", (float)(n2_end - n2_start) / (o2_end - o2_start));
        printf("---------------------------------------------------\n");
        printf(" STAGE 3: 2D DCT (4xY, 1xCb, 1xCr)\n");
        printf("  - Naive Cycles : %u\n", n3_end - n3_start);
        printf("  - Opt   Cycles : %u\n", o3_end - o3_start);
        printf("  - Speedup      : %.2fx\n", (float)(n3_end - n3_start) / (o3_end - o3_start));
        printf("---------------------------------------------------\n");
        uint32_t total_n = (n1_end - n1_start) + (n2_end - n2_start) + (n3_end - n3_start);
        uint32_t total_o = (o1_end - o1_start) + (o2_end - o2_start) + (o3_end - o3_start);
        printf(" TOTAL FULL PIPELINE\n");
        printf("  - Total Naive  : %u\n", total_n);
        printf("  - Total Opt    : %u\n", total_o);
        printf("  - Overall Gain : %.2fx\n", (float)total_n / total_o);
        printf("---------------------------------------------------\n");
        printf(" VERIFICATION S2 (Chroma) : %s\n", (err_s2 == 0) ? "[SUCCESS]" : "[FAILED]");
        printf(" VERIFICATION S3 (2D DCT) : %s\n", (err_dct == 0) ? "[SUCCESS]" : "[FAILED]");
        printf("===================================================\n\n");

    }

    snrt_cluster_hw_barrier();
    return 0;
}
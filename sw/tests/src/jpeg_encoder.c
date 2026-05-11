#include "snrt.h"
#include <stdint.h>
#include <stdio.h>
#include <math.h>

/* --- Configuration --- */
#define ROW_SIZE 256
#define ALIGN_UP_TCDM(addr) ((((addr) + ROW_SIZE - 1) / ROW_SIZE) * ROW_SIZE)
#define W 16                 
#define H 16                 
#define TOTAL_PIXELS (W * H)

/* --- Tables in L2 --- */
const double dct_m_l2[8][8] __attribute__((aligned(64))) = {
    {0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553, 0.353553},
    {0.490393, 0.415735, 0.277785, 0.097545, -0.097545, -0.277785, -0.415735, -0.490393},
    {0.461940, 0.191342, -0.191342, -0.461940, -0.461940, -0.191342, 0.191342, 0.461940},
    {0.415735, -0.097545, -0.490393, -0.277785, 0.277785, 0.490393, 0.097545, -0.415735},
    {0.353553, -0.353553, -0.353553, 0.353553, 0.353553, -0.353553, -0.353553, 0.353553},
    {0.277785, -0.490393, 0.097545, 0.415735, -0.415735, -0.097545, 0.490393, -0.277785},
    {0.191342, -0.461940, 0.461940, -0.191342, -0.191342, 0.461940, -0.461940, 0.191342},
    {0.097545, -0.277785, 0.415735, -0.490393, 0.490393, -0.415735, 0.277785, -0.097545}
};

const double quant_recip_lum_l2[64] __attribute__((aligned(64))) = {
    0.062500, 0.090909, 0.100000, 0.062500, 0.041667, 0.025000, 0.019608, 0.016393,
    0.083333, 0.083333, 0.071429, 0.052632, 0.038462, 0.017241, 0.016667, 0.018182,
    0.071429, 0.076923, 0.062500, 0.041667, 0.025000, 0.017544, 0.014493, 0.017857,
    0.071429, 0.058824, 0.045455, 0.034483, 0.019608, 0.011494, 0.012500, 0.016129,
    0.055556, 0.045455, 0.027027, 0.017857, 0.014706, 0.009174, 0.009709, 0.012987,
    0.041667, 0.028571, 0.018182, 0.015625, 0.012346, 0.009615, 0.008850, 0.010870,
    0.020408, 0.015625, 0.012821, 0.011494, 0.009709, 0.008264, 0.008333, 0.009901,
    0.013889, 0.010870, 0.010526, 0.010204, 0.008929, 0.010000, 0.009709, 0.010101
};

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

const uint8_t zigzag_table_l2[64] __attribute__((aligned(64))) = {
    0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63
};

/* ========================================================================
 * 2. PIPELINE KERNELS
 * ======================================================================== */

// --- 2.1 Color Conversion (RGB -> YCbCr) ---
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

// --- 2.2 Chroma Subsampling (4:2:0) ---
void opt_chroma_420_2d_ssr(double *in_cbcr, double *out_cbcr_final) {
    register double quarter asm("ft11") = 0.25;
    uint32_t b0 = W * 2;                 
    uint32_t b1 = H / 2;                 
    uint32_t s0 = sizeof(double);        
    uint32_t s1 = 2 * W * 2 * sizeof(double); 

    snrt_ssr_loop_2d(SNRT_SSR_DM0, b0, b1, s0, s1);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, in_cbcr);
    snrt_ssr_loop_2d(SNRT_SSR_DM2, b0, b1, s0, s1);
    snrt_ssr_read(SNRT_SSR_DM2, SNRT_SSR_2D, &in_cbcr[W * 2]);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, (W/2) * (H/2) * 2, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, out_cbcr_final);

    snrt_cluster_hw_barrier();
    snrt_ssr_enable();

    asm volatile(
        "frep.o %[total_blocks], 16, 0, 0 \n"  
        "fmv.d  ft8,  ft0 \n" "fmv.d  ft9,  ft0 \n" "fmv.d  ft3,  ft0 \n" "fmv.d  ft10, ft0 \n"         
        "fmv.d  ft4,  ft2 \n" "fmv.d  ft7,  ft2 \n" "fmv.d  ft5,  ft2 \n" "fmv.d  ft6,  ft2 \n"         
        "fadd.d ft8, ft8, ft3 \n" "fadd.d ft9, ft9, ft10 \n"    
        "fadd.d ft4, ft4, ft5 \n" "fadd.d ft7, ft7, ft6 \n"     
        "fadd.d ft8, ft8, ft4 \n" "fadd.d ft9, ft9, ft7 \n"     
        "fmul.d ft1, ft8, %[q] \n" "fmul.d ft1, ft9, %[q] \n"    
        : 
        : [total_blocks] "r" ((W/2 * H/2) - 1), [q] "f" (quarter)
        : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "ft8", "ft9", "ft10", "memory"
    );
    snrt_fpu_fence();
    snrt_ssr_disable();
}
/* --- Macroblock Extraction Helper --- */
void extract_8x8_block(double *src_planar, double *dst) {

    snrt_ssr_loop_4d(SNRT_SSR_DM0, 8,8,2,2, 8, 128, 64, 1024);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, (void*)src_planar);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 256, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)dst);
	snrt_ssr_enable();
    asm volatile(
        "frep.o %[n], 1, 0, 0 \n"
        "fmv.d ft1, ft0 \n" 
        : : [n] "r"(255) : "ft0", "ft1", "memory"
    );
    snrt_fpu_fence();
    snrt_ssr_disable();

}
// --- 2.3 Discrete Cosine Transform (DCT) ---
void opt_dct_forward(double *in, double *out, double *tmp, const double *dct_m) {
    register double offset asm("ft3") = -128;

    // Stage 0: Level Shift
    snrt_ssr_loop_1d(SNRT_SSR_DM0, 64, sizeof(double));
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)in);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 64, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)out);

    snrt_ssr_enable();
    asm volatile(
        "frep.o %[n], 1, 0, 0 \n"
        "fadd.d ft1, ft0, %[offset] \n" 
        : : [n] "r"(63), [offset] "f"(offset) : "ft0", "ft1", "memory"
    );
    snrt_fpu_fence();
    snrt_ssr_disable();

    // Stage 1: Pass 1 (Rows)
    snrt_ssr_loop_4d(SNRT_SSR_DM0, 8,8,1,8, 0, 8, 0, 64);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, (void*)out);
    snrt_ssr_loop_4d(SNRT_SSR_DM1, 8,8,1,8, 64,8,128,0);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, (void*)dct_m);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 64, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, (void*)tmp);

    snrt_ssr_enable();
    asm volatile(
        "frep.o %[cnts], 24, 0, 0 \n"
        "fsub.d f3, f3, f3 \n" "fsub.d f4, f4, f4 \n" "fsub.d f5, f5, f5 \n" "fsub.d f6, f6, f6 \n"
        "fsub.d f7, f7, f7 \n" "fsub.d f28, f28, f28 \n" "fsub.d f29, f29, f29 \n" "fsub.d f30, f30, f30 \n"
        "frep.o %[cnt], 8, 0, 0 \n"
        "fmadd.d f3, f0, f1, f3 \n" "fmadd.d f4, f0, f1, f4 \n" "fmadd.d f5, f0, f1, f5 \n" "fmadd.d f6, f0, f1, f6 \n"
        "fmadd.d f7, f0, f1, f7 \n" "fmadd.d f28, f0, f1, f28 \n" "fmadd.d f29, f0, f1, f29 \n" "fmadd.d f30, f0, f1, f30 \n"
        "fmadd.d f2, f0, f1, f3 \n" "fmadd.d f2, f0, f1, f4 \n" "fmadd.d f2, f0, f1, f5 \n" "fmadd.d f2, f0, f1, f6 \n"
        "fmadd.d f2, f0, f1, f7 \n" "fmadd.d f2, f0, f1, f28 \n" "fmadd.d f2, f0, f1, f29 \n" "fmadd.d f2, f0, f1, f30 \n"
        : : [cnt] "r"(6),[cnts]"r"(7) : "f0","f1","f2","f3","f4","f5","f6","f7","f28","f29","f30","memory"
    );
    snrt_fpu_fence();
    snrt_ssr_disable();
	
    // Stage 2: Pass 2 (Columns)
    snrt_ssr_loop_4d(SNRT_SSR_DM0, 8,8,1,8, 0, 8, 0, 64);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, (void*)dct_m);
    snrt_ssr_loop_4d(SNRT_SSR_DM1, 8,8,1,8, 8,64,16,0);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, (void*)tmp);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 64, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, (void*)out);

    snrt_ssr_enable();
    asm volatile(
        "frep.o %[cnts], 24, 0, 0 \n"
        "fsub.d f3, f3, f3 \n" "fsub.d f4, f4, f4 \n" "fsub.d f5, f5, f5 \n" "fsub.d f6, f6, f6 \n"
        "fsub.d f7, f7, f7 \n" "fsub.d f28, f28, f28 \n" "fsub.d f29, f29, f29 \n" "fsub.d f30, f30, f30 \n"
        "frep.o %[cnt], 8, 0, 0 \n"
        "fmadd.d f3, f0, f1, f3 \n" "fmadd.d f4, f0, f1, f4 \n" "fmadd.d f5, f0, f1, f5 \n" "fmadd.d f6, f0, f1, f6 \n"
        "fmadd.d f7, f0, f1, f7 \n" "fmadd.d f28, f0, f1, f28 \n" "fmadd.d f29, f0, f1, f29 \n" "fmadd.d f30, f0, f1, f30 \n"
        "fmadd.d f2, f0, f1, f3 \n" "fmadd.d f2, f0, f1, f4 \n" "fmadd.d f2, f0, f1, f5 \n" "fmadd.d f2, f0, f1, f6 \n"
        "fmadd.d f2, f0, f1, f7 \n" "fmadd.d f2, f0, f1, f28 \n" "fmadd.d f2, f0, f1, f29 \n" "fmadd.d f2, f0, f1, f30 \n"
        : : [cnt] "r"(6), [cnts]"r"(7) : "f0","f1","f2","f3","f4","f5","f6","f7","f28","f29","f30","memory"
    );
    snrt_fpu_fence();
    snrt_ssr_disable();
}

// --- 2.4 Quantization ---
void opt_quant(double *in, double *out, const double *recip) {
    register double magic asm("ft10") = 6755399441055744.0;
    snrt_ssr_loop_1d(SNRT_SSR_DM0, 64, sizeof(double));
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)in);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 64, sizeof(double));
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)recip);
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 64, sizeof(double));
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, (void*)out);

    snrt_ssr_enable(); 
    asm volatile(
        "frep.o %[cnt], 6, 0, 0    \n"  
        "fmul.d  ft3, ft0, ft1     \n"  
        "fmul.d  ft4, ft0, ft1     \n"  
        "fadd.d  ft5, ft3, ft10, rne \n" 
        "fadd.d  ft6, ft4, ft10, rne \n" 
        "fsub.d  ft2, ft5, ft10, rne \n" 
        "fsub.d  ft2, ft6, ft10, rne \n" 
        : 
        : [cnt] "r"(31), "f"(magic)     
        : "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft10", "ft11", "memory" 
    );
    snrt_fpu_fence();  
    snrt_ssr_disable(); 
}

// --- 2.5 Zig-Zag & Run-Length Encoding ---
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

// --- 2.6 Huffman Bit-Packing ---
uint32_t bit_buffer = 0;
int bit_count = 0;

inline void push_bits(uint32_t code, int length, uint8_t **stream_out) {
    bit_buffer = (bit_buffer << length) | (code & ((1 << length) - 1));
    bit_count += length;
    while (bit_count >= 8) {
        bit_count -= 8;
        uint8_t byte = (bit_buffer >> bit_count) & 0xFF;
        *(*stream_out)++ = byte;
        if (byte == 0xFF) *(*stream_out)++ = 0x00;
    }
}

inline int get_vli_size(int16_t val) {
    if (val == 0) return 0;
    int32_t abs_v = (val < 0) ? -val : val;
    return 32 - __builtin_clz(abs_v);
}

inline uint32_t get_vli_bits(int16_t val) {
    return (val > 0) ? val : (val - 1);
}

void get_huffman_prefix(int run, int size, uint32_t *code, int *len) {
    // Mock Table fallback (Replace with full logic for production)
    if (run == 0 && size == 4) { *code = 0x0B; *len = 4; return; } 
    if (run == 1 && size == 2) { *code = 0x1B; *len = 5; return; } 
    if (run == 1 && size == 1) { *code = 0x0C; *len = 4; return; } 
    if (run == 0 && size == 2) { *code = 0x01; *len = 2; return; } 
    if (run == 0 && size == 0) { *code = 0x0A; *len = 4; return; } 
    *code = 0x00; *len = 0; 
}

int process_huffman(int16_t *rle_in, uint8_t *jpg_stream) {
    uint8_t *stream_ptr = jpg_stream;
    bit_buffer = 0;
    bit_count = 0;

    for (int i = 0; i < 128; i += 2) {
        int16_t run = rle_in[i];
        int16_t val = rle_in[i+1];

        if (run == 0 && val == 0) {
            uint32_t prefix_code; int prefix_len;
            get_huffman_prefix(0, 0, &prefix_code, &prefix_len);
            push_bits(prefix_code, prefix_len, &stream_ptr);
            break; 
        }

        int size = get_vli_size(val);
        uint32_t vli_bits = get_vli_bits(val);

        uint32_t prefix_code; int prefix_len;
        get_huffman_prefix(run, size, &prefix_code, &prefix_len);

        push_bits(prefix_code, prefix_len, &stream_ptr);
        push_bits(vli_bits, size, &stream_ptr);
    }

    if (bit_count > 0) {
        push_bits((1 << (8 - bit_count)) - 1, 8 - bit_count, &stream_ptr);
    }
    
    return (int)(stream_ptr - jpg_stream); // Return bytes written
}

/* ========================================================================
 * 3. ORCHESTRATION (THE PIPELINE)
 * ======================================================================== */
int main() {
	uint32_t core_id = snrt_cluster_core_idx();
	uintptr_t base   = (uintptr_t)snrt_l1_next();

	// ── Shared read-only data (same for all cores) ──────────────────────────
	double  *lut     = (double  *)ALIGN_UP_TCDM(base);
	uint8_t *rgb_in  = (uint8_t *)ALIGN_UP_TCDM((uintptr_t)lut    + 256          * sizeof(double));

	// ── Pipeline buffers (single macroblock, single core for now) ───────────
	double  *planar_Y          = (double  *)ALIGN_UP_TCDM((uintptr_t)rgb_in           + TOTAL_PIXELS * 3);
	double  *planar_CbCr       = (double  *)ALIGN_UP_TCDM((uintptr_t)planar_Y         + TOTAL_PIXELS     * sizeof(double));
	double  *subsampled_CbCr   = (double  *)ALIGN_UP_TCDM((uintptr_t)planar_CbCr      + TOTAL_PIXELS * 2 * sizeof(double));
	double  *straightened16x16 = (double  *)ALIGN_UP_TCDM((uintptr_t)subsampled_CbCr  + 128               * sizeof(double));

	// ── Per-core working buffers (will be bank-local when parallelized) ──────
	// Each of these is written and read by a single core during block processing
	double  *temp8x8  = (double  *)snrt_l1_alloc_compute_core_local(64  * sizeof(double),  sizeof(double));
	double  *dct_out  = (double  *)snrt_l1_alloc_compute_core_local(64  * sizeof(double),  sizeof(double));
	int16_t *rle_out  = (int16_t *)snrt_l1_alloc_compute_core_local(128 * sizeof(int16_t), sizeof(int16_t));
	uint8_t *jpg_stream = (uint8_t *)snrt_l1_alloc_compute_core_local(2048 * sizeof(uint8_t), sizeof(uint8_t));

	// ── Shared lookup tables (DMA'd from L2, read-only during compute) ───────
	double  *l1_dct   = (double  *)ALIGN_UP_TCDM((uintptr_t)straightened16x16 + 256 * sizeof(double));
	double  *l1_q_lum = (double  *)ALIGN_UP_TCDM((uintptr_t)l1_dct   + 64 * sizeof(double));
	double  *l1_q_chr = (double  *)ALIGN_UP_TCDM((uintptr_t)l1_q_lum + 64 * sizeof(double));
	uint8_t *l1_zz    = (uint8_t *)ALIGN_UP_TCDM((uintptr_t)l1_q_chr + 64 * sizeof(double));

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l1_dct,   (void*)dct_m_l2,           64 * sizeof(double));
        snrt_dma_start_1d(l1_q_lum, (void*)quant_recip_lum_l2, 64 * sizeof(double));
        snrt_dma_start_1d(l1_q_chr, (void*)quant_recip_chr_l2, 64 * sizeof(double));
        snrt_dma_start_1d(l1_zz,    (void*)zigzag_table_l2,    64 * sizeof(uint8_t));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core() && core_id == 0) {
        for (int i = 0; i < 256; i++) lut[i] = (double)i;
        for (int i = 0; i < TOTAL_PIXELS * 3; i++) rgb_in[i] = (uint8_t)(i % 256);

        uint32_t t0, t1;
        uint32_t cyc_ycbcr, cyc_chroma, cyc_extract;
        uint32_t cyc_y_dct[4], cyc_y_quant[4], cyc_y_zz[4], cyc_y_huff[4];
        uint32_t cyc_cb_dct, cyc_cb_quant, cyc_cb_zz, cyc_cb_huff;
        uint32_t cyc_cr_dct, cyc_cr_quant, cyc_cr_zz, cyc_cr_huff;
        int bytes_total = 0;

        // ── Stage A ──────────────────────────────────────────────────
        t0 = snrt_mcycle();
        opt_YCBCR_conversion(rgb_in, planar_Y, planar_CbCr, lut, TOTAL_PIXELS);
        t1 = snrt_mcycle();
        cyc_ycbcr = t1 - t0;

        t0 = snrt_mcycle();
        opt_chroma_420_2d_ssr(planar_CbCr, subsampled_CbCr);
        t1 = snrt_mcycle();
        cyc_chroma = t1 - t0;

        t0 = snrt_mcycle();
        extract_8x8_block(planar_Y, straightened16x16);
        t1 = snrt_mcycle();
        cyc_extract = t1 - t0;

        // ── Stage B: 4 Luma blocks ───────────────────────────────────
        for (int i = 0; i < 4; i++) {
            double *in = straightened16x16 + (i * 64);

            t0 = snrt_mcycle();
            opt_dct_forward(in, dct_out, temp8x8, l1_dct);
            t1 = snrt_mcycle();
            cyc_y_dct[i] = t1 - t0;

            t0 = snrt_mcycle();
            opt_quant(dct_out, temp8x8, l1_q_lum);
            t1 = snrt_mcycle();
            cyc_y_quant[i] = t1 - t0;

            t0 = snrt_mcycle();
            process_zigzag_rle(temp8x8, rle_out, l1_zz);
            t1 = snrt_mcycle();
            cyc_y_zz[i] = t1 - t0;

            t0 = snrt_mcycle();
            bytes_total += process_huffman(rle_out, jpg_stream + bytes_total);
            t1 = snrt_mcycle();
            cyc_y_huff[i] = t1 - t0;
        }

        // ── Stage C: Cb block (i=0) ──────────────────────────────────
        t0 = snrt_mcycle();
        opt_dct_forward(&subsampled_CbCr[0 * 64], dct_out, temp8x8, l1_dct);
        t1 = snrt_mcycle();
        cyc_cb_dct = t1 - t0;

        t0 = snrt_mcycle();
        opt_quant(dct_out, temp8x8, l1_q_chr);
        t1 = snrt_mcycle();
        cyc_cb_quant = t1 - t0;

        t0 = snrt_mcycle();
        process_zigzag_rle(temp8x8, rle_out, l1_zz);
        t1 = snrt_mcycle();
        cyc_cb_zz = t1 - t0;

        t0 = snrt_mcycle();
        bytes_total += process_huffman(rle_out, jpg_stream + bytes_total);
        t1 = snrt_mcycle();
        cyc_cb_huff = t1 - t0;

        // ── Stage C: Cr block (i=1) ──────────────────────────────────
        t0 = snrt_mcycle();
        opt_dct_forward(&subsampled_CbCr[1 * 64], dct_out, temp8x8, l1_dct);
        t1 = snrt_mcycle();
        cyc_cr_dct = t1 - t0;

        t0 = snrt_mcycle();
        opt_quant(dct_out, temp8x8, l1_q_chr);
        t1 = snrt_mcycle();
        cyc_cr_quant = t1 - t0;

        t0 = snrt_mcycle();
        process_zigzag_rle(temp8x8, rle_out, l1_zz);
        t1 = snrt_mcycle();
        cyc_cr_zz = t1 - t0;

        t0 = snrt_mcycle();
        bytes_total += process_huffman(rle_out, jpg_stream + bytes_total);
        t1 = snrt_mcycle();
        cyc_cr_huff = t1 - t0;

        // ── Print Results ────────────────────────────────────────────
        printf("\n");
        printf("+==============================================+\n");
        printf("|    16x16 MACROBLOCK CYCLE BREAKDOWN         |\n");
        printf("+==============================================+\n");
        printf("| STAGE A                                      |\n");
        printf("|   YCbCr Conversion   : %8u cycles       |\n", cyc_ycbcr);
        printf("|   Chroma 4:2:0       : %8u cycles       |\n", cyc_chroma);
        printf("|   Extract 8x8 blocks : %8u cycles       |\n", cyc_extract);
        printf("+----------------------------------------------+\n");
        printf("| STAGE B - LUMA (Y) - 4x 8x8 blocks          |\n");
        printf("|  Block |     DCT |   Quant |  ZigZag | Huff  |\n");
        printf("|--------|---------|---------|---------|-------|\n");
        for (int i = 0; i < 4; i++) {
            printf("|   Y[%d] | %7u | %7u | %7u | %5u |\n",
                   i, cyc_y_dct[i], cyc_y_quant[i], cyc_y_zz[i], cyc_y_huff[i]);
        }
        printf("+----------------------------------------------+\n");
        printf("| STAGE C - CHROMA                             |\n");
        printf("|  Block |     DCT |   Quant |  ZigZag | Huff  |\n");
        printf("|--------|---------|---------|---------|-------|\n");
        printf("|     Cb | %7u | %7u | %7u | %5u |\n",
               cyc_cb_dct, cyc_cb_quant, cyc_cb_zz, cyc_cb_huff);
        printf("|     Cr | %7u | %7u | %7u | %5u |\n",
               cyc_cr_dct, cyc_cr_quant, cyc_cr_zz, cyc_cr_huff);
        printf("+----------------------------------------------+\n");

        uint32_t total =
            cyc_ycbcr + cyc_chroma + cyc_extract +
            cyc_y_dct[0] + cyc_y_dct[1] + cyc_y_dct[2] + cyc_y_dct[3] +
            cyc_y_quant[0] + cyc_y_quant[1] + cyc_y_quant[2] + cyc_y_quant[3] +
            cyc_y_zz[0] + cyc_y_zz[1] + cyc_y_zz[2] + cyc_y_zz[3] +
            cyc_y_huff[0] + cyc_y_huff[1] + cyc_y_huff[2] + cyc_y_huff[3] +
            cyc_cb_dct + cyc_cb_quant + cyc_cb_zz + cyc_cb_huff +
            cyc_cr_dct + cyc_cr_quant + cyc_cr_zz + cyc_cr_huff;

        printf("| Total Cycles  : %8u                     |\n", total);
        printf("| Output Bytes  : %8d                     |\n", bytes_total);
        printf("+==============================================+\n\n");
    }

    snrt_cluster_hw_barrier();
    return 0;
}
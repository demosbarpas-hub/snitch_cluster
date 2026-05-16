#include "snrt.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* --- Configuration --- */
#define ROW_SIZE 256
#define ALIGN_UP_TCDM(addr) ((((addr) + ROW_SIZE - 1) / ROW_SIZE) * ROW_SIZE)

typedef uint16_t float16_t;

// Helper conversions
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

/* --- Σταθεροί Πίνακες DCT (L2) --- */
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

// Κανονικός Πίνακας DCT-II (8x8)
// Κανονικός Πίνακας DCT-II (8x8) - Row-Packed
const uint64_t dct_m[16] __attribute__((aligned(64))) = {
    0x35a835a835a835a8, 0x35a835a835a835a8, // Σειρά 0
    0x2e3f34e93aa73be2, 0xbbe2baa7b4e9ae3f, // Σειρά 1
    0xbaabb21632163b64, 0x3b643216b216baab, // Σειρά 2
    0xb8e3bc53b2403aa7, 0xbaa732403c5338e3, // Σειρά 3
    0x35a8b5a8b5a835a8, 0x35a8b5a8b5a835a8, // Σειρά 4
    0x3aa73240bbe234e9, 0xb4e93be2b240baa7, // Σειρά 5
    0xb2163696b6963216, 0x3216b6963696b216, // Σειρά 6
    0xbbe23b64baa72e3f, 0xae3f3aa7bb643be2  // Σειρά 7
};

// Transposed Πίνακας DCT (M^T)
// Transposed Πίνακας DCT (M^T) - Row-Packed
const uint64_t dct_mt[16] __attribute__((aligned(64))) = {
    0x3aa73b643be235a8, 0x2e3f321634e935a8, // Σειρά 0
    0xb24032163aa735a8, 0xbaa7b696bbe2b5a8, // Σειρά 1
    0xbc53b21634e935a8, 0x3b6436963240b5a8, // Σειρά 2
    0xb8e3baab2e3f35a8, 0xbbe2b2163aa735a8, // Σειρά 3
    0x38e3baabae3f35a8, 0x3be2b216baa735a8, // Σειρά 4
    0x3c53b216b4e935a8, 0xbb643696b240b5a8, // Σειρά 5
    0x32403216baa735a8, 0x3aa7b6963be2b5a8, // Σειρά 6
    0xbaa73b64bbe235a8, 0xae3f3216b4e935a8  // Σειρά 7
};
// ========================================================================
// SIMD PACKED DCT (FP16) - TWO PASSES
// ========================================================================
void simd_dct_forward(float16_t *in, float16_t *out, float16_t *tmp, const float16_t *dct_m) {
    float offset_f = -128.0f;
    
    // --- STAGE 0: Level Shift (in -> out) ---
    snrt_ssr_loop_1d(SNRT_SSR_DM0, 16, 8); 
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, (void*)in);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 16, 8);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, (void*)out);
    snrt_ssr_enable();
    asm volatile(
    "fcvt.h.s f10, %0 \n"           // Μετατροπή του offset σε FP16 (Lane 0)
    "frep.o %[n], 1, 0, 0 \n"         // Επανάληψη για 16 chunks (64 pixels)
    "vfadd.r.h f1, f0, f10 \n"      // f1 = f0 (Vector) + f10 (Scalar Broadcast)
    :: "f"(offset_f), [n]"r"(15) 
    : "f1", "f0", "f10"
	);
    snrt_fpu_fence(); snrt_ssr_disable();

    // --- STAGE 1: Pass 1 - Rows (out -> tmp) ---
    // DM0: Data (Rows)
    snrt_ssr_loop_2d(SNRT_SSR_DM0, 2, 8, 8, 8, 0, 16); //8 φορές την κάθε γραμμή, μετά από κάτω
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_3D, (void*)out);
    // DM1: Matrix (Rows).
    snrt_ssr_loop_2d(SNRT_SSR_DM1, 16, 8, 8, 0);//2 τιμές της γραμμής, μετά από κάτω για όλες τις 16 τιμές, μετά ξανά από την αρχή
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_2D, (void*)dct_m);//DCT_m ο μη αναστραμένος πίνακας
    // DM2: Sequential Write to tmp
    snrt_ssr_loop_2d(SNRT_SSR_DM2, 16,8);
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, (void*)tmp);
    
    snrt_ssr_enable();
	asm volatile(
		"frep.o %[t0], 27, 0, 0 \n"  

		"fsub.d f10,f10,f10 \n"
		"fsub.d f11,f11,f11 \n"
		"fsub.d f12,f12,f12 \n"
		"fsub.d f13,f13,f13 \n"

		"vfmac.h f10, f0, f1 \n"
		"vfmac.h f10, f0, f1 \n"
		"vfsum.h f10,f10 \n"
		"vfsum.h f10,f10 \n"
		
		"vfmac.h f11, f0, f1 \n"
		"vfmac.h f11, f0, f1 \n"
		"vfsum.h f11,f11 \n"
		"vfsum.h f11,f11 \n"

		"vfmac.h f12, f0, f1 \n"
		"vfmac.h f12, f0, f1 \n"
		"vfsum.h f12,f12 \n"
		"vfsum.h f12,f12 \n"
				
		"vfmac.h f13, f0, f1 \n"
		"vfmac.h f13, f0, f1 \n"
		"vfsum.h f13,f13 \n"
		"vfsum.h f13,f13 \n"

			// 5. FPU PACKING PHASE
		// Μετατροπή των FP16 scalars σε FP32 για να τα δεχτεί η μονάδα packing
		"fcvt.s.h  f20, f10 \n"
		"fcvt.s.h  f21, f11 \n"
		"fcvt.s.h  f22, f12 \n"
		"fcvt.s.h  f23, f13 \n"
		
			// Συσκευασία στον f24
		"vfcpka.h.s f24, f20, f21 \n"          
		"vfcpkb.h.s f24, f22, f23 \n"          

		"fmv.d f2, f24 \n" 

		:: [t0] "r"(15)
		: "f10", "f11", "f12", "f13", "ft0", "ft1", 
		"f20", "f21", "f22", "f23", "f24", "memory"
	);

    snrt_fpu_fence();
    snrt_ssr_disable();

    // --- STAGE 2: Pass 2 - Columns (tmp -> out) ---
    // DM0: Matrix. Sequential rows.
    snrt_ssr_loop_3d(SNRT_SSR_DM0, 2, 8, 8, 8, 0, 16);
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_3D, (void*)dct_m);
    // DM1: Data from tmp. We access COLUMNS using stride 16 (8 pixels * 2 bytes).
    // Loop 0: 2 chunks (8 pixels), Loop 1: 4 rows, Loop 2: 8 columns/freqs
    snrt_ssr_loop_3d(SNRT_SSR_DM1, 8, 2,8, 16, 8,0); 
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_3D, (void*)tmp);
    // DM2: Write back to out
    snrt_ssr_loop_1d(SNRT_SSR_DM2, 16, 8);
    snrt_ssr_write(SNRT_SSR_DM2, SNRT_SSR_1D, (void*)out);

    snrt_ssr_enable();
	asm volatile(
		"frep.o %[t0], 10, 0, 0 \n"  // 10 εντολές block

		// 1. Reset Accumulator (f10)
		"fsub.d f10,f10,f10 \n"

		// 2. Vertical Dot Product (8 κύκλοι)
		// f0: Matrix Coefficients (Broadcasted/Interleaved από DM0)
		// f1: Column Chunks (4 pixels από 4 στήλες από DM1)
		"vfmac.h f10, f0, f1 \n"
		"vfmac.h f10, f0, f1 \n"
		"vfmac.h f10, f0, f1 \n"
		"vfmac.h f10, f0, f1 \n"
		"vfmac.h f10, f0, f1 \n"
		"vfmac.h f10, f0, f1 \n"
		"vfmac.h f10, f0, f1 \n"
		"vfmac.h f10, f0, f1 \n"

		// 3. Store Results
		// Μετά από 8 vfmac, τα lanes του f10 έχουν:
		// Lane 0: Col 0 result, Lane 1: Col 1 result, κλπ.
		"fmv.d f2, f10 \n" 

		:: [t0] "r"(15) 
		: "f10", "memory"
	);

    snrt_fpu_fence();
    snrt_ssr_disable();
}

// --- Naive DCT για Verification ---
void naive_dct(double *in, double *out, const double dct_m[8][8]) {
    double s[8][8], tmp[8][8];
    for (int i = 0; i < 64; i++) ((double*)s)[i] = in[i] - 128.0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            double sum = 0;
            for (int k = 0; k < 8; k++) sum += s[r][k] * dct_m[c][k];
            tmp[r][c] = sum;
        }
    }
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            double sum = 0;
            for (int k = 0; k < 8; k++) sum += dct_m[r][k] * tmp[k][c];
            out[r * 8 + c] = sum;
        }
    }
}

int main() {
    uint32_t core_id = snrt_cluster_core_idx();
    uintptr_t base = (uintptr_t)snrt_l1_next();
    
    // Memory allocation
    double *dct_in    = (double *)ALIGN_UP_TCDM(base);
    double *dct_out_n = (double *)ALIGN_UP_TCDM((uintptr_t)dct_in  + 64 * sizeof(double));
    double *l1_matrix = (double *)ALIGN_UP_TCDM((uintptr_t)dct_out_n + 64 * sizeof(double));
    
    float16_t *in_h   = (float16_t *)ALIGN_UP_TCDM((uintptr_t)l1_matrix + 64 * sizeof(double));
    float16_t *out_h  = (float16_t *)ALIGN_UP_TCDM((uintptr_t)in_h + 64 * sizeof(float16_t));
    float16_t *tmp_h  = (float16_t *)ALIGN_UP_TCDM((uintptr_t)out_h + 64 * sizeof(float16_t));
    float16_t *mat_h  = (float16_t *)ALIGN_UP_TCDM((uintptr_t)tmp_h + 64 * sizeof(float16_t));

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l1_matrix, (void*)dct_m_l2, 64 * sizeof(double));
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core() && core_id == 0) {
        // Initialize data
        for (int i = 0; i < 64; i++) {
            dct_in[i] = (double)((i*13 + (i/8)*27) % 256);
            in_h[i] = float_to_fp16(dct_in[i]);
            mat_h[i] = float_to_fp16(l1_matrix[i]);
        }

        // Run Naive (Reference)
        naive_dct(dct_in, dct_out_n, (const double(*)[8])l1_matrix);

        // Run SIMD (Optimized)
        uint32_t start = snrt_mcycle();
        simd_dct_forward(in_h, out_h, tmp_h, mat_h);
        uint32_t end = snrt_mcycle();

        printf("\nIndex | Naive Out  | SIMD Out (FP16 Result)\n");
        printf("--------------------------------------------\n");
        for (int i = 0; i < 10; i++) {
            printf("%5d | %10.3f | %10.3f\n", i, dct_out_n[i], fp16_to_float(out_h[i]));
        }

        printf("\nSIMD Cycles: %u\n", end - start);
    }

    snrt_cluster_hw_barrier();
    return 0;
}
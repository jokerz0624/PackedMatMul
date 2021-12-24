
#include <iostream>
#include "timelog.h"
typedef __fp16 FLOAT16;
#define load(x,y) sum += x[0];
#define str(x,y) x[0] = sum;

#define ASM_TEST_CORRECTNESS
#define ASM_TIME_PROFILE_PACK12
#define ASM_TIME_PROFILE_PACK24
extern "C" {
    void MatMulPack12(FLOAT16 *C, FLOAT16 *A, FLOAT16 *B, size_t eP, size_t l, size_t hP);
    void MatMulPack24(FLOAT16 *C, FLOAT16 *A, FLOAT16 *B, size_t eP, size_t l, size_t hP);
}

// C++ version for assembly logic
void MatMulPack12_CPP(FLOAT16 *C, FLOAT16 *A, FLOAT16 *B, size_t eP, size_t l, size_t hP) {
    //x0: C, x1:A, x2:B, x3:eP, x4:l, x5:hP

    if (hP < 1) {
        return;
    }
    FLOAT16 sum = 0;

    size_t c_stride = eP * 12 * 8;
    size_t b_stride = l * 16;
    
    do {
        FLOAT16 *A_offset = A;
        FLOAT16 *C_offset = C + c_stride;
        do {
            FLOAT16 *B_offset = B;
            load(B_offset, 16);
            B_offset += 16;
            load(A_offset, 12);
            A_offset += 12;
            //mul;
            l--;
            while(l > 0) {
                load(B_offset, 16);
                B_offset += 16;
                load(A_offset, 12);
                A_offset += 12;
                //fma;
                l--;
            } 
            str(C, 8 * 12);
            C += 96;
            str(C_offset, 8 * 12);
            C_offset += 96;
            eP--;
        } while(eP > 0);
        C += c_stride;
        B += b_stride;
        hP--;
    } while(hP > 0);
}

void MatMulPack24_CPP(FLOAT16 *C, FLOAT16 *A, FLOAT16 *B, size_t eP, size_t l, size_t hP) {
    //x0: C, x1:A, x2:B, x3:eP, x4:l, x5:hP

    if (hP < 1) {
        return;
    }
    FLOAT16 sum = 0;

    size_t b_stride = l * 16;
    
    do {
        FLOAT16 *A_offset = A;
        do {
            FLOAT16 *B_offset = B;
            load(B_offset, 16);
            B_offset += 16;
            load(A_offset, 12);
            A_offset += 12;
            //mul;
            l--;
            while(l > 0) {
                load(B_offset, 16);
                B_offset += 16;
                load(A_offset, 12);
                A_offset += 12;
                //fma;
                l--;
            } 
            str(C, 8 * 24);
            C += 192;
            eP--;
        } while(eP > 0);
        hP--;
        B += b_stride;
    } while(hP > 0);
}

// pack the matrix B/C
void PackUNIT(FLOAT16* dst, const FLOAT16* src, size_t area, size_t depth, size_t UNIT) {
    int depthCUnit  = depth / UNIT;
    int z, x, y;
    const FLOAT16* srcChannel[UNIT];
    const FLOAT16* srcOffset = src;
    for(z = 0; z < depthCUnit; ++z) {
        for(y = 0; y < UNIT; ++y) {
            srcChannel[y] = srcOffset + area * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < UNIT; ++y) {
                dst[0] = srcChannel[y][0];
                srcChannel[y]++;
                dst++;
            }
        }
        srcOffset += area * UNIT;
    }
}

// pack the matrix A for every ePack length data
void PackA(FLOAT16* dst, const FLOAT16* src, size_t l, size_t eP, size_t ePack) {
    for(int i = 0; i < eP; ++i) {
        for(int j = 0; j < l; ++j) {
            for(int k = 0; k < ePack; ++k) {
                dst[i * l * ePack + j * ePack + k] = src[j * eP * ePack 
                + i * ePack + k];
            }
        }
    }
}

// common matmul
void MatMul(FLOAT16 *C, FLOAT16 *A, FLOAT16 *B, size_t e, size_t l, size_t h) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < e; ++j) {
            FLOAT16 sum = 0;
            for (int k = 0; k < l; ++k) {
                sum += B[i * l + k] * A[k * e + j];
            }
            C[i * e + j] = sum;
        }
    }
}
int main(int argc, char **argv) {

#ifdef ASM_TEST_CORRECTNESS
    int e = 48;
    int l = 4;
    int h = 16;
#else
    int e = 600000;
    int l = 128;
    int h = 32;
#endif

    int ePack1 = 12;
    int hPack1 = 16;

    int ePack2 = 24;
    int hPack2 = 8;

    int eP1 = e / ePack1;
    int hP1 = h / hPack1;

    int eP2 = e / ePack2;
    int hP2 = h / hPack2;

    FLOAT16 *C = new FLOAT16[e * h];
    FLOAT16 *A = new FLOAT16[e * l];
    FLOAT16 *B = new FLOAT16[h * l];
    FLOAT16 *CPack = new FLOAT16[e * h];

    FLOAT16 *C1 = new FLOAT16[e * h];
    FLOAT16 *A1 = new FLOAT16[e * l];
    FLOAT16 *B1 = new FLOAT16[h * l];

    FLOAT16 *C2 = new FLOAT16[e * h];
    FLOAT16 *A2 = new FLOAT16[e * l];
    FLOAT16 *B2 = new FLOAT16[h * l];
    
#ifdef ASM_TEST_CORRECTNESS
    double ratio = 1.0 / (e * l);
    for (size_t i = 0; i < e * l; ++i) {
        // random data between 0~1
        int i_mod7 = i % 7;
        int i_mod3 = i % 3;
        double iData = i * ratio * i_mod7 / 7 - (1 - i * ratio) * i_mod3 / 3;
        A[i] = static_cast<FLOAT16>(iData);
        
        // regular data
        // A[i] = (FLOAT16)i;
    }
    PackA(A1, A, l, eP1, ePack1);
    PackA(A2, A, l, eP2, ePack2);


    for(size_t i = 0; i < l * h; ++i) {
        B[i] = (FLOAT16)i;
    }
    PackUNIT(B1, B, l, h, hPack1);
    PackUNIT(B2, B, l, h, hPack2);
    

    MatMul(C, A, B, e, l, h);
    PackUNIT(CPack, C, e, h, 8);

    MatMulPack12(C1, A1, B1, eP1, l, hP1);
    
    bool flag = true;
    for(size_t i = 0; i < e * h; ++i) {
        if (CPack[i] != C1[i]) {
            flag = false;
            printf("error data index: %zu\n", i);
            printf(" C1 is %f\n", (float)(C1[i]));
            printf(" CPack is %f\n\n", (float)(CPack[i]));
        }
    }
    if (flag) printf("MatMulPack12 asm correctness test has passed\n");

    flag = true;
    MatMulPack24(C2, A2, B2, eP2, l, hP2);
    for(size_t i = 0; i < e * h; ++i) {
        if (CPack[i] != C2[i]) {
            flag = false;
            printf("error data index: %zu\n", i);
            printf(" C2 is %f\n", (float)(C2[i]));
            printf(" CPack is %f\n\n", (float)(CPack[i]));
        }
    }
    if (flag) printf("MatMulPack24 asm correctness test has passed\n");
#else

// We can't test MatMulPack12 and MatMulPack24 together because of cache line
Timer timer;
#if defined(ASM_TIME_PROFILE_PACK12)
    timer.reset();
    MatMulPack12(C1, A1, B1, eP1, l, hP1);
    printf("MatMulPack12 time cost: %f ms\n", timer.cost());
#elif defined(ASM_TIME_PROFILE_PACK24)
    timer.reset();
    MatMulPack24(C2, A2, B2, eP2, l, hP2);
    printf("MatMulPack24 time cost: %f ms\n", timer.cost());
#endif

#endif
    delete[] C1;   
    delete[] C2;
    delete[] C;
    delete[] CPack;
    delete[] B1;
    delete[] B2;
    delete[] B;
    delete[] A1;
    delete[] A2;
    delete[] A;

    return 0;
}

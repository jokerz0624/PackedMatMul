#if defined(__aarch64__)
#include "ArmAsmGlobal.h"

.text
.align 5

asm_function MatMulPack12
//void MatMulPack12(float *C, float *A, float *B, size_t eP, size_t l, size_t hP)
//Auto load:
//x0: C, x1:A, x2:B, x3:eP, x4:l, x5:hP

cbz x3, End
cbz x4, End
cbz x5, End

sub sp, sp, #128
st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [sp], #64

// x8: c_stride * sizeof(fp16)
add	x8, x3, x3, lsl #1
lsl	x8, x8, #6

// x9: b_stride * sizeof(fp16)
lsl	x9, x4, #5

LoopHP:
    // x10: A_offset
    mov x10, x1

    // x11: C_offset
    add x11, x0, x8

    // x13 = eP
    mov x13, x3
LoopEP:
    // x14 = B_offset
    mov x14, x2
    
    ld1 {v3.8h, v4.8h}, [x14], #32
    ld1 {v0.4h, v1.4h, v2.4h}, [x10], #24

    fmul v8.8h, v3.8h, v0.h[0]
    fmul v9.8h, v3.8h, v0.h[1]
    fmul v10.8h, v3.8h, v0.h[2]
    fmul v11.8h, v3.8h, v0.h[3]
    fmul v12.8h, v3.8h, v1.h[0]
    fmul v13.8h, v3.8h, v1.h[1]
    fmul v14.8h, v3.8h, v1.h[2]
    fmul v15.8h, v3.8h, v1.h[3]
    fmul v16.8h, v3.8h, v2.h[0]
    fmul v17.8h, v3.8h, v2.h[1]
    fmul v18.8h, v3.8h, v2.h[2]
    fmul v19.8h, v3.8h, v2.h[3]

    fmul v20.8h, v4.8h, v0.h[0]
    fmul v21.8h, v4.8h, v0.h[1]
    fmul v22.8h, v4.8h, v0.h[2]
    fmul v23.8h, v4.8h, v0.h[3]
    fmul v24.8h, v4.8h, v1.h[0]
    fmul v25.8h, v4.8h, v1.h[1]
    fmul v26.8h, v4.8h, v1.h[2]
    fmul v27.8h, v4.8h, v1.h[3]
    fmul v28.8h, v4.8h, v2.h[0]
    fmul v29.8h, v4.8h, v2.h[1]
    fmul v30.8h, v4.8h, v2.h[2]
    fmul v31.8h, v4.8h, v2.h[3]

    // x12 = l - 1 
    mov x12, x4
    subs x12, x12, #1
    beq LoopEPRemain
LoopL:
    ld1 {v3.8h, v4.8h}, [x14], #32
    ld1 {v0.4h, v1.4h, v2.4h}, [x10], #24

    fmla v8.8h, v3.8h, v0.h[0]
    fmla v9.8h, v3.8h, v0.h[1]
    fmla v10.8h, v3.8h, v0.h[2]
    fmla v11.8h, v3.8h, v0.h[3]
    fmla v12.8h, v3.8h, v1.h[0]
    fmla v13.8h, v3.8h, v1.h[1]
    fmla v14.8h, v3.8h, v1.h[2]
    fmla v15.8h, v3.8h, v1.h[3]
    fmla v16.8h, v3.8h, v2.h[0]
    fmla v17.8h, v3.8h, v2.h[1]
    fmla v18.8h, v3.8h, v2.h[2]
    fmla v19.8h, v3.8h, v2.h[3]

    fmla v20.8h, v4.8h, v0.h[0]
    fmla v21.8h, v4.8h, v0.h[1]
    fmla v22.8h, v4.8h, v0.h[2]
    fmla v23.8h, v4.8h, v0.h[3]
    fmla v24.8h, v4.8h, v1.h[0]
    fmla v25.8h, v4.8h, v1.h[1]
    fmla v26.8h, v4.8h, v1.h[2]
    fmla v27.8h, v4.8h, v1.h[3]
    fmla v28.8h, v4.8h, v2.h[0]
    fmla v29.8h, v4.8h, v2.h[1]
    fmla v30.8h, v4.8h, v2.h[2]
    fmla v31.8h, v4.8h, v2.h[3]
    
    subs x12, x12, #1
    bne LoopL

LoopEPRemain:

    st1 {v8.8h, v9.8h, v10.8h, v11.8h}, [x0], #64
    st1 {v12.8h, v13.8h, v14.8h, v15.8h}, [x0], #64
    st1 {v16.8h, v17.8h, v18.8h, v19.8h}, [x0], #64

    st1 {v20.8h, v21.8h, v22.8h, v23.8h}, [x11], #64
    st1 {v24.8h, v25.8h, v26.8h, v27.8h}, [x11], #64
    st1 {v28.8h, v29.8h, v30.8h, v31.8h}, [x11], #64

    subs x13, x13, #1
    bne LoopEP

    add x0, x0, x8
    add x2, x2, x9
    subs x5, x5, #1
    bne LoopHP

sub sp, sp, #128
ld1 {v8.8h, v9.8h, v10.8h, v11.8h}, [sp], #64
ld1 {v12.8h, v13.8h, v14.8h, v15.8h}, [sp], #64

End:
ret
#endif
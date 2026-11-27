.arch_extension sve2
.text
.p2align 4
.hidden nc_av1_grain_high_luma_sve2
.global nc_av1_grain_high_luma_sve2
.type nc_av1_grain_high_luma_sve2, %function
nc_av1_grain_high_luma_sve2:
    // x0 source, x1 grain, x2 Q15 table, x3 output, x4 sample count.
    // Activate four indices so each gather works at any SVE vector length.
    ptrue   p0.s, vl4
    movi    v4.8h, #0
    mov     w7, #4095
    dup     v5.8h, w7
    dup     v6.8h, w5
    dup     v7.8h, w6
1:
    ldr     q0, [x0], #16
    ldr     q3, [x1], #16
    smax    v1.8h, v0.8h, v4.8h
    smin    v1.8h, v1.8h, v5.8h
    uxtl2   v2.4s, v1.8h
    uxtl    v1.4s, v1.4h
    ld1h    {z1.s}, p0/z, [x2, z1.s, uxtw #1]
    ld1h    {z2.s}, p0/z, [x2, z2.s, uxtw #1]
    xtn     v1.4h, v1.4s
    xtn2    v1.8h, v2.4s
    sqrdmulh v1.8h, v1.8h, v3.8h
    add     v0.8h, v0.8h, v1.8h
    smax    v0.8h, v0.8h, v6.8h
    smin    v0.8h, v0.8h, v7.8h
    str     q0, [x3], #16
    subs    x4, x4, #8
    b.ne    1b
    ret
.size nc_av1_grain_high_luma_sve2, .-nc_av1_grain_high_luma_sve2

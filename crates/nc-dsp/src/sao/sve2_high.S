.arch_extension sve2
.text
.p2align 4
.hidden nc_sao_edge_u16_sve2
.global nc_sao_edge_u16_sve2
.type nc_sao_edge_u16_sve2, %function
nc_sao_edge_u16_sve2:
    ldp     x1, x2, [x0]
    ldp     x3, x4, [x0, #16]
    ldp     x5, x6, [x0, #32]
    ldp     x7, x8, [x0, #48]
    ldrh    w9, [x0, #64]
    add     x10, x0, #66
    ptrue   p0.h
    ptrue   p2.h, vl8
    ld1h    {z4.h}, p2/z, [x10]
    dup     z19.h, w9
    mov     z16.h, #-1
    mov     z17.h, #1
    mov     z18.h, #0
    cmp     x6, x3
    b.ne    1f
    neg     x13, x3
    cmp     x5, x13
    b.eq    .Lvertical
1:
    add     x11, x1, x5
    add     x12, x1, x6
    mov     x9, #0
    whilelo p1.h, x9, x7
2:
    ld1h    {z0.h}, p1/z, [x1, x9, lsl #1]
    ld1h    {z1.h}, p1/z, [x11, x9, lsl #1]
    ld1h    {z2.h}, p1/z, [x12, x9, lsl #1]
    sub     z1.h, z0.h, z1.h
    sub     z2.h, z0.h, z2.h
    smax    z1.h, p0/m, z1.h, z16.h
    smax    z2.h, p0/m, z2.h, z16.h
    smin    z1.h, p0/m, z1.h, z17.h
    smin    z2.h, p0/m, z2.h, z17.h
    add     z1.h, z1.h, z2.h
    add     z1.h, z1.h, #2
    tbl     z1.h, {z4.h}, z1.h
    sqadd   z0.h, z0.h, z1.h
    smax    z0.h, p0/m, z0.h, z18.h
    smin    z0.h, p0/m, z0.h, z19.h
    st1h    {z0.h}, p1, [x2, x9, lsl #1]
    inch    x9
    whilelo p1.h, x9, x7
    b.ne    2b
    add     x1, x1, x3
    add     x2, x2, x4
    subs    x8, x8, #1
    b.ne    1b
    ret
.macro nc_sao_vertical_row current, next, previous_sign, next_sign
    ld1h    {\next\().h}, p1/z, [x13]
    sub     \next_sign\().h, \current\().h, \next\().h
    smax    \next_sign\().h, p0/m, \next_sign\().h, z16.h
    smin    \next_sign\().h, p0/m, \next_sign\().h, z17.h
    sub     z5.h, \next_sign\().h, \previous_sign\().h
    add     z5.h, z5.h, #2
    tbl     z5.h, {z4.h}, z5.h
    sqadd   z6.h, \current\().h, z5.h
    smax    z6.h, p0/m, z6.h, z18.h
    smin    z6.h, p0/m, z6.h, z19.h
    st1h    {z6.h}, p1, [x11]
    add     x13, x13, x3
    add     x11, x11, x4
.endm

// Each forward sign is subtracted at the next row. Alternating
// current and next registers keeps the pair entirely in registers.
.Lvertical:
    mov     x9, #0
    and     x15, x8, #1
    whilelo p1.h, x9, x7
.Lvertical_columns:
    add     x10, x1, x9, lsl #1
    add     x11, x2, x9, lsl #1
    sub     x12, x10, x3
    add     x13, x10, x3
    ld1h    {z0.h}, p1/z, [x10]
    ld1h    {z1.h}, p1/z, [x12]
    sub     z1.h, z1.h, z0.h
    smax    z1.h, p0/m, z1.h, z16.h
    smin    z1.h, p0/m, z1.h, z17.h
    lsr     x14, x8, #1
    cbz     x14, .Lvertical_last
.Lvertical_rows:
    nc_sao_vertical_row z0, z2, z1, z3
    nc_sao_vertical_row z2, z0, z3, z1
    subs    x14, x14, #1
    b.ne    .Lvertical_rows
.Lvertical_last:
    cbz     x15, .Lvertical_advance
    nc_sao_vertical_row z0, z2, z1, z3
.Lvertical_advance:
    inch    x9
    whilelo p1.h, x9, x7
    b.ne    .Lvertical_columns
    ret
.purgem nc_sao_vertical_row
.size nc_sao_edge_u16_sve2, .-nc_sao_edge_u16_sve2


/root/cs254a/project20/analysis-existing-buffers/single.o:	file format elf32-littleriscv

Disassembly of section .text.kernel_main:

00000000 <kernel_main>:
       0:      	addi	sp, sp, -0x20
       4:      	sw	a0, 0x4(sp)
       8:      	lw	a3, 0x4(a0)
       c:      	lw	s0, 0x8(a0)
      10:      	lw	a4, 0x20(a0)
      14:      	fmv.w.x	fa5, zero
      18:      	csrr	a1, 0xcd3
      1c:      	csrr	a2, 0xcd9
      20:      	csrr	a6, 0xcd7
      24:      	csrr	a5, 0xcd6
      28:      	csrr	a0, 0xcdf
      2c:      	csrr	a7, 0xcd0
      30:      	csrr	t0, 0xcd2
      34:      	fmv.s	ft0, fa5
      38:      	fmv.s	ft1, fa5
      3c:      	fmv.s	ft2, fa5
      40:      	fmv.s	ft3, fa5
      44:      	fmv.s	ft4, fa5
      48:      	fmv.s	ft5, fa5
      4c:      	fmv.s	ft6, fa5
      50:      	fmv.s	ft7, fa5
      54:      	fmv.s	fs0, fa5
      58:      	fmv.s	fs1, fa5
      5c:      	fmv.s	fa0, fa5
      60:      	fmv.s	fa1, fa5
      64:      	fmv.s	fa2, fa5
      68:      	fmv.s	fa3, fa5
      6c:      	fmv.s	fa4, fa5
      70:      	fmv.s	fa5, fa5
      74:      	csrr	t1, 0xcd1
      78:      	srli	a2, a2, 0x1
      7c:      	slli	a5, a5, 0x5
      80:      	andi	a2, a2, -0x10
      84:      	mul	a6, a6, a2
      88:      	slli	a2, a2, 0x5
      8c:      	add	a2, a0, a2
      90:      	bnez	t1, 0x90 <kernel_main+0x90>
      94:      	li	t2, 0x0
      98:      	mv	t6, a0
      9c:      	lui	t3, 0x80000
      a0:      	addi	t3, t3, 0x2
      a4:      	<unknown>
      a8:      	slli	t3, a7, 0x4
      ac:      	vx_tex	t6, t3, t2, a6
      b0:      	addi	t3, t3, 0x1
      b4:      	<unknown>
      b8:      	mv	t6, a2
      bc:      	vx_tex	t6, t3, a5, t2
      c0:      	<unknown>
      c4:      	li	t3, 0x10
      c8:      	srli	t2, a1, 0x5
      cc:      	bltu	s0, t3, 0xcc <kernel_main+0xcc>
      d0:      	vx_bar	a7, t0
      d4:      	csrr	a1, ng
      d8:      	csrr	t3, tid
      dc:      	slli	t4, t2, 0x9
      e0:      	add	t4, a0, t4
      e4:      	slli	t5, t3, 0x1
      e8:      	slli	t3, t3, 0x2
      ec:      	andi	t5, t5, 0x6
      f0:      	andi	t3, t3, -0x10
      f4:      	slli	t6, t5, 0x1
      f8:      	slli	s1, t3, 0x1
      fc:      	or	t3, t3, t5
     100:      	add	t6, t4, t6
     104:      	slli	t5, t3, 0x1
     108:      	add	t3, t6, s1
     10c:      	add	t4, t5, t4
     110:      	flw	fs8, 0x0(t3)
     114:      	flw	fs9, 0x10(t4)
     118:      	flw	fs10, 0x100(t4)
     11c:      	flw	fs11, 0x110(t4)
     120:      	srli	t5, s0, 0x4
     124:      	sub	a1, a2, a1
     128:      	li	t6, 0x20
     12c:      	<unknown>
     130:      	bltu	s0, t6, 0x130 <kernel_main+0x130>
     134:      	vx_bar	a7, t0
     138:      	li	t6, 0x1
     13c:      	bne	t5, t6, 0x13c <kernel_main+0x13c>
     140:      	slli	t2, t2, 0x4
     144:      	slli	a5, a5, 0x2
     148:      	add	a6, a6, t2
     14c:      	csrr	a0, tid
     150:      	mul	a1, a6, a3
     154:      	slli	a1, a1, 0x2
     158:      	add	a1, a4, a1
     15c:      	slli	a2, a3, 0x5
     160:      	add	a1, a1, a5
     164:      	srli	a4, a0, 0x2
     168:      	andi	a0, a0, 0x3
     16c:      	mul	a3, a4, a3
     170:      	slli	a0, a0, 0x2
     174:      	slli	a3, a3, 0x2
     178:      	add	a0, a1, a0
     17c:      	add	a0, a0, a3
     180:      	add	a2, a0, a2
     184:      	fsw	ft0, 0x0(a0)
     188:      	fsw	ft1, 0x0(a2)
     18c:      	fsw	ft2, 0x10(a0)
     190:      	fsw	ft3, 0x10(a2)
     194:      	fsw	ft4, 0x20(a0)
     198:      	fsw	ft6, 0x30(a0)
     19c:      	fsw	ft5, 0x20(a2)
     1a0:      	fsw	ft7, 0x30(a2)
     1a4:      	fsw	fs0, 0x40(a0)
     1a8:      	fsw	fs1, 0x40(a2)
     1ac:      	fsw	fa0, 0x50(a0)
     1b0:      	fsw	fa1, 0x50(a2)
     1b4:      	fsw	fa2, 0x60(a0)
     1b8:      	fsw	fa4, 0x70(a0)
     1bc:      	fsw	fa3, 0x60(a2)
     1c0:      	fsw	fa5, 0x70(a2)
     1c4:      	addi	sp, sp, 0x20
     1c8:      	ret
     1cc:      	slli	s0, a7, 0x4
     1d0:      	addi	s1, t5, 0x1
     1d4:      	li	s2, 0x2
     1d8:      	li	s3, 0x10
     1dc:      	lui	s5, 0x80000
     1e0:      	addi	s4, s0, 0x1
     1e4:      	addi	s5, s5, 0x2
     1e8:      	j	0x1e8 <kernel_main+0x1e8>
     1ec:      	addi	s2, s2, 0x1
     1f0:      	addi	s3, s3, 0x10
     1f4:      	beq	s2, s1, 0x1f4 <kernel_main+0x1f4>
     1f8:      	bnez	t1, 0x1f8 <kernel_main+0x1f8>
     1fc:      	<unknown>
     200:      	mv	t6, a0
     204:      	vx_tex	t6, s0, s3, a6
     208:      	<unknown>
     20c:      	mv	t6, a2
     210:      	vx_tex	t6, s4, a5, s3
     214:      	<unknown>
     218:      	vx_bar	a7, t0
     21c:      	flw	fs8, 0x0(t3)
     220:      	flw	fs9, 0x10(t4)
     224:      	flw	fs10, 0x100(t4)
     228:      	flw	fs11, 0x110(t4)
     22c:      	<unknown>
     230:      	bgeu	s2, t5, 0x230 <kernel_main+0x230>
     234:      	vx_bar	a7, t0
     238:      	j	0x238 <kernel_main+0x238>


/root/cs254a/project20/analysis-existing-buffers/double.o:	file format elf32-littleriscv

Disassembly of section .text.kernel_main:

00000000 <kernel_main>:
       0:      	addi	sp, sp, -0x20
       4:      	sw	a0, 0x4(sp)
       8:      	lw	a4, 0x4(a0)
       c:      	lw	a2, 0x8(a0)
      10:      	lw	a5, 0x20(a0)
      14:      	fmv.w.x	fa5, zero
      18:      	csrr	t3, 0xcd3
      1c:      	csrr	a1, 0xcd9
      20:      	csrr	a3, 0xcd7
      24:      	csrr	a6, 0xcd6
      28:      	csrr	a0, 0xcdf
      2c:      	csrr	t2, 0xcd0
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
      78:      	srli	a1, a1, 0x1
      7c:      	slli	a6, a6, 0x5
      80:      	andi	a1, a1, -0x10
      84:      	mul	a7, a3, a1
      88:      	slli	a3, a1, 0x5
      8c:      	add	a1, a0, a3
      90:      	bnez	t1, 0x90 <kernel_main+0x90>
      94:      	li	t4, 0x0
      98:      	mv	t6, a0
      9c:      	lui	t5, 0x80000
      a0:      	addi	t5, t5, 0x2
      a4:      	<unknown>
      a8:      	slli	t5, t2, 0x4
      ac:      	vx_tex	t6, t5, t4, a7
      b0:      	addi	t5, t5, 0x1
      b4:      	<unknown>
      b8:      	mv	t6, a1
      bc:      	vx_tex	t6, t5, a6, t4
      c0:      	<unknown>
      c4:      	li	t4, 0x10
      c8:      	srli	t3, t3, 0x5
      cc:      	bgeu	a2, t4, 0xcc <kernel_main+0xcc>
      d0:      	slli	t3, t3, 0x4
      d4:      	slli	a6, a6, 0x2
      d8:      	add	a7, a7, t3
      dc:      	csrr	a0, tid
      e0:      	mul	a1, a7, a4
      e4:      	slli	a1, a1, 0x2
      e8:      	add	a1, a5, a1
      ec:      	slli	a2, a4, 0x5
      f0:      	add	a1, a1, a6
      f4:      	srli	a3, a0, 0x2
      f8:      	andi	a0, a0, 0x3
      fc:      	mul	a3, a3, a4
     100:      	slli	a0, a0, 0x2
     104:      	slli	a3, a3, 0x2
     108:      	add	a0, a1, a0
     10c:      	add	a0, a0, a3
     110:      	add	a2, a0, a2
     114:      	fsw	ft0, 0x0(a0)
     118:      	fsw	ft1, 0x0(a2)
     11c:      	fsw	ft2, 0x10(a0)
     120:      	fsw	ft3, 0x10(a2)
     124:      	fsw	ft4, 0x20(a0)
     128:      	fsw	ft6, 0x30(a0)
     12c:      	fsw	ft5, 0x20(a2)
     130:      	fsw	ft7, 0x30(a2)
     134:      	fsw	fs0, 0x40(a0)
     138:      	fsw	fs1, 0x40(a2)
     13c:      	fsw	fa0, 0x50(a0)
     140:      	fsw	fa1, 0x50(a2)
     144:      	fsw	fa2, 0x60(a0)
     148:      	fsw	fa4, 0x70(a0)
     14c:      	fsw	fa3, 0x60(a2)
     150:      	fsw	fa5, 0x70(a2)
     154:      	addi	sp, sp, 0x20
     158:      	ret
     15c:      	addi	t6, a1, 0x400
     160:      	srli	t5, a2, 0x4
     164:      	slli	s0, t3, 0x9
     168:      	li	s1, 0x1
     16c:      	lui	s4, 0x80000
     170:      	add	a3, t6, a3
     174:      	xor	s2, a0, t6
     178:      	neg	s3, t5
     17c:      	addi	s4, s4, 0x2
     180:      	xor	s5, a3, a1
     184:      	li	s6, 0x1
     188:      	j	0x188 <kernel_main+0x188>
     18c:      	add	t6, s3, s6
     190:      	addi	t4, t4, 0x10
     194:      	mv	a0, a2
     198:      	mv	a1, a3
     19c:      	mv	t2, s7
     1a0:      	beq	t6, s1, 0x1a0 <kernel_main+0x1a0>
     1a4:      	xor	a2, s2, a0
     1a8:      	xor	a3, a1, s5
     1ac:      	xori	s7, t2, 0x100
     1b0:      	bnez	t1, 0x1b0 <kernel_main+0x1b0>
     1b4:      	bgeu	s6, t5, 0x1b4 <kernel_main+0x1b4>
     1b8:      	mv	t6, a2
     1bc:      	<unknown>
     1c0:      	slli	s8, s7, 0x4
     1c4:      	vx_tex	t6, s8, t4, a7
     1c8:      	addi	s8, s8, 0x1
     1cc:      	<unknown>
     1d0:      	mv	t6, a3
     1d4:      	vx_tex	t6, s8, a6, t4
     1d8:      	<unknown>
     1dc:      	vx_bar	t2, t0
     1e0:      	csrr	t6, ng
     1e4:      	csrr	s8, tid
     1e8:      	add	a0, a0, s0
     1ec:      	slli	s9, s8, 0x1
     1f0:      	slli	s8, s8, 0x2
     1f4:      	andi	s9, s9, 0x6
     1f8:      	andi	s8, s8, -0x10
     1fc:      	slli	s10, s9, 0x1
     200:      	or	s9, s8, s9
     204:      	slli	s8, s8, 0x1
     208:      	add	s10, a0, s10
     20c:      	slli	s9, s9, 0x1
     210:      	add	s8, s10, s8
     214:      	add	a0, s9, a0
     218:      	flw	fs8, 0x0(s8)
     21c:      	flw	fs9, 0x10(a0)
     220:      	flw	fs10, 0x100(a0)
     224:      	flw	fs11, 0x110(a0)
     228:      	sub	a1, a1, t6
     22c:      	addi	s6, s6, 0x1
     230:      	<unknown>
     234:      	bgeu	s6, t5, 0x234 <kernel_main+0x234>
     238:      	vx_bar	t2, t0
     23c:      	j	0x23c <kernel_main+0x23c>

.section .text
.globl main
.type main,@function
main:
   csrr t0, 0x803
   lw   t1, 4(t0)
   lw   t2, 8(t1)

   li   t3, 0xff00ffff
   sw   t3, 0(t2)
   sw   t1, 4(t2)
   sw   t2, 8(t2)
   lw   t3, 48(t1)
   sw   t3, 12(t2)

   la   t0, tohost
   li   t1, 1
   sw   t1, 0(t0)
   sw   zero, 4(t0)
1:
   j    1b

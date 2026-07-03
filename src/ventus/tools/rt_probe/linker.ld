OUTPUT_ARCH(riscv)
ENTRY(main)

SECTIONS
{
   . = 0x80000000;

   .text : {
      *(.text .text.*)
   }

   . = ALIGN(16);
   .eh_frame : {
      *(.eh_frame .eh_frame.*)
   }

   . = ALIGN(16);
   .rodata : {
      *(.rodata .rodata.*)
   }

   . = ALIGN(16);
   .data : {
      *(.data .data.*)
      *(.sdata .sdata.*)
   }

   . = ALIGN(16);
   .bss : {
      *(.bss .bss.*)
      *(COMMON)
   }

   . = ALIGN(8);
   .tohost : {
      tohost = .;
      . += 8;
      fromhost = .;
      . += 8;
   }
}

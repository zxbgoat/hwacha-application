/*======================================================================*/
/* Proxy kernel linker script                                           */
/*======================================================================*/
/* This is the linker script used when building the proxy kernel. */

/*----------------------------------------------------------------------*/
/* Setup                                                                */
/*----------------------------------------------------------------------*/

/* The OUTPUT_ARCH command specifies the machine architecture where the
   argument is one of the names used in the BFD library. More
   specifically one of the entires in bfd/cpu-mips.c */

OUTPUT_ARCH( "riscv" )
ENTRY(_start)

/*----------------------------------------------------------------------*/
/* Sections                                                             */
/*----------------------------------------------------------------------*/

SECTIONS
{

  /* text: test code section */
  . = 0x80000000;
  .text.init : { *(.text.init) }

  . = ALIGN(0x1000);
  .tohost : { *(.tohost) }

  /* runtime (crt.S / syscalls.c) and the host stay within JAL reach of .text.init */
  .text.crt : { *crt.o(.text .text.*) *syscalls.o(.text .text.*) *tv_main.o(.text .text.*) }

  /* first half of the model weights, in front of the code (see split_weights.py) */
  . = ALIGN(0x1000);
  .weights_lo : { *(.weights_lo) }

  . = ALIGN(0x1000);
  .text : { *(.text .text.*) }
  /* read-only data placed explicitly: an orphan section would land after .weights_hi, past _image_end,
     and the host's arena would grow over it */
  .rodata : { *(.rodata .rodata.* .gnu.linkonce.r.*) }
  .eh_frame : { *(.eh_frame) }

  /* data segment */
  .data : { *(.data) }

  .sdata : {
    __global_pointer$ = . + 0x800;
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  }

  /* bss segment */
  .sbss : {
    *(.sbss .sbss.* .gnu.linkonce.sb.*)
    *(.scommon)
  }
  .bss : { *(.bss) }

  /* thread-local data segment */
  .tdata :
  {
    _tdata_begin = .;
    *(.tdata)
    _tdata_end = .;
  }
  .tbss :
  {
    *(.tbss)
    _tbss_end = .;
  }

  /* End of uninitalized data segement */
  _end = .;

  /* second half of the weights, past the crt stacks (128 KB per hart from _end); the host's arena
     starts at _image_end */
  . = _end + 0x200000;
  .weights_hi : { *(.weights_hi) }
  _image_end = .;
}


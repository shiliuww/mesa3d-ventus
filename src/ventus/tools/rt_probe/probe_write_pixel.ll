target triple = "riscv32-unknown-elf"

@tohost = external global i64, align 8

declare i32 @llvm.riscv.ventus.kernel.metadata()

define i32 @main() {
entry:
  %metadata = call i32 @llvm.riscv.ventus.kernel.metadata()
  %args_slot_addr = add i32 %metadata, 4
  %args_slot_ptr = inttoptr i32 %args_slot_addr to ptr addrspace(1)
  %args = load i32, ptr addrspace(1) %args_slot_ptr, align 4

  %image_slot_addr = add i32 %args, 8
  %image_slot_ptr = inttoptr i32 %image_slot_addr to ptr addrspace(1)
  %image = load i32, ptr addrspace(1) %image_slot_ptr, align 4

  %pixel0 = inttoptr i32 %image to ptr addrspace(1)
  store i32 -16711681, ptr addrspace(1) %pixel0, align 4

  %pixel1_addr = add i32 %image, 4
  %pixel1 = inttoptr i32 %pixel1_addr to ptr addrspace(1)
  store i32 %args, ptr addrspace(1) %pixel1, align 4

  %pixel2_addr = add i32 %image, 8
  %pixel2 = inttoptr i32 %pixel2_addr to ptr addrspace(1)
  store i32 %image, ptr addrspace(1) %pixel2, align 4

  %width_slot_addr = add i32 %args, 48
  %width_slot_ptr = inttoptr i32 %width_slot_addr to ptr addrspace(1)
  %width = load i32, ptr addrspace(1) %width_slot_ptr, align 4

  %pixel3_addr = add i32 %image, 12
  %pixel3 = inttoptr i32 %pixel3_addr to ptr addrspace(1)
  store i32 %width, ptr addrspace(1) %pixel3, align 4

  store volatile i64 1, ptr @tohost, align 8
  br label %done

done:
  br label %done
}

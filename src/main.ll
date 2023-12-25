define i32 @main() {
entry:
  %a = alloca i32, align 4
  store i32 1, ptr %a, align 4
  %b = alloca i32, align 4
  store i32 1, ptr %b, align 4
  %a1 = load ptr, ptr %a, align 8
  %b2 = load ptr, ptr %b, align 8
  %equal = icmp eq ptr %a1, %b2
  br i1 %equal, label %main_entry_then_block, label %main_entry_else_block

main_entry_then_block:                            ; preds = %entry
  ret i32 1

main_entry_else_block:                            ; preds = %entry
  ret i32 0

}
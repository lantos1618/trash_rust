; ModuleID = 'main'
source_filename = "main"

define void @main() {
main_entry_block:
  %a = alloca i32, align 4
  store i32 1, ptr %a, align 4
  ret void
}
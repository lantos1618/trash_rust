; ModuleID = 'main'
source_filename = "main"

define i32 @main(i32 %0) {
main:
  %a = alloca i32, align 4
  store i32 %0, ptr %a, align 4
  %b = alloca i32, align 4
  %add_left = load i32, ptr %a, align 4
  %add = add i32 %add_left, 10
  store i32 %add, ptr %b, align 4
  %b1 = load i32, ptr %b, align 4
  ret i32 %b1
}
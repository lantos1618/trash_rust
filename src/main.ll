; ModuleID = 'my_module'
source_filename = "my_module"

declare i32 @printf(i8*, ...)

define i64 @main() {
entry:
  %x = alloca i64, align 8
  store i64 42, i64* %x, align 8
  br label %loop

loop:                                             ; preds = %loop, %entry
  %x1 = load i64, i64* %x, align 8
  %inc = add i64 %x1, 1
  store i64 %inc, i64* %x, align 8
  %lttmp = icmp slt i64 %inc, 100
  br i1 %lttmp, label %loop, label %afterloop

afterloop:                                        ; preds = %loop
  %x2 = load i64, i64* %x, align 8
  %fmt = getelementptr [5 x i8], [5 x i8]* @.str, i32 0, i32 0
  call i32 (i8*, ...) @printf(i8* %fmt, i64 %x2)
  ret i64 %x2
}

@.str = private unnamed_addr constant [5 x i8] c"%ld\0A\00", align 1

; ; ModuleID = 'my_module'
; source_filename = "my_module"

; declare i32 @printf(i8*, ...)

; define i64 @main() {
; entry:
;   %x = alloca i64, align 8
;   store i64 42, i64* %x, align 8
;   br label %loop

; loop:                                             ; preds = %loop, %entry
;   %x1 = load i64, i64* %x, align 8
;   %inc = add i64 %x1, 1
;   store i64 %inc, i64* %x, align 8
;   %lttmp = icmp slt i64 %inc, 100
;   %loop_cond = icmp ne i1 %lttmp, 0
;   br i1 %loop_cond, label %loop, label %afterloop

; afterloop:                                        ; preds = %loop
;   %x2 = load i64, i64* %x, align 8
;   %fmt = getelementptr [5 x i8], [5 x i8]* @.str, i32 0, i32 0
;   call i32 (i8*, ...) @printf(i8* %fmt, i64 %x2)
;   ret i64 %x2
; }

; @.str = private unnamed_addr constant [5 x i8] c"%ld\0A\00", align 1




; declare i32 @printf(i8*, ...)

; define i64 @main() {
; entry:
;   %x = alloca i64, align 8
;   store i64 42, ptr %x, align 8
;   br label %loop

; loop:                                             ; preds = %loop, %entry
;   %x1 = load i64, ptr %x, align 8
;   %inc = add i64 %x1, 1
;   store i64 %inc, ptr %x, align 8
;   %lttmp = icmp slt i64 %x1, 100
;   %loop_cond = icmp ne i1 %lttmp, 0
;   br i1 %loop_cond, label %loop, label %after_loop

; after_loop:                                       ; preds = %loop
;   %fmt = getelementptr [14 x i8], [14 x i8]* @.str, i32 0, i32 0
;   %calltmp = call i32 (ptr, i8, ...) @printf(i8* %fmt, i8 0)
;   %x2 = load i64, ptr %x, align 8
;   ret i64 %x2
; }

; @.str = private unnamed_addr constant [14 x i8] c"Hello, world!\0A" align 1


;  ModuleID = 'main'
source_filename = "main"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

declare i32 @printf(i8*, ...)

define i64 @main() {
entry:
  %x = alloca i64, align 8
  store i64 0, ptr %x, align 8
  br label %loop

loop:                                             ; preds = %loop, %entry
  %x1 = load i64, ptr %x, align 8
  %addtmp = add i64 %x1, 1
  store i64 %addtmp, ptr %x, align 8
  %x2 = load i64, ptr %x, align 8
  %lttmp = icmp slt i64 %x2, 420000000
  br i1 %lttmp, label %loop, label %after_loop

after_loop:                                       ; preds = %loop
  %x3 = load i64, ptr %x, align 8
  %fmt = getelementptr [5 x i8], [5 x i8]* @.str, i32 0, i32 0
  %calltmp = call i32 (i8*, ...) @printf(i8* %fmt, i64 %x3)

  ret i64 %x3
}

@.str = private unnamed_addr constant [5 x i8] c"%ld\0A\00", align 1
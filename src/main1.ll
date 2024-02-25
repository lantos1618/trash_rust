; ModuleID = 'src/main.c'
source_filename = "src/main.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i64, align 8
  store i32 0, ptr %1, align 4
  store i64 0, ptr %2, align 8
  br label %3

3:                                                ; preds = %6, %0
  %4 = load i64, ptr %2, align 8
  %5 = icmp slt i64 %4, 420000000
  br i1 %5, label %6, label %9

6:                                                ; preds = %3
  %7 = load i64, ptr %2, align 8
  %8 = add nsw i64 %7, 1
  store i64 %8, ptr %2, align 8
  br label %3, !llvm.loop !5

9:                                                ; preds = %3
  %10 = load i64, ptr %2, align 8
  %11 = trunc i64 %10 to i32
  ret i32 %11
}

attributes #0 = { noinline nounwind optnone ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 1}
!3 = !{i32 7, !"frame-pointer", i32 1}
!4 = !{!"Homebrew clang version 17.0.6"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}

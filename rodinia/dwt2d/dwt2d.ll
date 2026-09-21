; ModuleID = 'dwt2d/dwt2d.cl'
source_filename = "dwt2d/dwt2d.cl"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

%struct.FDWT53 = type { i32, i32, %struct.FDWT53Column, %struct.TransformBuffer, i32 }
%struct.FDWT53Column = type { i8, %struct.VerticalDWTPixelLoader, i32, i32, i32, i32 }
%struct.VerticalDWTPixelLoader = type { i8, i32 }
%struct.TransformBuffer = type { i32, i32, i32, i32, i32, i32, i32, [2182 x i32] }
%struct.VerticalDWTPixelIO = type { i8, i32, i32 }

@c_CopySrcToComponents.sData = internal unnamed_addr global [768 x i8] undef, align 1
@c_CopySrcToComponent.sData = internal unnamed_addr global [256 x i8] undef, align 1
@cl_fdwt53Kernel.fdwt53 = internal global %struct.FDWT53 undef, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local signext i32 @divRndUp(i32 noundef signext %0, i32 noundef signext %1) local_unnamed_addr #0 {
  %3 = sdiv i32 %0, %1
  %4 = mul i32 %3, %1
  %5 = sub i32 %0, %4
  %6 = icmp ne i32 %5, 0
  %7 = zext i1 %6 to i32
  %8 = add nsw i32 %3, %7
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define dso_local void @storeComponents(ptr nofree noundef writeonly captures(none) %0, ptr nofree noundef writeonly captures(none) %1, ptr nofree noundef writeonly captures(none) %2, i32 noundef signext %3, i32 noundef signext %4, i32 noundef signext %5, i32 noundef signext %6) local_unnamed_addr #1 {
  %8 = add nsw i32 %3, -128
  %9 = sext i32 %6 to i64
  %10 = getelementptr inbounds [4 x i8], ptr %0, i64 %9
  store i32 %8, ptr %10, align 4, !tbaa !11
  %11 = add nsw i32 %4, -128
  %12 = getelementptr inbounds [4 x i8], ptr %1, i64 %9
  store i32 %11, ptr %12, align 4, !tbaa !11
  %13 = add nsw i32 %5, -128
  %14 = getelementptr inbounds [4 x i8], ptr %2, i64 %9
  store i32 %13, ptr %14, align 4, !tbaa !11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define dso_local void @storeComponent(ptr nofree noundef writeonly captures(none) %0, i32 noundef signext %1, i32 noundef signext %2) local_unnamed_addr #1 {
  %4 = add nsw i32 %1, -128
  %5 = sext i32 %2 to i64
  %6 = getelementptr inbounds [4 x i8], ptr %0, i64 %5
  store i32 %4, ptr %6, align 4, !tbaa !11
  ret void
}

; Function Attrs: convergent norecurse nounwind
define dso_local void @c_CopySrcToComponents(ptr nofree noundef writeonly align 4 captures(none) %0, ptr nofree noundef writeonly align 4 captures(none) %1, ptr nofree noundef writeonly align 4 captures(none) %2, ptr nofree noundef readonly align 1 captures(none) %3, i32 noundef %4) local_unnamed_addr #2 !kernel_arg_addr_space !12 !kernel_arg_access_qual !13 !kernel_arg_type !14 !kernel_arg_base_type !14 !kernel_arg_type_qual !15 {
  %6 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #14
  %7 = trunc i64 %6 to i32
  %8 = tail call i64 @_Z14get_local_sizej(i32 noundef signext 0) #14
  %9 = tail call i64 @_Z12get_group_idj(i32 noundef signext 0) #14
  %10 = mul i64 %9, %8
  %11 = trunc i64 %10 to i32
  %12 = add i32 %11, %7
  %13 = mul i32 %12, 3
  %14 = sext i32 %13 to i64
  %15 = getelementptr inbounds i8, ptr %3, i64 %14
  %16 = load i8, ptr %15, align 1, !tbaa !16
  %17 = mul i64 %6, 12884901888
  %18 = ashr exact i64 %17, 32
  %19 = getelementptr inbounds i8, ptr @c_CopySrcToComponents.sData, i64 %18
  store i8 %16, ptr %19, align 1, !tbaa !16
  %20 = getelementptr i8, ptr %15, i64 1
  %21 = load i8, ptr %20, align 1, !tbaa !16
  %22 = getelementptr i8, ptr %19, i64 1
  store i8 %21, ptr %22, align 1, !tbaa !16
  %23 = getelementptr i8, ptr %15, i64 2
  %24 = load i8, ptr %23, align 1, !tbaa !16
  %25 = getelementptr i8, ptr %19, i64 2
  store i8 %24, ptr %25, align 1, !tbaa !16
  tail call void @_Z7barrierj(i32 noundef signext 1) #15
  %26 = icmp slt i32 %12, %4
  br i1 %26, label %27, label %41

27:                                               ; preds = %5
  %28 = load i8, ptr %25, align 1, !tbaa !16
  %29 = zext i8 %28 to i32
  %30 = load i8, ptr %22, align 1, !tbaa !16
  %31 = zext i8 %30 to i32
  %32 = load i8, ptr %19, align 1, !tbaa !16
  %33 = zext i8 %32 to i32
  %34 = add nsw i32 %33, -128
  %35 = sext i32 %12 to i64
  %36 = getelementptr inbounds [4 x i8], ptr %0, i64 %35
  store i32 %34, ptr %36, align 4, !tbaa !11
  %37 = add nsw i32 %31, -128
  %38 = getelementptr inbounds [4 x i8], ptr %1, i64 %35
  store i32 %37, ptr %38, align 4, !tbaa !11
  %39 = add nsw i32 %29, -128
  %40 = getelementptr inbounds [4 x i8], ptr %2, i64 %35
  store i32 %39, ptr %40, align 4, !tbaa !11
  br label %41

41:                                               ; preds = %5, %27
  ret void
}

; Function Attrs: alwaysinline convergent norecurse nounwind
define dso_local void @__clang_ocl_kern_imp_c_CopySrcToComponents(ptr nofree noundef writeonly align 4 captures(none) %0, ptr nofree noundef writeonly align 4 captures(none) %1, ptr nofree noundef writeonly align 4 captures(none) %2, ptr nofree noundef readonly align 1 captures(none) %3, i32 noundef signext %4) local_unnamed_addr #3 !kernel_arg_addr_space !12 !kernel_arg_access_qual !13 !kernel_arg_type !14 !kernel_arg_base_type !14 !kernel_arg_type_qual !15 {
  %6 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #14
  %7 = trunc i64 %6 to i32
  %8 = tail call i64 @_Z14get_local_sizej(i32 noundef signext 0) #14
  %9 = tail call i64 @_Z12get_group_idj(i32 noundef signext 0) #14
  %10 = mul i64 %9, %8
  %11 = trunc i64 %10 to i32
  %12 = add i32 %11, %7
  %13 = mul i32 %12, 3
  %14 = sext i32 %13 to i64
  %15 = getelementptr inbounds i8, ptr %3, i64 %14
  %16 = load i8, ptr %15, align 1, !tbaa !16
  %17 = mul i64 %6, 12884901888
  %18 = ashr exact i64 %17, 32
  %19 = getelementptr inbounds i8, ptr @c_CopySrcToComponents.sData, i64 %18
  store i8 %16, ptr %19, align 1, !tbaa !16
  %20 = getelementptr i8, ptr %15, i64 1
  %21 = load i8, ptr %20, align 1, !tbaa !16
  %22 = getelementptr i8, ptr %19, i64 1
  store i8 %21, ptr %22, align 1, !tbaa !16
  %23 = getelementptr i8, ptr %15, i64 2
  %24 = load i8, ptr %23, align 1, !tbaa !16
  %25 = getelementptr i8, ptr %19, i64 2
  store i8 %24, ptr %25, align 1, !tbaa !16
  tail call void @_Z7barrierj(i32 noundef signext 1) #15
  %26 = icmp slt i32 %12, %4
  br i1 %26, label %27, label %41

27:                                               ; preds = %5
  %28 = load i8, ptr %25, align 1, !tbaa !16
  %29 = zext i8 %28 to i32
  %30 = load i8, ptr %22, align 1, !tbaa !16
  %31 = zext i8 %30 to i32
  %32 = load i8, ptr %19, align 1, !tbaa !16
  %33 = zext i8 %32 to i32
  %34 = add nsw i32 %33, -128
  %35 = sext i32 %12 to i64
  %36 = getelementptr inbounds [4 x i8], ptr %0, i64 %35
  store i32 %34, ptr %36, align 4, !tbaa !11
  %37 = add nsw i32 %31, -128
  %38 = getelementptr inbounds [4 x i8], ptr %1, i64 %35
  store i32 %37, ptr %38, align 4, !tbaa !11
  %39 = add nsw i32 %29, -128
  %40 = getelementptr inbounds [4 x i8], ptr %2, i64 %35
  store i32 %39, ptr %40, align 4, !tbaa !11
  br label %41

41:                                               ; preds = %27, %5
  ret void
}

; Function Attrs: convergent norecurse nounwind
define dso_local void @c_CopySrcToComponent(ptr nofree noundef writeonly align 4 captures(none) %0, ptr nofree noundef readonly align 1 captures(none) %1, i32 noundef %2) local_unnamed_addr #2 !kernel_arg_addr_space !17 !kernel_arg_access_qual !18 !kernel_arg_type !19 !kernel_arg_base_type !19 !kernel_arg_type_qual !20 {
  %4 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #14
  %5 = trunc i64 %4 to i32
  %6 = tail call i64 @_Z14get_local_sizej(i32 noundef signext 0) #14
  %7 = tail call i64 @_Z12get_group_idj(i32 noundef signext 0) #14
  %8 = mul i64 %7, %6
  %9 = trunc i64 %8 to i32
  %10 = add nsw i32 %9, %5
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds i8, ptr %1, i64 %11
  %13 = load i8, ptr %12, align 1, !tbaa !16
  %14 = shl i64 %4, 32
  %15 = ashr exact i64 %14, 32
  %16 = getelementptr inbounds i8, ptr @c_CopySrcToComponent.sData, i64 %15
  store i8 %13, ptr %16, align 1, !tbaa !16
  tail call void @_Z7barrierj(i32 noundef signext 1) #15
  %17 = icmp slt i32 %10, %2
  br i1 %17, label %18, label %23

18:                                               ; preds = %3
  %19 = load i8, ptr %16, align 1, !tbaa !16
  %20 = zext i8 %19 to i32
  %21 = add nsw i32 %20, -128
  %22 = getelementptr inbounds [4 x i8], ptr %0, i64 %11
  store i32 %21, ptr %22, align 4, !tbaa !11
  br label %23

23:                                               ; preds = %3, %18
  ret void
}

; Function Attrs: alwaysinline convergent norecurse nounwind
define dso_local void @__clang_ocl_kern_imp_c_CopySrcToComponent(ptr nofree noundef writeonly align 4 captures(none) %0, ptr nofree noundef readonly align 1 captures(none) %1, i32 noundef signext %2) local_unnamed_addr #3 !kernel_arg_addr_space !17 !kernel_arg_access_qual !18 !kernel_arg_type !19 !kernel_arg_base_type !19 !kernel_arg_type_qual !20 {
  %4 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #14
  %5 = trunc i64 %4 to i32
  %6 = tail call i64 @_Z14get_local_sizej(i32 noundef signext 0) #14
  %7 = tail call i64 @_Z12get_group_idj(i32 noundef signext 0) #14
  %8 = mul i64 %7, %6
  %9 = trunc i64 %8 to i32
  %10 = add nsw i32 %9, %5
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds i8, ptr %1, i64 %11
  %13 = load i8, ptr %12, align 1, !tbaa !16
  %14 = shl i64 %4, 32
  %15 = ashr exact i64 %14, 32
  %16 = getelementptr inbounds i8, ptr @c_CopySrcToComponent.sData, i64 %15
  store i8 %13, ptr %16, align 1, !tbaa !16
  tail call void @_Z7barrierj(i32 noundef signext 1) #15
  %17 = icmp slt i32 %10, %2
  br i1 %17, label %18, label %23

18:                                               ; preds = %3
  %19 = load i8, ptr %16, align 1, !tbaa !16
  %20 = zext i8 %19 to i32
  %21 = add nsw i32 %20, -128
  %22 = getelementptr inbounds [4 x i8], ptr %0, i64 %11
  store i32 %21, ptr %22, align 4, !tbaa !11
  br label %23

23:                                               ; preds = %18, %3
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define dso_local signext i32 @initialize_PixelIO(ptr nofree noundef writeonly captures(none) initializes((0, 1), (4, 12)) %0, i1 noundef zeroext %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4, i32 noundef signext %5) local_unnamed_addr #1 {
  %7 = zext i1 %1 to i8
  store i8 %7, ptr %0, align 4, !tbaa !21
  %8 = mul nsw i32 %3, %2
  %9 = add nsw i32 %8, %4
  %10 = select i1 %1, i32 %9, i32 0
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 %10, ptr %11, align 4, !tbaa !24
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i32 %2, ptr %12, align 4, !tbaa !25
  %13 = mul nsw i32 %5, %2
  %14 = add nsw i32 %13, %4
  ret i32 %14
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define dso_local void @init_PixelLoader(ptr nofree noundef writeonly captures(none) initializes((4, 8)) %0, i32 noundef signext %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4, ptr nofree noundef writeonly captures(none) initializes((0, 1), (4, 12)) %5, i1 noundef zeroext %6) local_unnamed_addr #1 {
  %8 = icmp slt i32 %3, %1
  br i1 %8, label %12, label %9

9:                                                ; preds = %7
  %10 = shl nsw i32 %1, 1
  %11 = add nsw i32 %10, -2
  br label %14

12:                                               ; preds = %7
  %13 = icmp slt i32 %3, 0
  br i1 %13, label %14, label %17

14:                                               ; preds = %12, %9
  %15 = phi i32 [ %11, %9 ], [ 0, %12 ]
  %16 = sub i32 %15, %3
  br label %17

17:                                               ; preds = %12, %14
  %18 = phi i32 [ %16, %14 ], [ %3, %12 ]
  %19 = zext i1 %6 to i8
  store i8 %19, ptr %5, align 4, !tbaa !21
  %20 = mul nsw i32 %2, %1
  %21 = add nsw i32 %18, %20
  %22 = select i1 %6, i32 %21, i32 0
  %23 = getelementptr inbounds nuw i8, ptr %5, i64 4
  store i32 %22, ptr %23, align 4, !tbaa !24
  %24 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i32 %1, ptr %24, align 4, !tbaa !25
  %25 = add i32 %4, -1
  %26 = mul i32 %25, %1
  %27 = add i32 %26, %18
  %28 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 %27, ptr %28, align 4, !tbaa !26
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define dso_local void @clear_PixelLoader(ptr nofree noundef writeonly captures(none) initializes((4, 8)) %0, ptr nofree noundef writeonly captures(none) initializes((4, 12)) %1) local_unnamed_addr #1 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 0, ptr %3, align 4, !tbaa !26
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 4
  store i32 0, ptr %4, align 4, !tbaa !24
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 0, ptr %5, align 4, !tbaa !25
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local signext i32 @loadFrom(ptr nofree noundef captures(none) %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef captures(none) %2, i32 noundef signext %3) local_unnamed_addr #4 {
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %6 = load i32, ptr %5, align 4, !tbaa !25
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %8 = load i32, ptr %7, align 4, !tbaa !26
  %9 = add nsw i32 %8, %6
  store i32 %9, ptr %7, align 4, !tbaa !26
  %10 = icmp eq i32 %3, 0
  br i1 %10, label %19, label %11

11:                                               ; preds = %4
  %12 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %13 = load i32, ptr %12, align 4, !tbaa !24
  %14 = icmp eq i32 %9, %13
  br i1 %14, label %15, label %19

15:                                               ; preds = %11
  %16 = shl nsw i32 %6, 1
  %17 = sub nsw i32 %9, %16
  store i32 %17, ptr %7, align 4, !tbaa !26
  %18 = sub nsw i32 0, %6
  store i32 %18, ptr %5, align 4, !tbaa !25
  br label %19

19:                                               ; preds = %15, %11, %4
  %20 = phi i32 [ %17, %15 ], [ %9, %11 ], [ %9, %4 ]
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds [4 x i8], ptr %1, i64 %21
  %23 = load i32, ptr %22, align 4, !tbaa !11
  ret i32 %23
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local signext i32 @initialize_BandIO(ptr nofree noundef captures(none) initializes((4, 16)) %0, i32 noundef signext %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4) local_unnamed_addr #4 {
  %6 = sdiv i32 %3, 2
  %7 = and i32 %3, 1
  %8 = icmp eq i32 %7, 0
  %9 = sdiv i32 %1, 2
  %10 = and i32 %1, 1
  %11 = add nsw i32 %9, %10
  %12 = sdiv i32 %2, 2
  %13 = and i32 %2, 1
  %14 = add nsw i32 %12, %13
  br i1 %8, label %20, label %15

15:                                               ; preds = %5
  %16 = mul nsw i32 %14, %11
  %17 = add nsw i32 %16, %6
  %18 = mul nsw i32 %2, %1
  %19 = sdiv i32 %18, 2
  br label %22

20:                                               ; preds = %5
  %21 = mul nsw i32 %14, %1
  br label %22

22:                                               ; preds = %20, %15
  %23 = phi i32 [ %21, %20 ], [ %19, %15 ]
  %24 = phi i32 [ %6, %20 ], [ %17, %15 ]
  %25 = phi i32 [ %11, %20 ], [ %9, %15 ]
  %26 = getelementptr inbounds nuw i8, ptr %0, i64 12
  store i32 %23, ptr %26, align 4, !tbaa !28
  %27 = sub nsw i32 %25, %23
  %28 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i32 %27, ptr %28, align 4, !tbaa !30
  %29 = load i8, ptr %0, align 4, !tbaa !31, !range !32, !noundef !33
  %30 = trunc nuw i8 %29 to i1
  br i1 %30, label %31, label %38

31:                                               ; preds = %22
  %32 = sdiv i32 %2, 2
  %33 = mul nsw i32 %25, %32
  %34 = add nsw i32 %33, %24
  %35 = trunc i32 %2 to i1
  %36 = select i1 %35, i32 %23, i32 0
  %37 = add nsw i32 %34, %36
  br label %38

38:                                               ; preds = %22, %31
  %39 = phi i32 [ %37, %31 ], [ 0, %22 ]
  %40 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 %39, ptr %40, align 4, !tbaa !34
  %41 = sdiv i32 %4, 2
  %42 = mul nsw i32 %25, %41
  %43 = add nsw i32 %42, %24
  %44 = trunc i32 %4 to i1
  %45 = select i1 %44, i32 %23, i32 0
  %46 = add nsw i32 %43, %45
  ret i32 %46
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #5

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local signext i32 @saveAndUpdate(ptr nofree noundef captures(none) initializes((0, 1)) %0, i1 noundef zeroext %1, ptr nofree noundef readonly captures(none) %2, ptr nofree noundef writeonly captures(none) %3, ptr nofree noundef readonly captures(none) %4, ptr nofree noundef readonly captures(none) %5) local_unnamed_addr #4 {
  %7 = zext i1 %1 to i8
  store i8 %7, ptr %0, align 4, !tbaa !35
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %9 = load i32, ptr %8, align 4, !tbaa !37
  br i1 %1, label %10, label %14

10:                                               ; preds = %6
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %12 = load i32, ptr %11, align 4, !tbaa !34
  %13 = icmp eq i32 %9, %12
  br i1 %13, label %22, label %14

14:                                               ; preds = %6, %10
  %15 = load i32, ptr %4, align 4, !tbaa !11
  %16 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %17 = sext i32 %9 to i64
  %18 = getelementptr inbounds [4 x i8], ptr %3, i64 %17
  store i32 %15, ptr %18, align 4, !tbaa !11
  %19 = load i32, ptr %5, align 4, !tbaa !11
  %20 = load i32, ptr %16, align 4, !tbaa !37
  %21 = add nsw i32 %20, %19
  store i32 %21, ptr %16, align 4, !tbaa !37
  br label %22

22:                                               ; preds = %14, %10
  %23 = phi i32 [ %21, %14 ], [ %9, %10 ]
  ret i32 %23
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define dso_local void @clear_BandWriter(ptr nofree noundef writeonly captures(none) initializes((4, 8)) %0, ptr nofree noundef writeonly captures(none) initializes((4, 16)) %1) local_unnamed_addr #1 {
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 4
  store i32 0, ptr %3, align 4, !tbaa !34
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 0, ptr %4, align 4, !tbaa !30
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 12
  store i32 0, ptr %5, align 4, !tbaa !28
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 0, ptr %6, align 4, !tbaa !37
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local void @init_BandWriter(ptr nofree noundef writeonly captures(none) initializes((4, 8)) %0, ptr nofree noundef captures(none) initializes((4, 16)) %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4, i32 noundef signext %5) local_unnamed_addr #4 {
  %7 = icmp slt i32 %4, %2
  br i1 %7, label %8, label %47

8:                                                ; preds = %6
  %9 = sdiv i32 %4, 2
  %10 = and i32 %4, 1
  %11 = icmp eq i32 %10, 0
  %12 = sdiv i32 %2, 2
  %13 = and i32 %2, 1
  %14 = add nsw i32 %12, %13
  %15 = sdiv i32 %3, 2
  %16 = and i32 %3, 1
  %17 = add nsw i32 %15, %16
  br i1 %11, label %23, label %18

18:                                               ; preds = %8
  %19 = mul nsw i32 %17, %14
  %20 = add nsw i32 %19, %9
  %21 = mul nsw i32 %3, %2
  %22 = sdiv i32 %21, 2
  br label %25

23:                                               ; preds = %8
  %24 = mul nsw i32 %17, %2
  br label %25

25:                                               ; preds = %23, %18
  %26 = phi i32 [ %24, %23 ], [ %22, %18 ]
  %27 = phi i32 [ %9, %23 ], [ %20, %18 ]
  %28 = phi i32 [ %14, %23 ], [ %12, %18 ]
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 12
  store i32 %26, ptr %29, align 4, !tbaa !28
  %30 = sub nsw i32 %28, %26
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 %30, ptr %31, align 4, !tbaa !30
  %32 = load i8, ptr %1, align 4, !tbaa !31, !range !32, !noundef !33
  %33 = trunc nuw i8 %32 to i1
  %34 = mul nsw i32 %28, %15
  %35 = trunc i32 %3 to i1
  %36 = select i1 %35, i32 %26, i32 0
  %37 = add i32 %36, %27
  %38 = add i32 %37, %34
  %39 = select i1 %33, i32 %38, i32 0
  %40 = getelementptr inbounds nuw i8, ptr %1, i64 4
  store i32 %39, ptr %40, align 4, !tbaa !34
  %41 = sdiv i32 %5, 2
  %42 = mul nsw i32 %28, %41
  %43 = trunc i32 %5 to i1
  %44 = select i1 %43, i32 %26, i32 0
  %45 = add i32 %44, %27
  %46 = add i32 %45, %42
  br label %51

47:                                               ; preds = %6
  %48 = getelementptr inbounds nuw i8, ptr %1, i64 4
  store i32 0, ptr %48, align 4, !tbaa !34
  %49 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 0, ptr %49, align 4, !tbaa !30
  %50 = getelementptr inbounds nuw i8, ptr %1, i64 12
  store i32 0, ptr %50, align 4, !tbaa !28
  br label %51

51:                                               ; preds = %47, %25
  %52 = phi i32 [ 0, %47 ], [ %46, %25 ]
  %53 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 %52, ptr %53, align 4, !tbaa !37
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local signext i32 @writeLowInto(ptr nofree noundef captures(none) %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef writeonly captures(none) %2, ptr nofree noundef readonly captures(none) %3) local_unnamed_addr #4 {
  %5 = load i8, ptr %0, align 4, !tbaa !35, !range !32, !noundef !33
  %6 = trunc nuw i8 %5 to i1
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %9 = load i32, ptr %8, align 4, !tbaa !37
  br i1 %6, label %10, label %14

10:                                               ; preds = %4
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %12 = load i32, ptr %11, align 4, !tbaa !34
  %13 = icmp eq i32 %9, %12
  br i1 %13, label %21, label %14

14:                                               ; preds = %10, %4
  %15 = load i32, ptr %3, align 4, !tbaa !11
  %16 = sext i32 %9 to i64
  %17 = getelementptr inbounds [4 x i8], ptr %2, i64 %16
  store i32 %15, ptr %17, align 4, !tbaa !11
  %18 = load i32, ptr %7, align 4, !tbaa !11
  %19 = load i32, ptr %8, align 4, !tbaa !37
  %20 = add nsw i32 %19, %18
  store i32 %20, ptr %8, align 4, !tbaa !37
  br label %21

21:                                               ; preds = %10, %14
  %22 = phi i32 [ %20, %14 ], [ %9, %10 ]
  ret i32 %22
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local signext i32 @writeHighInto(ptr nofree noundef captures(none) %0, ptr nofree noundef readonly captures(none) %1, ptr nofree noundef writeonly captures(none) %2, ptr nofree noundef readonly captures(none) %3) local_unnamed_addr #4 {
  %5 = load i8, ptr %0, align 4, !tbaa !35, !range !32, !noundef !33
  %6 = trunc nuw i8 %5 to i1
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %9 = load i32, ptr %8, align 4, !tbaa !37
  br i1 %6, label %10, label %14

10:                                               ; preds = %4
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %12 = load i32, ptr %11, align 4, !tbaa !34
  %13 = icmp eq i32 %9, %12
  br i1 %13, label %21, label %14

14:                                               ; preds = %10, %4
  %15 = load i32, ptr %3, align 4, !tbaa !11
  %16 = sext i32 %9 to i64
  %17 = getelementptr inbounds [4 x i8], ptr %2, i64 %16
  store i32 %15, ptr %17, align 4, !tbaa !11
  %18 = load i32, ptr %7, align 4, !tbaa !11
  %19 = load i32, ptr %8, align 4, !tbaa !37
  %20 = add nsw i32 %19, %18
  store i32 %20, ptr %8, align 4, !tbaa !37
  br label %21

21:                                               ; preds = %10, %14
  %22 = phi i32 [ %20, %14 ], [ %9, %10 ]
  ret i32 %22
}

; Function Attrs: convergent nofree norecurse nounwind memory(argmem: readwrite)
define dso_local void @horizontalStep(ptr nofree noundef captures(none) %0, i32 noundef signext %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4, i32 noundef signext %5) local_unnamed_addr #6 {
  %7 = load i32, ptr %0, align 4, !tbaa !38
  %8 = freeze i32 %7
  %9 = sdiv i32 %1, %8
  %10 = mul i32 %9, %8
  %11 = sub i32 %1, %10
  %12 = sub nsw i32 %1, %11
  %13 = icmp sgt i32 %9, 0
  br i1 %13, label %14, label %16

14:                                               ; preds = %6
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 28
  br label %20

16:                                               ; preds = %48, %6
  %17 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #14
  %18 = sext i32 %11 to i64
  %19 = icmp ult i64 %17, %18
  br i1 %19, label %51, label %77

20:                                               ; preds = %14, %48
  %21 = phi i32 [ 0, %14 ], [ %49, %48 ]
  %22 = load i32, ptr %0, align 4, !tbaa !38
  %23 = mul nsw i32 %22, %21
  %24 = add nsw i32 %23, %2
  %25 = sext i32 %24 to i64
  %26 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #14
  %27 = getelementptr [4 x i8], ptr %15, i64 %26
  %28 = getelementptr [4 x i8], ptr %27, i64 %25
  %29 = load i32, ptr %28, align 4, !tbaa !11
  %30 = add nsw i32 %23, %4
  %31 = sext i32 %30 to i64
  %32 = getelementptr [4 x i8], ptr %27, i64 %31
  %33 = load i32, ptr %32, align 4, !tbaa !11
  %34 = add nsw i32 %23, %3
  %35 = sext i32 %34 to i64
  %36 = getelementptr [4 x i8], ptr %27, i64 %35
  switch i32 %5, label %48 [
    i32 0, label %37
    i32 1, label %40
  ]

37:                                               ; preds = %20
  %38 = add nsw i32 %33, %29
  %39 = sdiv i32 %38, -2
  br label %44

40:                                               ; preds = %20
  %41 = add nsw i32 %33, %29
  %42 = add nsw i32 %41, 2
  %43 = sdiv i32 %42, 4
  br label %44

44:                                               ; preds = %37, %40
  %45 = phi i32 [ %43, %40 ], [ %39, %37 ]
  %46 = load i32, ptr %36, align 4, !tbaa !11
  %47 = add i32 %46, %45
  store i32 %47, ptr %36, align 4, !tbaa !11
  br label %48

48:                                               ; preds = %44, %20
  %49 = add nuw nsw i32 %21, 1
  %50 = icmp eq i32 %49, %9
  br i1 %50, label %16, label %20, !llvm.loop !40

51:                                               ; preds = %16
  %52 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %53 = add nsw i32 %12, %2
  %54 = sext i32 %53 to i64
  %55 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #14
  %56 = getelementptr [4 x i8], ptr %52, i64 %55
  %57 = getelementptr [4 x i8], ptr %56, i64 %54
  %58 = load i32, ptr %57, align 4, !tbaa !11
  %59 = add nsw i32 %12, %4
  %60 = sext i32 %59 to i64
  %61 = getelementptr [4 x i8], ptr %56, i64 %60
  %62 = load i32, ptr %61, align 4, !tbaa !11
  %63 = add nsw i32 %12, %3
  %64 = sext i32 %63 to i64
  %65 = getelementptr [4 x i8], ptr %56, i64 %64
  switch i32 %5, label %77 [
    i32 0, label %66
    i32 1, label %69
  ]

66:                                               ; preds = %51
  %67 = add nsw i32 %62, %58
  %68 = sdiv i32 %67, -2
  br label %73

69:                                               ; preds = %51
  %70 = add nsw i32 %62, %58
  %71 = add nsw i32 %70, 2
  %72 = sdiv i32 %71, 4
  br label %73

73:                                               ; preds = %69, %66
  %74 = phi i32 [ %68, %66 ], [ %72, %69 ]
  %75 = load i32, ptr %65, align 4, !tbaa !11
  %76 = add i32 %75, %74
  store i32 %76, ptr %65, align 4, !tbaa !11
  br label %77

77:                                               ; preds = %73, %51, %16
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local i64 @_Z12get_local_idj(i32 noundef signext) local_unnamed_addr #7

; Function Attrs: convergent nofree norecurse nounwind memory(argmem: readwrite)
define dso_local void @forEachHorizontalOdd(ptr nofree noundef captures(none) %0, i32 noundef signext %1, i32 noundef signext %2, i32 noundef signext %3) local_unnamed_addr #6 {
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = load i32, ptr %5, align 4, !tbaa !42
  %7 = mul nsw i32 %6, %2
  %8 = add nsw i32 %7, -1
  %9 = mul nsw i32 %6, %1
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %11 = load i32, ptr %10, align 4, !tbaa !43
  %12 = add nsw i32 %9, %11
  %13 = add nsw i32 %9, 1
  tail call void @horizontalStep(ptr noundef %0, i32 noundef signext %8, i32 noundef signext %9, i32 noundef signext %12, i32 noundef signext %13, i32 noundef signext %3) #15
  ret void
}

; Function Attrs: convergent nofree norecurse nounwind memory(argmem: readwrite)
define dso_local void @forEachHorizontalEven(ptr nofree noundef captures(none) %0, i32 noundef signext %1, i32 noundef signext %2, i32 noundef signext %3) local_unnamed_addr #6 {
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = load i32, ptr %5, align 4, !tbaa !42
  %7 = mul nsw i32 %6, %2
  %8 = add nsw i32 %7, -1
  %9 = mul nsw i32 %6, %1
  %10 = add nsw i32 %9, 1
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %12 = load i32, ptr %11, align 4, !tbaa !43
  %13 = add nsw i32 %9, %12
  %14 = add nsw i32 %13, 1
  tail call void @horizontalStep(ptr noundef %0, i32 noundef signext %8, i32 noundef signext %13, i32 noundef signext %10, i32 noundef signext %14, i32 noundef signext %3) #15
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local void @forEachVerticalOdd(ptr nofree noundef captures(none) %0, i32 noundef signext %1, i32 noundef signext %2) local_unnamed_addr #8 {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %5 = load i32, ptr %4, align 4, !tbaa !44
  %6 = add nsw i32 %5, -1
  %7 = sdiv i32 %6, 2
  %8 = icmp sgt i32 %5, 2
  br i1 %8, label %9, label %15

9:                                                ; preds = %3
  %10 = icmp eq i32 %2, 0
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br i1 %10, label %13, label %15

13:                                               ; preds = %9
  %14 = tail call i32 @llvm.smax.i32(i32 %7, i32 1)
  br label %16

15:                                               ; preds = %16, %9, %3
  ret void

16:                                               ; preds = %13, %16
  %17 = phi i32 [ %40, %16 ], [ 0, %13 ]
  %18 = shl nuw nsw i32 %17, 1
  %19 = add nuw nsw i32 %18, 2
  %20 = load i32, ptr %12, align 4, !tbaa !42
  %21 = mul nsw i32 %20, %19
  %22 = add nsw i32 %21, %1
  %23 = sext i32 %22 to i64
  %24 = getelementptr inbounds [4 x i8], ptr %11, i64 %23
  %25 = load i32, ptr %24, align 4, !tbaa !11
  %26 = or disjoint i32 %18, 1
  %27 = mul nsw i32 %20, %18
  %28 = add nsw i32 %27, %1
  %29 = sext i32 %28 to i64
  %30 = getelementptr inbounds [4 x i8], ptr %11, i64 %29
  %31 = load i32, ptr %30, align 4, !tbaa !11
  %32 = add nsw i32 %31, %25
  %33 = sdiv i32 %32, -2
  %34 = mul nsw i32 %20, %26
  %35 = add nsw i32 %34, %1
  %36 = sext i32 %35 to i64
  %37 = getelementptr inbounds [4 x i8], ptr %11, i64 %36
  %38 = load i32, ptr %37, align 4, !tbaa !11
  %39 = add i32 %33, %38
  store i32 %39, ptr %37, align 4, !tbaa !11
  %40 = add nuw nsw i32 %17, 1
  %41 = icmp eq i32 %40, %14
  br i1 %41, label %15, label %16, !llvm.loop !45
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local void @forEachVerticalEven(ptr nofree noundef captures(none) %0, i32 noundef signext %1, i32 noundef signext %2) local_unnamed_addr #8 {
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %5 = load i32, ptr %4, align 4, !tbaa !44
  %6 = icmp sgt i32 %5, 3
  br i1 %6, label %7, label %42

7:                                                ; preds = %3
  %8 = icmp eq i32 %2, 1
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br i1 %8, label %11, label %42

11:                                               ; preds = %7
  %12 = lshr i32 %5, 1
  %13 = add nsw i32 %12, -2
  br label %14

14:                                               ; preds = %11, %14
  %15 = phi i32 [ %40, %14 ], [ 0, %11 ]
  %16 = shl nuw nsw i32 %15, 1
  %17 = add nuw nsw i32 %16, 3
  %18 = load i32, ptr %10, align 4, !tbaa !42
  %19 = mul nsw i32 %18, %17
  %20 = add nsw i32 %19, %1
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds [4 x i8], ptr %9, i64 %21
  %23 = load i32, ptr %22, align 4, !tbaa !11
  %24 = add nuw nsw i32 %16, 2
  %25 = or disjoint i32 %16, 1
  %26 = mul nsw i32 %18, %25
  %27 = add nsw i32 %26, %1
  %28 = sext i32 %27 to i64
  %29 = getelementptr inbounds [4 x i8], ptr %9, i64 %28
  %30 = load i32, ptr %29, align 4, !tbaa !11
  %31 = add i32 %23, 2
  %32 = add i32 %31, %30
  %33 = sdiv i32 %32, 4
  %34 = mul nsw i32 %18, %24
  %35 = add nsw i32 %34, %1
  %36 = sext i32 %35 to i64
  %37 = getelementptr inbounds [4 x i8], ptr %9, i64 %36
  %38 = load i32, ptr %37, align 4, !tbaa !11
  %39 = add nsw i32 %33, %38
  store i32 %39, ptr %37, align 4, !tbaa !11
  %40 = add nuw nsw i32 %15, 1
  %41 = icmp eq i32 %15, %13
  br i1 %41, label %42, label %14, !llvm.loop !46

42:                                               ; preds = %14, %7, %3
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define dso_local void @clear_FDWT53Column(ptr nofree noundef writeonly captures(none) initializes((8, 28)) %0, ptr nofree noundef writeonly captures(none) initializes((4, 12)) %1) local_unnamed_addr #1 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 4
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) %3, i8 0, i64 20, i1 false)
  store i32 0, ptr %4, align 4, !tbaa !24
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 0, ptr %5, align 4, !tbaa !25
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define dso_local signext i32 @getColumnOffset(i32 noundef signext %0, ptr nofree noundef readonly captures(none) %1) local_unnamed_addr #9 {
  %3 = add nsw i32 %0, 2
  %4 = sdiv i32 %3, 2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %6 = load i32, ptr %5, align 4, !tbaa !43
  %7 = trunc i32 %0 to i1
  %8 = select i1 %7, i32 %6, i32 0
  %9 = add nsw i32 %8, %4
  ret i32 %9
}

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define dso_local void @initColumn(ptr nofree noundef readonly captures(none) %0, ptr nofree noundef writeonly captures(none) initializes((0, 1), (8, 28)) %1, i1 noundef zeroext %2, ptr nofree noundef readonly captures(none) %3, i32 noundef signext %4, i32 noundef signext %5, i32 noundef signext %6, i32 noundef signext %7, ptr nofree noundef writeonly captures(none) initializes((0, 1), (4, 12)) %8) local_unnamed_addr #10 {
  %10 = zext i1 %2 to i8
  store i8 %10, ptr %1, align 4, !tbaa !47
  %11 = add nsw i32 %6, 2
  %12 = sdiv i32 %11, 2
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 60
  %14 = load i32, ptr %13, align 4, !tbaa !43
  %15 = trunc i32 %6 to i1
  %16 = select i1 %15, i32 %14, i32 0
  %17 = add nsw i32 %16, %12
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 12
  store i32 %17, ptr %18, align 4, !tbaa !49
  %19 = tail call i64 @_Z12get_group_idj(i32 noundef signext 0) #14
  %20 = load i32, ptr %0, align 4, !tbaa !50
  %21 = trunc i64 %19 to i32
  %22 = mul i32 %20, %21
  %23 = add i32 %22, %6
  %24 = tail call i64 @_Z12get_group_idj(i32 noundef signext 1) #14
  %25 = icmp eq i64 %24, 0
  %26 = icmp slt i32 %23, %4
  br i1 %25, label %27, label %99

27:                                               ; preds = %9
  br i1 %26, label %31, label %28

28:                                               ; preds = %27
  %29 = shl nsw i32 %4, 1
  %30 = add nsw i32 %29, -2
  br label %33

31:                                               ; preds = %27
  %32 = icmp slt i32 %23, 0
  br i1 %32, label %33, label %36

33:                                               ; preds = %31, %28
  %34 = phi i32 [ %30, %28 ], [ 0, %31 ]
  %35 = sub i32 %34, %23
  br label %36

36:                                               ; preds = %31, %33
  %37 = phi i32 [ %35, %33 ], [ %23, %31 ]
  store i8 %10, ptr %8, align 4, !tbaa !21
  %38 = mul nsw i32 %5, %4
  %39 = add nsw i32 %37, %38
  %40 = select i1 %2, i32 %39, i32 0
  %41 = getelementptr inbounds nuw i8, ptr %8, i64 4
  store i32 %40, ptr %41, align 4, !tbaa !24
  %42 = getelementptr inbounds nuw i8, ptr %8, i64 8
  store i32 %4, ptr %42, align 4, !tbaa !25
  %43 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %44 = mul i32 %7, %4
  %45 = add i32 %44, %37
  store i32 %45, ptr %43, align 4, !tbaa !26
  %46 = icmp eq i32 %45, %40
  %47 = select i1 %2, i1 %46, i1 false
  br i1 %47, label %48, label %52

48:                                               ; preds = %36
  %49 = shl nsw i32 %4, 1
  %50 = sub nsw i32 %40, %49
  store i32 %50, ptr %43, align 4, !tbaa !26
  %51 = sub nsw i32 0, %4
  store i32 %51, ptr %42, align 4, !tbaa !25
  br label %52

52:                                               ; preds = %36, %48
  %53 = phi i32 [ %51, %48 ], [ %4, %36 ]
  %54 = phi i32 [ %50, %48 ], [ %45, %36 ]
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds [4 x i8], ptr %3, i64 %55
  %57 = load i32, ptr %56, align 4, !tbaa !11
  %58 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i32 %57, ptr %58, align 4, !tbaa !52
  %59 = add nsw i32 %54, %53
  store i32 %59, ptr %43, align 4, !tbaa !26
  %60 = icmp eq i32 %59, %40
  %61 = select i1 %2, i1 %60, i1 false
  br i1 %61, label %62, label %66

62:                                               ; preds = %52
  %63 = shl nsw i32 %53, 1
  %64 = sub nsw i32 %40, %63
  store i32 %64, ptr %43, align 4, !tbaa !26
  %65 = sub nsw i32 0, %53
  store i32 %65, ptr %42, align 4, !tbaa !25
  br label %66

66:                                               ; preds = %52, %62
  %67 = phi i32 [ %65, %62 ], [ %53, %52 ]
  %68 = phi i32 [ %64, %62 ], [ %59, %52 ]
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds [4 x i8], ptr %3, i64 %69
  %71 = load i32, ptr %70, align 4, !tbaa !11
  %72 = getelementptr inbounds nuw i8, ptr %1, i64 20
  store i32 %71, ptr %72, align 4, !tbaa !53
  %73 = add nsw i32 %68, %67
  store i32 %73, ptr %43, align 4, !tbaa !26
  %74 = icmp eq i32 %73, %40
  %75 = select i1 %2, i1 %74, i1 false
  br i1 %75, label %76, label %80

76:                                               ; preds = %66
  %77 = shl nsw i32 %67, 1
  %78 = sub nsw i32 %40, %77
  store i32 %78, ptr %43, align 4, !tbaa !26
  %79 = sub nsw i32 0, %67
  store i32 %79, ptr %42, align 4, !tbaa !25
  br label %80

80:                                               ; preds = %66, %76
  %81 = phi i32 [ %78, %76 ], [ %73, %66 ]
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds [4 x i8], ptr %3, i64 %82
  %84 = load i32, ptr %83, align 4, !tbaa !11
  %85 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i32 %84, ptr %85, align 4, !tbaa !54
  br i1 %26, label %89, label %86

86:                                               ; preds = %80
  %87 = shl nsw i32 %4, 1
  %88 = add nsw i32 %87, -2
  br label %91

89:                                               ; preds = %80
  %90 = icmp slt i32 %23, 0
  br i1 %90, label %91, label %94

91:                                               ; preds = %89, %86
  %92 = phi i32 [ %88, %86 ], [ 0, %89 ]
  %93 = sub i32 %92, %23
  br label %94

94:                                               ; preds = %89, %91
  %95 = phi i32 [ %93, %91 ], [ %23, %89 ]
  store i8 %10, ptr %8, align 4, !tbaa !21
  %96 = add nsw i32 %95, %38
  %97 = select i1 %2, i32 %96, i32 0
  store i32 %97, ptr %41, align 4, !tbaa !24
  store i32 %4, ptr %42, align 4, !tbaa !25
  %98 = add i32 %95, %44
  store i32 %98, ptr %43, align 4, !tbaa !26
  br label %159

99:                                               ; preds = %9
  br i1 %26, label %103, label %100

100:                                              ; preds = %99
  %101 = shl nsw i32 %4, 1
  %102 = add nsw i32 %101, -2
  br label %105

103:                                              ; preds = %99
  %104 = icmp slt i32 %23, 0
  br i1 %104, label %105, label %108

105:                                              ; preds = %103, %100
  %106 = phi i32 [ %102, %100 ], [ 0, %103 ]
  %107 = sub i32 %106, %23
  br label %108

108:                                              ; preds = %103, %105
  %109 = phi i32 [ %107, %105 ], [ %23, %103 ]
  store i8 %10, ptr %8, align 4, !tbaa !21
  %110 = mul nsw i32 %5, %4
  %111 = add nsw i32 %109, %110
  %112 = select i1 %2, i32 %111, i32 0
  %113 = getelementptr inbounds nuw i8, ptr %8, i64 4
  store i32 %112, ptr %113, align 4, !tbaa !24
  %114 = getelementptr inbounds nuw i8, ptr %8, i64 8
  store i32 %4, ptr %114, align 4, !tbaa !25
  %115 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %116 = add i32 %7, -2
  %117 = mul i32 %116, %4
  %118 = add i32 %117, %109
  store i32 %118, ptr %115, align 4, !tbaa !26
  %119 = icmp eq i32 %118, %112
  %120 = select i1 %2, i1 %119, i1 false
  br i1 %120, label %121, label %125

121:                                              ; preds = %108
  %122 = shl nsw i32 %4, 1
  %123 = sub nsw i32 %112, %122
  store i32 %123, ptr %115, align 4, !tbaa !26
  %124 = sub nsw i32 0, %4
  store i32 %124, ptr %114, align 4, !tbaa !25
  br label %125

125:                                              ; preds = %108, %121
  %126 = phi i32 [ %124, %121 ], [ %4, %108 ]
  %127 = phi i32 [ %123, %121 ], [ %118, %108 ]
  %128 = sext i32 %127 to i64
  %129 = getelementptr inbounds [4 x i8], ptr %3, i64 %128
  %130 = load i32, ptr %129, align 4, !tbaa !11
  %131 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i32 %130, ptr %131, align 4, !tbaa !54
  %132 = add nsw i32 %127, %126
  store i32 %132, ptr %115, align 4, !tbaa !26
  %133 = icmp eq i32 %132, %112
  %134 = select i1 %2, i1 %133, i1 false
  br i1 %134, label %135, label %139

135:                                              ; preds = %125
  %136 = shl nsw i32 %126, 1
  %137 = sub nsw i32 %112, %136
  store i32 %137, ptr %115, align 4, !tbaa !26
  %138 = sub nsw i32 0, %126
  store i32 %138, ptr %114, align 4, !tbaa !25
  br label %139

139:                                              ; preds = %125, %135
  %140 = phi i32 [ %138, %135 ], [ %126, %125 ]
  %141 = phi i32 [ %137, %135 ], [ %132, %125 ]
  %142 = sext i32 %141 to i64
  %143 = getelementptr inbounds [4 x i8], ptr %3, i64 %142
  %144 = load i32, ptr %143, align 4, !tbaa !11
  %145 = getelementptr inbounds nuw i8, ptr %1, i64 20
  store i32 %144, ptr %145, align 4, !tbaa !53
  %146 = add nsw i32 %141, %140
  store i32 %146, ptr %115, align 4, !tbaa !26
  %147 = icmp eq i32 %146, %112
  %148 = select i1 %2, i1 %147, i1 false
  br i1 %148, label %149, label %153

149:                                              ; preds = %139
  %150 = shl nsw i32 %140, 1
  %151 = sub nsw i32 %112, %150
  store i32 %151, ptr %115, align 4, !tbaa !26
  %152 = sub nsw i32 0, %140
  store i32 %152, ptr %114, align 4, !tbaa !25
  br label %153

153:                                              ; preds = %139, %149
  %154 = phi i32 [ %151, %149 ], [ %146, %139 ]
  %155 = sext i32 %154 to i64
  %156 = getelementptr inbounds [4 x i8], ptr %3, i64 %155
  %157 = load i32, ptr %156, align 4, !tbaa !11
  %158 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i32 %157, ptr %158, align 4, !tbaa !52
  br label %159

159:                                              ; preds = %153, %94
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local i64 @_Z12get_group_idj(i32 noundef signext) local_unnamed_addr #7

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local void @loadAndVerticallyTransform(ptr nofree noundef captures(none) %0, ptr nofree noundef captures(none) %1, i1 noundef zeroext %2, ptr nofree noundef readonly captures(none) %3, ptr nofree noundef captures(none) %4) local_unnamed_addr #8 {
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %7 = load i32, ptr %6, align 4, !tbaa !54
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %10 = load i32, ptr %9, align 4, !tbaa !49
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8792
  %12 = sext i32 %10 to i64
  %13 = getelementptr inbounds [4 x i8], ptr %8, i64 %12
  store i32 %7, ptr %13, align 4, !tbaa !11
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %15 = load i32, ptr %14, align 4, !tbaa !53
  %16 = load i32, ptr %9, align 4, !tbaa !49
  %17 = load i32, ptr %11, align 4, !tbaa !55
  %18 = add nsw i32 %17, %16
  %19 = sext i32 %18 to i64
  %20 = getelementptr inbounds [4 x i8], ptr %8, i64 %19
  store i32 %15, ptr %20, align 4, !tbaa !11
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %22 = load i32, ptr %21, align 4, !tbaa !52
  %23 = load i32, ptr %9, align 4, !tbaa !49
  %24 = load i32, ptr %11, align 4, !tbaa !55
  %25 = shl nsw i32 %24, 1
  %26 = add nsw i32 %25, %23
  %27 = sext i32 %26 to i64
  %28 = getelementptr inbounds [4 x i8], ptr %8, i64 %27
  store i32 %22, ptr %28, align 4, !tbaa !11
  %29 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %30 = load i32, ptr %29, align 4, !tbaa !56
  %31 = icmp sgt i32 %30, 0
  br i1 %31, label %34, label %32

32:                                               ; preds = %5
  %33 = add nsw i32 %30, 2
  br label %38

34:                                               ; preds = %5
  %35 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %37 = getelementptr inbounds nuw i8, ptr %4, i64 4
  br label %128

38:                                               ; preds = %140, %32
  %39 = phi i32 [ %33, %32 ], [ %153, %140 ]
  %40 = phi i32 [ %30, %32 ], [ %152, %140 ]
  %41 = load i32, ptr %9, align 4, !tbaa !49
  %42 = load i32, ptr %11, align 4, !tbaa !55
  %43 = mul nsw i32 %42, %40
  %44 = add nsw i32 %43, %41
  %45 = sext i32 %44 to i64
  %46 = getelementptr inbounds [4 x i8], ptr %8, i64 %45
  %47 = load i32, ptr %46, align 4, !tbaa !11
  store i32 %47, ptr %6, align 4, !tbaa !54
  %48 = add nsw i32 %40, 1
  %49 = mul nsw i32 %42, %48
  %50 = add nsw i32 %49, %41
  %51 = sext i32 %50 to i64
  %52 = getelementptr inbounds [4 x i8], ptr %8, i64 %51
  %53 = load i32, ptr %52, align 4, !tbaa !11
  store i32 %53, ptr %14, align 4, !tbaa !53
  %54 = mul nsw i32 %42, %39
  %55 = add nsw i32 %54, %41
  %56 = sext i32 %55 to i64
  %57 = getelementptr inbounds [4 x i8], ptr %8, i64 %56
  %58 = load i32, ptr %57, align 4, !tbaa !11
  store i32 %58, ptr %21, align 4, !tbaa !52
  %59 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %60 = load i32, ptr %59, align 4, !tbaa !44
  %61 = icmp sgt i32 %60, 2
  br i1 %61, label %62, label %127

62:                                               ; preds = %38
  %63 = add nsw i32 %60, -1
  %64 = lshr i32 %63, 1
  %65 = getelementptr inbounds nuw i8, ptr %0, i64 44
  br label %66

66:                                               ; preds = %66, %62
  %67 = phi i32 [ %90, %66 ], [ 0, %62 ]
  %68 = shl nuw nsw i32 %67, 1
  %69 = add nuw nsw i32 %68, 2
  %70 = load i32, ptr %65, align 4, !tbaa !42
  %71 = mul nsw i32 %69, %70
  %72 = add nsw i32 %71, %41
  %73 = sext i32 %72 to i64
  %74 = getelementptr inbounds [4 x i8], ptr %8, i64 %73
  %75 = load i32, ptr %74, align 4, !tbaa !11
  %76 = or disjoint i32 %68, 1
  %77 = mul nsw i32 %70, %68
  %78 = add nsw i32 %77, %41
  %79 = sext i32 %78 to i64
  %80 = getelementptr inbounds [4 x i8], ptr %8, i64 %79
  %81 = load i32, ptr %80, align 4, !tbaa !11
  %82 = add nsw i32 %81, %75
  %83 = sdiv i32 %82, -2
  %84 = mul nsw i32 %76, %70
  %85 = add nsw i32 %84, %41
  %86 = sext i32 %85 to i64
  %87 = getelementptr inbounds [4 x i8], ptr %8, i64 %86
  %88 = load i32, ptr %87, align 4, !tbaa !11
  %89 = add i32 %83, %88
  store i32 %89, ptr %87, align 4, !tbaa !11
  %90 = add nuw nsw i32 %67, 1
  %91 = icmp eq i32 %90, %64
  br i1 %91, label %92, label %66, !llvm.loop !45

92:                                               ; preds = %66
  %93 = load i32, ptr %59, align 4, !tbaa !44
  %94 = load i32, ptr %9, align 4, !tbaa !49
  %95 = icmp sgt i32 %93, 3
  br i1 %95, label %96, label %127

96:                                               ; preds = %92
  %97 = lshr i32 %93, 1
  %98 = add nsw i32 %97, -2
  br label %99

99:                                               ; preds = %99, %96
  %100 = phi i32 [ %125, %99 ], [ 0, %96 ]
  %101 = shl nuw nsw i32 %100, 1
  %102 = add nuw nsw i32 %101, 3
  %103 = load i32, ptr %65, align 4, !tbaa !42
  %104 = mul nsw i32 %102, %103
  %105 = add nsw i32 %104, %94
  %106 = sext i32 %105 to i64
  %107 = getelementptr inbounds [4 x i8], ptr %8, i64 %106
  %108 = load i32, ptr %107, align 4, !tbaa !11
  %109 = add nuw nsw i32 %101, 2
  %110 = or disjoint i32 %101, 1
  %111 = mul nsw i32 %110, %103
  %112 = add nsw i32 %111, %94
  %113 = sext i32 %112 to i64
  %114 = getelementptr inbounds [4 x i8], ptr %8, i64 %113
  %115 = load i32, ptr %114, align 4, !tbaa !11
  %116 = add i32 %108, 2
  %117 = add i32 %116, %115
  %118 = sdiv i32 %117, 4
  %119 = mul nsw i32 %109, %103
  %120 = add nsw i32 %119, %94
  %121 = sext i32 %120 to i64
  %122 = getelementptr inbounds [4 x i8], ptr %8, i64 %121
  %123 = load i32, ptr %122, align 4, !tbaa !11
  %124 = add nsw i32 %118, %123
  store i32 %124, ptr %122, align 4, !tbaa !11
  %125 = add nuw nsw i32 %100, 1
  %126 = icmp eq i32 %100, %98
  br i1 %126, label %127, label %99, !llvm.loop !46

127:                                              ; preds = %99, %38, %92
  ret void

128:                                              ; preds = %34, %140
  %129 = phi i32 [ 3, %34 ], [ %151, %140 ]
  %130 = load i32, ptr %35, align 4, !tbaa !25
  %131 = load i32, ptr %36, align 4, !tbaa !26
  %132 = add nsw i32 %131, %130
  store i32 %132, ptr %36, align 4, !tbaa !26
  br i1 %2, label %133, label %140

133:                                              ; preds = %128
  %134 = load i32, ptr %37, align 4, !tbaa !24
  %135 = icmp eq i32 %132, %134
  br i1 %135, label %136, label %140

136:                                              ; preds = %133
  %137 = shl nsw i32 %130, 1
  %138 = sub nsw i32 %132, %137
  store i32 %138, ptr %36, align 4, !tbaa !26
  %139 = sub nsw i32 0, %130
  store i32 %139, ptr %35, align 4, !tbaa !25
  br label %140

140:                                              ; preds = %128, %133, %136
  %141 = phi i32 [ %138, %136 ], [ %132, %133 ], [ %132, %128 ]
  %142 = sext i32 %141 to i64
  %143 = getelementptr inbounds [4 x i8], ptr %3, i64 %142
  %144 = load i32, ptr %143, align 4, !tbaa !11
  %145 = load i32, ptr %9, align 4, !tbaa !49
  %146 = load i32, ptr %11, align 4, !tbaa !55
  %147 = mul nsw i32 %146, %129
  %148 = add nsw i32 %147, %145
  %149 = sext i32 %148 to i64
  %150 = getelementptr inbounds [4 x i8], ptr %8, i64 %149
  store i32 %144, ptr %150, align 4, !tbaa !11
  %151 = add nuw nsw i32 %129, 1
  %152 = load i32, ptr %29, align 4, !tbaa !56
  %153 = add nsw i32 %152, 2
  %154 = icmp slt i32 %129, %153
  br i1 %154, label %128, label %38, !llvm.loop !57
}

; Function Attrs: convergent norecurse nounwind
define dso_local void @transform(ptr nofree noundef captures(none) %0, i1 noundef zeroext %1, i1 noundef zeroext %2, ptr nofree noundef readonly captures(none) %3, ptr nofree noundef writeonly captures(none) %4, i32 noundef signext %5, i32 noundef signext %6, i32 noundef signext %7) local_unnamed_addr #2 {
  %9 = alloca %struct.FDWT53Column, align 4
  %10 = alloca %struct.VerticalDWTPixelIO, align 4
  %11 = alloca %struct.FDWT53Column, align 4
  %12 = alloca %struct.VerticalDWTPixelIO, align 4
  %13 = zext i1 %1 to i8
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #16
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #16
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #16
  store i8 %13, ptr %11, align 4, !tbaa !47
  call void @llvm.lifetime.start.p0(ptr nonnull %12) #16
  %14 = tail call i64 @_Z12get_group_idj(i32 noundef signext 1) #14
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %16 = load i32, ptr %15, align 4, !tbaa !56
  %17 = trunc i64 %14 to i32
  %18 = mul i32 %7, %17
  %19 = mul i32 %18, %16
  %20 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #14
  %21 = trunc i64 %20 to i32
  call void @initColumn(ptr noundef %0, ptr noundef nonnull %9, i1 noundef zeroext %1, ptr noundef %3, i32 noundef signext %5, i32 noundef signext %6, i32 noundef signext %21, i32 noundef signext %19, ptr noundef nonnull %10) #15
  %22 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %23 = getelementptr inbounds nuw i8, ptr %12, i64 4
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) %22, i8 0, i64 20, i1 false)
  store i32 0, ptr %23, align 4, !tbaa !24
  %24 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store i32 0, ptr %24, align 4, !tbaa !25
  %25 = icmp ult i64 %20, 3
  br i1 %25, label %26, label %36

26:                                               ; preds = %8
  %27 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #14
  %28 = icmp eq i64 %27, 0
  br i1 %28, label %29, label %32

29:                                               ; preds = %26
  %30 = load i32, ptr %0, align 4, !tbaa !50
  %31 = zext i32 %30 to i64
  br label %32

32:                                               ; preds = %26, %29
  %33 = phi i64 [ %31, %29 ], [ -3, %26 ]
  %34 = add i64 %33, %27
  %35 = trunc i64 %34 to i32
  call void @initColumn(ptr noundef nonnull %0, ptr noundef nonnull %11, i1 noundef zeroext %1, ptr noundef %3, i32 noundef signext %5, i32 noundef signext %6, i32 noundef signext %35, i32 noundef signext %19, ptr noundef nonnull %12) #15
  br label %36

36:                                               ; preds = %32, %8
  %37 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #14
  %38 = shl i64 %37, 1
  %39 = load i32, ptr %0, align 4, !tbaa !50
  %40 = add nsw i32 %39, -1
  %41 = zext i32 %40 to i64
  %42 = sdiv i32 %39, 2
  %43 = sext i32 %42 to i64
  %44 = udiv i64 %37, %43
  %45 = mul i64 %44, %41
  %46 = sub i64 %38, %45
  %47 = trunc i64 %46 to i32
  %48 = getelementptr inbounds nuw i8, ptr %0, i64 36
  %49 = add nsw i32 %47, 2
  %50 = sdiv i32 %49, 2
  %51 = getelementptr inbounds nuw i8, ptr %0, i64 60
  %52 = load i32, ptr %51, align 4, !tbaa !43
  %53 = trunc i64 %46 to i1
  %54 = select i1 %53, i32 %52, i32 0
  %55 = add nsw i32 %50, %54
  %56 = tail call i64 @_Z12get_group_idj(i32 noundef signext 0) #14
  %57 = zext i32 %39 to i64
  %58 = mul i64 %56, %57
  %59 = add i64 %58, %46
  %60 = trunc i64 %59 to i32
  %61 = icmp sgt i32 %5, %60
  br i1 %61, label %62, label %96

62:                                               ; preds = %36
  %63 = sdiv i32 %60, 2
  %64 = and i32 %60, 1
  %65 = icmp eq i32 %64, 0
  %66 = sdiv i32 %5, 2
  %67 = and i32 %5, 1
  %68 = add nsw i32 %66, %67
  %69 = sdiv i32 %6, 2
  %70 = and i32 %6, 1
  %71 = add nsw i32 %69, %70
  br i1 %65, label %77, label %72

72:                                               ; preds = %62
  %73 = mul nsw i32 %71, %68
  %74 = add nsw i32 %63, %73
  %75 = mul nsw i32 %6, %5
  %76 = sdiv i32 %75, 2
  br label %79

77:                                               ; preds = %62
  %78 = mul nsw i32 %71, %5
  br label %79

79:                                               ; preds = %77, %72
  %80 = phi i32 [ %78, %77 ], [ %76, %72 ]
  %81 = phi i32 [ %63, %77 ], [ %74, %72 ]
  %82 = phi i32 [ %68, %77 ], [ %66, %72 ]
  %83 = sub nsw i32 %82, %80
  %84 = mul nsw i32 %82, %69
  %85 = trunc i32 %6 to i1
  %86 = select i1 %85, i32 %80, i32 0
  %87 = add i32 %86, %81
  %88 = add i32 %87, %84
  %89 = select i1 %2, i32 %88, i32 0
  %90 = sdiv i32 %19, 2
  %91 = mul nsw i32 %82, %90
  %92 = trunc i32 %19 to i1
  %93 = select i1 %92, i32 %80, i32 0
  %94 = add i32 %93, %81
  %95 = add i32 %94, %91
  br label %96

96:                                               ; preds = %36, %79
  %97 = phi i32 [ %83, %79 ], [ 0, %36 ]
  %98 = phi i32 [ %89, %79 ], [ 0, %36 ]
  %99 = phi i32 [ %80, %79 ], [ 0, %36 ]
  %100 = phi i32 [ %95, %79 ], [ 0, %36 ]
  %101 = icmp sgt i32 %7, 0
  br i1 %101, label %102, label %105

102:                                              ; preds = %96
  %103 = getelementptr inbounds nuw i8, ptr %0, i64 44
  %104 = getelementptr inbounds nuw i8, ptr %0, i64 64
  br label %106

105:                                              ; preds = %132, %96
  call void @llvm.lifetime.end.p0(ptr nonnull %12) #16
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #16
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #16
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #16
  ret void

106:                                              ; preds = %102, %132
  %107 = phi i32 [ 0, %102 ], [ %134, %132 ]
  %108 = phi i32 [ %100, %102 ], [ %133, %132 ]
  call void @loadAndVerticallyTransform(ptr noundef nonnull %0, ptr noundef nonnull %9, i1 noundef zeroext %1, ptr noundef %3, ptr noundef nonnull %10) #17
  %109 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #14
  %110 = icmp ult i64 %109, 3
  br i1 %110, label %111, label %112

111:                                              ; preds = %106
  call void @loadAndVerticallyTransform(ptr noundef nonnull %0, ptr noundef nonnull %11, i1 noundef zeroext %1, ptr noundef %3, ptr noundef nonnull %12) #17
  br label %112

112:                                              ; preds = %111, %106
  tail call void @_Z7barrierj(i32 noundef signext 1) #15
  %113 = load i32, ptr %15, align 4, !tbaa !56
  %114 = load i32, ptr %103, align 4, !tbaa !42
  %115 = mul nsw i32 %114, %113
  %116 = add nsw i32 %115, -1
  %117 = shl nsw i32 %114, 1
  %118 = load i32, ptr %51, align 4, !tbaa !43
  %119 = add nsw i32 %117, %118
  %120 = or disjoint i32 %117, 1
  tail call void @horizontalStep(ptr noundef nonnull %48, i32 noundef signext %116, i32 noundef signext %117, i32 noundef signext %119, i32 noundef signext %120, i32 noundef signext 0) #15
  tail call void @_Z7barrierj(i32 noundef signext 1) #15
  %121 = load i32, ptr %15, align 4, !tbaa !56
  %122 = load i32, ptr %103, align 4, !tbaa !42
  %123 = mul nsw i32 %122, %121
  %124 = add nsw i32 %123, -1
  %125 = shl nsw i32 %122, 1
  %126 = or disjoint i32 %125, 1
  %127 = load i32, ptr %51, align 4, !tbaa !43
  %128 = add nsw i32 %125, %127
  %129 = add nsw i32 %128, 1
  tail call void @horizontalStep(ptr noundef nonnull %48, i32 noundef signext %124, i32 noundef signext %128, i32 noundef signext %126, i32 noundef signext %129, i32 noundef signext 1) #15
  tail call void @_Z7barrierj(i32 noundef signext 1) #15
  %130 = load i32, ptr %15, align 4, !tbaa !56
  %131 = icmp sgt i32 %130, 0
  br i1 %131, label %136, label %132

132:                                              ; preds = %164, %112
  %133 = phi i32 [ %108, %112 ], [ %165, %164 ]
  tail call void @_Z7barrierj(i32 noundef signext 1) #15
  %134 = add nuw nsw i32 %107, 1
  %135 = icmp eq i32 %134, %7
  br i1 %135, label %105, label %106, !llvm.loop !58

136:                                              ; preds = %112, %164
  %137 = phi i32 [ %166, %164 ], [ 2, %112 ]
  %138 = phi i32 [ %165, %164 ], [ %108, %112 ]
  %139 = icmp eq i32 %138, %98
  %140 = select i1 %2, i1 %139, i1 false
  br i1 %140, label %164, label %141

141:                                              ; preds = %136
  %142 = load i32, ptr %103, align 4, !tbaa !59
  %143 = mul nsw i32 %142, %137
  %144 = add nsw i32 %143, %55
  %145 = sext i32 %144 to i64
  %146 = getelementptr inbounds [4 x i8], ptr %104, i64 %145
  %147 = load i32, ptr %146, align 4, !tbaa !11
  %148 = sext i32 %138 to i64
  %149 = getelementptr inbounds [4 x i8], ptr %4, i64 %148
  store i32 %147, ptr %149, align 4, !tbaa !11
  %150 = add nsw i32 %138, %99
  %151 = icmp eq i32 %150, %98
  %152 = select i1 %2, i1 %151, i1 false
  br i1 %152, label %164, label %153

153:                                              ; preds = %141
  %154 = load i32, ptr %103, align 4, !tbaa !59
  %155 = or disjoint i32 %137, 1
  %156 = mul nsw i32 %154, %155
  %157 = add nsw i32 %156, %55
  %158 = sext i32 %157 to i64
  %159 = getelementptr inbounds [4 x i8], ptr %104, i64 %158
  %160 = load i32, ptr %159, align 4, !tbaa !11
  %161 = sext i32 %150 to i64
  %162 = getelementptr inbounds [4 x i8], ptr %4, i64 %161
  store i32 %160, ptr %162, align 4, !tbaa !11
  %163 = add nsw i32 %150, %97
  br label %164

164:                                              ; preds = %141, %136, %153
  %165 = phi i32 [ %98, %141 ], [ %163, %153 ], [ %98, %136 ]
  %166 = add nuw nsw i32 %137, 2
  %167 = load i32, ptr %15, align 4, !tbaa !56
  %168 = icmp slt i32 %137, %167
  br i1 %168, label %136, label %132, !llvm.loop !60
}

; Function Attrs: convergent nounwind
declare dso_local void @_Z7barrierj(i32 noundef signext) local_unnamed_addr #11

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local void @Forward53Predict(i32 noundef signext %0, ptr nofree noundef captures(none) %1, i32 noundef signext %2) local_unnamed_addr #4 {
  %4 = add nsw i32 %2, %0
  %5 = sdiv i32 %4, -2
  %6 = load i32, ptr %1, align 4, !tbaa !11
  %7 = add i32 %6, %5
  store i32 %7, ptr %1, align 4, !tbaa !11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local void @Forward53Update(i32 noundef signext %0, ptr nofree noundef captures(none) %1, i32 noundef signext %2) local_unnamed_addr #4 {
  %4 = add i32 %0, 2
  %5 = add i32 %4, %2
  %6 = sdiv i32 %5, 4
  %7 = load i32, ptr %1, align 4, !tbaa !11
  %8 = add nsw i32 %7, %6
  store i32 %8, ptr %1, align 4, !tbaa !11
  ret void
}

; Function Attrs: convergent norecurse nounwind
define dso_local void @cl_fdwt53Kernel(ptr nofree noundef readonly align 4 captures(none) %0, ptr nofree noundef writeonly align 4 captures(none) %1, i32 noundef %2, i32 noundef %3, i32 noundef %4, i32 noundef %5, i32 noundef %6) local_unnamed_addr #2 !kernel_arg_addr_space !61 !kernel_arg_access_qual !62 !kernel_arg_type !63 !kernel_arg_base_type !63 !kernel_arg_type_qual !64 {
  store i32 %5, ptr @cl_fdwt53Kernel.fdwt53, align 4, !tbaa !50
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 4), align 4, !tbaa !56
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(8728) getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 64), i8 0, i64 8728, i1 false), !tbaa !11
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 36), align 4, !tbaa !65
  %8 = add nsw i32 %6, 3
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 40), align 4, !tbaa !66
  %9 = sdiv i32 %5, 2
  %10 = add nsw i32 %9, 2
  store i32 %10, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 44), align 4, !tbaa !59
  store i32 32, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 48), align 4, !tbaa !67
  %11 = mul nsw i32 %10, %8
  store i32 %11, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 52), align 4, !tbaa !68
  %12 = add nsw i32 %11, 16
  %13 = srem i32 %12, 32
  %14 = sub nsw i32 32, %13
  store i32 %14, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 56), align 4, !tbaa !69
  %15 = add nsw i32 %14, %11
  store i32 %15, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 60), align 4, !tbaa !70
  store i32 %10, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 8792), align 4, !tbaa !55
  %16 = tail call i64 @_Z12get_group_idj(i32 noundef signext 0) #14
  %17 = tail call i64 @_Z12get_group_idj(i32 noundef signext 1) #14
  %18 = trunc i64 %17 to i32
  %19 = add i32 %18, 1
  %20 = mul i32 %6, %4
  %21 = mul i32 %20, %19
  %22 = add i32 %21, 1
  %23 = icmp slt i32 %22, %3
  br i1 %23, label %25, label %24

24:                                               ; preds = %7
  tail call void @transform(ptr noundef nonnull @cl_fdwt53Kernel.fdwt53, i1 noundef zeroext true, i1 noundef zeroext true, ptr noundef align 4 %0, ptr noundef align 4 %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4) #15
  br label %33

25:                                               ; preds = %7
  %26 = trunc i64 %16 to i32
  %27 = add i32 %26, 1
  %28 = mul i32 %27, %5
  %29 = add i32 %28, 1
  %30 = icmp slt i32 %29, %2
  br i1 %30, label %32, label %31

31:                                               ; preds = %25
  tail call void @transform(ptr noundef nonnull @cl_fdwt53Kernel.fdwt53, i1 noundef zeroext false, i1 noundef zeroext true, ptr noundef align 4 %0, ptr noundef align 4 %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4) #15
  br label %33

32:                                               ; preds = %25
  tail call void @transform(ptr noundef nonnull @cl_fdwt53Kernel.fdwt53, i1 noundef zeroext false, i1 noundef zeroext false, ptr noundef align 4 %0, ptr noundef align 4 %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4) #15
  br label %33

33:                                               ; preds = %24, %31, %32
  ret void
}

; Function Attrs: alwaysinline convergent norecurse nounwind
define dso_local void @__clang_ocl_kern_imp_cl_fdwt53Kernel(ptr nofree noundef readonly align 4 captures(none) %0, ptr nofree noundef writeonly align 4 captures(none) %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4, i32 noundef signext %5, i32 noundef signext %6) local_unnamed_addr #3 !kernel_arg_addr_space !61 !kernel_arg_access_qual !62 !kernel_arg_type !63 !kernel_arg_base_type !63 !kernel_arg_type_qual !64 {
  store i32 %5, ptr @cl_fdwt53Kernel.fdwt53, align 4, !tbaa !50
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 4), align 4, !tbaa !56
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(8728) getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 64), i8 0, i64 8728, i1 false), !tbaa !11
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 36), align 4, !tbaa !65
  %8 = add nsw i32 %6, 3
  store i32 %8, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 40), align 4, !tbaa !66
  %9 = sdiv i32 %5, 2
  %10 = add nsw i32 %9, 2
  store i32 %10, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 44), align 4, !tbaa !59
  store i32 32, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 48), align 4, !tbaa !67
  %11 = mul nsw i32 %10, %8
  store i32 %11, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 52), align 4, !tbaa !68
  %12 = add nsw i32 %11, 16
  %13 = srem i32 %12, 32
  %14 = sub nsw i32 32, %13
  store i32 %14, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 56), align 4, !tbaa !69
  %15 = add nsw i32 %14, %11
  store i32 %15, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 60), align 4, !tbaa !70
  store i32 %10, ptr getelementptr inbounds nuw (i8, ptr @cl_fdwt53Kernel.fdwt53, i64 8792), align 4, !tbaa !55
  %16 = tail call i64 @_Z12get_group_idj(i32 noundef signext 0) #14
  %17 = tail call i64 @_Z12get_group_idj(i32 noundef signext 1) #14
  %18 = trunc i64 %17 to i32
  %19 = add i32 %18, 1
  %20 = mul i32 %6, %4
  %21 = mul i32 %20, %19
  %22 = add i32 %21, 1
  %23 = icmp slt i32 %22, %3
  br i1 %23, label %25, label %24

24:                                               ; preds = %7
  tail call void @transform(ptr noundef nonnull @cl_fdwt53Kernel.fdwt53, i1 noundef zeroext true, i1 noundef zeroext true, ptr noundef %0, ptr noundef %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4) #15
  br label %33

25:                                               ; preds = %7
  %26 = trunc i64 %16 to i32
  %27 = add i32 %26, 1
  %28 = mul i32 %27, %5
  %29 = add i32 %28, 1
  %30 = icmp slt i32 %29, %2
  br i1 %30, label %32, label %31

31:                                               ; preds = %25
  tail call void @transform(ptr noundef nonnull @cl_fdwt53Kernel.fdwt53, i1 noundef zeroext false, i1 noundef zeroext true, ptr noundef %0, ptr noundef %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4) #15
  br label %33

32:                                               ; preds = %25
  tail call void @transform(ptr noundef nonnull @cl_fdwt53Kernel.fdwt53, i1 noundef zeroext false, i1 noundef zeroext false, ptr noundef %0, ptr noundef %1, i32 noundef signext %2, i32 noundef signext %3, i32 noundef signext %4) #15
  br label %33

33:                                               ; preds = %31, %32, %24
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local i64 @_Z14get_local_sizej(i32 noundef signext) local_unnamed_addr #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #12

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #13

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #2 = { convergent norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #3 = { alwaysinline convergent norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #4 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { convergent nofree norecurse nounwind memory(argmem: readwrite) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #7 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #8 = { nofree norecurse nosync nounwind memory(argmem: readwrite) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #9 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #10 = { convergent mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #11 = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #12 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #13 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
attributes #14 = { convergent nounwind willreturn memory(none) "uniform-work-group-size" }
attributes #15 = { convergent nounwind "uniform-work-group-size" }
attributes #16 = { nounwind }
attributes #17 = { nounwind "uniform-work-group-size" }

!llvm.module.flags = !{!0, !1, !3}
!opencl.ocl.version = !{!4}
!llvm.ident = !{!5}
!llvm.errno.tbaa = !{!6}

!0 = !{i32 1, !"target-abi", !"lp64d"}
!1 = !{i32 6, !"riscv-isa", !2}
!2 = !{!"rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0"}
!3 = !{i32 8, !"SmallDataLimit", i32 0}
!4 = !{i32 1, i32 2}
!5 = !{!"clang version 23.1.1 (https://github.com/conda-forge/clangdev-feedstock 262945a5823b5ab7dd4012acfcbaf228fbf53c91)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"__libc_errno", !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!8, !8, i64 0}
!12 = !{i32 1, i32 1, i32 1, i32 1, i32 0}
!13 = !{!"none", !"none", !"none", !"none", !"none"}
!14 = !{!"int*", !"int*", !"int*", !"uchar*", !"int"}
!15 = !{!"", !"", !"", !"", !""}
!16 = !{!9, !9, i64 0}
!17 = !{i32 1, i32 1, i32 0}
!18 = !{!"none", !"none", !"none"}
!19 = !{!"int*", !"uchar*", !"int"}
!20 = !{!"", !"", !""}
!21 = !{!22, !23, i64 0}
!22 = !{!"VerticalDWTPixelIO", !23, i64 0, !8, i64 4, !8, i64 8}
!23 = !{!"bool", !9, i64 0}
!24 = !{!22, !8, i64 4}
!25 = !{!22, !8, i64 8}
!26 = !{!27, !8, i64 4}
!27 = !{!"VerticalDWTPixelLoader", !23, i64 0, !8, i64 4}
!28 = !{!29, !8, i64 12}
!29 = !{!"VerticalDWTBandIO", !23, i64 0, !8, i64 4, !8, i64 8, !8, i64 12}
!30 = !{!29, !8, i64 8}
!31 = !{!29, !23, i64 0}
!32 = !{i8 0, i8 2}
!33 = !{}
!34 = !{!29, !8, i64 4}
!35 = !{!36, !23, i64 0}
!36 = !{!"VerticalDWTBandWriter", !23, i64 0, !8, i64 4}
!37 = !{!36, !8, i64 4}
!38 = !{!39, !8, i64 0}
!39 = !{!"TransformBuffer", !8, i64 0, !8, i64 4, !8, i64 8, !8, i64 12, !8, i64 16, !8, i64 20, !8, i64 24, !9, i64 28}
!40 = distinct !{!40, !41}
!41 = !{!"llvm.loop.unroll.disable"}
!42 = !{!39, !8, i64 8}
!43 = !{!39, !8, i64 24}
!44 = !{!39, !8, i64 4}
!45 = distinct !{!45, !41}
!46 = distinct !{!46, !41}
!47 = !{!48, !23, i64 0}
!48 = !{!"FDWT53Column", !23, i64 0, !27, i64 4, !8, i64 12, !8, i64 16, !8, i64 20, !8, i64 24}
!49 = !{!48, !8, i64 12}
!50 = !{!51, !8, i64 0}
!51 = !{!"FDWT53", !8, i64 0, !8, i64 4, !48, i64 8, !39, i64 36, !8, i64 8792}
!52 = !{!48, !8, i64 24}
!53 = !{!48, !8, i64 20}
!54 = !{!48, !8, i64 16}
!55 = !{!51, !8, i64 8792}
!56 = !{!51, !8, i64 4}
!57 = distinct !{!57, !41}
!58 = distinct !{!58, !41}
!59 = !{!51, !8, i64 44}
!60 = distinct !{!60, !41}
!61 = !{i32 1, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0}
!62 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!63 = !{!"int*", !"int*", !"int", !"int", !"int", !"int", !"int"}
!64 = !{!"const", !"", !"", !"", !"", !"", !""}
!65 = !{!51, !8, i64 36}
!66 = !{!51, !8, i64 40}
!67 = !{!51, !8, i64 48}
!68 = !{!51, !8, i64 52}
!69 = !{!51, !8, i64 56}
!70 = !{!51, !8, i64 60}

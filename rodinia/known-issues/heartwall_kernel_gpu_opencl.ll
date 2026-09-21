; ModuleID = 'heartwall/kernel/kernel_gpu_opencl.cl'
source_filename = "heartwall/kernel/kernel_gpu_opencl.cl"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

%struct.params_common = type { i32, i32, i32, i32, i32, i32, i32, float, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

; Function Attrs: convergent norecurse nounwind
define dso_local void @kernel_gpu_opencl(ptr nofree noundef readonly byval(%struct.params_common) align 4 %0, ptr nofree noundef readonly align 4 captures(none) %1, i32 noundef %2, ptr nofree noundef align 4 captures(none) %3, ptr nofree noundef align 4 captures(none) %4, ptr nofree noundef align 4 captures(none) %5, ptr nofree noundef align 4 captures(none) %6, ptr nofree noundef align 4 captures(none) %7, ptr nofree noundef align 4 captures(none) %8, ptr nofree noundef align 4 captures(none) %9, ptr nofree noundef align 4 captures(none) %10, ptr nofree noundef align 4 captures(none) %11, ptr nofree noundef align 4 captures(none) %12, ptr nofree noundef align 4 captures(none) %13, ptr nofree noundef align 4 captures(none) %14, ptr nofree noundef align 4 captures(none) %15, ptr nofree noundef align 4 captures(none) %16, ptr nofree noundef align 4 captures(none) %17, ptr nofree noundef align 4 captures(none) %18, ptr nofree noundef align 4 captures(none) %19, ptr nofree noundef align 4 captures(none) %20, ptr nofree noundef align 4 captures(none) %21, ptr nofree noundef align 4 captures(none) %22, ptr nofree noundef align 4 captures(none) %23, ptr nofree noundef align 4 captures(none) %24, ptr nofree noundef align 4 captures(none) %25, ptr nofree noundef align 4 captures(none) %26, ptr nofree noundef align 4 captures(none) %27, ptr nofree noundef align 4 captures(none) %28, ptr nofree noundef align 4 captures(none) %29, ptr nofree noundef align 4 captures(none) %30, ptr nofree noundef align 4 captures(none) %31, ptr nofree noundef align 4 captures(none) %32, ptr nofree noundef readnone align 4 captures(none) %33) local_unnamed_addr #0 !kernel_arg_addr_space !11 !kernel_arg_access_qual !12 !kernel_arg_type !13 !kernel_arg_base_type !14 !kernel_arg_type_qual !15 {
  %35 = tail call i64 @_Z12get_group_idj(i32 noundef signext 0) #6
  %36 = trunc i64 %35 to i32
  %37 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #6
  %38 = trunc i64 %37 to i32
  %39 = getelementptr inbounds nuw i8, ptr %0, i64 52
  %40 = load i32, ptr %39, align 4, !tbaa !16
  %41 = icmp sgt i32 %40, %36
  br i1 %41, label %42, label %48

42:                                               ; preds = %34
  %43 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %44 = load i32, ptr %43, align 4, !tbaa !19
  %45 = mul nsw i32 %44, %36
  %46 = sext i32 %45 to i64
  %47 = getelementptr inbounds [4 x i8], ptr %11, i64 %46
  br label %57

48:                                               ; preds = %34
  %49 = sub nsw i32 %36, %40
  %50 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %51 = load i32, ptr %50, align 4, !tbaa !19
  %52 = mul nsw i32 %51, %49
  %53 = sext i32 %52 to i64
  %54 = getelementptr inbounds [4 x i8], ptr %12, i64 %53
  %55 = mul nsw i32 %51, %36
  %56 = sext i32 %55 to i64
  br label %57

57:                                               ; preds = %48, %42
  %58 = phi i64 [ %56, %48 ], [ %46, %42 ]
  %59 = phi i32 [ %49, %48 ], [ %36, %42 ]
  %60 = phi ptr [ %7, %48 ], [ %3, %42 ]
  %61 = phi ptr [ %8, %48 ], [ %4, %42 ]
  %62 = phi ptr [ %9, %48 ], [ %5, %42 ]
  %63 = phi ptr [ %10, %48 ], [ %6, %42 ]
  %64 = phi ptr [ %54, %48 ], [ %47, %42 ]
  %65 = getelementptr inbounds nuw i8, ptr %0, i64 100
  %66 = load i32, ptr %65, align 4, !tbaa !20
  %67 = mul nsw i32 %66, %36
  %68 = sext i32 %67 to i64
  %69 = getelementptr inbounds [4 x i8], ptr %13, i64 %68
  %70 = getelementptr inbounds nuw i8, ptr %0, i64 116
  %71 = load i32, ptr %70, align 4, !tbaa !21
  %72 = mul nsw i32 %71, %36
  %73 = sext i32 %72 to i64
  %74 = getelementptr inbounds [4 x i8], ptr %14, i64 %73
  %75 = getelementptr inbounds nuw i8, ptr %0, i64 148
  %76 = load i32, ptr %75, align 4, !tbaa !22
  %77 = mul nsw i32 %76, %36
  %78 = sext i32 %77 to i64
  %79 = getelementptr inbounds [4 x i8], ptr %15, i64 %78
  %80 = getelementptr inbounds nuw i8, ptr %0, i64 164
  %81 = load i32, ptr %80, align 4, !tbaa !23
  %82 = mul nsw i32 %81, %36
  %83 = sext i32 %82 to i64
  %84 = getelementptr inbounds [4 x i8], ptr %16, i64 %83
  %85 = getelementptr inbounds nuw i8, ptr %0, i64 212
  %86 = load i32, ptr %85, align 4, !tbaa !24
  %87 = mul nsw i32 %86, %36
  %88 = sext i32 %87 to i64
  %89 = getelementptr inbounds [4 x i8], ptr %17, i64 %88
  %90 = getelementptr inbounds nuw i8, ptr %0, i64 228
  %91 = load i32, ptr %90, align 4, !tbaa !25
  %92 = mul nsw i32 %91, %36
  %93 = sext i32 %92 to i64
  %94 = getelementptr inbounds [4 x i8], ptr %18, i64 %93
  %95 = getelementptr inbounds nuw i8, ptr %0, i64 276
  %96 = load i32, ptr %95, align 4, !tbaa !26
  %97 = mul nsw i32 %96, %36
  %98 = sext i32 %97 to i64
  %99 = getelementptr inbounds [4 x i8], ptr %19, i64 %98
  %100 = getelementptr inbounds nuw i8, ptr %0, i64 292
  %101 = load i32, ptr %100, align 4, !tbaa !27
  %102 = mul nsw i32 %101, %36
  %103 = sext i32 %102 to i64
  %104 = getelementptr inbounds [4 x i8], ptr %20, i64 %103
  %105 = getelementptr inbounds nuw i8, ptr %0, i64 308
  %106 = load i32, ptr %105, align 4, !tbaa !28
  %107 = mul nsw i32 %106, %36
  %108 = sext i32 %107 to i64
  %109 = getelementptr inbounds [4 x i8], ptr %21, i64 %108
  %110 = getelementptr inbounds nuw i8, ptr %0, i64 324
  %111 = load i32, ptr %110, align 4, !tbaa !29
  %112 = mul nsw i32 %111, %36
  %113 = sext i32 %112 to i64
  %114 = getelementptr inbounds [4 x i8], ptr %22, i64 %113
  %115 = getelementptr inbounds nuw i8, ptr %0, i64 340
  %116 = load i32, ptr %115, align 4, !tbaa !30
  %117 = mul nsw i32 %116, %36
  %118 = sext i32 %117 to i64
  %119 = getelementptr inbounds [4 x i8], ptr %23, i64 %118
  %120 = getelementptr inbounds nuw i8, ptr %0, i64 372
  %121 = load i32, ptr %120, align 4, !tbaa !31
  %122 = mul nsw i32 %121, %36
  %123 = sext i32 %122 to i64
  %124 = getelementptr inbounds [4 x i8], ptr %24, i64 %123
  %125 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %126 = getelementptr inbounds [4 x i8], ptr %25, i64 %58
  %127 = getelementptr inbounds nuw i8, ptr %0, i64 76
  %128 = load i32, ptr %127, align 4, !tbaa !32
  %129 = mul nsw i32 %128, %36
  %130 = sext i32 %129 to i64
  %131 = getelementptr inbounds [4 x i8], ptr %26, i64 %130
  %132 = getelementptr inbounds nuw i8, ptr %0, i64 316
  %133 = load i32, ptr %132, align 4, !tbaa !33
  %134 = mul nsw i32 %133, %36
  %135 = sext i32 %134 to i64
  %136 = getelementptr inbounds [4 x i8], ptr %27, i64 %135
  %137 = getelementptr inbounds nuw i8, ptr %0, i64 364
  %138 = load i32, ptr %137, align 4, !tbaa !34
  %139 = mul nsw i32 %138, %36
  %140 = sext i32 %139 to i64
  %141 = getelementptr inbounds [4 x i8], ptr %28, i64 %140
  %142 = getelementptr inbounds [4 x i8], ptr %29, i64 %140
  %143 = shl i64 %35, 32
  %144 = ashr exact i64 %143, 32
  %145 = getelementptr inbounds [4 x i8], ptr %30, i64 %144
  %146 = getelementptr inbounds [4 x i8], ptr %31, i64 %144
  %147 = getelementptr inbounds [4 x i8], ptr %32, i64 %144
  %148 = icmp eq i32 %2, 0
  br i1 %148, label %149, label %205

149:                                              ; preds = %57
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %150 = icmp eq i32 %38, 0
  br i1 %150, label %151, label %163

151:                                              ; preds = %149
  %152 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %153 = load i32, ptr %152, align 4, !tbaa !35
  %154 = mul nsw i32 %153, %59
  %155 = sext i32 %59 to i64
  %156 = getelementptr inbounds [4 x i8], ptr %60, i64 %155
  %157 = load i32, ptr %156, align 4, !tbaa !36
  %158 = sext i32 %154 to i64
  %159 = getelementptr inbounds [4 x i8], ptr %62, i64 %158
  store i32 %157, ptr %159, align 4, !tbaa !36
  %160 = getelementptr inbounds [4 x i8], ptr %61, i64 %155
  %161 = load i32, ptr %160, align 4, !tbaa !36
  %162 = getelementptr inbounds [4 x i8], ptr %63, i64 %158
  store i32 %161, ptr %162, align 4, !tbaa !36
  br label %163

163:                                              ; preds = %151, %149
  %164 = load i32, ptr %125, align 4, !tbaa !19
  %165 = icmp sgt i32 %164, %38
  br i1 %165, label %166, label %204

166:                                              ; preds = %163
  %167 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %168 = load i32, ptr %167, align 4, !tbaa !37
  %169 = sext i32 %59 to i64
  %170 = getelementptr inbounds [4 x i8], ptr %60, i64 %169
  %171 = load i32, ptr %170, align 4, !tbaa !36
  %172 = getelementptr inbounds [4 x i8], ptr %61, i64 %169
  %173 = load i32, ptr %172, align 4, !tbaa !36
  %174 = add i32 %173, -26
  %175 = getelementptr inbounds nuw i8, ptr %0, i64 36
  %176 = load i32, ptr %175, align 4, !tbaa !38
  %177 = add i32 %171, -26
  br label %178

178:                                              ; preds = %166, %178
  %179 = phi i32 [ %38, %166 ], [ %202, %178 ]
  %180 = add nsw i32 %179, 1
  %181 = freeze i32 %180
  %182 = freeze i32 %168
  %183 = sdiv i32 %181, %182
  %184 = mul i32 %183, %182
  %185 = sub i32 %181, %184
  %186 = icmp eq i32 %185, 0
  %187 = sext i1 %186 to i32
  %188 = add nsw i32 %183, %187
  %189 = select i1 %186, i32 %168, i32 %185
  %190 = add nsw i32 %189, -1
  %191 = add i32 %174, %188
  %192 = mul nsw i32 %191, %176
  %193 = add i32 %177, %190
  %194 = add nsw i32 %193, %192
  %195 = sext i32 %194 to i64
  %196 = getelementptr inbounds [4 x i8], ptr %1, i64 %195
  %197 = load float, ptr %196, align 4, !tbaa !39
  %198 = mul nsw i32 %188, %168
  %199 = add nsw i32 %198, %190
  %200 = sext i32 %199 to i64
  %201 = getelementptr inbounds [4 x i8], ptr %64, i64 %200
  store float %197, ptr %201, align 4, !tbaa !39
  %202 = add nsw i32 %179, 256
  %203 = icmp slt i32 %202, %164
  br i1 %203, label %178, label %204, !llvm.loop !40

204:                                              ; preds = %178, %163
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  br label %1451

205:                                              ; preds = %57
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %206 = sext i32 %59 to i64
  %207 = getelementptr inbounds [4 x i8], ptr %60, i64 %206
  %208 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %209 = getelementptr inbounds [4 x i8], ptr %61, i64 %206
  %210 = load i32, ptr %65, align 4, !tbaa !20
  %211 = icmp sgt i32 %210, %38
  br i1 %211, label %212, label %250

212:                                              ; preds = %205
  %213 = load i32, ptr %209, align 4, !tbaa !36
  %214 = load i32, ptr %208, align 4, !tbaa !42
  %215 = load i32, ptr %207, align 4, !tbaa !36
  %216 = getelementptr inbounds nuw i8, ptr %0, i64 92
  %217 = load i32, ptr %216, align 4, !tbaa !43
  %218 = xor i32 %214, -1
  %219 = add i32 %213, %218
  %220 = getelementptr inbounds nuw i8, ptr %0, i64 36
  %221 = load i32, ptr %220, align 4, !tbaa !38
  %222 = add i32 %215, -2
  %223 = sub i32 %222, %214
  %224 = shl i64 %37, 32
  %225 = ashr exact i64 %224, 32
  %226 = sext i32 %210 to i64
  br label %227

227:                                              ; preds = %212, %227
  %228 = phi i64 [ %225, %212 ], [ %248, %227 ]
  %229 = trunc i64 %228 to i32
  %230 = add i32 %229, 1
  %231 = freeze i32 %230
  %232 = freeze i32 %217
  %233 = sdiv i32 %231, %232
  %234 = mul i32 %233, %232
  %235 = sub i32 %231, %234
  %236 = icmp eq i32 %235, 0
  %237 = sext i1 %236 to i32
  %238 = select i1 %236, i32 %217, i32 %235
  %239 = add i32 %219, %233
  %240 = add i32 %239, %237
  %241 = mul nsw i32 %240, %221
  %242 = add i32 %223, %238
  %243 = add nsw i32 %242, %241
  %244 = sext i32 %243 to i64
  %245 = getelementptr inbounds [4 x i8], ptr %1, i64 %244
  %246 = load float, ptr %245, align 4, !tbaa !39
  %247 = getelementptr inbounds [4 x i8], ptr %69, i64 %228
  store float %246, ptr %247, align 4, !tbaa !39
  %248 = add nsw i64 %228, 256
  %249 = icmp slt i64 %248, %226
  br i1 %249, label %227, label %250, !llvm.loop !44

250:                                              ; preds = %227, %205
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %251 = load i32, ptr %125, align 4, !tbaa !19
  %252 = icmp sgt i32 %251, %38
  br i1 %252, label %253, label %283

253:                                              ; preds = %250
  %254 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %255 = load i32, ptr %254, align 4, !tbaa !37
  %256 = shl i64 %37, 32
  %257 = ashr exact i64 %256, 32
  %258 = sext i32 %251 to i64
  br label %259

259:                                              ; preds = %253, %259
  %260 = phi i64 [ %257, %253 ], [ %281, %259 ]
  %261 = trunc i64 %260 to i32
  %262 = add i32 %261, 1
  %263 = freeze i32 %262
  %264 = freeze i32 %255
  %265 = sdiv i32 %263, %264
  %266 = mul i32 %265, %264
  %267 = sub i32 %263, %266
  %268 = icmp eq i32 %267, 0
  %269 = sext i1 %268 to i32
  %270 = add nsw i32 %265, %269
  %271 = sub i32 %255, %267
  %272 = select i1 %268, i32 0, i32 %271
  %273 = xor i32 %270, -1
  %274 = add i32 %255, %273
  %275 = mul nsw i32 %274, %255
  %276 = add nsw i32 %275, %272
  %277 = sext i32 %276 to i64
  %278 = getelementptr inbounds [4 x i8], ptr %64, i64 %277
  %279 = load float, ptr %278, align 4, !tbaa !39
  %280 = getelementptr inbounds [4 x i8], ptr %126, i64 %260
  store float %279, ptr %280, align 4, !tbaa !39
  %281 = add nsw i64 %260, 256
  %282 = icmp slt i64 %281, %258
  br i1 %282, label %259, label %283, !llvm.loop !45

283:                                              ; preds = %259, %250
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %284 = load i32, ptr %70, align 4, !tbaa !21
  %285 = icmp sgt i32 %284, %38
  br i1 %285, label %286, label %371

286:                                              ; preds = %283
  %287 = getelementptr inbounds nuw i8, ptr %0, i64 108
  %288 = load i32, ptr %287, align 4, !tbaa !46
  %289 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %290 = load i32, ptr %289, align 4, !tbaa !47
  %291 = getelementptr inbounds nuw i8, ptr %0, i64 96
  %292 = load i32, ptr %291, align 4, !tbaa !48
  %293 = load i32, ptr %127, align 4, !tbaa !32
  %294 = getelementptr inbounds nuw i8, ptr %0, i64 124
  %295 = load i32, ptr %294, align 4, !tbaa !49
  %296 = getelementptr inbounds nuw i8, ptr %0, i64 92
  %297 = load i32, ptr %296, align 4, !tbaa !43
  %298 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %299 = load i32, ptr %298, align 4, !tbaa !37
  %300 = sext i32 %299 to i64
  %301 = shl i64 %37, 32
  %302 = ashr exact i64 %301, 32
  %303 = sext i32 %284 to i64
  %304 = sub i32 1, %292
  br label %305

305:                                              ; preds = %286, %366
  %306 = phi i64 [ %302, %286 ], [ %369, %366 ]
  %307 = trunc i64 %306 to i32
  %308 = add i32 %307, 1
  %309 = freeze i32 %308
  %310 = freeze i32 %288
  %311 = sdiv i32 %309, %310
  %312 = mul i32 %311, %310
  %313 = sub i32 %309, %312
  %314 = icmp ne i32 %313, 0
  %315 = zext i1 %314 to i32
  %316 = add nsw i32 %311, %315
  %317 = select i1 %314, i32 %313, i32 %288
  %318 = add nsw i32 %316, %290
  %319 = icmp sgt i32 %292, %318
  %320 = add i32 %318, %304
  %321 = select i1 %319, i32 1, i32 %320
  %322 = tail call i32 @llvm.smin.i32(i32 %293, i32 %318)
  %323 = add i32 %295, %317
  %324 = add i32 %323, 1
  %325 = icmp sgt i32 %321, %322
  br i1 %325, label %366, label %326

326:                                              ; preds = %305
  %327 = tail call i32 @llvm.smin.i32(i32 %299, i32 %323)
  %328 = icmp sgt i32 %297, %323
  %329 = sub i32 %324, %297
  %330 = select i1 %328, i32 1, i32 %329
  %331 = icmp sgt i32 %330, %327
  %332 = sext i32 %330 to i64
  %333 = sext i32 %327 to i64
  %334 = sext i32 %321 to i64
  %335 = sext i32 %322 to i64
  br label %336

336:                                              ; preds = %326, %362
  %337 = phi i64 [ %334, %326 ], [ %364, %362 ]
  %338 = phi float [ 0.000000e+00, %326 ], [ %363, %362 ]
  br i1 %331, label %362, label %339

339:                                              ; preds = %336
  %340 = add nsw i64 %337, -1
  %341 = mul nsw i64 %340, %300
  %342 = trunc nsw i64 %337 to i32
  %343 = sub i32 %318, %342
  %344 = mul nsw i32 %343, %297
  %345 = add i32 %344, %324
  %346 = getelementptr [4 x i8], ptr %126, i64 %341
  br label %347

347:                                              ; preds = %339, %347
  %348 = phi i64 [ %332, %339 ], [ %360, %347 ]
  %349 = phi float [ %338, %339 ], [ %359, %347 ]
  %350 = getelementptr [4 x i8], ptr %346, i64 %348
  %351 = getelementptr i8, ptr %350, i64 -4
  %352 = load float, ptr %351, align 4, !tbaa !39
  %353 = trunc nsw i64 %348 to i32
  %354 = sub i32 %345, %353
  %355 = sext i32 %354 to i64
  %356 = getelementptr [4 x i8], ptr %69, i64 %355
  %357 = getelementptr i8, ptr %356, i64 -4
  %358 = load float, ptr %357, align 4, !tbaa !39
  %359 = tail call float @llvm.fmuladd.f32(float %352, float %358, float %349)
  %360 = add nuw nsw i64 %348, 1
  %361 = icmp slt i64 %348, %333
  br i1 %361, label %347, label %362, !llvm.loop !50

362:                                              ; preds = %347, %336
  %363 = phi float [ %338, %336 ], [ %359, %347 ]
  %364 = add nuw nsw i64 %337, 1
  %365 = icmp slt i64 %337, %335
  br i1 %365, label %336, label %366, !llvm.loop !51

366:                                              ; preds = %362, %305
  %367 = phi float [ 0.000000e+00, %305 ], [ %363, %362 ]
  %368 = getelementptr inbounds [4 x i8], ptr %74, i64 %306
  store float %367, ptr %368, align 4, !tbaa !39
  %369 = add nsw i64 %306, 256
  %370 = icmp slt i64 %369, %303
  br i1 %370, label %305, label %371, !llvm.loop !52

371:                                              ; preds = %366, %283
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %372 = load i32, ptr %75, align 4, !tbaa !22
  %373 = icmp sgt i32 %372, %38
  br i1 %373, label %374, label %424

374:                                              ; preds = %371
  %375 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %376 = load i32, ptr %375, align 4, !tbaa !53
  %377 = getelementptr inbounds nuw i8, ptr %0, i64 132
  %378 = load i32, ptr %377, align 4, !tbaa !54
  %379 = getelementptr inbounds nuw i8, ptr %0, i64 92
  %380 = load i32, ptr %379, align 4
  %381 = add nsw i32 %380, %378
  %382 = getelementptr inbounds nuw i8, ptr %0, i64 136
  %383 = load i32, ptr %382, align 4
  %384 = getelementptr inbounds nuw i8, ptr %0, i64 96
  %385 = load i32, ptr %384, align 4
  %386 = add nsw i32 %385, %383
  %387 = xor i32 %378, -1
  %388 = shl i64 %37, 32
  %389 = ashr exact i64 %388, 32
  %390 = sext i32 %372 to i64
  br label %391

391:                                              ; preds = %374, %419
  %392 = phi i64 [ %389, %374 ], [ %422, %419 ]
  %393 = trunc i64 %392 to i32
  %394 = add i32 %393, 1
  %395 = freeze i32 %394
  %396 = freeze i32 %376
  %397 = sdiv i32 %395, %396
  %398 = mul i32 %397, %396
  %399 = sub i32 %395, %398
  %400 = icmp eq i32 %399, 0
  %401 = sext i1 %400 to i32
  %402 = add nsw i32 %397, %401
  %403 = select i1 %400, i32 %376, i32 %399
  %404 = icmp sgt i32 %403, %378
  %405 = icmp sle i32 %403, %381
  %406 = select i1 %404, i1 %405, i1 false
  %407 = icmp sge i32 %402, %383
  %408 = select i1 %406, i1 %407, i1 false
  %409 = icmp slt i32 %402, %386
  %410 = select i1 %408, i1 %409, i1 false
  br i1 %410, label %411, label %419

411:                                              ; preds = %391
  %412 = add i32 %403, %387
  %413 = sub nsw i32 %402, %383
  %414 = mul nsw i32 %413, %380
  %415 = add nsw i32 %412, %414
  %416 = sext i32 %415 to i64
  %417 = getelementptr inbounds [4 x i8], ptr %69, i64 %416
  %418 = load float, ptr %417, align 4, !tbaa !39
  br label %419

419:                                              ; preds = %391, %411
  %420 = phi float [ %418, %411 ], [ 0.000000e+00, %391 ]
  %421 = getelementptr inbounds [4 x i8], ptr %79, i64 %392
  store float %420, ptr %421, align 4, !tbaa !39
  %422 = add nsw i64 %392, 256
  %423 = icmp slt i64 %422, %390
  br i1 %423, label %391, label %424, !llvm.loop !55

424:                                              ; preds = %419, %371
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %425 = getelementptr inbounds nuw i8, ptr %0, i64 144
  %426 = load i32, ptr %425, align 4, !tbaa !56
  %427 = icmp sgt i32 %426, %38
  br i1 %427, label %428, label %457

428:                                              ; preds = %424
  %429 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %430 = load i32, ptr %429, align 4, !tbaa !53
  %431 = mul i32 %430, %38
  %432 = shl i32 %430, 8
  %433 = shl i64 %37, 32
  %434 = ashr exact i64 %433, 32
  %435 = sext i32 %426 to i64
  %436 = sext i32 %430 to i64
  %437 = icmp sgt i32 %430, 0
  br label %438

438:                                              ; preds = %428, %453
  %439 = phi i64 [ %434, %428 ], [ %454, %453 ]
  %440 = phi i32 [ %431, %428 ], [ %456, %453 ]
  %441 = add nsw i64 %439, 1
  %442 = mul i64 %441, %436
  br i1 %437, label %443, label %453

443:                                              ; preds = %438
  %444 = sext i32 %440 to i64
  br label %445

445:                                              ; preds = %443, %445
  %446 = phi i64 [ %444, %443 ], [ %451, %445 ]
  %447 = phi float [ 0.000000e+00, %443 ], [ %450, %445 ]
  %448 = getelementptr inbounds [4 x i8], ptr %79, i64 %446
  %449 = load float, ptr %448, align 4, !tbaa !39
  %450 = fadd float %447, %449
  store float %450, ptr %448, align 4, !tbaa !39
  %451 = add nsw i64 %446, 1
  %452 = icmp slt i64 %451, %442
  br i1 %452, label %445, label %453, !llvm.loop !57

453:                                              ; preds = %445, %438
  %454 = add nsw i64 %439, 256
  %455 = icmp slt i64 %454, %435
  %456 = add i32 %440, %432
  br i1 %455, label %438, label %457, !llvm.loop !58

457:                                              ; preds = %453, %424
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %458 = load i32, ptr %80, align 4, !tbaa !23
  %459 = icmp sgt i32 %458, %38
  br i1 %459, label %460, label %497

460:                                              ; preds = %457
  %461 = getelementptr inbounds nuw i8, ptr %0, i64 156
  %462 = load i32, ptr %461, align 4, !tbaa !59
  %463 = getelementptr inbounds nuw i8, ptr %0, i64 172
  %464 = load i32, ptr %463, align 4, !tbaa !60
  %465 = getelementptr inbounds nuw i8, ptr %0, i64 180
  %466 = load i32, ptr %465, align 4, !tbaa !61
  %467 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %468 = load i32, ptr %467, align 4, !tbaa !53
  %469 = add i32 %464, -2
  %470 = shl i64 %37, 32
  %471 = ashr exact i64 %470, 32
  %472 = sext i32 %458 to i64
  br label %473

473:                                              ; preds = %460, %473
  %474 = phi i64 [ %471, %460 ], [ %495, %473 ]
  %475 = trunc i64 %474 to i32
  %476 = add i32 %475, 1
  %477 = freeze i32 %476
  %478 = freeze i32 %462
  %479 = sdiv i32 %477, %478
  %480 = mul i32 %479, %478
  %481 = sub i32 %477, %480
  %482 = icmp eq i32 %481, 0
  %483 = sext i1 %482 to i32
  %484 = select i1 %482, i32 %462, i32 %481
  %485 = add i32 %479, -1
  %486 = add i32 %485, %483
  %487 = add i32 %486, %466
  %488 = mul nsw i32 %487, %468
  %489 = add i32 %484, %469
  %490 = add nsw i32 %489, %488
  %491 = sext i32 %490 to i64
  %492 = getelementptr inbounds [4 x i8], ptr %79, i64 %491
  %493 = load float, ptr %492, align 4, !tbaa !39
  %494 = getelementptr inbounds [4 x i8], ptr %84, i64 %474
  store float %493, ptr %494, align 4, !tbaa !39
  %495 = add nsw i64 %474, 256
  %496 = icmp slt i64 %495, %472
  br i1 %496, label %473, label %497, !llvm.loop !62

497:                                              ; preds = %473, %457
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %498 = load i32, ptr %85, align 4, !tbaa !24
  %499 = icmp sgt i32 %498, %38
  br i1 %499, label %500, label %537

500:                                              ; preds = %497
  %501 = getelementptr inbounds nuw i8, ptr %0, i64 204
  %502 = load i32, ptr %501, align 4, !tbaa !63
  %503 = getelementptr inbounds nuw i8, ptr %0, i64 188
  %504 = load i32, ptr %503, align 4, !tbaa !64
  %505 = getelementptr inbounds nuw i8, ptr %0, i64 196
  %506 = load i32, ptr %505, align 4, !tbaa !65
  %507 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %508 = load i32, ptr %507, align 4, !tbaa !53
  %509 = add i32 %504, -2
  %510 = shl i64 %37, 32
  %511 = ashr exact i64 %510, 32
  %512 = sext i32 %498 to i64
  br label %513

513:                                              ; preds = %500, %513
  %514 = phi i64 [ %511, %500 ], [ %535, %513 ]
  %515 = trunc i64 %514 to i32
  %516 = add i32 %515, 1
  %517 = freeze i32 %516
  %518 = freeze i32 %502
  %519 = sdiv i32 %517, %518
  %520 = mul i32 %519, %518
  %521 = sub i32 %517, %520
  %522 = icmp eq i32 %521, 0
  %523 = sext i1 %522 to i32
  %524 = select i1 %522, i32 %502, i32 %521
  %525 = add i32 %519, -1
  %526 = add i32 %525, %523
  %527 = add i32 %526, %506
  %528 = mul nsw i32 %527, %508
  %529 = add i32 %524, %509
  %530 = add nsw i32 %529, %528
  %531 = sext i32 %530 to i64
  %532 = getelementptr inbounds [4 x i8], ptr %79, i64 %531
  %533 = load float, ptr %532, align 4, !tbaa !39
  %534 = getelementptr inbounds [4 x i8], ptr %89, i64 %514
  store float %533, ptr %534, align 4, !tbaa !39
  %535 = add nsw i64 %514, 256
  %536 = icmp slt i64 %535, %512
  br i1 %536, label %513, label %537, !llvm.loop !66

537:                                              ; preds = %513, %497
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %538 = load i32, ptr %85, align 4, !tbaa !24
  %539 = icmp sgt i32 %538, %38
  br i1 %539, label %540, label %553

540:                                              ; preds = %537
  %541 = shl i64 %37, 32
  %542 = ashr exact i64 %541, 32
  %543 = sext i32 %538 to i64
  br label %544

544:                                              ; preds = %540, %544
  %545 = phi i64 [ %542, %540 ], [ %551, %544 ]
  %546 = getelementptr inbounds [4 x i8], ptr %84, i64 %545
  %547 = load float, ptr %546, align 4, !tbaa !39
  %548 = getelementptr inbounds [4 x i8], ptr %89, i64 %545
  %549 = load float, ptr %548, align 4, !tbaa !39
  %550 = fsub float %547, %549
  store float %550, ptr %548, align 4, !tbaa !39
  %551 = add nsw i64 %545, 256
  %552 = icmp slt i64 %551, %543
  br i1 %552, label %544, label %553, !llvm.loop !67

553:                                              ; preds = %544, %537
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %554 = getelementptr inbounds nuw i8, ptr %0, i64 204
  %555 = load i32, ptr %554, align 4, !tbaa !63
  %556 = icmp sgt i32 %555, %38
  br i1 %556, label %557, label %578

557:                                              ; preds = %553
  %558 = load i32, ptr %85, align 4, !tbaa !24
  %559 = shl i64 %37, 32
  %560 = ashr exact i64 %559, 32
  %561 = sext i32 %555 to i64
  %562 = sext i32 %558 to i64
  %563 = icmp sgt i32 %558, 0
  br label %564

564:                                              ; preds = %557, %575
  %565 = phi i64 [ %560, %557 ], [ %576, %575 ]
  %566 = add nsw i64 %565, %562
  br i1 %563, label %567, label %575

567:                                              ; preds = %564, %567
  %568 = phi i64 [ %573, %567 ], [ %565, %564 ]
  %569 = phi float [ %572, %567 ], [ 0.000000e+00, %564 ]
  %570 = getelementptr inbounds [4 x i8], ptr %89, i64 %568
  %571 = load float, ptr %570, align 4, !tbaa !39
  %572 = fadd float %569, %571
  store float %572, ptr %570, align 4, !tbaa !39
  %573 = add nsw i64 %568, %561
  %574 = icmp slt i64 %573, %566
  br i1 %574, label %567, label %575, !llvm.loop !68

575:                                              ; preds = %567, %564
  %576 = add nsw i64 %565, 256
  %577 = icmp slt i64 %576, %561
  br i1 %577, label %564, label %578, !llvm.loop !69

578:                                              ; preds = %575, %553
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %579 = load i32, ptr %90, align 4, !tbaa !25
  %580 = icmp sgt i32 %579, %38
  br i1 %580, label %581, label %617

581:                                              ; preds = %578
  %582 = getelementptr inbounds nuw i8, ptr %0, i64 220
  %583 = load i32, ptr %582, align 4, !tbaa !70
  %584 = getelementptr inbounds nuw i8, ptr %0, i64 236
  %585 = load i32, ptr %584, align 4, !tbaa !71
  %586 = getelementptr inbounds nuw i8, ptr %0, i64 244
  %587 = load i32, ptr %586, align 4, !tbaa !72
  %588 = load i32, ptr %554, align 4, !tbaa !63
  %589 = add i32 %585, -2
  %590 = shl i64 %37, 32
  %591 = ashr exact i64 %590, 32
  %592 = sext i32 %579 to i64
  br label %593

593:                                              ; preds = %581, %593
  %594 = phi i64 [ %591, %581 ], [ %615, %593 ]
  %595 = trunc i64 %594 to i32
  %596 = add i32 %595, 1
  %597 = freeze i32 %596
  %598 = freeze i32 %583
  %599 = sdiv i32 %597, %598
  %600 = mul i32 %599, %598
  %601 = sub i32 %597, %600
  %602 = icmp eq i32 %601, 0
  %603 = sext i1 %602 to i32
  %604 = select i1 %602, i32 %583, i32 %601
  %605 = add i32 %599, -1
  %606 = add i32 %605, %603
  %607 = add i32 %606, %587
  %608 = mul nsw i32 %607, %588
  %609 = add i32 %604, %589
  %610 = add nsw i32 %609, %608
  %611 = sext i32 %610 to i64
  %612 = getelementptr inbounds [4 x i8], ptr %89, i64 %611
  %613 = load float, ptr %612, align 4, !tbaa !39
  %614 = getelementptr inbounds [4 x i8], ptr %94, i64 %594
  store float %613, ptr %614, align 4, !tbaa !39
  %615 = add nsw i64 %594, 256
  %616 = icmp slt i64 %615, %592
  br i1 %616, label %593, label %617, !llvm.loop !73

617:                                              ; preds = %593, %578
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %618 = load i32, ptr %95, align 4, !tbaa !26
  %619 = icmp sgt i32 %618, %38
  br i1 %619, label %620, label %656

620:                                              ; preds = %617
  %621 = getelementptr inbounds nuw i8, ptr %0, i64 268
  %622 = load i32, ptr %621, align 4, !tbaa !74
  %623 = getelementptr inbounds nuw i8, ptr %0, i64 252
  %624 = load i32, ptr %623, align 4, !tbaa !75
  %625 = getelementptr inbounds nuw i8, ptr %0, i64 260
  %626 = load i32, ptr %625, align 4, !tbaa !76
  %627 = load i32, ptr %554, align 4, !tbaa !63
  %628 = add i32 %624, -2
  %629 = shl i64 %37, 32
  %630 = ashr exact i64 %629, 32
  %631 = sext i32 %618 to i64
  br label %632

632:                                              ; preds = %620, %632
  %633 = phi i64 [ %630, %620 ], [ %654, %632 ]
  %634 = trunc i64 %633 to i32
  %635 = add i32 %634, 1
  %636 = freeze i32 %635
  %637 = freeze i32 %622
  %638 = sdiv i32 %636, %637
  %639 = mul i32 %638, %637
  %640 = sub i32 %636, %639
  %641 = icmp eq i32 %640, 0
  %642 = sext i1 %641 to i32
  %643 = select i1 %641, i32 %622, i32 %640
  %644 = add i32 %638, -1
  %645 = add i32 %644, %642
  %646 = add i32 %645, %626
  %647 = mul nsw i32 %646, %627
  %648 = add i32 %643, %628
  %649 = add nsw i32 %648, %647
  %650 = sext i32 %649 to i64
  %651 = getelementptr inbounds [4 x i8], ptr %89, i64 %650
  %652 = load float, ptr %651, align 4, !tbaa !39
  %653 = getelementptr inbounds [4 x i8], ptr %99, i64 %633
  store float %652, ptr %653, align 4, !tbaa !39
  %654 = add nsw i64 %633, 256
  %655 = icmp slt i64 %654, %631
  br i1 %655, label %632, label %656, !llvm.loop !77

656:                                              ; preds = %632, %617
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %657 = load i32, ptr %95, align 4, !tbaa !26
  %658 = icmp sgt i32 %657, %38
  br i1 %658, label %659, label %672

659:                                              ; preds = %656
  %660 = shl i64 %37, 32
  %661 = ashr exact i64 %660, 32
  %662 = sext i32 %657 to i64
  br label %663

663:                                              ; preds = %659, %663
  %664 = phi i64 [ %661, %659 ], [ %670, %663 ]
  %665 = getelementptr inbounds [4 x i8], ptr %94, i64 %664
  %666 = load float, ptr %665, align 4, !tbaa !39
  %667 = getelementptr inbounds [4 x i8], ptr %99, i64 %664
  %668 = load float, ptr %667, align 4, !tbaa !39
  %669 = fsub float %666, %668
  store float %669, ptr %667, align 4, !tbaa !39
  %670 = add nsw i64 %664, 256
  %671 = icmp slt i64 %670, %662
  br i1 %671, label %663, label %672, !llvm.loop !78

672:                                              ; preds = %663, %656
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %673 = load i32, ptr %100, align 4, !tbaa !27
  %674 = icmp sgt i32 %673, %38
  br i1 %674, label %675, label %687

675:                                              ; preds = %672
  %676 = shl i64 %37, 32
  %677 = ashr exact i64 %676, 32
  %678 = sext i32 %673 to i64
  br label %679

679:                                              ; preds = %675, %679
  %680 = phi i64 [ %677, %675 ], [ %685, %679 ]
  %681 = getelementptr inbounds [4 x i8], ptr %69, i64 %680
  %682 = load float, ptr %681, align 4, !tbaa !39
  %683 = fmul float %682, %682
  %684 = getelementptr inbounds [4 x i8], ptr %104, i64 %680
  store float %683, ptr %684, align 4, !tbaa !39
  %685 = add nsw i64 %680, 256
  %686 = icmp slt i64 %685, %678
  br i1 %686, label %679, label %687, !llvm.loop !79

687:                                              ; preds = %679, %672
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %688 = load i32, ptr %75, align 4, !tbaa !22
  %689 = icmp sgt i32 %688, %38
  br i1 %689, label %690, label %740

690:                                              ; preds = %687
  %691 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %692 = load i32, ptr %691, align 4, !tbaa !53
  %693 = getelementptr inbounds nuw i8, ptr %0, i64 132
  %694 = load i32, ptr %693, align 4, !tbaa !54
  %695 = getelementptr inbounds nuw i8, ptr %0, i64 284
  %696 = load i32, ptr %695, align 4
  %697 = add nsw i32 %696, %694
  %698 = getelementptr inbounds nuw i8, ptr %0, i64 136
  %699 = load i32, ptr %698, align 4
  %700 = getelementptr inbounds nuw i8, ptr %0, i64 288
  %701 = load i32, ptr %700, align 4
  %702 = add nsw i32 %701, %699
  %703 = xor i32 %694, -1
  %704 = shl i64 %37, 32
  %705 = ashr exact i64 %704, 32
  %706 = sext i32 %688 to i64
  br label %707

707:                                              ; preds = %690, %735
  %708 = phi i64 [ %705, %690 ], [ %738, %735 ]
  %709 = trunc i64 %708 to i32
  %710 = add i32 %709, 1
  %711 = freeze i32 %710
  %712 = freeze i32 %692
  %713 = sdiv i32 %711, %712
  %714 = mul i32 %713, %712
  %715 = sub i32 %711, %714
  %716 = icmp eq i32 %715, 0
  %717 = sext i1 %716 to i32
  %718 = add nsw i32 %713, %717
  %719 = select i1 %716, i32 %692, i32 %715
  %720 = icmp sgt i32 %719, %694
  %721 = icmp sle i32 %719, %697
  %722 = select i1 %720, i1 %721, i1 false
  %723 = icmp sge i32 %718, %699
  %724 = select i1 %722, i1 %723, i1 false
  %725 = icmp slt i32 %718, %702
  %726 = select i1 %724, i1 %725, i1 false
  br i1 %726, label %727, label %735

727:                                              ; preds = %707
  %728 = add i32 %719, %703
  %729 = sub nsw i32 %718, %699
  %730 = mul nsw i32 %729, %696
  %731 = add nsw i32 %728, %730
  %732 = sext i32 %731 to i64
  %733 = getelementptr inbounds [4 x i8], ptr %104, i64 %732
  %734 = load float, ptr %733, align 4, !tbaa !39
  br label %735

735:                                              ; preds = %707, %727
  %736 = phi float [ %734, %727 ], [ 0.000000e+00, %707 ]
  %737 = getelementptr inbounds [4 x i8], ptr %79, i64 %708
  store float %736, ptr %737, align 4, !tbaa !39
  %738 = add nsw i64 %708, 256
  %739 = icmp slt i64 %738, %706
  br i1 %739, label %707, label %740, !llvm.loop !80

740:                                              ; preds = %735, %687
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %741 = load i32, ptr %425, align 4, !tbaa !56
  %742 = icmp sgt i32 %741, %38
  br i1 %742, label %743, label %772

743:                                              ; preds = %740
  %744 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %745 = load i32, ptr %744, align 4, !tbaa !53
  %746 = mul i32 %745, %38
  %747 = shl i32 %745, 8
  %748 = shl i64 %37, 32
  %749 = ashr exact i64 %748, 32
  %750 = sext i32 %741 to i64
  %751 = sext i32 %745 to i64
  %752 = icmp sgt i32 %745, 0
  br label %753

753:                                              ; preds = %743, %768
  %754 = phi i64 [ %749, %743 ], [ %769, %768 ]
  %755 = phi i32 [ %746, %743 ], [ %771, %768 ]
  %756 = add nsw i64 %754, 1
  %757 = mul i64 %756, %751
  br i1 %752, label %758, label %768

758:                                              ; preds = %753
  %759 = sext i32 %755 to i64
  br label %760

760:                                              ; preds = %758, %760
  %761 = phi i64 [ %759, %758 ], [ %766, %760 ]
  %762 = phi float [ 0.000000e+00, %758 ], [ %765, %760 ]
  %763 = getelementptr inbounds [4 x i8], ptr %79, i64 %761
  %764 = load float, ptr %763, align 4, !tbaa !39
  %765 = fadd float %762, %764
  store float %765, ptr %763, align 4, !tbaa !39
  %766 = add nsw i64 %761, 1
  %767 = icmp slt i64 %766, %757
  br i1 %767, label %760, label %768, !llvm.loop !81

768:                                              ; preds = %760, %753
  %769 = add nsw i64 %754, 256
  %770 = icmp slt i64 %769, %750
  %771 = add i32 %755, %747
  br i1 %770, label %753, label %772, !llvm.loop !82

772:                                              ; preds = %768, %740
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %773 = load i32, ptr %80, align 4, !tbaa !23
  %774 = icmp sgt i32 %773, %38
  br i1 %774, label %775, label %812

775:                                              ; preds = %772
  %776 = getelementptr inbounds nuw i8, ptr %0, i64 156
  %777 = load i32, ptr %776, align 4, !tbaa !59
  %778 = getelementptr inbounds nuw i8, ptr %0, i64 172
  %779 = load i32, ptr %778, align 4, !tbaa !60
  %780 = getelementptr inbounds nuw i8, ptr %0, i64 180
  %781 = load i32, ptr %780, align 4, !tbaa !61
  %782 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %783 = load i32, ptr %782, align 4, !tbaa !53
  %784 = add i32 %779, -2
  %785 = shl i64 %37, 32
  %786 = ashr exact i64 %785, 32
  %787 = sext i32 %773 to i64
  br label %788

788:                                              ; preds = %775, %788
  %789 = phi i64 [ %786, %775 ], [ %810, %788 ]
  %790 = trunc i64 %789 to i32
  %791 = add i32 %790, 1
  %792 = freeze i32 %791
  %793 = freeze i32 %777
  %794 = sdiv i32 %792, %793
  %795 = mul i32 %794, %793
  %796 = sub i32 %792, %795
  %797 = icmp eq i32 %796, 0
  %798 = sext i1 %797 to i32
  %799 = select i1 %797, i32 %777, i32 %796
  %800 = add i32 %794, -1
  %801 = add i32 %800, %798
  %802 = add i32 %801, %781
  %803 = mul nsw i32 %802, %783
  %804 = add i32 %799, %784
  %805 = add nsw i32 %804, %803
  %806 = sext i32 %805 to i64
  %807 = getelementptr inbounds [4 x i8], ptr %79, i64 %806
  %808 = load float, ptr %807, align 4, !tbaa !39
  %809 = getelementptr inbounds [4 x i8], ptr %84, i64 %789
  store float %808, ptr %809, align 4, !tbaa !39
  %810 = add nsw i64 %789, 256
  %811 = icmp slt i64 %810, %787
  br i1 %811, label %788, label %812, !llvm.loop !83

812:                                              ; preds = %788, %772
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %813 = load i32, ptr %85, align 4, !tbaa !24
  %814 = icmp sgt i32 %813, %38
  br i1 %814, label %815, label %851

815:                                              ; preds = %812
  %816 = load i32, ptr %554, align 4, !tbaa !63
  %817 = getelementptr inbounds nuw i8, ptr %0, i64 188
  %818 = load i32, ptr %817, align 4, !tbaa !64
  %819 = getelementptr inbounds nuw i8, ptr %0, i64 196
  %820 = load i32, ptr %819, align 4, !tbaa !65
  %821 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %822 = load i32, ptr %821, align 4, !tbaa !53
  %823 = add i32 %818, -2
  %824 = shl i64 %37, 32
  %825 = ashr exact i64 %824, 32
  %826 = sext i32 %813 to i64
  br label %827

827:                                              ; preds = %815, %827
  %828 = phi i64 [ %825, %815 ], [ %849, %827 ]
  %829 = trunc i64 %828 to i32
  %830 = add i32 %829, 1
  %831 = freeze i32 %830
  %832 = freeze i32 %816
  %833 = sdiv i32 %831, %832
  %834 = mul i32 %833, %832
  %835 = sub i32 %831, %834
  %836 = icmp eq i32 %835, 0
  %837 = sext i1 %836 to i32
  %838 = select i1 %836, i32 %816, i32 %835
  %839 = add i32 %833, -1
  %840 = add i32 %839, %837
  %841 = add i32 %840, %820
  %842 = mul nsw i32 %841, %822
  %843 = add i32 %838, %823
  %844 = add nsw i32 %843, %842
  %845 = sext i32 %844 to i64
  %846 = getelementptr inbounds [4 x i8], ptr %79, i64 %845
  %847 = load float, ptr %846, align 4, !tbaa !39
  %848 = getelementptr inbounds [4 x i8], ptr %89, i64 %828
  store float %847, ptr %848, align 4, !tbaa !39
  %849 = add nsw i64 %828, 256
  %850 = icmp slt i64 %849, %826
  br i1 %850, label %827, label %851, !llvm.loop !84

851:                                              ; preds = %827, %812
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %852 = load i32, ptr %85, align 4, !tbaa !24
  %853 = icmp sgt i32 %852, %38
  br i1 %853, label %854, label %867

854:                                              ; preds = %851
  %855 = shl i64 %37, 32
  %856 = ashr exact i64 %855, 32
  %857 = sext i32 %852 to i64
  br label %858

858:                                              ; preds = %854, %858
  %859 = phi i64 [ %856, %854 ], [ %865, %858 ]
  %860 = getelementptr inbounds [4 x i8], ptr %84, i64 %859
  %861 = load float, ptr %860, align 4, !tbaa !39
  %862 = getelementptr inbounds [4 x i8], ptr %89, i64 %859
  %863 = load float, ptr %862, align 4, !tbaa !39
  %864 = fsub float %861, %863
  store float %864, ptr %862, align 4, !tbaa !39
  %865 = add nsw i64 %859, 256
  %866 = icmp slt i64 %865, %857
  br i1 %866, label %858, label %867, !llvm.loop !85

867:                                              ; preds = %858, %851
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %868 = load i32, ptr %554, align 4, !tbaa !63
  %869 = icmp sgt i32 %868, %38
  br i1 %869, label %870, label %891

870:                                              ; preds = %867
  %871 = load i32, ptr %85, align 4, !tbaa !24
  %872 = shl i64 %37, 32
  %873 = ashr exact i64 %872, 32
  %874 = sext i32 %868 to i64
  %875 = sext i32 %871 to i64
  %876 = icmp sgt i32 %871, 0
  br label %877

877:                                              ; preds = %870, %888
  %878 = phi i64 [ %873, %870 ], [ %889, %888 ]
  %879 = add nsw i64 %878, %875
  br i1 %876, label %880, label %888

880:                                              ; preds = %877, %880
  %881 = phi i64 [ %886, %880 ], [ %878, %877 ]
  %882 = phi float [ %885, %880 ], [ 0.000000e+00, %877 ]
  %883 = getelementptr inbounds [4 x i8], ptr %89, i64 %881
  %884 = load float, ptr %883, align 4, !tbaa !39
  %885 = fadd float %882, %884
  store float %885, ptr %883, align 4, !tbaa !39
  %886 = add nsw i64 %881, %874
  %887 = icmp slt i64 %886, %879
  br i1 %887, label %880, label %888, !llvm.loop !86

888:                                              ; preds = %880, %877
  %889 = add nsw i64 %878, 256
  %890 = icmp slt i64 %889, %874
  br i1 %890, label %877, label %891, !llvm.loop !87

891:                                              ; preds = %888, %867
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %892 = load i32, ptr %90, align 4, !tbaa !25
  %893 = icmp sgt i32 %892, %38
  br i1 %893, label %894, label %930

894:                                              ; preds = %891
  %895 = getelementptr inbounds nuw i8, ptr %0, i64 220
  %896 = load i32, ptr %895, align 4, !tbaa !70
  %897 = getelementptr inbounds nuw i8, ptr %0, i64 236
  %898 = load i32, ptr %897, align 4, !tbaa !71
  %899 = getelementptr inbounds nuw i8, ptr %0, i64 244
  %900 = load i32, ptr %899, align 4, !tbaa !72
  %901 = load i32, ptr %554, align 4, !tbaa !63
  %902 = add i32 %898, -2
  %903 = shl i64 %37, 32
  %904 = ashr exact i64 %903, 32
  %905 = sext i32 %892 to i64
  br label %906

906:                                              ; preds = %894, %906
  %907 = phi i64 [ %904, %894 ], [ %928, %906 ]
  %908 = trunc i64 %907 to i32
  %909 = add i32 %908, 1
  %910 = freeze i32 %909
  %911 = freeze i32 %896
  %912 = sdiv i32 %910, %911
  %913 = mul i32 %912, %911
  %914 = sub i32 %910, %913
  %915 = icmp eq i32 %914, 0
  %916 = sext i1 %915 to i32
  %917 = select i1 %915, i32 %896, i32 %914
  %918 = add i32 %912, -1
  %919 = add i32 %918, %916
  %920 = add i32 %919, %900
  %921 = mul nsw i32 %920, %901
  %922 = add i32 %917, %902
  %923 = add nsw i32 %922, %921
  %924 = sext i32 %923 to i64
  %925 = getelementptr inbounds [4 x i8], ptr %89, i64 %924
  %926 = load float, ptr %925, align 4, !tbaa !39
  %927 = getelementptr inbounds [4 x i8], ptr %94, i64 %907
  store float %926, ptr %927, align 4, !tbaa !39
  %928 = add nsw i64 %907, 256
  %929 = icmp slt i64 %928, %905
  br i1 %929, label %906, label %930, !llvm.loop !88

930:                                              ; preds = %906, %891
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %931 = load i32, ptr %95, align 4, !tbaa !26
  %932 = icmp sgt i32 %931, %38
  br i1 %932, label %933, label %969

933:                                              ; preds = %930
  %934 = getelementptr inbounds nuw i8, ptr %0, i64 268
  %935 = load i32, ptr %934, align 4, !tbaa !74
  %936 = getelementptr inbounds nuw i8, ptr %0, i64 252
  %937 = load i32, ptr %936, align 4, !tbaa !75
  %938 = getelementptr inbounds nuw i8, ptr %0, i64 260
  %939 = load i32, ptr %938, align 4, !tbaa !76
  %940 = load i32, ptr %554, align 4, !tbaa !63
  %941 = add i32 %937, -2
  %942 = shl i64 %37, 32
  %943 = ashr exact i64 %942, 32
  %944 = sext i32 %931 to i64
  br label %945

945:                                              ; preds = %933, %945
  %946 = phi i64 [ %943, %933 ], [ %967, %945 ]
  %947 = trunc i64 %946 to i32
  %948 = add i32 %947, 1
  %949 = freeze i32 %948
  %950 = freeze i32 %935
  %951 = sdiv i32 %949, %950
  %952 = mul i32 %951, %950
  %953 = sub i32 %949, %952
  %954 = icmp eq i32 %953, 0
  %955 = sext i1 %954 to i32
  %956 = select i1 %954, i32 %935, i32 %953
  %957 = add i32 %951, -1
  %958 = add i32 %957, %955
  %959 = add i32 %958, %939
  %960 = mul nsw i32 %959, %940
  %961 = add i32 %956, %941
  %962 = add nsw i32 %961, %960
  %963 = sext i32 %962 to i64
  %964 = getelementptr inbounds [4 x i8], ptr %89, i64 %963
  %965 = load float, ptr %964, align 4, !tbaa !39
  %966 = getelementptr inbounds [4 x i8], ptr %109, i64 %946
  store float %965, ptr %966, align 4, !tbaa !39
  %967 = add nsw i64 %946, 256
  %968 = icmp slt i64 %967, %944
  br i1 %968, label %945, label %969, !llvm.loop !89

969:                                              ; preds = %945, %930
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %970 = load i32, ptr %95, align 4, !tbaa !26
  %971 = icmp sgt i32 %970, %38
  br i1 %971, label %972, label %985

972:                                              ; preds = %969
  %973 = shl i64 %37, 32
  %974 = ashr exact i64 %973, 32
  %975 = sext i32 %970 to i64
  br label %976

976:                                              ; preds = %972, %976
  %977 = phi i64 [ %974, %972 ], [ %983, %976 ]
  %978 = getelementptr inbounds [4 x i8], ptr %94, i64 %977
  %979 = load float, ptr %978, align 4, !tbaa !39
  %980 = getelementptr inbounds [4 x i8], ptr %109, i64 %977
  %981 = load float, ptr %980, align 4, !tbaa !39
  %982 = fsub float %979, %981
  store float %982, ptr %980, align 4, !tbaa !39
  %983 = add nsw i64 %977, 256
  %984 = icmp slt i64 %983, %975
  br i1 %984, label %976, label %985, !llvm.loop !90

985:                                              ; preds = %976, %969
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %986 = load i32, ptr %95, align 4, !tbaa !26
  %987 = icmp sgt i32 %986, %38
  br i1 %987, label %988, label %1008

988:                                              ; preds = %985
  %989 = load i32, ptr %125, align 4, !tbaa !19
  %990 = sitofp i32 %989 to float
  %991 = shl i64 %37, 32
  %992 = ashr exact i64 %991, 32
  %993 = sext i32 %986 to i64
  br label %994

994:                                              ; preds = %988, %994
  %995 = phi i64 [ %992, %988 ], [ %1006, %994 ]
  %996 = getelementptr inbounds [4 x i8], ptr %99, i64 %995
  %997 = load float, ptr %996, align 4, !tbaa !39
  %998 = getelementptr inbounds [4 x i8], ptr %109, i64 %995
  %999 = load float, ptr %998, align 4, !tbaa !39
  %1000 = fmul float %997, %997
  %1001 = fdiv float %1000, %990, !fpmath !91
  %1002 = fsub float %999, %1001
  %1003 = fcmp olt float %1002, 0.000000e+00
  %1004 = select i1 %1003, float 0.000000e+00, float %1002
  %1005 = tail call float @_Z4sqrtf(float noundef %1004) #6, !fpmath !92
  store float %1005, ptr %998, align 4, !tbaa !39
  %1006 = add nsw i64 %995, 256
  %1007 = icmp slt i64 %1006, %993
  br i1 %1007, label %994, label %1008, !llvm.loop !93

1008:                                             ; preds = %994, %985
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1009 = load i32, ptr %110, align 4, !tbaa !29
  %1010 = icmp sgt i32 %1009, %38
  br i1 %1010, label %1011, label %1023

1011:                                             ; preds = %1008
  %1012 = shl i64 %37, 32
  %1013 = ashr exact i64 %1012, 32
  %1014 = sext i32 %1009 to i64
  br label %1015

1015:                                             ; preds = %1011, %1015
  %1016 = phi i64 [ %1013, %1011 ], [ %1021, %1015 ]
  %1017 = getelementptr inbounds [4 x i8], ptr %64, i64 %1016
  %1018 = load float, ptr %1017, align 4, !tbaa !39
  %1019 = fmul float %1018, %1018
  %1020 = getelementptr inbounds [4 x i8], ptr %114, i64 %1016
  store float %1019, ptr %1020, align 4, !tbaa !39
  %1021 = add nsw i64 %1016, 256
  %1022 = icmp slt i64 %1021, %1014
  br i1 %1022, label %1015, label %1023, !llvm.loop !94

1023:                                             ; preds = %1015, %1008
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1024 = load i32, ptr %127, align 4, !tbaa !32
  %1025 = icmp sgt i32 %1024, %38
  br i1 %1025, label %1026, label %1053

1026:                                             ; preds = %1023
  %1027 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %1028 = load i32, ptr %1027, align 4, !tbaa !37
  %1029 = icmp sgt i32 %1028, 0
  %1030 = shl i64 %37, 32
  %1031 = ashr exact i64 %1030, 32
  %1032 = sext i32 %1028 to i64
  %1033 = sext i32 %1024 to i64
  %1034 = zext nneg i32 %1028 to i64
  br label %1035

1035:                                             ; preds = %1026, %1048
  %1036 = phi i64 [ %1031, %1026 ], [ %1051, %1048 ]
  br i1 %1029, label %1037, label %1048

1037:                                             ; preds = %1035
  %1038 = mul nsw i64 %1036, %1032
  %1039 = getelementptr [4 x i8], ptr %64, i64 %1038
  br label %1040

1040:                                             ; preds = %1037, %1040
  %1041 = phi i64 [ 0, %1037 ], [ %1046, %1040 ]
  %1042 = phi float [ 0.000000e+00, %1037 ], [ %1045, %1040 ]
  %1043 = getelementptr [4 x i8], ptr %1039, i64 %1041
  %1044 = load float, ptr %1043, align 4, !tbaa !39
  %1045 = fadd float %1042, %1044
  %1046 = add nuw nsw i64 %1041, 1
  %1047 = icmp eq i64 %1046, %1034
  br i1 %1047, label %1048, label %1040, !llvm.loop !95

1048:                                             ; preds = %1040, %1035
  %1049 = phi float [ 0.000000e+00, %1035 ], [ %1045, %1040 ]
  %1050 = getelementptr inbounds [4 x i8], ptr %131, i64 %1036
  store float %1049, ptr %1050, align 4, !tbaa !39
  %1051 = add nsw i64 %1036, 256
  %1052 = icmp slt i64 %1051, %1033
  br i1 %1052, label %1035, label %1053, !llvm.loop !96

1053:                                             ; preds = %1048, %1023
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1054 = load i32, ptr %132, align 4, !tbaa !33
  %1055 = icmp sgt i32 %1054, %38
  br i1 %1055, label %1056, label %1082

1056:                                             ; preds = %1053
  %1057 = getelementptr inbounds nuw i8, ptr %0, i64 320
  %1058 = load i32, ptr %1057, align 4, !tbaa !97
  %1059 = icmp sgt i32 %1058, 0
  %1060 = sext i32 %1054 to i64
  %1061 = shl i64 %37, 32
  %1062 = ashr exact i64 %1061, 32
  %1063 = zext nneg i32 %1058 to i64
  br label %1064

1064:                                             ; preds = %1056, %1077
  %1065 = phi i64 [ %1062, %1056 ], [ %1080, %1077 ]
  br i1 %1059, label %1066, label %1077

1066:                                             ; preds = %1064
  %1067 = getelementptr [4 x i8], ptr %114, i64 %1065
  br label %1068

1068:                                             ; preds = %1066, %1068
  %1069 = phi i64 [ 0, %1066 ], [ %1075, %1068 ]
  %1070 = phi float [ 0.000000e+00, %1066 ], [ %1074, %1068 ]
  %1071 = mul nsw i64 %1069, %1060
  %1072 = getelementptr [4 x i8], ptr %1067, i64 %1071
  %1073 = load float, ptr %1072, align 4, !tbaa !39
  %1074 = fadd float %1070, %1073
  %1075 = add nuw nsw i64 %1069, 1
  %1076 = icmp eq i64 %1075, %1063
  br i1 %1076, label %1077, label %1068, !llvm.loop !98

1077:                                             ; preds = %1068, %1064
  %1078 = phi float [ 0.000000e+00, %1064 ], [ %1074, %1068 ]
  %1079 = getelementptr inbounds [4 x i8], ptr %136, i64 %1065
  store float %1078, ptr %1079, align 4, !tbaa !39
  %1080 = add nsw i64 %1065, 256
  %1081 = icmp slt i64 %1080, %1060
  br i1 %1081, label %1064, label %1082, !llvm.loop !99

1082:                                             ; preds = %1077, %1053
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1083 = icmp eq i32 %38, 0
  br i1 %1083, label %1084, label %1097

1084:                                             ; preds = %1082
  store float 0.000000e+00, ptr %145, align 4, !tbaa !39
  %1085 = load i32, ptr %127, align 4, !tbaa !32
  %1086 = icmp sgt i32 %1085, 0
  br i1 %1086, label %1087, label %1113

1087:                                             ; preds = %1084
  %1088 = zext nneg i32 %1085 to i64
  br label %1089

1089:                                             ; preds = %1087, %1089
  %1090 = phi i64 [ 0, %1087 ], [ %1095, %1089 ]
  %1091 = phi float [ 0.000000e+00, %1087 ], [ %1094, %1089 ]
  %1092 = getelementptr inbounds nuw [4 x i8], ptr %131, i64 %1090
  %1093 = load float, ptr %1092, align 4, !tbaa !39
  %1094 = fadd float %1091, %1093
  store float %1094, ptr %145, align 4, !tbaa !39
  %1095 = add nuw nsw i64 %1090, 1
  %1096 = icmp eq i64 %1095, %1088
  br i1 %1096, label %1113, label %1089, !llvm.loop !100

1097:                                             ; preds = %1082
  %1098 = icmp eq i32 %38, 1
  br i1 %1098, label %1099, label %1113

1099:                                             ; preds = %1097
  store float 0.000000e+00, ptr %146, align 4, !tbaa !39
  %1100 = getelementptr inbounds nuw i8, ptr %0, i64 320
  %1101 = load i32, ptr %1100, align 4, !tbaa !97
  %1102 = icmp sgt i32 %1101, 0
  br i1 %1102, label %1103, label %1113

1103:                                             ; preds = %1099
  %1104 = zext nneg i32 %1101 to i64
  br label %1105

1105:                                             ; preds = %1103, %1105
  %1106 = phi i64 [ 0, %1103 ], [ %1111, %1105 ]
  %1107 = phi float [ 0.000000e+00, %1103 ], [ %1110, %1105 ]
  %1108 = getelementptr inbounds nuw [4 x i8], ptr %136, i64 %1106
  %1109 = load float, ptr %1108, align 4, !tbaa !39
  %1110 = fadd float %1107, %1109
  store float %1110, ptr %146, align 4, !tbaa !39
  %1111 = add nuw nsw i64 %1106, 1
  %1112 = icmp eq i64 %1111, %1104
  br i1 %1112, label %1113, label %1105, !llvm.loop !101

1113:                                             ; preds = %1105, %1089, %1099, %1084, %1097
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  br i1 %1083, label %1114, label %1128

1114:                                             ; preds = %1113
  %1115 = load float, ptr %145, align 4, !tbaa !39
  %1116 = load i32, ptr %125, align 4, !tbaa !19
  %1117 = sitofp i32 %1116 to float
  %1118 = fdiv float %1115, %1117, !fpmath !91
  %1119 = fmul float %1118, %1118
  %1120 = load float, ptr %146, align 4, !tbaa !39
  %1121 = fdiv float %1120, %1117, !fpmath !91
  %1122 = fsub float %1121, %1119
  %1123 = tail call float @_Z4sqrtf(float noundef %1122) #6, !fpmath !92
  %1124 = add nsw i32 %1116, -1
  %1125 = sitofp i32 %1124 to float
  %1126 = tail call float @_Z4sqrtf(float noundef %1125) #6, !fpmath !92
  %1127 = fmul float %1123, %1126
  store float %1127, ptr %147, align 4, !tbaa !39
  br label %1128

1128:                                             ; preds = %1114, %1113
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1129 = load i32, ptr %95, align 4, !tbaa !26
  %1130 = icmp sgt i32 %1129, %38
  br i1 %1130, label %1131, label %1143

1131:                                             ; preds = %1128
  %1132 = shl i64 %37, 32
  %1133 = ashr exact i64 %1132, 32
  %1134 = sext i32 %1129 to i64
  br label %1135

1135:                                             ; preds = %1131, %1135
  %1136 = phi i64 [ %1133, %1131 ], [ %1141, %1135 ]
  %1137 = getelementptr inbounds [4 x i8], ptr %109, i64 %1136
  %1138 = load float, ptr %1137, align 4, !tbaa !39
  %1139 = load float, ptr %147, align 4, !tbaa !39
  %1140 = fmul float %1138, %1139
  store float %1140, ptr %1137, align 4, !tbaa !39
  %1141 = add nsw i64 %1136, 256
  %1142 = icmp slt i64 %1141, %1134
  br i1 %1142, label %1135, label %1143, !llvm.loop !102

1143:                                             ; preds = %1135, %1128
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1144 = load i32, ptr %70, align 4, !tbaa !21
  %1145 = icmp sgt i32 %1144, %38
  br i1 %1145, label %1146, label %1164

1146:                                             ; preds = %1143
  %1147 = load i32, ptr %125, align 4, !tbaa !19
  %1148 = sitofp i32 %1147 to float
  %1149 = shl i64 %37, 32
  %1150 = ashr exact i64 %1149, 32
  %1151 = sext i32 %1144 to i64
  br label %1152

1152:                                             ; preds = %1146, %1152
  %1153 = phi i64 [ %1150, %1146 ], [ %1162, %1152 ]
  %1154 = getelementptr inbounds [4 x i8], ptr %74, i64 %1153
  %1155 = load float, ptr %1154, align 4, !tbaa !39
  %1156 = getelementptr inbounds [4 x i8], ptr %99, i64 %1153
  %1157 = load float, ptr %1156, align 4, !tbaa !39
  %1158 = load float, ptr %145, align 4, !tbaa !39
  %1159 = fmul float %1157, %1158
  %1160 = fdiv float %1159, %1148, !fpmath !91
  %1161 = fsub float %1155, %1160
  store float %1161, ptr %1154, align 4, !tbaa !39
  %1162 = add nsw i64 %1153, 256
  %1163 = icmp slt i64 %1162, %1151
  br i1 %1163, label %1152, label %1164, !llvm.loop !103

1164:                                             ; preds = %1152, %1143
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1165 = load i32, ptr %95, align 4, !tbaa !26
  %1166 = icmp sgt i32 %1165, %38
  br i1 %1166, label %1167, label %1180

1167:                                             ; preds = %1164
  %1168 = shl i64 %37, 32
  %1169 = ashr exact i64 %1168, 32
  %1170 = sext i32 %1165 to i64
  br label %1171

1171:                                             ; preds = %1167, %1171
  %1172 = phi i64 [ %1169, %1167 ], [ %1178, %1171 ]
  %1173 = getelementptr inbounds [4 x i8], ptr %74, i64 %1172
  %1174 = load float, ptr %1173, align 4, !tbaa !39
  %1175 = getelementptr inbounds [4 x i8], ptr %109, i64 %1172
  %1176 = load float, ptr %1175, align 4, !tbaa !39
  %1177 = fdiv float %1174, %1176, !fpmath !91
  store float %1177, ptr %1175, align 4, !tbaa !39
  %1178 = add nsw i64 %1172, 256
  %1179 = icmp slt i64 %1178, %1170
  br i1 %1179, label %1171, label %1180, !llvm.loop !104

1180:                                             ; preds = %1171, %1164
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1181 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %1182 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %1183 = load i32, ptr %115, align 4, !tbaa !30
  %1184 = icmp sgt i32 %1183, %38
  br i1 %1184, label %1185, label %1222

1185:                                             ; preds = %1180
  %1186 = add i32 %2, -1
  %1187 = load i32, ptr %1182, align 4, !tbaa !35
  %1188 = mul nsw i32 %1187, %59
  %1189 = add i32 %1186, %1188
  %1190 = sext i32 %1189 to i64
  %1191 = getelementptr inbounds [4 x i8], ptr %63, i64 %1190
  %1192 = load i32, ptr %1191, align 4, !tbaa !36
  %1193 = load i32, ptr %208, align 4, !tbaa !42
  %1194 = add i32 %1193, 1
  %1195 = load i32, ptr %1181, align 4, !tbaa !105
  %1196 = add i32 %1194, %1195
  %1197 = add nsw i32 %1192, %1196
  %1198 = load i32, ptr %209, align 4, !tbaa !36
  %1199 = xor i32 %1198, -1
  %1200 = add i32 %1197, %1199
  %1201 = getelementptr inbounds [4 x i8], ptr %62, i64 %1190
  %1202 = load i32, ptr %1201, align 4, !tbaa !36
  %1203 = add nsw i32 %1202, %1196
  %1204 = load i32, ptr %207, align 4, !tbaa !36
  %1205 = xor i32 %1204, -1
  %1206 = add i32 %1203, %1205
  %1207 = getelementptr inbounds nuw i8, ptr %0, i64 332
  %1208 = load i32, ptr %1207, align 4, !tbaa !106
  %1209 = mul nsw i32 %1208, %1200
  %1210 = add nsw i32 %1206, %1209
  %1211 = shl i64 %37, 32
  %1212 = ashr exact i64 %1211, 32
  %1213 = sext i32 %1183 to i64
  %1214 = sext i32 %1210 to i64
  br label %1215

1215:                                             ; preds = %1185, %1215
  %1216 = phi i64 [ %1212, %1185 ], [ %1220, %1215 ]
  %1217 = icmp eq i64 %1216, %1214
  %1218 = select i1 %1217, float 1.000000e+00, float 0.000000e+00
  %1219 = getelementptr inbounds [4 x i8], ptr %119, i64 %1216
  store float %1218, ptr %1219, align 4, !tbaa !39
  %1220 = add nsw i64 %1216, 256
  %1221 = icmp slt i64 %1220, %1213
  br i1 %1221, label %1215, label %1222, !llvm.loop !107

1222:                                             ; preds = %1215, %1180
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1223 = load i32, ptr %120, align 4, !tbaa !31
  %1224 = icmp sgt i32 %1223, %38
  br i1 %1224, label %1225, label %1303

1225:                                             ; preds = %1222
  %1226 = load i32, ptr %137, align 4, !tbaa !34
  %1227 = getelementptr inbounds nuw i8, ptr %0, i64 384
  %1228 = load i32, ptr %1227, align 4, !tbaa !108
  %1229 = getelementptr inbounds nuw i8, ptr %0, i64 352
  %1230 = load i32, ptr %1229, align 4, !tbaa !109
  %1231 = getelementptr inbounds nuw i8, ptr %0, i64 336
  %1232 = load i32, ptr %1231, align 4, !tbaa !110
  %1233 = getelementptr inbounds nuw i8, ptr %0, i64 380
  %1234 = load i32, ptr %1233, align 4, !tbaa !111
  %1235 = getelementptr inbounds nuw i8, ptr %0, i64 348
  %1236 = load i32, ptr %1235, align 4, !tbaa !112
  %1237 = getelementptr inbounds nuw i8, ptr %0, i64 332
  %1238 = load i32, ptr %1237, align 4, !tbaa !106
  %1239 = sext i32 %1238 to i64
  %1240 = shl i64 %37, 32
  %1241 = ashr exact i64 %1240, 32
  %1242 = sext i32 %1223 to i64
  %1243 = sub i32 1, %1230
  %1244 = sub i32 1, %1236
  br label %1245

1245:                                             ; preds = %1225, %1295
  %1246 = phi i64 [ %1241, %1225 ], [ %1301, %1295 ]
  %1247 = trunc i64 %1246 to i32
  %1248 = add i32 %1247, 1
  %1249 = freeze i32 %1248
  %1250 = freeze i32 %1226
  %1251 = sdiv i32 %1249, %1250
  %1252 = mul i32 %1251, %1250
  %1253 = sub i32 %1249, %1252
  %1254 = icmp ne i32 %1253, 0
  %1255 = zext i1 %1254 to i32
  %1256 = add nsw i32 %1251, %1255
  %1257 = add nsw i32 %1256, %1228
  %1258 = icmp sgt i32 %1230, %1257
  %1259 = add i32 %1257, %1243
  %1260 = select i1 %1258, i32 1, i32 %1259
  %1261 = tail call i32 @llvm.smin.i32(i32 %1232, i32 %1257)
  %1262 = icmp sgt i32 %1260, %1261
  br i1 %1262, label %1295, label %1263

1263:                                             ; preds = %1245
  %1264 = select i1 %1254, i32 %1253, i32 %1226
  %1265 = add i32 %1234, %1264
  %1266 = tail call i32 @llvm.smin.i32(i32 %1238, i32 %1265)
  %1267 = icmp sgt i32 %1236, %1265
  %1268 = add i32 %1265, %1244
  %1269 = select i1 %1267, i32 1, i32 %1268
  %1270 = icmp sgt i32 %1269, %1266
  %1271 = sext i32 %1269 to i64
  %1272 = sext i32 %1266 to i64
  %1273 = sext i32 %1260 to i64
  %1274 = sext i32 %1261 to i64
  br label %1275

1275:                                             ; preds = %1263, %1291
  %1276 = phi i64 [ %1273, %1263 ], [ %1293, %1291 ]
  %1277 = phi float [ 0.000000e+00, %1263 ], [ %1292, %1291 ]
  br i1 %1270, label %1291, label %1278

1278:                                             ; preds = %1275
  %1279 = add nsw i64 %1276, -1
  %1280 = mul nsw i64 %1279, %1239
  %1281 = getelementptr [4 x i8], ptr %119, i64 %1280
  br label %1282

1282:                                             ; preds = %1278, %1282
  %1283 = phi i64 [ %1271, %1278 ], [ %1289, %1282 ]
  %1284 = phi float [ %1277, %1278 ], [ %1288, %1282 ]
  %1285 = getelementptr [4 x i8], ptr %1281, i64 %1283
  %1286 = getelementptr i8, ptr %1285, i64 -4
  %1287 = load float, ptr %1286, align 4, !tbaa !39
  %1288 = fadd float %1284, %1287
  %1289 = add nuw nsw i64 %1283, 1
  %1290 = icmp slt i64 %1283, %1272
  br i1 %1290, label %1282, label %1291, !llvm.loop !113

1291:                                             ; preds = %1282, %1275
  %1292 = phi float [ %1277, %1275 ], [ %1288, %1282 ]
  %1293 = add nuw nsw i64 %1276, 1
  %1294 = icmp slt i64 %1276, %1274
  br i1 %1294, label %1275, label %1295, !llvm.loop !114

1295:                                             ; preds = %1291, %1245
  %1296 = phi float [ 0.000000e+00, %1245 ], [ %1292, %1291 ]
  %1297 = getelementptr inbounds [4 x i8], ptr %109, i64 %1246
  %1298 = load float, ptr %1297, align 4, !tbaa !39
  %1299 = fmul float %1296, %1298
  %1300 = getelementptr inbounds [4 x i8], ptr %124, i64 %1246
  store float %1299, ptr %1300, align 4, !tbaa !39
  %1301 = add nsw i64 %1246, 256
  %1302 = icmp slt i64 %1301, %1242
  br i1 %1302, label %1245, label %1303, !llvm.loop !115

1303:                                             ; preds = %1295, %1222
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1304 = load i32, ptr %137, align 4, !tbaa !34
  %1305 = icmp sgt i32 %1304, %38
  br i1 %1305, label %1306, label %1341

1306:                                             ; preds = %1303
  %1307 = getelementptr inbounds nuw i8, ptr %0, i64 368
  %1308 = load i32, ptr %1307, align 4, !tbaa !116
  %1309 = icmp sgt i32 %1308, 0
  %1310 = shl i64 %37, 32
  %1311 = ashr exact i64 %1310, 32
  %1312 = sext i32 %1304 to i64
  %1313 = zext nneg i32 %1308 to i64
  br label %1314

1314:                                             ; preds = %1306, %1334
  %1315 = phi i64 [ %1311, %1306 ], [ %1339, %1334 ]
  %1316 = phi float [ 0.000000e+00, %1306 ], [ %1336, %1334 ]
  %1317 = phi i32 [ 0, %1306 ], [ %1335, %1334 ]
  br i1 %1309, label %1318, label %1334

1318:                                             ; preds = %1314
  %1319 = mul nsw i64 %1315, %1312
  br label %1320

1320:                                             ; preds = %1318, %1320
  %1321 = phi i64 [ 0, %1318 ], [ %1332, %1320 ]
  %1322 = phi float [ %1316, %1318 ], [ %1331, %1320 ]
  %1323 = phi i32 [ %1317, %1318 ], [ %1330, %1320 ]
  %1324 = add nsw i64 %1321, %1319
  %1325 = getelementptr inbounds [4 x i8], ptr %124, i64 %1324
  %1326 = load float, ptr %1325, align 4, !tbaa !39
  %1327 = tail call float @_Z4fabsf(float noundef %1326) #6
  %1328 = fcmp ogt float %1327, %1322
  %1329 = trunc nsw i64 %1324 to i32
  %1330 = select i1 %1328, i32 %1329, i32 %1323
  %1331 = select i1 %1328, float %1327, float %1322
  %1332 = add nuw nsw i64 %1321, 1
  %1333 = icmp eq i64 %1332, %1313
  br i1 %1333, label %1334, label %1320, !llvm.loop !117

1334:                                             ; preds = %1320, %1314
  %1335 = phi i32 [ %1317, %1314 ], [ %1330, %1320 ]
  %1336 = phi float [ %1316, %1314 ], [ %1331, %1320 ]
  %1337 = getelementptr inbounds [4 x i8], ptr %142, i64 %1315
  store i32 %1335, ptr %1337, align 4, !tbaa !36
  %1338 = getelementptr inbounds [4 x i8], ptr %141, i64 %1315
  store float %1336, ptr %1338, align 4, !tbaa !39
  %1339 = add nsw i64 %1315, 256
  %1340 = icmp slt i64 %1339, %1312
  br i1 %1340, label %1314, label %1341, !llvm.loop !118

1341:                                             ; preds = %1334, %1303
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  br i1 %1083, label %1342, label %1396

1342:                                             ; preds = %1341
  %1343 = load i32, ptr %137, align 4, !tbaa !34
  %1344 = icmp sgt i32 %1343, 0
  br i1 %1344, label %1345, label %1364

1345:                                             ; preds = %1342
  %1346 = zext nneg i32 %1343 to i64
  br label %1347

1347:                                             ; preds = %1345, %1357
  %1348 = phi i64 [ 0, %1345 ], [ %1360, %1357 ]
  %1349 = phi i32 [ 0, %1345 ], [ %1359, %1357 ]
  %1350 = phi float [ 0.000000e+00, %1345 ], [ %1358, %1357 ]
  %1351 = getelementptr inbounds nuw [4 x i8], ptr %141, i64 %1348
  %1352 = load float, ptr %1351, align 4, !tbaa !39
  %1353 = fcmp ogt float %1352, %1350
  br i1 %1353, label %1354, label %1357

1354:                                             ; preds = %1347
  %1355 = getelementptr inbounds nuw [4 x i8], ptr %142, i64 %1348
  %1356 = load i32, ptr %1355, align 4, !tbaa !36
  br label %1357

1357:                                             ; preds = %1354, %1347
  %1358 = phi float [ %1352, %1354 ], [ %1350, %1347 ]
  %1359 = phi i32 [ %1356, %1354 ], [ %1349, %1347 ]
  %1360 = add nuw nsw i64 %1348, 1
  %1361 = icmp eq i64 %1360, %1346
  br i1 %1361, label %1362, label %1347, !llvm.loop !119

1362:                                             ; preds = %1357
  %1363 = add nsw i32 %1359, 1
  br label %1364

1364:                                             ; preds = %1362, %1342
  %1365 = phi i32 [ 1, %1342 ], [ %1363, %1362 ]
  %1366 = freeze i32 %1365
  %1367 = freeze i32 %1343
  %1368 = sdiv i32 %1366, %1367
  %1369 = mul i32 %1368, %1367
  %1370 = sub i32 %1366, %1369
  %1371 = icmp eq i32 %1370, 0
  %1372 = select i1 %1371, i32 %1343, i32 %1370
  %1373 = sext i1 %1371 to i32
  %1374 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %1375 = load i32, ptr %1374, align 4, !tbaa !37
  %1376 = load i32, ptr %208, align 4, !tbaa !42
  %1377 = load i32, ptr %1181, align 4, !tbaa !105
  %1378 = sub i32 %1377, %1376
  %1379 = load i32, ptr %127, align 4, !tbaa !32
  %1380 = load i32, ptr %1182, align 4, !tbaa !35
  %1381 = mul nsw i32 %1380, %59
  %1382 = add nsw i32 %1381, %2
  %1383 = load i32, ptr %207, align 4, !tbaa !36
  %1384 = sub i32 %1372, %1375
  %1385 = add i32 %1384, %1378
  %1386 = add nsw i32 %1385, %1383
  %1387 = sext i32 %1382 to i64
  %1388 = getelementptr inbounds [4 x i8], ptr %62, i64 %1387
  store i32 %1386, ptr %1388, align 4, !tbaa !36
  %1389 = load i32, ptr %209, align 4, !tbaa !36
  %1390 = add i32 %1368, 1
  %1391 = add i32 %1390, %1373
  %1392 = add i32 %1391, %1378
  %1393 = sub i32 %1392, %1379
  %1394 = add nsw i32 %1393, %1389
  %1395 = getelementptr inbounds [4 x i8], ptr %63, i64 %1387
  store i32 %1394, ptr %1395, align 4, !tbaa !36
  br label %1396

1396:                                             ; preds = %1364, %1341
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1397 = srem i32 %2, 10
  %1398 = icmp eq i32 %1397, 0
  br i1 %1398, label %1399, label %1451

1399:                                             ; preds = %1396
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1400 = load i32, ptr %1182, align 4, !tbaa !35
  %1401 = mul nsw i32 %1400, %59
  %1402 = add nsw i32 %1401, %2
  %1403 = sext i32 %1402 to i64
  %1404 = getelementptr inbounds [4 x i8], ptr %62, i64 %1403
  %1405 = load i32, ptr %1404, align 4, !tbaa !36
  store i32 %1405, ptr %207, align 4, !tbaa !36
  %1406 = getelementptr inbounds [4 x i8], ptr %63, i64 %1403
  %1407 = load i32, ptr %1406, align 4, !tbaa !36
  store i32 %1407, ptr %209, align 4, !tbaa !36
  %1408 = load i32, ptr %125, align 4, !tbaa !19
  %1409 = icmp sgt i32 %1408, %38
  br i1 %1409, label %1410, label %1450

1410:                                             ; preds = %1399
  %1411 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %1412 = load i32, ptr %1411, align 4, !tbaa !37
  %1413 = load i32, ptr %207, align 4, !tbaa !36
  %1414 = add i32 %1407, -26
  %1415 = getelementptr inbounds nuw i8, ptr %0, i64 36
  %1416 = load i32, ptr %1415, align 4, !tbaa !38
  %1417 = add i32 %1413, -27
  %1418 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %1419 = load float, ptr %1418, align 4, !tbaa !120
  %1420 = fsub float 1.000000e+00, %1419
  %1421 = shl i64 %37, 32
  %1422 = ashr exact i64 %1421, 32
  %1423 = sext i32 %1408 to i64
  br label %1424

1424:                                             ; preds = %1410, %1424
  %1425 = phi i64 [ %1422, %1410 ], [ %1448, %1424 ]
  %1426 = trunc i64 %1425 to i32
  %1427 = add i32 %1426, 1
  %1428 = freeze i32 %1427
  %1429 = freeze i32 %1412
  %1430 = sdiv i32 %1428, %1429
  %1431 = mul i32 %1430, %1429
  %1432 = sub i32 %1428, %1431
  %1433 = icmp eq i32 %1432, 0
  %1434 = sext i1 %1433 to i32
  %1435 = select i1 %1433, i32 %1412, i32 %1432
  %1436 = add i32 %1414, %1430
  %1437 = add i32 %1436, %1434
  %1438 = mul nsw i32 %1437, %1416
  %1439 = add i32 %1435, %1417
  %1440 = add nsw i32 %1439, %1438
  %1441 = getelementptr inbounds [4 x i8], ptr %64, i64 %1425
  %1442 = load float, ptr %1441, align 4, !tbaa !39
  %1443 = sext i32 %1440 to i64
  %1444 = getelementptr inbounds [4 x i8], ptr %1, i64 %1443
  %1445 = load float, ptr %1444, align 4, !tbaa !39
  %1446 = fmul float %1420, %1445
  %1447 = tail call float @llvm.fmuladd.f32(float %1419, float %1442, float %1446)
  store float %1447, ptr %1441, align 4, !tbaa !39
  %1448 = add nsw i64 %1425, 256
  %1449 = icmp slt i64 %1448, %1423
  br i1 %1449, label %1424, label %1450, !llvm.loop !121

1450:                                             ; preds = %1424, %1399
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  br label %1451

1451:                                             ; preds = %204, %1396, %1450
  ret void
}

; Function Attrs: alwaysinline convergent norecurse nounwind
define dso_local void @__clang_ocl_kern_imp_kernel_gpu_opencl(ptr nofree noundef readonly align 4 dead_on_return %0, ptr nofree noundef readonly align 4 captures(none) %1, i32 noundef signext %2, ptr nofree noundef align 4 captures(none) %3, ptr nofree noundef align 4 captures(none) %4, ptr nofree noundef align 4 captures(none) %5, ptr nofree noundef align 4 captures(none) %6, ptr nofree noundef align 4 captures(none) %7, ptr nofree noundef align 4 captures(none) %8, ptr nofree noundef align 4 captures(none) %9, ptr nofree noundef align 4 captures(none) %10, ptr nofree noundef align 4 captures(none) %11, ptr nofree noundef align 4 captures(none) %12, ptr nofree noundef align 4 captures(none) %13, ptr nofree noundef align 4 captures(none) %14, ptr nofree noundef align 4 captures(none) %15, ptr nofree noundef align 4 captures(none) %16, ptr nofree noundef align 4 captures(none) %17, ptr nofree noundef align 4 captures(none) %18, ptr nofree noundef align 4 captures(none) %19, ptr nofree noundef align 4 captures(none) %20, ptr nofree noundef align 4 captures(none) %21, ptr nofree noundef align 4 captures(none) %22, ptr nofree noundef align 4 captures(none) %23, ptr nofree noundef align 4 captures(none) %24, ptr nofree noundef align 4 captures(none) %25, ptr nofree noundef align 4 captures(none) %26, ptr nofree noundef align 4 captures(none) %27, ptr nofree noundef align 4 captures(none) %28, ptr nofree noundef align 4 captures(none) %29, ptr nofree noundef align 4 captures(none) %30, ptr nofree noundef align 4 captures(none) %31, ptr nofree noundef align 4 captures(none) %32, ptr nofree noundef readnone align 4 captures(none) %33) local_unnamed_addr #1 !kernel_arg_addr_space !11 !kernel_arg_access_qual !12 !kernel_arg_type !13 !kernel_arg_base_type !14 !kernel_arg_type_qual !15 {
  %35 = tail call i64 @_Z12get_group_idj(i32 noundef signext 0) #6
  %36 = trunc i64 %35 to i32
  %37 = tail call i64 @_Z12get_local_idj(i32 noundef signext 0) #6
  %38 = trunc i64 %37 to i32
  %39 = getelementptr inbounds nuw i8, ptr %0, i64 52
  %40 = load i32, ptr %39, align 4, !tbaa !16
  %41 = icmp sgt i32 %40, %36
  br i1 %41, label %42, label %48

42:                                               ; preds = %34
  %43 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %44 = load i32, ptr %43, align 4, !tbaa !19
  %45 = mul nsw i32 %44, %36
  %46 = sext i32 %45 to i64
  %47 = getelementptr inbounds [4 x i8], ptr %11, i64 %46
  br label %57

48:                                               ; preds = %34
  %49 = sub nsw i32 %36, %40
  %50 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %51 = load i32, ptr %50, align 4, !tbaa !19
  %52 = mul nsw i32 %51, %49
  %53 = sext i32 %52 to i64
  %54 = getelementptr inbounds [4 x i8], ptr %12, i64 %53
  %55 = mul nsw i32 %51, %36
  %56 = sext i32 %55 to i64
  br label %57

57:                                               ; preds = %48, %42
  %58 = phi i64 [ %56, %48 ], [ %46, %42 ]
  %59 = phi i32 [ %49, %48 ], [ %36, %42 ]
  %60 = phi ptr [ %7, %48 ], [ %3, %42 ]
  %61 = phi ptr [ %8, %48 ], [ %4, %42 ]
  %62 = phi ptr [ %9, %48 ], [ %5, %42 ]
  %63 = phi ptr [ %10, %48 ], [ %6, %42 ]
  %64 = phi ptr [ %54, %48 ], [ %47, %42 ]
  %65 = getelementptr inbounds nuw i8, ptr %0, i64 100
  %66 = load i32, ptr %65, align 4, !tbaa !20
  %67 = mul nsw i32 %66, %36
  %68 = sext i32 %67 to i64
  %69 = getelementptr inbounds [4 x i8], ptr %13, i64 %68
  %70 = getelementptr inbounds nuw i8, ptr %0, i64 116
  %71 = load i32, ptr %70, align 4, !tbaa !21
  %72 = mul nsw i32 %71, %36
  %73 = sext i32 %72 to i64
  %74 = getelementptr inbounds [4 x i8], ptr %14, i64 %73
  %75 = getelementptr inbounds nuw i8, ptr %0, i64 148
  %76 = load i32, ptr %75, align 4, !tbaa !22
  %77 = mul nsw i32 %76, %36
  %78 = sext i32 %77 to i64
  %79 = getelementptr inbounds [4 x i8], ptr %15, i64 %78
  %80 = getelementptr inbounds nuw i8, ptr %0, i64 164
  %81 = load i32, ptr %80, align 4, !tbaa !23
  %82 = mul nsw i32 %81, %36
  %83 = sext i32 %82 to i64
  %84 = getelementptr inbounds [4 x i8], ptr %16, i64 %83
  %85 = getelementptr inbounds nuw i8, ptr %0, i64 212
  %86 = load i32, ptr %85, align 4, !tbaa !24
  %87 = mul nsw i32 %86, %36
  %88 = sext i32 %87 to i64
  %89 = getelementptr inbounds [4 x i8], ptr %17, i64 %88
  %90 = getelementptr inbounds nuw i8, ptr %0, i64 228
  %91 = load i32, ptr %90, align 4, !tbaa !25
  %92 = mul nsw i32 %91, %36
  %93 = sext i32 %92 to i64
  %94 = getelementptr inbounds [4 x i8], ptr %18, i64 %93
  %95 = getelementptr inbounds nuw i8, ptr %0, i64 276
  %96 = load i32, ptr %95, align 4, !tbaa !26
  %97 = mul nsw i32 %96, %36
  %98 = sext i32 %97 to i64
  %99 = getelementptr inbounds [4 x i8], ptr %19, i64 %98
  %100 = getelementptr inbounds nuw i8, ptr %0, i64 292
  %101 = load i32, ptr %100, align 4, !tbaa !27
  %102 = mul nsw i32 %101, %36
  %103 = sext i32 %102 to i64
  %104 = getelementptr inbounds [4 x i8], ptr %20, i64 %103
  %105 = getelementptr inbounds nuw i8, ptr %0, i64 308
  %106 = load i32, ptr %105, align 4, !tbaa !28
  %107 = mul nsw i32 %106, %36
  %108 = sext i32 %107 to i64
  %109 = getelementptr inbounds [4 x i8], ptr %21, i64 %108
  %110 = getelementptr inbounds nuw i8, ptr %0, i64 324
  %111 = load i32, ptr %110, align 4, !tbaa !29
  %112 = mul nsw i32 %111, %36
  %113 = sext i32 %112 to i64
  %114 = getelementptr inbounds [4 x i8], ptr %22, i64 %113
  %115 = getelementptr inbounds nuw i8, ptr %0, i64 340
  %116 = load i32, ptr %115, align 4, !tbaa !30
  %117 = mul nsw i32 %116, %36
  %118 = sext i32 %117 to i64
  %119 = getelementptr inbounds [4 x i8], ptr %23, i64 %118
  %120 = getelementptr inbounds nuw i8, ptr %0, i64 372
  %121 = load i32, ptr %120, align 4, !tbaa !31
  %122 = mul nsw i32 %121, %36
  %123 = sext i32 %122 to i64
  %124 = getelementptr inbounds [4 x i8], ptr %24, i64 %123
  %125 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %126 = getelementptr inbounds [4 x i8], ptr %25, i64 %58
  %127 = getelementptr inbounds nuw i8, ptr %0, i64 76
  %128 = load i32, ptr %127, align 4, !tbaa !32
  %129 = mul nsw i32 %128, %36
  %130 = sext i32 %129 to i64
  %131 = getelementptr inbounds [4 x i8], ptr %26, i64 %130
  %132 = getelementptr inbounds nuw i8, ptr %0, i64 316
  %133 = load i32, ptr %132, align 4, !tbaa !33
  %134 = mul nsw i32 %133, %36
  %135 = sext i32 %134 to i64
  %136 = getelementptr inbounds [4 x i8], ptr %27, i64 %135
  %137 = getelementptr inbounds nuw i8, ptr %0, i64 364
  %138 = load i32, ptr %137, align 4, !tbaa !34
  %139 = mul nsw i32 %138, %36
  %140 = sext i32 %139 to i64
  %141 = getelementptr inbounds [4 x i8], ptr %28, i64 %140
  %142 = getelementptr inbounds [4 x i8], ptr %29, i64 %140
  %143 = shl i64 %35, 32
  %144 = ashr exact i64 %143, 32
  %145 = getelementptr inbounds [4 x i8], ptr %30, i64 %144
  %146 = getelementptr inbounds [4 x i8], ptr %31, i64 %144
  %147 = getelementptr inbounds [4 x i8], ptr %32, i64 %144
  %148 = icmp eq i32 %2, 0
  br i1 %148, label %149, label %205

149:                                              ; preds = %57
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %150 = icmp eq i32 %38, 0
  br i1 %150, label %151, label %163

151:                                              ; preds = %149
  %152 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %153 = load i32, ptr %152, align 4, !tbaa !35
  %154 = mul nsw i32 %153, %59
  %155 = sext i32 %59 to i64
  %156 = getelementptr inbounds [4 x i8], ptr %60, i64 %155
  %157 = load i32, ptr %156, align 4, !tbaa !36
  %158 = sext i32 %154 to i64
  %159 = getelementptr inbounds [4 x i8], ptr %62, i64 %158
  store i32 %157, ptr %159, align 4, !tbaa !36
  %160 = getelementptr inbounds [4 x i8], ptr %61, i64 %155
  %161 = load i32, ptr %160, align 4, !tbaa !36
  %162 = getelementptr inbounds [4 x i8], ptr %63, i64 %158
  store i32 %161, ptr %162, align 4, !tbaa !36
  br label %163

163:                                              ; preds = %151, %149
  %164 = load i32, ptr %125, align 4, !tbaa !19
  %165 = icmp sgt i32 %164, %38
  br i1 %165, label %166, label %204

166:                                              ; preds = %163
  %167 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %168 = load i32, ptr %167, align 4, !tbaa !37
  %169 = sext i32 %59 to i64
  %170 = getelementptr inbounds [4 x i8], ptr %60, i64 %169
  %171 = load i32, ptr %170, align 4, !tbaa !36
  %172 = getelementptr inbounds [4 x i8], ptr %61, i64 %169
  %173 = load i32, ptr %172, align 4, !tbaa !36
  %174 = add i32 %173, -26
  %175 = getelementptr inbounds nuw i8, ptr %0, i64 36
  %176 = load i32, ptr %175, align 4, !tbaa !38
  %177 = add i32 %171, -26
  br label %178

178:                                              ; preds = %166, %178
  %179 = phi i32 [ %38, %166 ], [ %202, %178 ]
  %180 = add nsw i32 %179, 1
  %181 = freeze i32 %180
  %182 = freeze i32 %168
  %183 = sdiv i32 %181, %182
  %184 = mul i32 %183, %182
  %185 = sub i32 %181, %184
  %186 = icmp eq i32 %185, 0
  %187 = sext i1 %186 to i32
  %188 = add nsw i32 %183, %187
  %189 = select i1 %186, i32 %168, i32 %185
  %190 = add nsw i32 %189, -1
  %191 = add i32 %174, %188
  %192 = mul nsw i32 %191, %176
  %193 = add i32 %177, %190
  %194 = add nsw i32 %193, %192
  %195 = sext i32 %194 to i64
  %196 = getelementptr inbounds [4 x i8], ptr %1, i64 %195
  %197 = load float, ptr %196, align 4, !tbaa !39
  %198 = mul nsw i32 %188, %168
  %199 = add nsw i32 %198, %190
  %200 = sext i32 %199 to i64
  %201 = getelementptr inbounds [4 x i8], ptr %64, i64 %200
  store float %197, ptr %201, align 4, !tbaa !39
  %202 = add nsw i32 %179, 256
  %203 = icmp slt i32 %202, %164
  br i1 %203, label %178, label %204, !llvm.loop !40

204:                                              ; preds = %178, %163
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  br label %1454

205:                                              ; preds = %57
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %206 = sext i32 %59 to i64
  %207 = getelementptr inbounds [4 x i8], ptr %60, i64 %206
  %208 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %209 = getelementptr inbounds [4 x i8], ptr %61, i64 %206
  %210 = load i32, ptr %65, align 4, !tbaa !20
  %211 = icmp sgt i32 %210, %38
  br i1 %211, label %212, label %250

212:                                              ; preds = %205
  %213 = load i32, ptr %209, align 4, !tbaa !36
  %214 = load i32, ptr %208, align 4, !tbaa !42
  %215 = load i32, ptr %207, align 4, !tbaa !36
  %216 = getelementptr inbounds nuw i8, ptr %0, i64 92
  %217 = load i32, ptr %216, align 4, !tbaa !43
  %218 = xor i32 %214, -1
  %219 = add i32 %213, %218
  %220 = getelementptr inbounds nuw i8, ptr %0, i64 36
  %221 = load i32, ptr %220, align 4, !tbaa !38
  %222 = add i32 %215, -2
  %223 = sub i32 %222, %214
  %224 = shl i64 %37, 32
  %225 = ashr exact i64 %224, 32
  %226 = sext i32 %210 to i64
  br label %227

227:                                              ; preds = %212, %227
  %228 = phi i64 [ %225, %212 ], [ %248, %227 ]
  %229 = trunc i64 %228 to i32
  %230 = add i32 %229, 1
  %231 = freeze i32 %230
  %232 = freeze i32 %217
  %233 = sdiv i32 %231, %232
  %234 = mul i32 %233, %232
  %235 = sub i32 %231, %234
  %236 = icmp eq i32 %235, 0
  %237 = sext i1 %236 to i32
  %238 = select i1 %236, i32 %217, i32 %235
  %239 = add i32 %219, %233
  %240 = add i32 %239, %237
  %241 = mul nsw i32 %240, %221
  %242 = add i32 %223, %238
  %243 = add nsw i32 %242, %241
  %244 = sext i32 %243 to i64
  %245 = getelementptr inbounds [4 x i8], ptr %1, i64 %244
  %246 = load float, ptr %245, align 4, !tbaa !39
  %247 = getelementptr inbounds [4 x i8], ptr %69, i64 %228
  store float %246, ptr %247, align 4, !tbaa !39
  %248 = add nsw i64 %228, 256
  %249 = icmp slt i64 %248, %226
  br i1 %249, label %227, label %250, !llvm.loop !44

250:                                              ; preds = %227, %205
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %251 = load i32, ptr %125, align 4, !tbaa !19
  %252 = icmp sgt i32 %251, %38
  br i1 %252, label %253, label %283

253:                                              ; preds = %250
  %254 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %255 = load i32, ptr %254, align 4, !tbaa !37
  %256 = shl i64 %37, 32
  %257 = ashr exact i64 %256, 32
  %258 = sext i32 %251 to i64
  br label %259

259:                                              ; preds = %253, %259
  %260 = phi i64 [ %257, %253 ], [ %281, %259 ]
  %261 = trunc i64 %260 to i32
  %262 = add i32 %261, 1
  %263 = freeze i32 %262
  %264 = freeze i32 %255
  %265 = sdiv i32 %263, %264
  %266 = mul i32 %265, %264
  %267 = sub i32 %263, %266
  %268 = icmp eq i32 %267, 0
  %269 = sext i1 %268 to i32
  %270 = add nsw i32 %265, %269
  %271 = sub i32 %255, %267
  %272 = select i1 %268, i32 0, i32 %271
  %273 = xor i32 %270, -1
  %274 = add i32 %255, %273
  %275 = mul nsw i32 %274, %255
  %276 = add nsw i32 %275, %272
  %277 = sext i32 %276 to i64
  %278 = getelementptr inbounds [4 x i8], ptr %64, i64 %277
  %279 = load float, ptr %278, align 4, !tbaa !39
  %280 = getelementptr inbounds [4 x i8], ptr %126, i64 %260
  store float %279, ptr %280, align 4, !tbaa !39
  %281 = add nsw i64 %260, 256
  %282 = icmp slt i64 %281, %258
  br i1 %282, label %259, label %283, !llvm.loop !45

283:                                              ; preds = %259, %250
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %284 = load i32, ptr %70, align 4, !tbaa !21
  %285 = icmp sgt i32 %284, %38
  br i1 %285, label %286, label %371

286:                                              ; preds = %283
  %287 = getelementptr inbounds nuw i8, ptr %0, i64 108
  %288 = load i32, ptr %287, align 4, !tbaa !46
  %289 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %290 = load i32, ptr %289, align 4, !tbaa !47
  %291 = getelementptr inbounds nuw i8, ptr %0, i64 96
  %292 = load i32, ptr %291, align 4, !tbaa !48
  %293 = load i32, ptr %127, align 4, !tbaa !32
  %294 = getelementptr inbounds nuw i8, ptr %0, i64 124
  %295 = load i32, ptr %294, align 4, !tbaa !49
  %296 = getelementptr inbounds nuw i8, ptr %0, i64 92
  %297 = load i32, ptr %296, align 4, !tbaa !43
  %298 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %299 = load i32, ptr %298, align 4, !tbaa !37
  %300 = sext i32 %299 to i64
  %301 = shl i64 %37, 32
  %302 = ashr exact i64 %301, 32
  %303 = sext i32 %284 to i64
  %304 = sub i32 1, %292
  br label %305

305:                                              ; preds = %286, %366
  %306 = phi i64 [ %302, %286 ], [ %369, %366 ]
  %307 = trunc i64 %306 to i32
  %308 = add i32 %307, 1
  %309 = freeze i32 %308
  %310 = freeze i32 %288
  %311 = sdiv i32 %309, %310
  %312 = mul i32 %311, %310
  %313 = sub i32 %309, %312
  %314 = icmp ne i32 %313, 0
  %315 = zext i1 %314 to i32
  %316 = add nsw i32 %311, %315
  %317 = select i1 %314, i32 %313, i32 %288
  %318 = add nsw i32 %316, %290
  %319 = icmp sgt i32 %292, %318
  %320 = add i32 %318, %304
  %321 = select i1 %319, i32 1, i32 %320
  %322 = tail call i32 @llvm.smin.i32(i32 %293, i32 %318)
  %323 = add i32 %295, %317
  %324 = add i32 %323, 1
  %325 = icmp sgt i32 %321, %322
  br i1 %325, label %366, label %326

326:                                              ; preds = %305
  %327 = tail call i32 @llvm.smin.i32(i32 %299, i32 %323)
  %328 = icmp sgt i32 %297, %323
  %329 = sub i32 %324, %297
  %330 = select i1 %328, i32 1, i32 %329
  %331 = icmp sgt i32 %330, %327
  %332 = sext i32 %330 to i64
  %333 = sext i32 %327 to i64
  %334 = sext i32 %321 to i64
  %335 = sext i32 %322 to i64
  br label %336

336:                                              ; preds = %326, %362
  %337 = phi i64 [ %334, %326 ], [ %364, %362 ]
  %338 = phi float [ 0.000000e+00, %326 ], [ %363, %362 ]
  br i1 %331, label %362, label %339

339:                                              ; preds = %336
  %340 = add nsw i64 %337, -1
  %341 = mul nsw i64 %340, %300
  %342 = trunc nsw i64 %337 to i32
  %343 = sub i32 %318, %342
  %344 = mul nsw i32 %343, %297
  %345 = add i32 %344, %324
  %346 = getelementptr [4 x i8], ptr %126, i64 %341
  br label %347

347:                                              ; preds = %339, %347
  %348 = phi i64 [ %332, %339 ], [ %360, %347 ]
  %349 = phi float [ %338, %339 ], [ %359, %347 ]
  %350 = getelementptr [4 x i8], ptr %346, i64 %348
  %351 = getelementptr i8, ptr %350, i64 -4
  %352 = load float, ptr %351, align 4, !tbaa !39
  %353 = trunc nsw i64 %348 to i32
  %354 = sub i32 %345, %353
  %355 = sext i32 %354 to i64
  %356 = getelementptr [4 x i8], ptr %69, i64 %355
  %357 = getelementptr i8, ptr %356, i64 -4
  %358 = load float, ptr %357, align 4, !tbaa !39
  %359 = tail call float @llvm.fmuladd.f32(float %352, float %358, float %349)
  %360 = add nuw nsw i64 %348, 1
  %361 = icmp slt i64 %348, %333
  br i1 %361, label %347, label %362, !llvm.loop !50

362:                                              ; preds = %347, %336
  %363 = phi float [ %338, %336 ], [ %359, %347 ]
  %364 = add nuw nsw i64 %337, 1
  %365 = icmp slt i64 %337, %335
  br i1 %365, label %336, label %366, !llvm.loop !51

366:                                              ; preds = %362, %305
  %367 = phi float [ 0.000000e+00, %305 ], [ %363, %362 ]
  %368 = getelementptr inbounds [4 x i8], ptr %74, i64 %306
  store float %367, ptr %368, align 4, !tbaa !39
  %369 = add nsw i64 %306, 256
  %370 = icmp slt i64 %369, %303
  br i1 %370, label %305, label %371, !llvm.loop !52

371:                                              ; preds = %366, %283
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %372 = load i32, ptr %75, align 4, !tbaa !22
  %373 = icmp sgt i32 %372, %38
  br i1 %373, label %374, label %424

374:                                              ; preds = %371
  %375 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %376 = load i32, ptr %375, align 4, !tbaa !53
  %377 = getelementptr inbounds nuw i8, ptr %0, i64 132
  %378 = load i32, ptr %377, align 4, !tbaa !54
  %379 = getelementptr inbounds nuw i8, ptr %0, i64 92
  %380 = getelementptr inbounds nuw i8, ptr %0, i64 136
  %381 = getelementptr inbounds nuw i8, ptr %0, i64 96
  %382 = xor i32 %378, -1
  %383 = shl i64 %37, 32
  %384 = ashr exact i64 %383, 32
  %385 = sext i32 %372 to i64
  br label %386

386:                                              ; preds = %374, %419
  %387 = phi i64 [ %384, %374 ], [ %422, %419 ]
  %388 = trunc i64 %387 to i32
  %389 = add i32 %388, 1
  %390 = freeze i32 %389
  %391 = freeze i32 %376
  %392 = sdiv i32 %390, %391
  %393 = mul i32 %392, %391
  %394 = sub i32 %390, %393
  %395 = icmp eq i32 %394, 0
  %396 = sext i1 %395 to i32
  %397 = add nsw i32 %392, %396
  %398 = select i1 %395, i32 %376, i32 %394
  %399 = icmp sgt i32 %398, %378
  br i1 %399, label %400, label %419

400:                                              ; preds = %386
  %401 = load i32, ptr %379, align 4, !tbaa !43
  %402 = add nsw i32 %401, %378
  %403 = icmp sgt i32 %398, %402
  br i1 %403, label %419, label %404

404:                                              ; preds = %400
  %405 = load i32, ptr %380, align 4, !tbaa !122
  %406 = icmp slt i32 %397, %405
  br i1 %406, label %419, label %407

407:                                              ; preds = %404
  %408 = load i32, ptr %381, align 4, !tbaa !48
  %409 = add nsw i32 %408, %405
  %410 = icmp slt i32 %397, %409
  br i1 %410, label %411, label %419

411:                                              ; preds = %407
  %412 = add i32 %398, %382
  %413 = sub nsw i32 %397, %405
  %414 = mul nsw i32 %413, %401
  %415 = add nsw i32 %412, %414
  %416 = sext i32 %415 to i64
  %417 = getelementptr inbounds [4 x i8], ptr %69, i64 %416
  %418 = load float, ptr %417, align 4, !tbaa !39
  br label %419

419:                                              ; preds = %386, %400, %404, %407, %411
  %420 = phi float [ %418, %411 ], [ 0.000000e+00, %407 ], [ 0.000000e+00, %404 ], [ 0.000000e+00, %400 ], [ 0.000000e+00, %386 ]
  %421 = getelementptr inbounds [4 x i8], ptr %79, i64 %387
  store float %420, ptr %421, align 4, !tbaa !39
  %422 = add nsw i64 %387, 256
  %423 = icmp slt i64 %422, %385
  br i1 %423, label %386, label %424, !llvm.loop !55

424:                                              ; preds = %419, %371
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %425 = getelementptr inbounds nuw i8, ptr %0, i64 144
  %426 = load i32, ptr %425, align 4, !tbaa !56
  %427 = icmp sgt i32 %426, %38
  br i1 %427, label %428, label %457

428:                                              ; preds = %424
  %429 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %430 = load i32, ptr %429, align 4, !tbaa !53
  %431 = mul i32 %430, %38
  %432 = shl i32 %430, 8
  %433 = shl i64 %37, 32
  %434 = ashr exact i64 %433, 32
  %435 = sext i32 %430 to i64
  %436 = sext i32 %426 to i64
  %437 = icmp sgt i32 %430, 0
  br label %438

438:                                              ; preds = %428, %453
  %439 = phi i64 [ %434, %428 ], [ %454, %453 ]
  %440 = phi i32 [ %431, %428 ], [ %456, %453 ]
  %441 = add nsw i64 %439, 1
  %442 = mul i64 %441, %435
  br i1 %437, label %443, label %453

443:                                              ; preds = %438
  %444 = sext i32 %440 to i64
  br label %445

445:                                              ; preds = %443, %445
  %446 = phi i64 [ %444, %443 ], [ %451, %445 ]
  %447 = phi float [ 0.000000e+00, %443 ], [ %450, %445 ]
  %448 = getelementptr inbounds [4 x i8], ptr %79, i64 %446
  %449 = load float, ptr %448, align 4, !tbaa !39
  %450 = fadd float %447, %449
  store float %450, ptr %448, align 4, !tbaa !39
  %451 = add nsw i64 %446, 1
  %452 = icmp slt i64 %451, %442
  br i1 %452, label %445, label %453, !llvm.loop !57

453:                                              ; preds = %445, %438
  %454 = add nsw i64 %439, 256
  %455 = icmp slt i64 %454, %436
  %456 = add i32 %440, %432
  br i1 %455, label %438, label %457, !llvm.loop !58

457:                                              ; preds = %453, %424
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %458 = load i32, ptr %80, align 4, !tbaa !23
  %459 = icmp sgt i32 %458, %38
  br i1 %459, label %460, label %497

460:                                              ; preds = %457
  %461 = getelementptr inbounds nuw i8, ptr %0, i64 156
  %462 = load i32, ptr %461, align 4, !tbaa !59
  %463 = getelementptr inbounds nuw i8, ptr %0, i64 172
  %464 = load i32, ptr %463, align 4, !tbaa !60
  %465 = getelementptr inbounds nuw i8, ptr %0, i64 180
  %466 = load i32, ptr %465, align 4, !tbaa !61
  %467 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %468 = load i32, ptr %467, align 4, !tbaa !53
  %469 = add i32 %464, -2
  %470 = shl i64 %37, 32
  %471 = ashr exact i64 %470, 32
  %472 = sext i32 %458 to i64
  br label %473

473:                                              ; preds = %460, %473
  %474 = phi i64 [ %471, %460 ], [ %495, %473 ]
  %475 = trunc i64 %474 to i32
  %476 = add i32 %475, 1
  %477 = freeze i32 %476
  %478 = freeze i32 %462
  %479 = sdiv i32 %477, %478
  %480 = mul i32 %479, %478
  %481 = sub i32 %477, %480
  %482 = icmp eq i32 %481, 0
  %483 = sext i1 %482 to i32
  %484 = select i1 %482, i32 %462, i32 %481
  %485 = add i32 %479, -1
  %486 = add i32 %485, %483
  %487 = add i32 %486, %466
  %488 = mul nsw i32 %487, %468
  %489 = add i32 %484, %469
  %490 = add nsw i32 %489, %488
  %491 = sext i32 %490 to i64
  %492 = getelementptr inbounds [4 x i8], ptr %79, i64 %491
  %493 = load float, ptr %492, align 4, !tbaa !39
  %494 = getelementptr inbounds [4 x i8], ptr %84, i64 %474
  store float %493, ptr %494, align 4, !tbaa !39
  %495 = add nsw i64 %474, 256
  %496 = icmp slt i64 %495, %472
  br i1 %496, label %473, label %497, !llvm.loop !62

497:                                              ; preds = %473, %457
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %498 = load i32, ptr %85, align 4, !tbaa !24
  %499 = icmp sgt i32 %498, %38
  br i1 %499, label %500, label %537

500:                                              ; preds = %497
  %501 = getelementptr inbounds nuw i8, ptr %0, i64 204
  %502 = load i32, ptr %501, align 4, !tbaa !63
  %503 = getelementptr inbounds nuw i8, ptr %0, i64 188
  %504 = load i32, ptr %503, align 4, !tbaa !64
  %505 = getelementptr inbounds nuw i8, ptr %0, i64 196
  %506 = load i32, ptr %505, align 4, !tbaa !65
  %507 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %508 = load i32, ptr %507, align 4, !tbaa !53
  %509 = add i32 %504, -2
  %510 = shl i64 %37, 32
  %511 = ashr exact i64 %510, 32
  %512 = sext i32 %498 to i64
  br label %513

513:                                              ; preds = %500, %513
  %514 = phi i64 [ %511, %500 ], [ %535, %513 ]
  %515 = trunc i64 %514 to i32
  %516 = add i32 %515, 1
  %517 = freeze i32 %516
  %518 = freeze i32 %502
  %519 = sdiv i32 %517, %518
  %520 = mul i32 %519, %518
  %521 = sub i32 %517, %520
  %522 = icmp eq i32 %521, 0
  %523 = sext i1 %522 to i32
  %524 = select i1 %522, i32 %502, i32 %521
  %525 = add i32 %519, -1
  %526 = add i32 %525, %523
  %527 = add i32 %526, %506
  %528 = mul nsw i32 %527, %508
  %529 = add i32 %524, %509
  %530 = add nsw i32 %529, %528
  %531 = sext i32 %530 to i64
  %532 = getelementptr inbounds [4 x i8], ptr %79, i64 %531
  %533 = load float, ptr %532, align 4, !tbaa !39
  %534 = getelementptr inbounds [4 x i8], ptr %89, i64 %514
  store float %533, ptr %534, align 4, !tbaa !39
  %535 = add nsw i64 %514, 256
  %536 = icmp slt i64 %535, %512
  br i1 %536, label %513, label %537, !llvm.loop !66

537:                                              ; preds = %513, %497
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %538 = load i32, ptr %85, align 4, !tbaa !24
  %539 = icmp sgt i32 %538, %38
  br i1 %539, label %540, label %553

540:                                              ; preds = %537
  %541 = shl i64 %37, 32
  %542 = ashr exact i64 %541, 32
  %543 = sext i32 %538 to i64
  br label %544

544:                                              ; preds = %540, %544
  %545 = phi i64 [ %542, %540 ], [ %551, %544 ]
  %546 = getelementptr inbounds [4 x i8], ptr %84, i64 %545
  %547 = load float, ptr %546, align 4, !tbaa !39
  %548 = getelementptr inbounds [4 x i8], ptr %89, i64 %545
  %549 = load float, ptr %548, align 4, !tbaa !39
  %550 = fsub float %547, %549
  store float %550, ptr %548, align 4, !tbaa !39
  %551 = add nsw i64 %545, 256
  %552 = icmp slt i64 %551, %543
  br i1 %552, label %544, label %553, !llvm.loop !67

553:                                              ; preds = %544, %537
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %554 = getelementptr inbounds nuw i8, ptr %0, i64 204
  %555 = load i32, ptr %554, align 4, !tbaa !63
  %556 = icmp sgt i32 %555, %38
  br i1 %556, label %557, label %578

557:                                              ; preds = %553
  %558 = load i32, ptr %85, align 4, !tbaa !24
  %559 = shl i64 %37, 32
  %560 = ashr exact i64 %559, 32
  %561 = sext i32 %555 to i64
  %562 = sext i32 %558 to i64
  %563 = icmp sgt i32 %558, 0
  br label %564

564:                                              ; preds = %557, %575
  %565 = phi i64 [ %560, %557 ], [ %576, %575 ]
  %566 = add nsw i64 %565, %562
  br i1 %563, label %567, label %575

567:                                              ; preds = %564, %567
  %568 = phi i64 [ %573, %567 ], [ %565, %564 ]
  %569 = phi float [ %572, %567 ], [ 0.000000e+00, %564 ]
  %570 = getelementptr inbounds [4 x i8], ptr %89, i64 %568
  %571 = load float, ptr %570, align 4, !tbaa !39
  %572 = fadd float %569, %571
  store float %572, ptr %570, align 4, !tbaa !39
  %573 = add nsw i64 %568, %561
  %574 = icmp slt i64 %573, %566
  br i1 %574, label %567, label %575, !llvm.loop !68

575:                                              ; preds = %567, %564
  %576 = add nsw i64 %565, 256
  %577 = icmp slt i64 %576, %561
  br i1 %577, label %564, label %578, !llvm.loop !69

578:                                              ; preds = %575, %553
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %579 = load i32, ptr %90, align 4, !tbaa !25
  %580 = icmp sgt i32 %579, %38
  br i1 %580, label %581, label %617

581:                                              ; preds = %578
  %582 = getelementptr inbounds nuw i8, ptr %0, i64 220
  %583 = load i32, ptr %582, align 4, !tbaa !70
  %584 = getelementptr inbounds nuw i8, ptr %0, i64 236
  %585 = load i32, ptr %584, align 4, !tbaa !71
  %586 = getelementptr inbounds nuw i8, ptr %0, i64 244
  %587 = load i32, ptr %586, align 4, !tbaa !72
  %588 = load i32, ptr %554, align 4, !tbaa !63
  %589 = add i32 %585, -2
  %590 = shl i64 %37, 32
  %591 = ashr exact i64 %590, 32
  %592 = sext i32 %579 to i64
  br label %593

593:                                              ; preds = %581, %593
  %594 = phi i64 [ %591, %581 ], [ %615, %593 ]
  %595 = trunc i64 %594 to i32
  %596 = add i32 %595, 1
  %597 = freeze i32 %596
  %598 = freeze i32 %583
  %599 = sdiv i32 %597, %598
  %600 = mul i32 %599, %598
  %601 = sub i32 %597, %600
  %602 = icmp eq i32 %601, 0
  %603 = sext i1 %602 to i32
  %604 = select i1 %602, i32 %583, i32 %601
  %605 = add i32 %599, -1
  %606 = add i32 %605, %603
  %607 = add i32 %606, %587
  %608 = mul nsw i32 %607, %588
  %609 = add i32 %604, %589
  %610 = add nsw i32 %609, %608
  %611 = sext i32 %610 to i64
  %612 = getelementptr inbounds [4 x i8], ptr %89, i64 %611
  %613 = load float, ptr %612, align 4, !tbaa !39
  %614 = getelementptr inbounds [4 x i8], ptr %94, i64 %594
  store float %613, ptr %614, align 4, !tbaa !39
  %615 = add nsw i64 %594, 256
  %616 = icmp slt i64 %615, %592
  br i1 %616, label %593, label %617, !llvm.loop !73

617:                                              ; preds = %593, %578
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %618 = load i32, ptr %95, align 4, !tbaa !26
  %619 = icmp sgt i32 %618, %38
  br i1 %619, label %620, label %656

620:                                              ; preds = %617
  %621 = getelementptr inbounds nuw i8, ptr %0, i64 268
  %622 = load i32, ptr %621, align 4, !tbaa !74
  %623 = getelementptr inbounds nuw i8, ptr %0, i64 252
  %624 = load i32, ptr %623, align 4, !tbaa !75
  %625 = getelementptr inbounds nuw i8, ptr %0, i64 260
  %626 = load i32, ptr %625, align 4, !tbaa !76
  %627 = load i32, ptr %554, align 4, !tbaa !63
  %628 = add i32 %624, -2
  %629 = shl i64 %37, 32
  %630 = ashr exact i64 %629, 32
  %631 = sext i32 %618 to i64
  br label %632

632:                                              ; preds = %620, %632
  %633 = phi i64 [ %630, %620 ], [ %654, %632 ]
  %634 = trunc i64 %633 to i32
  %635 = add i32 %634, 1
  %636 = freeze i32 %635
  %637 = freeze i32 %622
  %638 = sdiv i32 %636, %637
  %639 = mul i32 %638, %637
  %640 = sub i32 %636, %639
  %641 = icmp eq i32 %640, 0
  %642 = sext i1 %641 to i32
  %643 = select i1 %641, i32 %622, i32 %640
  %644 = add i32 %638, -1
  %645 = add i32 %644, %642
  %646 = add i32 %645, %626
  %647 = mul nsw i32 %646, %627
  %648 = add i32 %643, %628
  %649 = add nsw i32 %648, %647
  %650 = sext i32 %649 to i64
  %651 = getelementptr inbounds [4 x i8], ptr %89, i64 %650
  %652 = load float, ptr %651, align 4, !tbaa !39
  %653 = getelementptr inbounds [4 x i8], ptr %99, i64 %633
  store float %652, ptr %653, align 4, !tbaa !39
  %654 = add nsw i64 %633, 256
  %655 = icmp slt i64 %654, %631
  br i1 %655, label %632, label %656, !llvm.loop !77

656:                                              ; preds = %632, %617
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %657 = load i32, ptr %95, align 4, !tbaa !26
  %658 = icmp sgt i32 %657, %38
  br i1 %658, label %659, label %672

659:                                              ; preds = %656
  %660 = shl i64 %37, 32
  %661 = ashr exact i64 %660, 32
  %662 = sext i32 %657 to i64
  br label %663

663:                                              ; preds = %659, %663
  %664 = phi i64 [ %661, %659 ], [ %670, %663 ]
  %665 = getelementptr inbounds [4 x i8], ptr %94, i64 %664
  %666 = load float, ptr %665, align 4, !tbaa !39
  %667 = getelementptr inbounds [4 x i8], ptr %99, i64 %664
  %668 = load float, ptr %667, align 4, !tbaa !39
  %669 = fsub float %666, %668
  store float %669, ptr %667, align 4, !tbaa !39
  %670 = add nsw i64 %664, 256
  %671 = icmp slt i64 %670, %662
  br i1 %671, label %663, label %672, !llvm.loop !78

672:                                              ; preds = %663, %656
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %673 = load i32, ptr %100, align 4, !tbaa !27
  %674 = icmp sgt i32 %673, %38
  br i1 %674, label %675, label %687

675:                                              ; preds = %672
  %676 = shl i64 %37, 32
  %677 = ashr exact i64 %676, 32
  %678 = sext i32 %673 to i64
  br label %679

679:                                              ; preds = %675, %679
  %680 = phi i64 [ %677, %675 ], [ %685, %679 ]
  %681 = getelementptr inbounds [4 x i8], ptr %69, i64 %680
  %682 = load float, ptr %681, align 4, !tbaa !39
  %683 = fmul float %682, %682
  %684 = getelementptr inbounds [4 x i8], ptr %104, i64 %680
  store float %683, ptr %684, align 4, !tbaa !39
  %685 = add nsw i64 %680, 256
  %686 = icmp slt i64 %685, %678
  br i1 %686, label %679, label %687, !llvm.loop !79

687:                                              ; preds = %679, %672
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %688 = load i32, ptr %75, align 4, !tbaa !22
  %689 = icmp sgt i32 %688, %38
  br i1 %689, label %690, label %740

690:                                              ; preds = %687
  %691 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %692 = load i32, ptr %691, align 4, !tbaa !53
  %693 = getelementptr inbounds nuw i8, ptr %0, i64 132
  %694 = load i32, ptr %693, align 4, !tbaa !54
  %695 = getelementptr inbounds nuw i8, ptr %0, i64 284
  %696 = getelementptr inbounds nuw i8, ptr %0, i64 136
  %697 = getelementptr inbounds nuw i8, ptr %0, i64 288
  %698 = xor i32 %694, -1
  %699 = shl i64 %37, 32
  %700 = ashr exact i64 %699, 32
  %701 = sext i32 %688 to i64
  br label %702

702:                                              ; preds = %690, %735
  %703 = phi i64 [ %700, %690 ], [ %738, %735 ]
  %704 = trunc i64 %703 to i32
  %705 = add i32 %704, 1
  %706 = freeze i32 %705
  %707 = freeze i32 %692
  %708 = sdiv i32 %706, %707
  %709 = mul i32 %708, %707
  %710 = sub i32 %706, %709
  %711 = icmp eq i32 %710, 0
  %712 = sext i1 %711 to i32
  %713 = add nsw i32 %708, %712
  %714 = select i1 %711, i32 %692, i32 %710
  %715 = icmp sgt i32 %714, %694
  br i1 %715, label %716, label %735

716:                                              ; preds = %702
  %717 = load i32, ptr %695, align 4, !tbaa !123
  %718 = add nsw i32 %717, %694
  %719 = icmp sgt i32 %714, %718
  br i1 %719, label %735, label %720

720:                                              ; preds = %716
  %721 = load i32, ptr %696, align 4, !tbaa !122
  %722 = icmp slt i32 %713, %721
  br i1 %722, label %735, label %723

723:                                              ; preds = %720
  %724 = load i32, ptr %697, align 4, !tbaa !124
  %725 = add nsw i32 %724, %721
  %726 = icmp slt i32 %713, %725
  br i1 %726, label %727, label %735

727:                                              ; preds = %723
  %728 = add i32 %714, %698
  %729 = sub nsw i32 %713, %721
  %730 = mul nsw i32 %729, %717
  %731 = add nsw i32 %728, %730
  %732 = sext i32 %731 to i64
  %733 = getelementptr inbounds [4 x i8], ptr %104, i64 %732
  %734 = load float, ptr %733, align 4, !tbaa !39
  br label %735

735:                                              ; preds = %702, %716, %720, %723, %727
  %736 = phi float [ %734, %727 ], [ 0.000000e+00, %723 ], [ 0.000000e+00, %720 ], [ 0.000000e+00, %716 ], [ 0.000000e+00, %702 ]
  %737 = getelementptr inbounds [4 x i8], ptr %79, i64 %703
  store float %736, ptr %737, align 4, !tbaa !39
  %738 = add nsw i64 %703, 256
  %739 = icmp slt i64 %738, %701
  br i1 %739, label %702, label %740, !llvm.loop !80

740:                                              ; preds = %735, %687
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %741 = load i32, ptr %425, align 4, !tbaa !56
  %742 = icmp sgt i32 %741, %38
  br i1 %742, label %743, label %772

743:                                              ; preds = %740
  %744 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %745 = load i32, ptr %744, align 4, !tbaa !53
  %746 = mul i32 %745, %38
  %747 = shl i32 %745, 8
  %748 = shl i64 %37, 32
  %749 = ashr exact i64 %748, 32
  %750 = sext i32 %745 to i64
  %751 = sext i32 %741 to i64
  %752 = icmp sgt i32 %745, 0
  br label %753

753:                                              ; preds = %743, %768
  %754 = phi i64 [ %749, %743 ], [ %769, %768 ]
  %755 = phi i32 [ %746, %743 ], [ %771, %768 ]
  %756 = add nsw i64 %754, 1
  %757 = mul i64 %756, %750
  br i1 %752, label %758, label %768

758:                                              ; preds = %753
  %759 = sext i32 %755 to i64
  br label %760

760:                                              ; preds = %758, %760
  %761 = phi i64 [ %759, %758 ], [ %766, %760 ]
  %762 = phi float [ 0.000000e+00, %758 ], [ %765, %760 ]
  %763 = getelementptr inbounds [4 x i8], ptr %79, i64 %761
  %764 = load float, ptr %763, align 4, !tbaa !39
  %765 = fadd float %762, %764
  store float %765, ptr %763, align 4, !tbaa !39
  %766 = add nsw i64 %761, 1
  %767 = icmp slt i64 %766, %757
  br i1 %767, label %760, label %768, !llvm.loop !81

768:                                              ; preds = %760, %753
  %769 = add nsw i64 %754, 256
  %770 = icmp slt i64 %769, %751
  %771 = add i32 %755, %747
  br i1 %770, label %753, label %772, !llvm.loop !82

772:                                              ; preds = %768, %740
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %773 = load i32, ptr %80, align 4, !tbaa !23
  %774 = icmp sgt i32 %773, %38
  br i1 %774, label %775, label %812

775:                                              ; preds = %772
  %776 = getelementptr inbounds nuw i8, ptr %0, i64 156
  %777 = load i32, ptr %776, align 4, !tbaa !59
  %778 = getelementptr inbounds nuw i8, ptr %0, i64 172
  %779 = load i32, ptr %778, align 4, !tbaa !60
  %780 = getelementptr inbounds nuw i8, ptr %0, i64 180
  %781 = load i32, ptr %780, align 4, !tbaa !61
  %782 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %783 = load i32, ptr %782, align 4, !tbaa !53
  %784 = add i32 %779, -2
  %785 = shl i64 %37, 32
  %786 = ashr exact i64 %785, 32
  %787 = sext i32 %773 to i64
  br label %788

788:                                              ; preds = %775, %788
  %789 = phi i64 [ %786, %775 ], [ %810, %788 ]
  %790 = trunc i64 %789 to i32
  %791 = add i32 %790, 1
  %792 = freeze i32 %791
  %793 = freeze i32 %777
  %794 = sdiv i32 %792, %793
  %795 = mul i32 %794, %793
  %796 = sub i32 %792, %795
  %797 = icmp eq i32 %796, 0
  %798 = sext i1 %797 to i32
  %799 = select i1 %797, i32 %777, i32 %796
  %800 = add i32 %794, -1
  %801 = add i32 %800, %798
  %802 = add i32 %801, %781
  %803 = mul nsw i32 %802, %783
  %804 = add i32 %799, %784
  %805 = add nsw i32 %804, %803
  %806 = sext i32 %805 to i64
  %807 = getelementptr inbounds [4 x i8], ptr %79, i64 %806
  %808 = load float, ptr %807, align 4, !tbaa !39
  %809 = getelementptr inbounds [4 x i8], ptr %84, i64 %789
  store float %808, ptr %809, align 4, !tbaa !39
  %810 = add nsw i64 %789, 256
  %811 = icmp slt i64 %810, %787
  br i1 %811, label %788, label %812, !llvm.loop !83

812:                                              ; preds = %788, %772
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %813 = load i32, ptr %85, align 4, !tbaa !24
  %814 = icmp sgt i32 %813, %38
  br i1 %814, label %815, label %851

815:                                              ; preds = %812
  %816 = load i32, ptr %554, align 4, !tbaa !63
  %817 = getelementptr inbounds nuw i8, ptr %0, i64 188
  %818 = load i32, ptr %817, align 4, !tbaa !64
  %819 = getelementptr inbounds nuw i8, ptr %0, i64 196
  %820 = load i32, ptr %819, align 4, !tbaa !65
  %821 = getelementptr inbounds nuw i8, ptr %0, i64 140
  %822 = load i32, ptr %821, align 4, !tbaa !53
  %823 = add i32 %818, -2
  %824 = shl i64 %37, 32
  %825 = ashr exact i64 %824, 32
  %826 = sext i32 %813 to i64
  br label %827

827:                                              ; preds = %815, %827
  %828 = phi i64 [ %825, %815 ], [ %849, %827 ]
  %829 = trunc i64 %828 to i32
  %830 = add i32 %829, 1
  %831 = freeze i32 %830
  %832 = freeze i32 %816
  %833 = sdiv i32 %831, %832
  %834 = mul i32 %833, %832
  %835 = sub i32 %831, %834
  %836 = icmp eq i32 %835, 0
  %837 = sext i1 %836 to i32
  %838 = select i1 %836, i32 %816, i32 %835
  %839 = add i32 %833, -1
  %840 = add i32 %839, %837
  %841 = add i32 %840, %820
  %842 = mul nsw i32 %841, %822
  %843 = add i32 %838, %823
  %844 = add nsw i32 %843, %842
  %845 = sext i32 %844 to i64
  %846 = getelementptr inbounds [4 x i8], ptr %79, i64 %845
  %847 = load float, ptr %846, align 4, !tbaa !39
  %848 = getelementptr inbounds [4 x i8], ptr %89, i64 %828
  store float %847, ptr %848, align 4, !tbaa !39
  %849 = add nsw i64 %828, 256
  %850 = icmp slt i64 %849, %826
  br i1 %850, label %827, label %851, !llvm.loop !84

851:                                              ; preds = %827, %812
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %852 = load i32, ptr %85, align 4, !tbaa !24
  %853 = icmp sgt i32 %852, %38
  br i1 %853, label %854, label %867

854:                                              ; preds = %851
  %855 = shl i64 %37, 32
  %856 = ashr exact i64 %855, 32
  %857 = sext i32 %852 to i64
  br label %858

858:                                              ; preds = %854, %858
  %859 = phi i64 [ %856, %854 ], [ %865, %858 ]
  %860 = getelementptr inbounds [4 x i8], ptr %84, i64 %859
  %861 = load float, ptr %860, align 4, !tbaa !39
  %862 = getelementptr inbounds [4 x i8], ptr %89, i64 %859
  %863 = load float, ptr %862, align 4, !tbaa !39
  %864 = fsub float %861, %863
  store float %864, ptr %862, align 4, !tbaa !39
  %865 = add nsw i64 %859, 256
  %866 = icmp slt i64 %865, %857
  br i1 %866, label %858, label %867, !llvm.loop !85

867:                                              ; preds = %858, %851
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %868 = load i32, ptr %554, align 4, !tbaa !63
  %869 = icmp sgt i32 %868, %38
  br i1 %869, label %870, label %891

870:                                              ; preds = %867
  %871 = load i32, ptr %85, align 4, !tbaa !24
  %872 = shl i64 %37, 32
  %873 = ashr exact i64 %872, 32
  %874 = sext i32 %868 to i64
  %875 = sext i32 %871 to i64
  %876 = icmp sgt i32 %871, 0
  br label %877

877:                                              ; preds = %870, %888
  %878 = phi i64 [ %873, %870 ], [ %889, %888 ]
  %879 = add nsw i64 %878, %875
  br i1 %876, label %880, label %888

880:                                              ; preds = %877, %880
  %881 = phi i64 [ %886, %880 ], [ %878, %877 ]
  %882 = phi float [ %885, %880 ], [ 0.000000e+00, %877 ]
  %883 = getelementptr inbounds [4 x i8], ptr %89, i64 %881
  %884 = load float, ptr %883, align 4, !tbaa !39
  %885 = fadd float %882, %884
  store float %885, ptr %883, align 4, !tbaa !39
  %886 = add nsw i64 %881, %874
  %887 = icmp slt i64 %886, %879
  br i1 %887, label %880, label %888, !llvm.loop !86

888:                                              ; preds = %880, %877
  %889 = add nsw i64 %878, 256
  %890 = icmp slt i64 %889, %874
  br i1 %890, label %877, label %891, !llvm.loop !87

891:                                              ; preds = %888, %867
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %892 = load i32, ptr %90, align 4, !tbaa !25
  %893 = icmp sgt i32 %892, %38
  br i1 %893, label %894, label %930

894:                                              ; preds = %891
  %895 = getelementptr inbounds nuw i8, ptr %0, i64 220
  %896 = load i32, ptr %895, align 4, !tbaa !70
  %897 = getelementptr inbounds nuw i8, ptr %0, i64 236
  %898 = load i32, ptr %897, align 4, !tbaa !71
  %899 = getelementptr inbounds nuw i8, ptr %0, i64 244
  %900 = load i32, ptr %899, align 4, !tbaa !72
  %901 = load i32, ptr %554, align 4, !tbaa !63
  %902 = add i32 %898, -2
  %903 = shl i64 %37, 32
  %904 = ashr exact i64 %903, 32
  %905 = sext i32 %892 to i64
  br label %906

906:                                              ; preds = %894, %906
  %907 = phi i64 [ %904, %894 ], [ %928, %906 ]
  %908 = trunc i64 %907 to i32
  %909 = add i32 %908, 1
  %910 = freeze i32 %909
  %911 = freeze i32 %896
  %912 = sdiv i32 %910, %911
  %913 = mul i32 %912, %911
  %914 = sub i32 %910, %913
  %915 = icmp eq i32 %914, 0
  %916 = sext i1 %915 to i32
  %917 = select i1 %915, i32 %896, i32 %914
  %918 = add i32 %912, -1
  %919 = add i32 %918, %916
  %920 = add i32 %919, %900
  %921 = mul nsw i32 %920, %901
  %922 = add i32 %917, %902
  %923 = add nsw i32 %922, %921
  %924 = sext i32 %923 to i64
  %925 = getelementptr inbounds [4 x i8], ptr %89, i64 %924
  %926 = load float, ptr %925, align 4, !tbaa !39
  %927 = getelementptr inbounds [4 x i8], ptr %94, i64 %907
  store float %926, ptr %927, align 4, !tbaa !39
  %928 = add nsw i64 %907, 256
  %929 = icmp slt i64 %928, %905
  br i1 %929, label %906, label %930, !llvm.loop !88

930:                                              ; preds = %906, %891
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %931 = load i32, ptr %95, align 4, !tbaa !26
  %932 = icmp sgt i32 %931, %38
  br i1 %932, label %933, label %969

933:                                              ; preds = %930
  %934 = getelementptr inbounds nuw i8, ptr %0, i64 268
  %935 = load i32, ptr %934, align 4, !tbaa !74
  %936 = getelementptr inbounds nuw i8, ptr %0, i64 252
  %937 = load i32, ptr %936, align 4, !tbaa !75
  %938 = getelementptr inbounds nuw i8, ptr %0, i64 260
  %939 = load i32, ptr %938, align 4, !tbaa !76
  %940 = load i32, ptr %554, align 4, !tbaa !63
  %941 = add i32 %937, -2
  %942 = shl i64 %37, 32
  %943 = ashr exact i64 %942, 32
  %944 = sext i32 %931 to i64
  br label %945

945:                                              ; preds = %933, %945
  %946 = phi i64 [ %943, %933 ], [ %967, %945 ]
  %947 = trunc i64 %946 to i32
  %948 = add i32 %947, 1
  %949 = freeze i32 %948
  %950 = freeze i32 %935
  %951 = sdiv i32 %949, %950
  %952 = mul i32 %951, %950
  %953 = sub i32 %949, %952
  %954 = icmp eq i32 %953, 0
  %955 = sext i1 %954 to i32
  %956 = select i1 %954, i32 %935, i32 %953
  %957 = add i32 %951, -1
  %958 = add i32 %957, %955
  %959 = add i32 %958, %939
  %960 = mul nsw i32 %959, %940
  %961 = add i32 %956, %941
  %962 = add nsw i32 %961, %960
  %963 = sext i32 %962 to i64
  %964 = getelementptr inbounds [4 x i8], ptr %89, i64 %963
  %965 = load float, ptr %964, align 4, !tbaa !39
  %966 = getelementptr inbounds [4 x i8], ptr %109, i64 %946
  store float %965, ptr %966, align 4, !tbaa !39
  %967 = add nsw i64 %946, 256
  %968 = icmp slt i64 %967, %944
  br i1 %968, label %945, label %969, !llvm.loop !89

969:                                              ; preds = %945, %930
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %970 = load i32, ptr %95, align 4, !tbaa !26
  %971 = icmp sgt i32 %970, %38
  br i1 %971, label %972, label %985

972:                                              ; preds = %969
  %973 = shl i64 %37, 32
  %974 = ashr exact i64 %973, 32
  %975 = sext i32 %970 to i64
  br label %976

976:                                              ; preds = %972, %976
  %977 = phi i64 [ %974, %972 ], [ %983, %976 ]
  %978 = getelementptr inbounds [4 x i8], ptr %94, i64 %977
  %979 = load float, ptr %978, align 4, !tbaa !39
  %980 = getelementptr inbounds [4 x i8], ptr %109, i64 %977
  %981 = load float, ptr %980, align 4, !tbaa !39
  %982 = fsub float %979, %981
  store float %982, ptr %980, align 4, !tbaa !39
  %983 = add nsw i64 %977, 256
  %984 = icmp slt i64 %983, %975
  br i1 %984, label %976, label %985, !llvm.loop !90

985:                                              ; preds = %976, %969
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %986 = load i32, ptr %95, align 4, !tbaa !26
  %987 = icmp sgt i32 %986, %38
  br i1 %987, label %988, label %1008

988:                                              ; preds = %985
  %989 = load i32, ptr %125, align 4, !tbaa !19
  %990 = sitofp i32 %989 to float
  %991 = shl i64 %37, 32
  %992 = ashr exact i64 %991, 32
  %993 = sext i32 %986 to i64
  br label %994

994:                                              ; preds = %988, %994
  %995 = phi i64 [ %992, %988 ], [ %1006, %994 ]
  %996 = getelementptr inbounds [4 x i8], ptr %99, i64 %995
  %997 = load float, ptr %996, align 4, !tbaa !39
  %998 = getelementptr inbounds [4 x i8], ptr %109, i64 %995
  %999 = load float, ptr %998, align 4, !tbaa !39
  %1000 = fmul float %997, %997
  %1001 = fdiv float %1000, %990, !fpmath !91
  %1002 = fsub float %999, %1001
  %1003 = fcmp olt float %1002, 0.000000e+00
  %1004 = select i1 %1003, float 0.000000e+00, float %1002
  %1005 = tail call float @_Z4sqrtf(float noundef %1004) #6, !fpmath !92
  store float %1005, ptr %998, align 4, !tbaa !39
  %1006 = add nsw i64 %995, 256
  %1007 = icmp slt i64 %1006, %993
  br i1 %1007, label %994, label %1008, !llvm.loop !93

1008:                                             ; preds = %994, %985
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1009 = load i32, ptr %110, align 4, !tbaa !29
  %1010 = icmp sgt i32 %1009, %38
  br i1 %1010, label %1011, label %1023

1011:                                             ; preds = %1008
  %1012 = shl i64 %37, 32
  %1013 = ashr exact i64 %1012, 32
  %1014 = sext i32 %1009 to i64
  br label %1015

1015:                                             ; preds = %1011, %1015
  %1016 = phi i64 [ %1013, %1011 ], [ %1021, %1015 ]
  %1017 = getelementptr inbounds [4 x i8], ptr %64, i64 %1016
  %1018 = load float, ptr %1017, align 4, !tbaa !39
  %1019 = fmul float %1018, %1018
  %1020 = getelementptr inbounds [4 x i8], ptr %114, i64 %1016
  store float %1019, ptr %1020, align 4, !tbaa !39
  %1021 = add nsw i64 %1016, 256
  %1022 = icmp slt i64 %1021, %1014
  br i1 %1022, label %1015, label %1023, !llvm.loop !94

1023:                                             ; preds = %1015, %1008
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1024 = load i32, ptr %127, align 4, !tbaa !32
  %1025 = icmp sgt i32 %1024, %38
  br i1 %1025, label %1026, label %1053

1026:                                             ; preds = %1023
  %1027 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %1028 = load i32, ptr %1027, align 4, !tbaa !37
  %1029 = icmp sgt i32 %1028, 0
  %1030 = shl i64 %37, 32
  %1031 = ashr exact i64 %1030, 32
  %1032 = sext i32 %1028 to i64
  %1033 = sext i32 %1024 to i64
  %1034 = zext nneg i32 %1028 to i64
  br label %1035

1035:                                             ; preds = %1026, %1048
  %1036 = phi i64 [ %1031, %1026 ], [ %1051, %1048 ]
  br i1 %1029, label %1037, label %1048

1037:                                             ; preds = %1035
  %1038 = mul nsw i64 %1036, %1032
  %1039 = getelementptr [4 x i8], ptr %64, i64 %1038
  br label %1040

1040:                                             ; preds = %1037, %1040
  %1041 = phi i64 [ 0, %1037 ], [ %1046, %1040 ]
  %1042 = phi float [ 0.000000e+00, %1037 ], [ %1045, %1040 ]
  %1043 = getelementptr [4 x i8], ptr %1039, i64 %1041
  %1044 = load float, ptr %1043, align 4, !tbaa !39
  %1045 = fadd float %1042, %1044
  %1046 = add nuw nsw i64 %1041, 1
  %1047 = icmp eq i64 %1046, %1034
  br i1 %1047, label %1048, label %1040, !llvm.loop !95

1048:                                             ; preds = %1040, %1035
  %1049 = phi float [ 0.000000e+00, %1035 ], [ %1045, %1040 ]
  %1050 = getelementptr inbounds [4 x i8], ptr %131, i64 %1036
  store float %1049, ptr %1050, align 4, !tbaa !39
  %1051 = add nsw i64 %1036, 256
  %1052 = icmp slt i64 %1051, %1033
  br i1 %1052, label %1035, label %1053, !llvm.loop !96

1053:                                             ; preds = %1048, %1023
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1054 = load i32, ptr %132, align 4, !tbaa !33
  %1055 = icmp sgt i32 %1054, %38
  br i1 %1055, label %1056, label %1082

1056:                                             ; preds = %1053
  %1057 = getelementptr inbounds nuw i8, ptr %0, i64 320
  %1058 = load i32, ptr %1057, align 4, !tbaa !97
  %1059 = icmp sgt i32 %1058, 0
  %1060 = sext i32 %1054 to i64
  %1061 = shl i64 %37, 32
  %1062 = ashr exact i64 %1061, 32
  %1063 = zext nneg i32 %1058 to i64
  br label %1064

1064:                                             ; preds = %1056, %1077
  %1065 = phi i64 [ %1062, %1056 ], [ %1080, %1077 ]
  br i1 %1059, label %1066, label %1077

1066:                                             ; preds = %1064
  %1067 = getelementptr [4 x i8], ptr %114, i64 %1065
  br label %1068

1068:                                             ; preds = %1066, %1068
  %1069 = phi i64 [ 0, %1066 ], [ %1075, %1068 ]
  %1070 = phi float [ 0.000000e+00, %1066 ], [ %1074, %1068 ]
  %1071 = mul nsw i64 %1069, %1060
  %1072 = getelementptr [4 x i8], ptr %1067, i64 %1071
  %1073 = load float, ptr %1072, align 4, !tbaa !39
  %1074 = fadd float %1070, %1073
  %1075 = add nuw nsw i64 %1069, 1
  %1076 = icmp eq i64 %1075, %1063
  br i1 %1076, label %1077, label %1068, !llvm.loop !98

1077:                                             ; preds = %1068, %1064
  %1078 = phi float [ 0.000000e+00, %1064 ], [ %1074, %1068 ]
  %1079 = getelementptr inbounds [4 x i8], ptr %136, i64 %1065
  store float %1078, ptr %1079, align 4, !tbaa !39
  %1080 = add nsw i64 %1065, 256
  %1081 = icmp slt i64 %1080, %1060
  br i1 %1081, label %1064, label %1082, !llvm.loop !99

1082:                                             ; preds = %1077, %1053
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1083 = icmp eq i32 %38, 0
  br i1 %1083, label %1084, label %1097

1084:                                             ; preds = %1082
  store float 0.000000e+00, ptr %145, align 4, !tbaa !39
  %1085 = load i32, ptr %127, align 4, !tbaa !32
  %1086 = icmp sgt i32 %1085, 0
  br i1 %1086, label %1087, label %1113

1087:                                             ; preds = %1084
  %1088 = zext nneg i32 %1085 to i64
  br label %1089

1089:                                             ; preds = %1087, %1089
  %1090 = phi i64 [ 0, %1087 ], [ %1095, %1089 ]
  %1091 = phi float [ 0.000000e+00, %1087 ], [ %1094, %1089 ]
  %1092 = getelementptr inbounds nuw [4 x i8], ptr %131, i64 %1090
  %1093 = load float, ptr %1092, align 4, !tbaa !39
  %1094 = fadd float %1091, %1093
  store float %1094, ptr %145, align 4, !tbaa !39
  %1095 = add nuw nsw i64 %1090, 1
  %1096 = icmp eq i64 %1095, %1088
  br i1 %1096, label %1113, label %1089, !llvm.loop !100

1097:                                             ; preds = %1082
  %1098 = icmp eq i32 %38, 1
  br i1 %1098, label %1099, label %1113

1099:                                             ; preds = %1097
  store float 0.000000e+00, ptr %146, align 4, !tbaa !39
  %1100 = getelementptr inbounds nuw i8, ptr %0, i64 320
  %1101 = load i32, ptr %1100, align 4, !tbaa !97
  %1102 = icmp sgt i32 %1101, 0
  br i1 %1102, label %1103, label %1113

1103:                                             ; preds = %1099
  %1104 = zext nneg i32 %1101 to i64
  br label %1105

1105:                                             ; preds = %1103, %1105
  %1106 = phi i64 [ 0, %1103 ], [ %1111, %1105 ]
  %1107 = phi float [ 0.000000e+00, %1103 ], [ %1110, %1105 ]
  %1108 = getelementptr inbounds nuw [4 x i8], ptr %136, i64 %1106
  %1109 = load float, ptr %1108, align 4, !tbaa !39
  %1110 = fadd float %1107, %1109
  store float %1110, ptr %146, align 4, !tbaa !39
  %1111 = add nuw nsw i64 %1106, 1
  %1112 = icmp eq i64 %1111, %1104
  br i1 %1112, label %1113, label %1105, !llvm.loop !101

1113:                                             ; preds = %1105, %1089, %1099, %1084, %1097
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  br i1 %1083, label %1114, label %1128

1114:                                             ; preds = %1113
  %1115 = load float, ptr %145, align 4, !tbaa !39
  %1116 = load i32, ptr %125, align 4, !tbaa !19
  %1117 = sitofp i32 %1116 to float
  %1118 = fdiv float %1115, %1117, !fpmath !91
  %1119 = fmul float %1118, %1118
  %1120 = load float, ptr %146, align 4, !tbaa !39
  %1121 = fdiv float %1120, %1117, !fpmath !91
  %1122 = fsub float %1121, %1119
  %1123 = tail call float @_Z4sqrtf(float noundef %1122) #6, !fpmath !92
  %1124 = add nsw i32 %1116, -1
  %1125 = sitofp i32 %1124 to float
  %1126 = tail call float @_Z4sqrtf(float noundef %1125) #6, !fpmath !92
  %1127 = fmul float %1123, %1126
  store float %1127, ptr %147, align 4, !tbaa !39
  br label %1128

1128:                                             ; preds = %1114, %1113
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1129 = load i32, ptr %95, align 4, !tbaa !26
  %1130 = icmp sgt i32 %1129, %38
  br i1 %1130, label %1131, label %1143

1131:                                             ; preds = %1128
  %1132 = shl i64 %37, 32
  %1133 = ashr exact i64 %1132, 32
  %1134 = sext i32 %1129 to i64
  br label %1135

1135:                                             ; preds = %1131, %1135
  %1136 = phi i64 [ %1133, %1131 ], [ %1141, %1135 ]
  %1137 = getelementptr inbounds [4 x i8], ptr %109, i64 %1136
  %1138 = load float, ptr %1137, align 4, !tbaa !39
  %1139 = load float, ptr %147, align 4, !tbaa !39
  %1140 = fmul float %1138, %1139
  store float %1140, ptr %1137, align 4, !tbaa !39
  %1141 = add nsw i64 %1136, 256
  %1142 = icmp slt i64 %1141, %1134
  br i1 %1142, label %1135, label %1143, !llvm.loop !102

1143:                                             ; preds = %1135, %1128
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1144 = load i32, ptr %70, align 4, !tbaa !21
  %1145 = icmp sgt i32 %1144, %38
  br i1 %1145, label %1146, label %1164

1146:                                             ; preds = %1143
  %1147 = load i32, ptr %125, align 4, !tbaa !19
  %1148 = sitofp i32 %1147 to float
  %1149 = shl i64 %37, 32
  %1150 = ashr exact i64 %1149, 32
  %1151 = sext i32 %1144 to i64
  br label %1152

1152:                                             ; preds = %1146, %1152
  %1153 = phi i64 [ %1150, %1146 ], [ %1162, %1152 ]
  %1154 = getelementptr inbounds [4 x i8], ptr %74, i64 %1153
  %1155 = load float, ptr %1154, align 4, !tbaa !39
  %1156 = getelementptr inbounds [4 x i8], ptr %99, i64 %1153
  %1157 = load float, ptr %1156, align 4, !tbaa !39
  %1158 = load float, ptr %145, align 4, !tbaa !39
  %1159 = fmul float %1157, %1158
  %1160 = fdiv float %1159, %1148, !fpmath !91
  %1161 = fsub float %1155, %1160
  store float %1161, ptr %1154, align 4, !tbaa !39
  %1162 = add nsw i64 %1153, 256
  %1163 = icmp slt i64 %1162, %1151
  br i1 %1163, label %1152, label %1164, !llvm.loop !103

1164:                                             ; preds = %1152, %1143
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1165 = load i32, ptr %95, align 4, !tbaa !26
  %1166 = icmp sgt i32 %1165, %38
  br i1 %1166, label %1167, label %1180

1167:                                             ; preds = %1164
  %1168 = shl i64 %37, 32
  %1169 = ashr exact i64 %1168, 32
  %1170 = sext i32 %1165 to i64
  br label %1171

1171:                                             ; preds = %1167, %1171
  %1172 = phi i64 [ %1169, %1167 ], [ %1178, %1171 ]
  %1173 = getelementptr inbounds [4 x i8], ptr %74, i64 %1172
  %1174 = load float, ptr %1173, align 4, !tbaa !39
  %1175 = getelementptr inbounds [4 x i8], ptr %109, i64 %1172
  %1176 = load float, ptr %1175, align 4, !tbaa !39
  %1177 = fdiv float %1174, %1176, !fpmath !91
  store float %1177, ptr %1175, align 4, !tbaa !39
  %1178 = add nsw i64 %1172, 256
  %1179 = icmp slt i64 %1178, %1170
  br i1 %1179, label %1171, label %1180, !llvm.loop !104

1180:                                             ; preds = %1171, %1164
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1181 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %1182 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %1183 = load i32, ptr %115, align 4, !tbaa !30
  %1184 = icmp sgt i32 %1183, %38
  br i1 %1184, label %1185, label %1222

1185:                                             ; preds = %1180
  %1186 = add i32 %2, -1
  %1187 = load i32, ptr %1182, align 4, !tbaa !35
  %1188 = mul nsw i32 %1187, %59
  %1189 = add i32 %1186, %1188
  %1190 = sext i32 %1189 to i64
  %1191 = getelementptr inbounds [4 x i8], ptr %63, i64 %1190
  %1192 = load i32, ptr %1191, align 4, !tbaa !36
  %1193 = load i32, ptr %208, align 4, !tbaa !42
  %1194 = add i32 %1193, 1
  %1195 = load i32, ptr %1181, align 4, !tbaa !105
  %1196 = add i32 %1194, %1195
  %1197 = add nsw i32 %1192, %1196
  %1198 = load i32, ptr %209, align 4, !tbaa !36
  %1199 = xor i32 %1198, -1
  %1200 = add i32 %1197, %1199
  %1201 = getelementptr inbounds [4 x i8], ptr %62, i64 %1190
  %1202 = load i32, ptr %1201, align 4, !tbaa !36
  %1203 = add nsw i32 %1202, %1196
  %1204 = load i32, ptr %207, align 4, !tbaa !36
  %1205 = xor i32 %1204, -1
  %1206 = add i32 %1203, %1205
  %1207 = getelementptr inbounds nuw i8, ptr %0, i64 332
  %1208 = load i32, ptr %1207, align 4, !tbaa !106
  %1209 = mul nsw i32 %1208, %1200
  %1210 = add nsw i32 %1206, %1209
  %1211 = shl i64 %37, 32
  %1212 = ashr exact i64 %1211, 32
  %1213 = sext i32 %1183 to i64
  %1214 = sext i32 %1210 to i64
  br label %1215

1215:                                             ; preds = %1185, %1215
  %1216 = phi i64 [ %1212, %1185 ], [ %1220, %1215 ]
  %1217 = icmp eq i64 %1216, %1214
  %1218 = select i1 %1217, float 1.000000e+00, float 0.000000e+00
  %1219 = getelementptr inbounds [4 x i8], ptr %119, i64 %1216
  store float %1218, ptr %1219, align 4, !tbaa !39
  %1220 = add nsw i64 %1216, 256
  %1221 = icmp slt i64 %1220, %1213
  br i1 %1221, label %1215, label %1222, !llvm.loop !107

1222:                                             ; preds = %1215, %1180
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1223 = load i32, ptr %120, align 4, !tbaa !31
  %1224 = icmp sgt i32 %1223, %38
  br i1 %1224, label %1225, label %1303

1225:                                             ; preds = %1222
  %1226 = load i32, ptr %137, align 4, !tbaa !34
  %1227 = getelementptr inbounds nuw i8, ptr %0, i64 384
  %1228 = load i32, ptr %1227, align 4, !tbaa !108
  %1229 = getelementptr inbounds nuw i8, ptr %0, i64 352
  %1230 = load i32, ptr %1229, align 4, !tbaa !109
  %1231 = getelementptr inbounds nuw i8, ptr %0, i64 336
  %1232 = load i32, ptr %1231, align 4, !tbaa !110
  %1233 = getelementptr inbounds nuw i8, ptr %0, i64 380
  %1234 = load i32, ptr %1233, align 4, !tbaa !111
  %1235 = getelementptr inbounds nuw i8, ptr %0, i64 348
  %1236 = load i32, ptr %1235, align 4, !tbaa !112
  %1237 = getelementptr inbounds nuw i8, ptr %0, i64 332
  %1238 = load i32, ptr %1237, align 4, !tbaa !106
  %1239 = sext i32 %1238 to i64
  %1240 = shl i64 %37, 32
  %1241 = ashr exact i64 %1240, 32
  %1242 = sext i32 %1223 to i64
  %1243 = sub i32 1, %1230
  %1244 = sub i32 1, %1236
  br label %1245

1245:                                             ; preds = %1225, %1295
  %1246 = phi i64 [ %1241, %1225 ], [ %1301, %1295 ]
  %1247 = trunc i64 %1246 to i32
  %1248 = add i32 %1247, 1
  %1249 = freeze i32 %1248
  %1250 = freeze i32 %1226
  %1251 = sdiv i32 %1249, %1250
  %1252 = mul i32 %1251, %1250
  %1253 = sub i32 %1249, %1252
  %1254 = icmp ne i32 %1253, 0
  %1255 = zext i1 %1254 to i32
  %1256 = add nsw i32 %1251, %1255
  %1257 = add nsw i32 %1256, %1228
  %1258 = icmp sgt i32 %1230, %1257
  %1259 = add i32 %1257, %1243
  %1260 = select i1 %1258, i32 1, i32 %1259
  %1261 = tail call i32 @llvm.smin.i32(i32 %1232, i32 %1257)
  %1262 = icmp sgt i32 %1260, %1261
  br i1 %1262, label %1295, label %1263

1263:                                             ; preds = %1245
  %1264 = select i1 %1254, i32 %1253, i32 %1226
  %1265 = add i32 %1234, %1264
  %1266 = tail call i32 @llvm.smin.i32(i32 %1238, i32 %1265)
  %1267 = icmp sgt i32 %1236, %1265
  %1268 = add i32 %1265, %1244
  %1269 = select i1 %1267, i32 1, i32 %1268
  %1270 = icmp sgt i32 %1269, %1266
  %1271 = sext i32 %1269 to i64
  %1272 = sext i32 %1266 to i64
  %1273 = sext i32 %1260 to i64
  %1274 = sext i32 %1261 to i64
  br label %1275

1275:                                             ; preds = %1263, %1291
  %1276 = phi i64 [ %1273, %1263 ], [ %1293, %1291 ]
  %1277 = phi float [ 0.000000e+00, %1263 ], [ %1292, %1291 ]
  br i1 %1270, label %1291, label %1278

1278:                                             ; preds = %1275
  %1279 = add nsw i64 %1276, -1
  %1280 = mul nsw i64 %1279, %1239
  %1281 = getelementptr [4 x i8], ptr %119, i64 %1280
  br label %1282

1282:                                             ; preds = %1278, %1282
  %1283 = phi i64 [ %1271, %1278 ], [ %1289, %1282 ]
  %1284 = phi float [ %1277, %1278 ], [ %1288, %1282 ]
  %1285 = getelementptr [4 x i8], ptr %1281, i64 %1283
  %1286 = getelementptr i8, ptr %1285, i64 -4
  %1287 = load float, ptr %1286, align 4, !tbaa !39
  %1288 = fadd float %1284, %1287
  %1289 = add nuw nsw i64 %1283, 1
  %1290 = icmp slt i64 %1283, %1272
  br i1 %1290, label %1282, label %1291, !llvm.loop !113

1291:                                             ; preds = %1282, %1275
  %1292 = phi float [ %1277, %1275 ], [ %1288, %1282 ]
  %1293 = add nuw nsw i64 %1276, 1
  %1294 = icmp slt i64 %1276, %1274
  br i1 %1294, label %1275, label %1295, !llvm.loop !114

1295:                                             ; preds = %1291, %1245
  %1296 = phi float [ 0.000000e+00, %1245 ], [ %1292, %1291 ]
  %1297 = getelementptr inbounds [4 x i8], ptr %109, i64 %1246
  %1298 = load float, ptr %1297, align 4, !tbaa !39
  %1299 = fmul float %1296, %1298
  %1300 = getelementptr inbounds [4 x i8], ptr %124, i64 %1246
  store float %1299, ptr %1300, align 4, !tbaa !39
  %1301 = add nsw i64 %1246, 256
  %1302 = icmp slt i64 %1301, %1242
  br i1 %1302, label %1245, label %1303, !llvm.loop !115

1303:                                             ; preds = %1295, %1222
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1304 = load i32, ptr %137, align 4, !tbaa !34
  %1305 = icmp sgt i32 %1304, %38
  br i1 %1305, label %1306, label %1344

1306:                                             ; preds = %1303
  %1307 = getelementptr inbounds nuw i8, ptr %0, i64 368
  %1308 = shl i64 %37, 32
  %1309 = ashr exact i64 %1308, 32
  br label %1310

1310:                                             ; preds = %1306, %1335
  %1311 = phi i64 [ %1309, %1306 ], [ %1340, %1335 ]
  %1312 = phi i32 [ %1304, %1306 ], [ %1341, %1335 ]
  %1313 = phi float [ 0.000000e+00, %1306 ], [ %1337, %1335 ]
  %1314 = phi i32 [ 0, %1306 ], [ %1336, %1335 ]
  %1315 = load i32, ptr %1307, align 4, !tbaa !116
  %1316 = icmp sgt i32 %1315, 0
  br i1 %1316, label %1317, label %1335

1317:                                             ; preds = %1310
  %1318 = sext i32 %1312 to i64
  %1319 = mul nsw i64 %1311, %1318
  %1320 = zext nneg i32 %1315 to i64
  br label %1321

1321:                                             ; preds = %1317, %1321
  %1322 = phi i64 [ 0, %1317 ], [ %1333, %1321 ]
  %1323 = phi float [ %1313, %1317 ], [ %1332, %1321 ]
  %1324 = phi i32 [ %1314, %1317 ], [ %1331, %1321 ]
  %1325 = add nsw i64 %1322, %1319
  %1326 = getelementptr inbounds [4 x i8], ptr %124, i64 %1325
  %1327 = load float, ptr %1326, align 4, !tbaa !39
  %1328 = tail call float @_Z4fabsf(float noundef %1327) #6
  %1329 = fcmp ogt float %1328, %1323
  %1330 = trunc nsw i64 %1325 to i32
  %1331 = select i1 %1329, i32 %1330, i32 %1324
  %1332 = select i1 %1329, float %1328, float %1323
  %1333 = add nuw nsw i64 %1322, 1
  %1334 = icmp eq i64 %1333, %1320
  br i1 %1334, label %1335, label %1321, !llvm.loop !117

1335:                                             ; preds = %1321, %1310
  %1336 = phi i32 [ %1314, %1310 ], [ %1331, %1321 ]
  %1337 = phi float [ %1313, %1310 ], [ %1332, %1321 ]
  %1338 = getelementptr inbounds [4 x i8], ptr %142, i64 %1311
  store i32 %1336, ptr %1338, align 4, !tbaa !36
  %1339 = getelementptr inbounds [4 x i8], ptr %141, i64 %1311
  store float %1337, ptr %1339, align 4, !tbaa !39
  %1340 = add nsw i64 %1311, 256
  %1341 = load i32, ptr %137, align 4, !tbaa !34
  %1342 = sext i32 %1341 to i64
  %1343 = icmp slt i64 %1340, %1342
  br i1 %1343, label %1310, label %1344, !llvm.loop !118

1344:                                             ; preds = %1335, %1303
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  br i1 %1083, label %1345, label %1399

1345:                                             ; preds = %1344
  %1346 = load i32, ptr %137, align 4, !tbaa !34
  %1347 = icmp sgt i32 %1346, 0
  br i1 %1347, label %1348, label %1367

1348:                                             ; preds = %1345
  %1349 = zext nneg i32 %1346 to i64
  br label %1350

1350:                                             ; preds = %1348, %1360
  %1351 = phi i64 [ 0, %1348 ], [ %1363, %1360 ]
  %1352 = phi i32 [ 0, %1348 ], [ %1362, %1360 ]
  %1353 = phi float [ 0.000000e+00, %1348 ], [ %1361, %1360 ]
  %1354 = getelementptr inbounds nuw [4 x i8], ptr %141, i64 %1351
  %1355 = load float, ptr %1354, align 4, !tbaa !39
  %1356 = fcmp ogt float %1355, %1353
  br i1 %1356, label %1357, label %1360

1357:                                             ; preds = %1350
  %1358 = getelementptr inbounds nuw [4 x i8], ptr %142, i64 %1351
  %1359 = load i32, ptr %1358, align 4, !tbaa !36
  br label %1360

1360:                                             ; preds = %1350, %1357
  %1361 = phi float [ %1355, %1357 ], [ %1353, %1350 ]
  %1362 = phi i32 [ %1359, %1357 ], [ %1352, %1350 ]
  %1363 = add nuw nsw i64 %1351, 1
  %1364 = icmp eq i64 %1363, %1349
  br i1 %1364, label %1365, label %1350, !llvm.loop !119

1365:                                             ; preds = %1360
  %1366 = add nsw i32 %1362, 1
  br label %1367

1367:                                             ; preds = %1365, %1345
  %1368 = phi i32 [ 1, %1345 ], [ %1366, %1365 ]
  %1369 = freeze i32 %1368
  %1370 = freeze i32 %1346
  %1371 = sdiv i32 %1369, %1370
  %1372 = mul i32 %1371, %1370
  %1373 = sub i32 %1369, %1372
  %1374 = icmp eq i32 %1373, 0
  %1375 = select i1 %1374, i32 %1346, i32 %1373
  %1376 = sext i1 %1374 to i32
  %1377 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %1378 = load i32, ptr %1377, align 4, !tbaa !37
  %1379 = load i32, ptr %208, align 4, !tbaa !42
  %1380 = load i32, ptr %1181, align 4, !tbaa !105
  %1381 = sub i32 %1380, %1379
  %1382 = load i32, ptr %127, align 4, !tbaa !32
  %1383 = load i32, ptr %1182, align 4, !tbaa !35
  %1384 = mul nsw i32 %1383, %59
  %1385 = add nsw i32 %1384, %2
  %1386 = load i32, ptr %207, align 4, !tbaa !36
  %1387 = sub i32 %1375, %1378
  %1388 = add i32 %1387, %1381
  %1389 = add nsw i32 %1388, %1386
  %1390 = sext i32 %1385 to i64
  %1391 = getelementptr inbounds [4 x i8], ptr %62, i64 %1390
  store i32 %1389, ptr %1391, align 4, !tbaa !36
  %1392 = load i32, ptr %209, align 4, !tbaa !36
  %1393 = add i32 %1371, 1
  %1394 = add i32 %1393, %1376
  %1395 = add i32 %1394, %1381
  %1396 = sub i32 %1395, %1382
  %1397 = add nsw i32 %1396, %1392
  %1398 = getelementptr inbounds [4 x i8], ptr %63, i64 %1390
  store i32 %1397, ptr %1398, align 4, !tbaa !36
  br label %1399

1399:                                             ; preds = %1367, %1344
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1400 = srem i32 %2, 10
  %1401 = icmp eq i32 %1400, 0
  br i1 %1401, label %1402, label %1454

1402:                                             ; preds = %1399
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  %1403 = load i32, ptr %1182, align 4, !tbaa !35
  %1404 = mul nsw i32 %1403, %59
  %1405 = add nsw i32 %1404, %2
  %1406 = sext i32 %1405 to i64
  %1407 = getelementptr inbounds [4 x i8], ptr %62, i64 %1406
  %1408 = load i32, ptr %1407, align 4, !tbaa !36
  store i32 %1408, ptr %207, align 4, !tbaa !36
  %1409 = getelementptr inbounds [4 x i8], ptr %63, i64 %1406
  %1410 = load i32, ptr %1409, align 4, !tbaa !36
  store i32 %1410, ptr %209, align 4, !tbaa !36
  %1411 = load i32, ptr %125, align 4, !tbaa !19
  %1412 = icmp sgt i32 %1411, %38
  br i1 %1412, label %1413, label %1453

1413:                                             ; preds = %1402
  %1414 = getelementptr inbounds nuw i8, ptr %0, i64 72
  %1415 = load i32, ptr %1414, align 4, !tbaa !37
  %1416 = load i32, ptr %207, align 4, !tbaa !36
  %1417 = add i32 %1410, -26
  %1418 = getelementptr inbounds nuw i8, ptr %0, i64 36
  %1419 = load i32, ptr %1418, align 4, !tbaa !38
  %1420 = add i32 %1416, -27
  %1421 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %1422 = shl i64 %37, 32
  %1423 = ashr exact i64 %1422, 32
  %1424 = sext i32 %1411 to i64
  br label %1425

1425:                                             ; preds = %1413, %1425
  %1426 = phi i64 [ %1423, %1413 ], [ %1451, %1425 ]
  %1427 = trunc i64 %1426 to i32
  %1428 = add i32 %1427, 1
  %1429 = freeze i32 %1428
  %1430 = freeze i32 %1415
  %1431 = sdiv i32 %1429, %1430
  %1432 = mul i32 %1431, %1430
  %1433 = sub i32 %1429, %1432
  %1434 = icmp eq i32 %1433, 0
  %1435 = sext i1 %1434 to i32
  %1436 = select i1 %1434, i32 %1415, i32 %1433
  %1437 = add i32 %1417, %1431
  %1438 = add i32 %1437, %1435
  %1439 = mul nsw i32 %1438, %1419
  %1440 = add i32 %1436, %1420
  %1441 = add nsw i32 %1440, %1439
  %1442 = load float, ptr %1421, align 4, !tbaa !120
  %1443 = getelementptr inbounds [4 x i8], ptr %64, i64 %1426
  %1444 = load float, ptr %1443, align 4, !tbaa !39
  %1445 = fsub float 1.000000e+00, %1442
  %1446 = sext i32 %1441 to i64
  %1447 = getelementptr inbounds [4 x i8], ptr %1, i64 %1446
  %1448 = load float, ptr %1447, align 4, !tbaa !39
  %1449 = fmul float %1445, %1448
  %1450 = tail call float @llvm.fmuladd.f32(float %1442, float %1444, float %1449)
  store float %1450, ptr %1443, align 4, !tbaa !39
  %1451 = add nsw i64 %1426, 256
  %1452 = icmp slt i64 %1451, %1424
  br i1 %1452, label %1425, label %1453, !llvm.loop !121

1453:                                             ; preds = %1425, %1402
  tail call void @_Z7barrierj(i32 noundef signext 3) #7
  br label %1454

1454:                                             ; preds = %204, %1453, %1399
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local i64 @_Z12get_group_idj(i32 noundef signext) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local i64 @_Z12get_local_idj(i32 noundef signext) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local void @_Z7barrierj(i32 noundef signext) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local float @_Z4sqrtf(float noundef) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare dso_local float @_Z4fabsf(float noundef) local_unnamed_addr #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #5

attributes #0 = { convergent norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #1 = { alwaysinline convergent norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #2 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #3 = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-smpmpmt,-experimental-svukte,-experimental-xqccmt,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-y,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvabd,-experimental-zvbc32e,-experimental-zvdot4a8i,-experimental-zvfbdota32f,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvfqwbdota8f,-experimental-zvfqwdota8f,-experimental-zvfwbdota16bf,-experimental-zvfwdota16bf,-experimental-zvkgs,-experimental-zvqwbdota16i,-experimental-zvqwbdota8i,-experimental-zvqwdota16i,-experimental-zvqwdota8i,-experimental-zvvfmm,-experimental-zvvmm,-experimental-zvvmtls,-experimental-zvvmttls,-experimental-zvzip,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svrsw60t59b,-svvptc,-v,-xaifet,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcheriot,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xqccmp,-xqci,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xsmtvdotii,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zalasr,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccid,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "uniform-work-group-size" }
attributes #4 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { convergent nounwind willreturn memory(none) "uniform-work-group-size" }
attributes #7 = { convergent nounwind "uniform-work-group-size" }

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
!11 = !{i32 0, i32 1, i32 0, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1}
!12 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!13 = !{!"params_common", !"float*", !"int", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"int*", !"float*", !"float*", !"float*", !"float*"}
!14 = !{!"struct params_common", !"float*", !"int", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"int*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"float*", !"int*", !"float*", !"float*", !"float*", !"float*"}
!15 = !{!"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !"", !""}
!16 = !{!17, !8, i64 52}
!17 = !{!"params_common", !8, i64 0, !8, i64 4, !8, i64 8, !8, i64 12, !8, i64 16, !8, i64 20, !8, i64 24, !18, i64 28, !8, i64 32, !8, i64 36, !8, i64 40, !8, i64 44, !8, i64 48, !8, i64 52, !8, i64 56, !8, i64 60, !8, i64 64, !8, i64 68, !8, i64 72, !8, i64 76, !8, i64 80, !8, i64 84, !8, i64 88, !8, i64 92, !8, i64 96, !8, i64 100, !8, i64 104, !8, i64 108, !8, i64 112, !8, i64 116, !8, i64 120, !8, i64 124, !8, i64 128, !8, i64 132, !8, i64 136, !8, i64 140, !8, i64 144, !8, i64 148, !8, i64 152, !8, i64 156, !8, i64 160, !8, i64 164, !8, i64 168, !8, i64 172, !8, i64 176, !8, i64 180, !8, i64 184, !8, i64 188, !8, i64 192, !8, i64 196, !8, i64 200, !8, i64 204, !8, i64 208, !8, i64 212, !8, i64 216, !8, i64 220, !8, i64 224, !8, i64 228, !8, i64 232, !8, i64 236, !8, i64 240, !8, i64 244, !8, i64 248, !8, i64 252, !8, i64 256, !8, i64 260, !8, i64 264, !8, i64 268, !8, i64 272, !8, i64 276, !8, i64 280, !8, i64 284, !8, i64 288, !8, i64 292, !8, i64 296, !8, i64 300, !8, i64 304, !8, i64 308, !8, i64 312, !8, i64 316, !8, i64 320, !8, i64 324, !8, i64 328, !8, i64 332, !8, i64 336, !8, i64 340, !8, i64 344, !8, i64 348, !8, i64 352, !8, i64 356, !8, i64 360, !8, i64 364, !8, i64 368, !8, i64 372, !8, i64 376, !8, i64 380, !8, i64 384}
!18 = !{!"float", !9, i64 0}
!19 = !{!17, !8, i64 80}
!20 = !{!17, !8, i64 100}
!21 = !{!17, !8, i64 116}
!22 = !{!17, !8, i64 148}
!23 = !{!17, !8, i64 164}
!24 = !{!17, !8, i64 212}
!25 = !{!17, !8, i64 228}
!26 = !{!17, !8, i64 276}
!27 = !{!17, !8, i64 292}
!28 = !{!17, !8, i64 308}
!29 = !{!17, !8, i64 324}
!30 = !{!17, !8, i64 340}
!31 = !{!17, !8, i64 372}
!32 = !{!17, !8, i64 76}
!33 = !{!17, !8, i64 316}
!34 = !{!17, !8, i64 364}
!35 = !{!17, !8, i64 32}
!36 = !{!8, !8, i64 0}
!37 = !{!17, !8, i64 72}
!38 = !{!17, !8, i64 36}
!39 = !{!18, !18, i64 0}
!40 = distinct !{!40, !41}
!41 = !{!"llvm.loop.unroll.disable"}
!42 = !{!17, !8, i64 16}
!43 = !{!17, !8, i64 92}
!44 = distinct !{!44, !41}
!45 = distinct !{!45, !41}
!46 = !{!17, !8, i64 108}
!47 = !{!17, !8, i64 128}
!48 = !{!17, !8, i64 96}
!49 = !{!17, !8, i64 124}
!50 = distinct !{!50, !41}
!51 = distinct !{!51, !41}
!52 = distinct !{!52, !41}
!53 = !{!17, !8, i64 140}
!54 = !{!17, !8, i64 132}
!55 = distinct !{!55, !41}
!56 = !{!17, !8, i64 144}
!57 = distinct !{!57, !41}
!58 = distinct !{!58, !41}
!59 = !{!17, !8, i64 156}
!60 = !{!17, !8, i64 172}
!61 = !{!17, !8, i64 180}
!62 = distinct !{!62, !41}
!63 = !{!17, !8, i64 204}
!64 = !{!17, !8, i64 188}
!65 = !{!17, !8, i64 196}
!66 = distinct !{!66, !41}
!67 = distinct !{!67, !41}
!68 = distinct !{!68, !41}
!69 = distinct !{!69, !41}
!70 = !{!17, !8, i64 220}
!71 = !{!17, !8, i64 236}
!72 = !{!17, !8, i64 244}
!73 = distinct !{!73, !41}
!74 = !{!17, !8, i64 268}
!75 = !{!17, !8, i64 252}
!76 = !{!17, !8, i64 260}
!77 = distinct !{!77, !41}
!78 = distinct !{!78, !41}
!79 = distinct !{!79, !41}
!80 = distinct !{!80, !41}
!81 = distinct !{!81, !41}
!82 = distinct !{!82, !41}
!83 = distinct !{!83, !41}
!84 = distinct !{!84, !41}
!85 = distinct !{!85, !41}
!86 = distinct !{!86, !41}
!87 = distinct !{!87, !41}
!88 = distinct !{!88, !41}
!89 = distinct !{!89, !41}
!90 = distinct !{!90, !41}
!91 = !{float 2.500000e+00}
!92 = !{float 3.000000e+00}
!93 = distinct !{!93, !41}
!94 = distinct !{!94, !41}
!95 = distinct !{!95, !41}
!96 = distinct !{!96, !41}
!97 = !{!17, !8, i64 320}
!98 = distinct !{!98, !41}
!99 = distinct !{!99, !41}
!100 = distinct !{!100, !41}
!101 = distinct !{!101, !41}
!102 = distinct !{!102, !41}
!103 = distinct !{!103, !41}
!104 = distinct !{!104, !41}
!105 = !{!17, !8, i64 20}
!106 = !{!17, !8, i64 332}
!107 = distinct !{!107, !41}
!108 = !{!17, !8, i64 384}
!109 = !{!17, !8, i64 352}
!110 = !{!17, !8, i64 336}
!111 = !{!17, !8, i64 380}
!112 = !{!17, !8, i64 348}
!113 = distinct !{!113, !41}
!114 = distinct !{!114, !41}
!115 = distinct !{!115, !41}
!116 = !{!17, !8, i64 368}
!117 = distinct !{!117, !41}
!118 = distinct !{!118, !41}
!119 = distinct !{!119, !41}
!120 = !{!17, !18, i64 28}
!121 = distinct !{!121, !41}
!122 = !{!17, !8, i64 136}
!123 = !{!17, !8, i64 284}
!124 = !{!17, !8, i64 288}

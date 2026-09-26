#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<() -> ()>
module {
  func.func @net(%arg0: tensor<1xf32>) -> tensor<f32> {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %cst_2 = arith.constant dense_resource<torch_tensor_2_2_torch.float32_1> : tensor<2x2xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_2_2_torch.float32> : tensor<2x2xf32>
    %cst_4 = arith.constant dense_resource<torch_tensor_2_torch.float32> : tensor<2xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_2_torch.float32_1> : tensor<2xf32>
    %0 = tensor.empty() : tensor<2xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%cst_4, %cst_5 : tensor<2xf32>, tensor<2xf32>) outs(%0 : tensor<2xf32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.subf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<2xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<2xf32>) outs(%0 : tensor<2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = math.fpowi %in, %c2_i64 : f32, i64
      linalg.yield %35 : f32
    } -> tensor<2xf32>
    %3 = tensor.empty() : tensor<f32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%3 : tensor<f32>) -> tensor<f32>
    %5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%2 : tensor<2xf32>) outs(%4 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %6 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<2xf32>) {
    ^bb0(%out: f32):
      %35 = linalg.index 0 : index
      %extracted = tensor.extract %cst_3[%35, %35] : tensor<2x2xf32>
      linalg.yield %extracted : f32
    } -> tensor<2xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%6 : tensor<2xf32>) outs(%4 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %8 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<2xf32>) {
    ^bb0(%out: f32):
      %35 = linalg.index 0 : index
      %extracted = tensor.extract %cst_2[%35, %35] : tensor<2x2xf32>
      linalg.yield %extracted : f32
    } -> tensor<2xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%8 : tensor<2xf32>) outs(%4 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %10 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%7, %9 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.addf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %11 = tensor.empty() : tensor<2x2xf32>
    %12 = linalg.fill ins(%cst_1 : f32) outs(%11 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %13 = linalg.matmul ins(%cst_3, %cst_2 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%12 : tensor<2x2xf32>) -> tensor<2x2xf32>
    %extracted_slice = tensor.extract_slice %13[0, 0] [1, 2] [1, 1] : tensor<2x2xf32> to tensor<1x2xf32>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x2xf32> into tensor<2xf32>
    %extracted_slice_6 = tensor.extract_slice %collapsed[0] [1] [1] : tensor<2xf32> to tensor<1xf32>
    %collapsed_7 = tensor.collapse_shape %extracted_slice_6 [] : tensor<1xf32> into tensor<f32>
    %extracted_slice_8 = tensor.extract_slice %13[1, 0] [1, 2] [1, 1] : tensor<2x2xf32> to tensor<1x2xf32>
    %collapsed_9 = tensor.collapse_shape %extracted_slice_8 [[0, 1]] : tensor<1x2xf32> into tensor<2xf32>
    %extracted_slice_10 = tensor.extract_slice %collapsed_9[1] [1] [1] : tensor<2xf32> to tensor<1xf32>
    %collapsed_11 = tensor.collapse_shape %extracted_slice_10 [] : tensor<1xf32> into tensor<f32>
    %14 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%collapsed_7, %collapsed_11 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.addf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %15 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%collapsed_7, %collapsed_11 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.mulf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %extracted_slice_12 = tensor.extract_slice %collapsed[1] [1] [1] : tensor<2xf32> to tensor<1xf32>
    %collapsed_13 = tensor.collapse_shape %extracted_slice_12 [] : tensor<1xf32> into tensor<f32>
    %extracted_slice_14 = tensor.extract_slice %collapsed_9[0] [1] [1] : tensor<2xf32> to tensor<1xf32>
    %collapsed_15 = tensor.collapse_shape %extracted_slice_14 [] : tensor<1xf32> into tensor<f32>
    %16 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%collapsed_13, %collapsed_15 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.mulf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %17 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%15, %16 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.subf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %18 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%14, %14 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.mulf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %19 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%17 : tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.mulf %in, %cst_0 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %20 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%18, %19 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.subf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %21 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%20 : tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.cmpf ult, %in, %cst_1 : f32
      %36 = arith.select %35, %cst_1, %in : f32
      linalg.yield %36 : f32
    } -> tensor<f32>
    %22 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%21 : tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = math.sqrt %in : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %23 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%14, %22 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.addf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %24 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%23 : tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.divf %in, %cst : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %25 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%14, %22 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.subf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %26 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%25 : tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.divf %in, %cst : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %27 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%24 : tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.cmpf ult, %in, %cst_1 : f32
      %36 = arith.select %35, %cst_1, %in : f32
      linalg.yield %36 : f32
    } -> tensor<f32>
    %28 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%27 : tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = math.sqrt %in : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %29 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%26 : tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.cmpf ult, %in, %cst_1 : f32
      %36 = arith.select %35, %cst_1, %in : f32
      linalg.yield %36 : f32
    } -> tensor<f32>
    %30 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%29 : tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = math.sqrt %in : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %31 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%28, %30 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.addf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %32 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%5, %10 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.addf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %33 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%31 : tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.mulf %in, %cst : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %34 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = []} ins(%32, %33 : tensor<f32>, tensor<f32>) outs(%3 : tensor<f32>) {
    ^bb0(%in: f32, %in_16: f32, %out: f32):
      %35 = arith.subf %in, %in_16 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    return %34 : tensor<f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_2_2_torch.float32_1: "0x040000006666663F0000000000000000CDCC8C3F",
      torch_tensor_2_2_torch.float32: "0x040000000000803FCDCCCC3DCDCCCC3D0000803F",
      torch_tensor_2_torch.float32: "0x040000000000803F00000040",
      torch_tensor_2_torch.float32_1: "0x040000000000C03F00002040"
    }
  }
#-}

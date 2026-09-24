#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> ()>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d0, 0)>
#map6 = affine_map<(d0, d1, d2) -> (d2)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map10 = affine_map<(d0, d1, d2) -> ()>
#map11 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map12 = affine_map<(d0, d1, d2) -> (d1)>
#map13 = affine_map<(d0) -> ()>
#map14 = affine_map<() -> ()>
module {
  func.func @net(%arg0: tensor<4x8xf32>) -> tensor<1xf32> {
    %c0_i64 = arith.constant 0 : i64
    %c8 = arith.constant 8 : index
    %c9223372036854775807_i64 = arith.constant 9223372036854775807 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %false = arith.constant false
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<f32>
    %cst_2 = arith.constant 8.000000e+00 : f32
    %cst_3 = arith.constant 4.000000e+00 : f32
    %c-1_i64 = arith.constant -1 : i64
    %cst_4 = arith.constant dense<-1> : tensor<i64>
    %cst_5 = arith.constant dense<0> : tensor<i64>
    %cst_6 = arith.constant dense<8> : tensor<i64>
    %cst_7 = arith.constant dense_resource<torch_tensor_4_8_torch.int64> : tensor<4x8xi64>
    %0 = tensor.empty() : tensor<8xi64>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<8xi64>) {
    ^bb0(%out: i64):
      %35 = linalg.index 0 : index
      %36 = arith.index_cast %35 : index to i64
      linalg.yield %36 : i64
    } -> tensor<8xi64>
    %2 = tensor.empty() : tensor<4x8xi1>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_7 : tensor<4x8xi64>) outs(%2 : tensor<4x8xi1>) {
    ^bb0(%in: i64, %out: i1):
      %35 = arith.cmpi eq, %in, %c-1_i64 : i64
      linalg.yield %35 : i1
    } -> tensor<4x8xi1>
    %4 = tensor.empty() : tensor<4x8xi64>
    %5 = linalg.generic {indexing_maps = [#map1, #map2, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %1, %cst_6 : tensor<4x8xi1>, tensor<8xi64>, tensor<i64>) outs(%4 : tensor<4x8xi64>) {
    ^bb0(%in: i1, %in_11: i64, %in_12: i64, %out: i64):
      %35 = arith.select %in, %in_11, %in_12 : i64
      linalg.yield %35 : i64
    } -> tensor<4x8xi64>
    %6 = tensor.empty() : tensor<4xi64>
    %7 = linalg.fill ins(%c0_i64 : i64) outs(%6 : tensor<4xi64>) -> tensor<4xi64>
    %8 = linalg.fill ins(%c9223372036854775807_i64 : i64) outs(%6 : tensor<4xi64>) -> tensor<4xi64>
    %9:2 = linalg.generic {indexing_maps = [#map1, #map4, #map4], iterator_types = ["parallel", "reduction"]} ins(%5 : tensor<4x8xi64>) outs(%8, %7 : tensor<4xi64>, tensor<4xi64>) {
    ^bb0(%in: i64, %out: i64, %out_11: i64):
      %35 = linalg.index 1 : index
      %36 = arith.index_cast %35 : index to i64
      %37 = arith.minsi %in, %out : i64
      %38 = arith.cmpi slt, %in, %out : i64
      %39 = arith.select %38, %36, %out_11 : i64
      linalg.yield %37, %39 : i64, i64
    } -> (tensor<4xi64>, tensor<4xi64>)
    %expanded = tensor.expand_shape %9#0 [[0, 1]] output_shape [4, 1] : tensor<4xi64> into tensor<4x1xi64>
    %10 = linalg.generic {indexing_maps = [#map2, #map5, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %expanded : tensor<8xi64>, tensor<4x1xi64>) outs(%2 : tensor<4x8xi1>) {
    ^bb0(%in: i64, %in_11: i64, %out: i1):
      %35 = arith.cmpi slt, %in, %in_11 : i64
      linalg.yield %35 : i1
    } -> tensor<4x8xi1>
    %11 = linalg.generic {indexing_maps = [#map1, #map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%10, %cst_7, %cst_5 : tensor<4x8xi1>, tensor<4x8xi64>, tensor<i64>) outs(%4 : tensor<4x8xi64>) {
    ^bb0(%in: i1, %in_11: i64, %in_12: i64, %out: i64):
      %35 = arith.select %in, %in_11, %in_12 : i64
      linalg.yield %35 : i64
    } -> tensor<4x8xi64>
    %12 = tensor.empty() : tensor<4x8xf32>
    %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<4x8xi64>) outs(%13 : tensor<4x8xf32>) {
    ^bb0(%in: i64, %out: f32):
      %35 = linalg.index 0 : index
      %36 = arith.index_cast %in : i64 to index
      %37 = arith.cmpi slt, %36, %c8 : index
      cf.assert %37, "index must be smaller than dim size"
      %38 = arith.cmpi sge, %in, %c0_i64 : i64
      cf.assert %38, "index must be larger or equal to 0"
      %extracted = tensor.extract %arg0[%35, %36] : tensor<4x8xf32>
      linalg.yield %extracted : f32
    } -> tensor<4x8xf32>
    %15 = linalg.generic {indexing_maps = [#map1, #map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%10, %cst_7, %cst_4 : tensor<4x8xi1>, tensor<4x8xi64>, tensor<i64>) outs(%4 : tensor<4x8xi64>) {
    ^bb0(%in: i1, %in_11: i64, %in_12: i64, %out: i64):
      %35 = arith.select %in, %in_11, %in_12 : i64
      linalg.yield %35 : i64
    } -> tensor<4x8xi64>
    %expanded_8 = tensor.expand_shape %15 [[0], [1, 2]] output_shape [4, 8, 1] : tensor<4x8xi64> into tensor<4x8x1xi64>
    %16 = tensor.empty() : tensor<4x8x8xi1>
    %17 = linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %expanded_8 : tensor<8xi64>, tensor<4x8x1xi64>) outs(%16 : tensor<4x8x8xi1>) {
    ^bb0(%in: i64, %in_11: i64, %out: i1):
      %35 = arith.cmpi eq, %in, %in_11 : i64
      linalg.yield %35 : i1
    } -> tensor<4x8x8xi1>
    %18 = linalg.fill ins(%false : i1) outs(%2 : tensor<4x8xi1>) -> tensor<4x8xi1>
    %19 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "reduction", "parallel"]} ins(%17 : tensor<4x8x8xi1>) outs(%18 : tensor<4x8xi1>) {
    ^bb0(%in: i1, %out: i1):
      %35 = arith.ori %in, %out : i1
      linalg.yield %35 : i1
    } -> tensor<4x8xi1>
    %20 = tensor.empty() : tensor<8x4xf32>
    %transposed = linalg.transpose ins(%14 : tensor<4x8xf32>) outs(%20 : tensor<8x4xf32>) permutation = [1, 0] 
    %expanded_9 = tensor.expand_shape %transposed [[0], [1, 2]] output_shape [8, 4, 1] : tensor<8x4xf32> into tensor<8x4x1xf32>
    %21 = tensor.empty() : tensor<8x4x1xf32>
    %22 = linalg.generic {indexing_maps = [#map10, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_1, %expanded_9 : tensor<f32>, tensor<8x4x1xf32>) outs(%21 : tensor<8x4x1xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %35 = arith.subf %in, %in_11 : f32
      linalg.yield %35 : f32
    } -> tensor<8x4x1xf32>
    %23 = tensor.empty() : tensor<8x4x8xf32>
    %24 = linalg.generic {indexing_maps = [#map7, #map11, #map8], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22, %arg0 : tensor<8x4x1xf32>, tensor<4x8xf32>) outs(%23 : tensor<8x4x8xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %35 = arith.addf %in, %in_11 : f32
      linalg.yield %35 : f32
    } -> tensor<8x4x8xf32>
    %25 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24 : tensor<8x4x8xf32>) outs(%23 : tensor<8x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.cmpf ult, %in, %cst : f32
      %36 = arith.select %35, %cst, %in : f32
      linalg.yield %36 : f32
    } -> tensor<8x4x8xf32>
    %26 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25 : tensor<8x4x8xf32>) outs(%23 : tensor<8x4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.divf %in, %cst_2 : f32
      linalg.yield %35 : f32
    } -> tensor<8x4x8xf32>
    %27 = linalg.generic {indexing_maps = [#map11, #map10, #map8, #map8], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %cst_0, %26 : tensor<4x8xi1>, tensor<f32>, tensor<8x4x8xf32>) outs(%23 : tensor<8x4x8xf32>) {
    ^bb0(%in: i1, %in_11: f32, %in_12: f32, %out: f32):
      %35 = arith.select %in, %in_11, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<8x4x8xf32>
    %28 = tensor.empty() : tensor<4xf32>
    %29 = linalg.fill ins(%cst : f32) outs(%28 : tensor<4xf32>) -> tensor<4xf32>
    %30 = linalg.generic {indexing_maps = [#map8, #map12], iterator_types = ["reduction", "parallel", "reduction"]} ins(%27 : tensor<8x4x8xf32>) outs(%29 : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    } -> tensor<4xf32>
    %31 = tensor.empty() : tensor<f32>
    %32 = linalg.fill ins(%cst : f32) outs(%31 : tensor<f32>) -> tensor<f32>
    %33 = linalg.generic {indexing_maps = [#map, #map13], iterator_types = ["reduction"]} ins(%30 : tensor<4xf32>) outs(%32 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.addf %in, %out : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %34 = linalg.generic {indexing_maps = [#map14, #map14], iterator_types = []} ins(%33 : tensor<f32>) outs(%31 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %35 = arith.divf %in, %cst_3 : f32
      linalg.yield %35 : f32
    } -> tensor<f32>
    %expanded_10 = tensor.expand_shape %34 [] output_shape [1] : tensor<f32> into tensor<1xf32>
    return %expanded_10 : tensor<1xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_4_8_torch.int64: "0x0800000003000000000000000000000000000000FFFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000000000000000000000000000000000003000000000000000000000000000000FFFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000000000000000000000000000000000003000000000000000000000000000000FFFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000000000000000000000000000000000003000000000000000000000000000000FFFFFFFFFFFFFFFF00000000000000000000000000000000000000000000000000000000000000000000000000000000"
    }
  }
#-}

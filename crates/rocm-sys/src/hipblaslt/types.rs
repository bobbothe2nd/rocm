/// Single precision floating point type"]
pub type hipblasLtFloat = f32;
/// Structure definition for hipblasLtHalf"]
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct _hipblasLtHalf {
    pub data: u16,
}
/// Structure definition for hipblasLtHalf"]
pub type hipblasLtHalf = _hipblasLtHalf;
pub type hipblasLtBfloat16 = u16;
pub type hipblasLtInt8 = i8;
pub type hipblasLtInt32 = i32;
#[repr(i32)]
/// Specifies the enumeration type to set the postprocessing options for the epilogue."]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLtEpilogue_t {
    ///<No special postprocessing. Scale and quantize the results if necessary."]
    HIPBLASLT_EPILOGUE_DEFAULT = 1,
    ///<Apply ReLU pointwise transform to the results (``x:=max(x, 0)``)"]
    HIPBLASLT_EPILOGUE_RELU = 2,
    ///<Apply bias from the bias vector, and broadcast to all columns if HIPBLAST_MATMUL_DESC_BIAS_BATCH_STRIDE = 0. The bias vector length must match the number of rows in matrix D, and it must be packed (so the stride between vector elements is one). The bias vector is broadcast to all columns if HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE is 0 and added before applying the final postprocessing."]
    HIPBLASLT_EPILOGUE_BIAS = 4,
    ///<Apply bias and then ReLU transform."]
    HIPBLASLT_EPILOGUE_RELU_BIAS = 6,
    ///<Apply GELU pointwise transform to the results (``x:=GELU(x)``)."]
    HIPBLASLT_EPILOGUE_GELU = 32,
    ///<Apply Bias and then GELU transform."]
    HIPBLASLT_EPILOGUE_GELU_BIAS = 36,
    ///<Output GEMM results before applying RELU transform."]
    HIPBLASLT_EPILOGUE_RELU_AUX = 130,
    ///<Output GEMM results after applying bias but before applying RELU transform."]
    HIPBLASLT_EPILOGUE_RELU_AUX_BIAS = 134,
    HIPBLASLT_EPILOGUE_DRELU = 136,
    ///<Apply gradient RELU transform and bias gradient to the results. Requires additional auxiliary input. */           /**<Apply gradient RELU transform. Requires additional auxiliary input."]
    HIPBLASLT_EPILOGUE_DRELU_BGRAD = 152,
    ///<Output GEMM results before applying GELU transform."]
    HIPBLASLT_EPILOGUE_GELU_AUX = 160,
    ///<Output GEMM results after applying bias but before applying GELU transform."]
    HIPBLASLT_EPILOGUE_GELU_AUX_BIAS = 164,
    ///<Apply gradient GELU transform. Requires additional auxiliary input."]
    HIPBLASLT_EPILOGUE_DGELU = 192,
    ///<Apply gradient GELU transform and bias gradient to the results. Requires additional auxiliary input."]
    HIPBLASLT_EPILOGUE_DGELU_BGRAD = 208,
    ///<Apply bias gradient to A and output GEMM result."]
    HIPBLASLT_EPILOGUE_BGRADA = 256,
    ///<Apply bias gradient to B and output GEMM result."]
    HIPBLASLT_EPILOGUE_BGRADB = 512,
    ///<Apply sigmoid activation function pointwise."]
    HIPBLASLT_EPILOGUE_SIGMOID = 1024,
    ///<Apply Swish pointwise transform to the results (``x:=Swish(x, 1)``)."]
    HIPBLASLT_EPILOGUE_SWISH_EXT = 65536,
    ///<Apply Bias and then Swish transform."]
    HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT = 65540,
    ///<Apply pointwise clamp to the results (``x:=max(alpha, min(x, beta))``)."]
    HIPBLASLT_EPILOGUE_CLAMP_EXT = 131072,
    ///<Apply Bias and then clamp."]
    HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT = 131076,
    ///<Output GEMM results before applying clamp transform."]
    HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT = 131200,
    ///<Output GEMM results after applying bias but before applying clamp transform."]
    HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT = 131204,
}
#[repr(i32)]
/// Specify the batch mode of the matrices."]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLtBatchMode_t {
    HIPBLASLT_BATCH_MODE_STRIDED = 0,
    HIPBLASLT_BATCH_MODE_POINTER_ARRAY = 1,
    ///<Grouped GEMM: each group has its own shape. Set only by ``hipblasLtGroupedMatrixLayoutCreate``."]
    HIPBLASLT_BATCH_MODE_GROUPED = 2,
}
#[repr(i32)]
/// Specifies the attributes that define the details of the matrix."]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLtMatrixLayoutAttribute_t {
    ///<Number of batches of this matrix. Default value is 1. Data type: ``int32_t``."]
    HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 0,
    ///<Stride (in elements) to the next matrix for the strided batch operation. Default value is 0. Data type: ``int64_t``."]
    HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 1,
    /// Data type. See ``hipDataType``.
    ///
    ///
    ///
    /// ``uint32_t``"]
    HIPBLASLT_MATRIX_LAYOUT_TYPE = 2,
    /// Memory order of the data. See ``hipblasLtOrder_t``.
    ///
    ///
    ///
    /// ``int32_t``, default: ``HIPBLASLT_ORDER_COL``."]
    HIPBLASLT_MATRIX_LAYOUT_ORDER = 3,
    /// Number of rows.
    ///
    ///
    ///
    /// Typically only values that can be expressed as ``int32_t`` are supported.
    ///
    ///
    ///
    /// ``uint64_t``"]
    HIPBLASLT_MATRIX_LAYOUT_ROWS = 4,
    /// Number of columns.
    ///
    ///
    ///
    /// Typically only values that can be expressed as ``int32_t`` are supported.
    ///
    ///
    ///
    /// ``uint64_t``"]
    HIPBLASLT_MATRIX_LAYOUT_COLS = 5,
    /// Matrix leading dimension.
    ///
    ///
    ///
    /// For ``HIPBLASLT_ORDER_COL``, this is the stride (in elements) of the matrix column. For more details and documentation for
    ///
    /// other memory orders, see the documentation for ``hipblasLtOrder_t`` values.
    ///
    ///
    ///
    /// Currently only non-negative values are supported. The value must be large enough so that matrix memory locations are not
    ///
    /// overlapping (that is, greater or equal to ``HIPBLASLT_MATRIX_LAYOUT_ROWS`` in the case of ``HIPBLASLT_ORDER_COL``).
    ///
    ///
    ///
    /// ``int64_t``"]
    HIPBLASLT_MATRIX_LAYOUT_LD = 6,
    /// Matrix Batch Mode.
    ///
    /// Batched GEMM can be either:
    ///
    /// 1. Strided Batch: Single contiguous memory allocation and stride between matrices in
    ///
    /// the batch is specified in terms of number of elements.
    ///
    /// 2. General Batched: This uses pointer array with each pointer storing the base address
    ///
    /// of the matrices in the batch.
    ///
    /// See hipblasLtBatchMode_t"]
    HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE = 7,
    /// Matrix Offset.
    ///
    ///
    ///
    /// For ``General Batched GEMM``, we can support for users to access a sub-matrix of
    ///
    /// the original matrix by adding an ``offset`` value (in elements) from the base address.
    ///
    /// Note that for non-batched or Strided Batch GEMM case, we can directly apply
    ///
    /// the offset value by using the strided-offset value."]
    HIPBLASLT_MATRIX_LAYOUT_OFFSET = 8,
    /// Grouped layouts only: device array of ``groupCount`` row counts, or NULL to use
    ///
    /// ``HIPBLASLT_MATRIX_LAYOUT_ROWS`` for every group.
    ///
    ///
    ///
    /// ``void*``"]
    HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY = 9,
    /// Grouped layouts only: device array of ``groupCount`` column counts, or NULL to
    ///
    /// use ``HIPBLASLT_MATRIX_LAYOUT_COLS`` for every group.
    ///
    ///
    ///
    /// ``void*``"]
    HIPBLASLT_GROUPED_MATRIX_LAYOUT_COLS_ARRAY = 10,
    /// Grouped layouts only: device array of ``groupCount`` leading dimensions, or
    ///
    /// NULL to use ``HIPBLASLT_MATRIX_LAYOUT_LD`` for every group.
    ///
    ///
    ///
    /// ``void*``"]
    HIPBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY = 11,
    /// Width of the integers in the rows and cols arrays. See ``hipblasLtIntegerWidth_t``.
    ///
    ///
    ///
    /// ``int32_t``, default: ``HIPBLASLT_INTEGER_WIDTH_32``."]
    HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH = 12,
    /// Width of the integers in the leading-dimension array. See ``hipblasLtIntegerWidth_t``.
    ///
    ///
    ///
    /// ``int32_t``, default: ``HIPBLASLT_INTEGER_WIDTH_32``."]
    HIPBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH = 13,
}
#[repr(i32)]
/// Width of the integers in the dimension arrays of a grouped matrix layout."]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLtIntegerWidth_t {
    ///<32-bit integers (``int32_t``)."]
    HIPBLASLT_INTEGER_WIDTH_32 = 0,
    ///<64-bit integers (``int64_t``)."]
    HIPBLASLT_INTEGER_WIDTH_64 = 1,
}
#[repr(i32)]
/// Pointer mode to use for alpha."]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLtPointerMode_t {
    ///<Targets host memory."]
    HIPBLASLT_POINTER_MODE_HOST = 0,
    ///<Targets device memory."]
    HIPBLASLT_POINTER_MODE_DEVICE = 1,
    ///<Alpha pointer targets a device memory vector of length equal to the number of rows of matrix D. Beta is a single value in host memory."]
    HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST = 4,
}
#[repr(i32)]
/// Block scale mode for A and B."]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLtMatmulMatrixScale_t {
    ///<Scaling factors are single-precision scalars applied to the whole tensors (this mode is the default for ``fp8``)."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F = 0,
    ///<Not supported yet. Scaling factors are tensors that contain a dedicated scaling factor stored as an 8-bit ``HIP_R_8F_E4M3`` value for each 16-element block in the innermost dimension of the corresponding data tensor."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 = 1,
    ///<Scaling factors are tensors that contain a dedicated scaling factor stored as an 8-bit ``R_8F_UE8M0`` value for each 32-element block in the innermost dimension of the corresponding data tensor."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 = 2,
    ///<Scaling factors are single-precision vectors. This mode is only applicable to matrices A and B, in which case the vectors are expected to have M and N elements respectively, and each (i, j)-th element of product of A and B is multiplied by i-th element of A scale and j-th element of B scale."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F = 3,
    ///<Not supported yet. Scaling factors are tensors that contain a dedicated ``FP32`` scaling factor for each 128-element block in the innermost dimension of the corresponding data tensor."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F = 4,
    ///<Not supported yet. Scaling factors are tensors that contain a dedicated ``FP32`` scaling factor for each 128x128-element block in the corresponding data tensor."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F = 5,
    ///< Scaling factors are tensors that contain a dedicated 8-bit ``R_8F_UE8M0`` value for each 32-element block in the innermost dimension of the corresponding data tensor. The scale data is pre-swizzled to match the memory access pattern expected by the kernel."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT = 1001,
    ///<Not supported yet. Scaling factors are tensors that contain a dedicated scaling factor stored as an 8-bit ``R_8F_UE8M0`` value for each 16-element block in the innermost dimension of the corresponding data tensor."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE8M0_EXT = 1002,
    ///<Not supported yet. Scaling factors are tensors that contain a dedicated scaling factor stored as an 8-bit ``HIP_R_8F_E4M3`` value for each 32-element block in the innermost dimension of the corresponding data tensor."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE4M3_EXT = 1003,
    ///<Not supported yet. Scaling factors are tensors that contain a dedicated scaling factor stored as an 8-bit ``HIP_R_8F_E5M3_EXT`` value for each 16-element block in the innermost dimension of the corresponding data tensor."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE5M3_EXT = 1004,
    ///<Not supported yet. Scaling factors are tensors that contain a dedicated scaling factor stored as an 8-bit ``HIP_R_8F_E5M3_EXT`` value for each 32-element block in the innermost dimension of the corresponding data tensor."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE5M3_EXT = 1005,
    ///<Scaling factors are a dense ``[M][ceil(K/32)]`` tensor that contains a dedicated ``HIP_R_16BF`` scaling factor for each 32-element block in the K dimension of the corresponding data tensor, with the K/32 dimension innermost. Symmetric quantization only (no zero-point). Currently supported for matrix A only, with A of type ``HIP_R_4I``, and the scale type named here must equal the type of matrix B."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_EXT = 1006,
    ///<Equivalent to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_EXT`` with a block size of 64."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_16BF_EXT = 1007,
    ///<Equivalent to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_EXT`` with a block size of 128."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_16BF_EXT = 1008,
    ///<Equivalent to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_EXT`` plus int4 zero-points (asymmetric quantization). The scale pointer addresses one allocation: the ``[M][ceil(K/32)]`` scales as in ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_EXT``, then, at the next 256-byte boundary, ``ceil(M/8) * ceil(K/32)`` 32-bit words of packed int4 zero-points. Word ``(m/8)*ceil(K/32) + k/32`` holds the zero-point of row ``m`` and K element ``k`` in bits ``4*(m%8)`` to ``4*(m%8)+3``, in the domain selected by ``HIPBLASLT_MATMUL_DESC_A_INT4_ENCODING_EXT``. The dequantized weight is ``(q - z) * s``. Currently supported for matrix A only, with A of type ``HIP_R_4I``."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_ZP_EXT = 1009,
    ///<Equivalent to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_ZP_EXT`` with a block size of 64."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_16BF_ZP_EXT = 1010,
    ///<Equivalent to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_ZP_EXT`` with a block size of 128."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_16BF_ZP_EXT = 1011,
    ///<Equivalent to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_EXT`` with ``HIP_R_16F`` scales."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16F_EXT = 1012,
    ///<Equivalent to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16F_EXT`` with a block size of 64."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_16F_EXT = 1013,
    ///<Equivalent to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16F_EXT`` with a block size of 128."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_16F_EXT = 1014,
    ///<Equivalent to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16BF_ZP_EXT`` with ``HIP_R_16F`` scales."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16F_ZP_EXT = 1015,
    ///<Equivalent to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16F_ZP_EXT`` with a block size of 64."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC64_16F_ZP_EXT = 1016,
    ///<Equivalent to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_16F_ZP_EXT`` with a block size of 128."]
    HIPBLASLT_MATMUL_MATRIX_SCALE_VEC128_16F_ZP_EXT = 1017,
    HIPBLASLT_MATMUL_MATRIX_SCALE_END = 1018,
}
#[repr(i32)]
/// Encoding of the int4 weights in matrix A on the w4a16 path; independent of the
///
/// scale mode."]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLtInt4Encoding_t {
    /// Two's-complement int4 in ``[-8, 7]``, nibbles in K order: element ``2n``
    ///
    /// in the low nibble of byte ``n``. Default."]
    HIPBLASLT_INT4_ENCODING_SIGNED_EXT = 0,
    /// Unsigned int4 in ``[0, 15]``, element ``2n`` in the low nibble of byte ``n``; implicit zero-point 8 when no zero-point tensor is supplied. Matches the GPTQ / compressed-tensors ``qweight`` byte layout."]
    HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT = 1,
    /// Unsigned int4 in ``[0, 15]``, element ``2n`` in the low nibble of byte ``n``; implicit zero-point 8 when no zero-point tensor is supplied. Matches the GPTQ / compressed-tensors ``qweight`` byte layout."]
    HIPBLASLT_INT4_ENCODING_END_EXT = 2,
}
#[repr(i32)]
/// Mode values for the ``HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT``
///
/// attribute and the C++ ext ``GemmPreference::setStreamKTileSchedulingMode``.
///
/// The attribute storage stays ``int32_t``; values outside ``{0, 1, 2}`` are
///
/// rejected by the setter with ``HIPBLAS_STATUS_INVALID_VALUE``."]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLtStreamKTileSchedulingMode_t {
    ///< Default hybrid assignment policy: StaticGrid unless the CU-budget heuristic runs. When ``HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET`` is positive the library heuristic still runs per launch to pick SK4 when appropriate."]
    HIPBLASLT_STREAMK_TILE_SCHEDULING_OFF = 0,
    ///< Always request DynamicWorkQueue on kernels compiled with WorkAssignment=Hybrid."]
    HIPBLASLT_STREAMK_TILE_SCHEDULING_ON = 1,
    ///< Always let hipBLASLt's heuristic pick StaticGrid or DynamicWorkQueue for each hybrid launch."]
    HIPBLASLT_STREAMK_TILE_SCHEDULING_AUTO = 2,
}
#[repr(i32)]
/// Specifies the attributes that define the specifics of the matrix multiply operation."]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLtMatmulDescAttributes_t {
    ///<Specifies the type of transformation operation that should be performed on matrix A. Default value is ``HIPBLAS_OP_N`` (for example, non-transpose operation). See ``hipblasOperation_t``. Data type: ``int32_t``."]
    HIPBLASLT_MATMUL_DESC_TRANSA = 0,
    ///<Specifies the type of transformation operation that should be performed on matrix B. Default value is ``HIPBLAS_OP_N`` (for example, non-transpose operation). See ``hipblasOperation_t``. Data type: ``int32_t``."]
    HIPBLASLT_MATMUL_DESC_TRANSB = 1,
    ///<Epilogue function. See ``hipblasLtEpilogue_t``. Default value is ``HIPBLASLT_EPILOGUE_DEFAULT``. Data type: ``uint32_t``."]
    HIPBLASLT_MATMUL_DESC_EPILOGUE = 2,
    ///<Bias or bias gradient vector pointer in the device memory. Data type: ``void*`` / ``const void*``."]
    HIPBLASLT_MATMUL_DESC_BIAS_POINTER = 3,
    ///<Type of the bias vector in the device memory. Can be set the same as the D matrix type or Scale type. Bias case: see ``HIPBLASLT_EPILOGUE_BIAS``. Data type: ``int32_t`` based on ``hipDataType``."]
    HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 4,
    ///<Device pointer to the scale factor value that converts data in matrix A to the compute data type range. The scaling factor must have the same type as the compute type. If not specified, or set to NULL, the scaling factor is assumed to be ``1``. If set for an unsupported matrix data, scale, and compute type combination, calling hipblasLtMatmul() will return ``HIPBLAS_INVALID_VALUE``. Default value: NULL. Data type: ``void*`` ``/const void*``."]
    HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER = 5,
    ///<Equivalent to ``HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER`` for matrix B. Default value: NULL. Data type: ``void*`` / ``const void*``."]
    HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER = 6,
    ///<Equivalent to ``HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER`` for matrix C. Default value: NULL. Data type: ``void*`` / ``const void*``."]
    HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER = 7,
    ///<Equivalent to ``HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER`` for matrix D. Default value: NULL. Data type: ``void*`` / ``const void*``."]
    HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER = 8,
    ///<Not supported: no kernel applies a scale to the AUX output. Setting a non-NULL pointer returns ``HIPBLAS_STATUS_NOT_SUPPORTED``; setting NULL succeeds. Default value: NULL. Data type: ``void*`` / ``const void*``."]
    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER = 9,
    ///<Epilogue auxiliary buffer pointer in the device memory. Data type: ``void*`` / ``const void*``."]
    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 10,
    ///<The leading dimension of the epilogue auxiliary buffer pointer in the device memory. Data type: ``int64_t``."]
    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD = 11,
    ///<The batch stride of the epilogue auxiliary buffer pointer in the device memory. Data type: ``int64_t``."]
    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE = 12,
    ///<Specifies that alpha and beta are passed by reference, whether they are scalars on the host or on the device, or device vectors. Default value is: ``HIPBLASLT_POINTER_MODE_HOST`` (on the host). Data type: ``int32_t`` based on ``hipblasLtPointerMode_t``."]
    HIPBLASLT_MATMUL_DESC_POINTER_MODE = 13,
    ///<Device pointer to the memory location that on completion will be set to the maximum of the absolute values in the output matrix. Data type: ``void*`` / ``const void*``."]
    HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER = 14,
    ///<Type of the auxiliary vector in the device memory. Default value is: ``HIPBLASLT_DATATYPE_INVALID`` (using D matrix type). Data type: ``int32_t`` based on ``hipDataType``."]
    HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE = 22,
    ///<The batch stride of the bias vector pointer in the device memory. This is only applicable for hipblasltBatchMode_t is 0 (Strided Batched GEMM) and hipblasltEpilogue_t is BIAS enabled. Default value is 0 meaning same bias value broadcast across all batches. Data type: ``int32_t``."]
    HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE = 23,
    ///<Scaling mode that defines how the matrix scaling factor for matrix A is interpreted. See ``hipblasLtMatmulMatrixScale_t``."]
    HIPBLASLT_MATMUL_DESC_A_SCALE_MODE = 31,
    ///<Scaling mode that defines how the matrix scaling factor for matrix B is interpreted. See ``hipblasLtMatmulMatrixScale_t``."]
    HIPBLASLT_MATMUL_DESC_B_SCALE_MODE = 32,
    ///<Target the matmul kernel selection and persistent-grid sizing for this many compute units (CUs). Set to ``0`` (the default) to use all CUs the device exposes. Negative values are rejected with ``HIPBLAS_STATUS_INVALID_VALUE``. This is a hint to the library heuristics; the launched grid is not guaranteed to use exactly this many CUs. Data type: ``int32_t``."]
    HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET = 33,
    ///<Grouped GEMM only. ``0`` (default): one alpha for all groups. ``1``: with ``HIPBLASLT_POINTER_MODE_DEVICE``, ``alpha`` is a device array of ``groupCount`` pointers, one per group. Data type: ``int64_t``."]
    HIPBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE = 34,
    ///<Grouped GEMM only. Same as ``HIPBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE`` for ``beta``. Data type: ``int64_t``."]
    HIPBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE = 35,
    ///<Compute input A types. Defines the data type used for the input A of a matrix multiply."]
    HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT = 100,
    ///<Compute input B types. Defines the data type used for the input B of a matrix multiply."]
    HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT = 101,
    ///<First extra argument for the activation function. Data type: ``float``."]
    HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT = 102,
    ///<Second extra argument for the activation function. Data type: ``float``."]
    HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT = 103,
    ///<Select the hipBLASLt StreamK tile scheduling mode for kernels compiled with TileProcessingStrategy=StreamK and WorkAssignment=Hybrid. Provided as an ``_EXT`` attribute. Accepts values from ``hipblasLtStreamKTileSchedulingMode_t``: ``0`` (``OFF``, default) uses the SK3 static sub-path; when ``HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET`` is set to a positive value the library heuristic still runs per launch to pick SK4 when appropriate; ``1`` (``ON``) always requests the SK4 dynamic work-queue sub-path when the selected kernel supports it; ``2`` (``AUTO``) always lets the library's heuristic pick between static and dynamic per launch. Values outside ``{0, 1, 2}`` are rejected with ``HIPBLAS_STATUS_INVALID_VALUE``. Data type: ``int32_t``."]
    HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT = 104,
    ///<Request a uniform summation order across the M dimension. Provided as an ``_EXT`` attribute. When enabled, hipBLASLt guarantees that if every row of matrix A is the identical vector, every row of the output matrix D is bitwise identical. This is uniformity across the M dimension within a single run; it is **not** run-to-run determinism. ``-1`` (default) inherits the handle-level request (``hipblasLtSetUniformSummationOrder``); ``0`` disables this GEMM; ``1`` enables this GEMM. Other values are rejected with ``HIPBLAS_STATUS_INVALID_VALUE``. Enabling the mode restricts kernel selection and the launch configuration, so it can reduce performance, and hipblasLtMatmul() returns ``HIPBLAS_STATUS_INVALID_VALUE`` when no uniform-safe configuration exists for the resolved launch rather than silently producing a non-uniform result. Data type: ``int32_t``."]
    HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT = 105,
    ///<Encoding of the int4 weights in matrix A. Provided as an ``_EXT`` attribute. Accepts values from ``hipblasLtInt4Encoding_t``, default ``HIPBLASLT_INT4_ENCODING_SIGNED_EXT``; others are rejected with ``HIPBLAS_STATUS_INVALID_VALUE``. Only used when A is ``HIP_R_4I`` with a w4a16 ``VEC*_16BF``/``VEC*_16F`` scale mode. Data type: ``int32_t``."]
    HIPBLASLT_MATMUL_DESC_A_INT4_ENCODING_EXT = 107,
    HIPBLASLT_MATMUL_DESC_MAX = 108,
}
#[repr(i32)]
/// This is an enumerated type used to apply algorithm search preferences while fine-tuning the heuristic function."]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLtMatmulPreferenceAttributes_t {
    ///<Search mode. Data type: ``uint32_t``."]
    HIPBLASLT_MATMUL_PREF_SEARCH_MODE = 0,
    ///<Maximum allowed workspace memory. Default is 0 (no workspace memory allowed). Values above ``UINT32_MAX`` (4 GiB - 1), such as ``SIZE_MAX`` for no limit, are clamped to ``UINT32_MAX``, which is the value read back. Data type: ``uint64_t``."]
    HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
    ///<Bias heuristic algorithm selection toward kernels that perform well at this targeted compute-unit count. ``0`` (default) means no constraint. Negative values are rejected with ``HIPBLAS_STATUS_INVALID_VALUE``. Data type: ``int32_t``."]
    HIPBLASLT_MATMUL_PREF_SM_COUNT_TARGET = 2,
    ///<Grouped GEMM only. Expected average number of rows of D per group. ``0`` (default) means not set. Data type: ``int64_t``."]
    HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS = 3,
    ///<Grouped GEMM only. Expected average number of columns of D per group. ``0`` (default) means not set. Data type: ``int64_t``."]
    HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS = 4,
    ///<Grouped GEMM only. Expected average K per group. ``0`` (default) means not set. Data type: ``int64_t``."]
    HIPBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM = 5,
    ///<Grouped GEMM only. Upper bound on the rows of D in any group. Sizes beyond the bounds are undefined behavior. ``0`` (default) means not set. Data type: ``int64_t``."]
    HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_ROWS_EXT = 100,
    ///<Grouped GEMM only. Upper bound on the columns of D in any group. Data type: ``int64_t``."]
    HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_COLS_EXT = 101,
    ///<Grouped GEMM only. Upper bound on the sum of the rows of D over all groups. Data type: ``int64_t``."]
    HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_TOTAL_ROWS_EXT = 102,
    ///<Grouped GEMM only. Upper bound on the sum of the columns of D over all groups. Data type: ``int64_t``."]
    HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_TOTAL_COLS_EXT = 103,
    HIPBLASLT_MATMUL_PREF_MAX = 104,
}
#[repr(i32)]
/// Enumeration for data ordering."]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLtOrder_t {
    /// Column-major.
    ///
    ///
    ///
    /// Leading dimension is the stride (in elements) to the beginning of the next column in memory."]
    HIPBLASLT_ORDER_COL = 0,
    /// Row-major.
    ///
    ///
    ///
    /// Leading dimension is the stride (in elements) to the beginning of the next row in memory."]
    HIPBLASLT_ORDER_ROW = 1,
    /// Data is ordered in column-major ordered tiles of composite tiles with a total of 32 columns and 128 rows.
    ///
    /// A tile is composed of 4 inner tiles in column-major with a total of 32 rows and 128 columns.
    ///
    /// The element offset within the tile is calculated as ``row%32+32*col+(row/32)*32*32``.
    ///
    /// Note that for this order, the number of columns (rows) of the tensor has to be a multiple of 32(128) or
    ///
    /// pre-padded to 32(128)."]
    HIPBLASLT_ORDER_COL16_4R32 = 99,
    /// Data is ordered in column-major ordered tiles of composite tiles with a total of 16 columns and 64 rows.
    ///
    /// A tile is composed of 4 inner tiles in column-major with a total of 16 rows and 16 columns.
    ///
    /// The element offset within the tile is calculated as ``row%16+16*col+(row/16)*16*16``.
    ///
    /// Note that for this order, the number of columns (rows) of the tensor has to be a multiple of 16(64) or
    ///
    /// pre-padded to 16(64)."]
    HIPBLASLT_ORDER_COL16_4R16 = 100,
    /// Data is ordered in column-major ordered tiles of composite tiles with a total of 16 columns and 32 rows.
    ///
    /// A tile is composed of 4 inner tiles in column-major with a total of 8 rows and 16 columns.
    ///
    /// Element offset within the tile is calculated as ``row%8+8*col+(row/8)*16*8``.
    ///
    /// Note that for this order, the number of columns (rows) of the tensor has to be a multiple of 16(32) or
    ///
    /// pre-padded to 16(32)."]
    HIPBLASLT_ORDER_COL16_4R8 = 101,
    /// Data is ordered in column-major ordered tiles of composite tiles with a total of 16 columns and 32 rows.
    ///
    /// A tile is composed of 4 inner tiles in column-major with a total of 8 rows and 16 columns.
    ///
    /// Element offset within the tile is calculated as ``row%8+8*col+(row/8)*16*8``.
    ///
    /// Note that for this order, the number of columns (rows) of the tensor has to be a multiple of 16(32) or
    ///
    /// pre-padded to 16(32)."]
    HIPBLASLT_ORDER_COL16_4R4 = 102,
    /// Data is ordered in column-major ordered tiles of composite tiles with a total of 16 columns and 32 rows.
    ///
    /// A tile is composed of 4 inner tiles in column-major with a total of 8 rows and 16 columns.
    ///
    /// Element offset within the tile is calculated as ``row%8+8*col+(row/8)*16*8``.
    ///
    /// Note that for this order, the number of columns (rows) of the tensor has to be a multiple of 16(32) or
    ///
    /// pre-padded to 16(32)."]
    HIPBLASLT_ORDER_COL16_4R2 = 103,
}
#[repr(i32)]
/// Matrix transform descriptor attributes to define details of the operation."]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum hipblasLtMatrixTransformDescAttributes_t {
    /// Scale type, see hipDataType. Inputs are converted to scale type for scaling and summation and results are then
    ///
    /// converted to output type to store in memory.
    ///
    ///
    ///
    /// int32_t"]
    HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE = 0,
    /// Pointer mode of alpha and beta, see hipblasLtPointerMode_t.
    ///
    ///
    ///
    /// int32_t, default: HIPBLASLT_POINTER_MODE_HOST"]
    HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE = 1,
    /// Transform of matrix A, see hipblasOperation_t.
    ///
    ///
    ///
    /// int32_t, default: HIPBLAS_OP_N"]
    HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA = 2,
    /// Transform of matrix B, see hipblasOperation_t.
    ///
    ///
    ///
    /// int32_t, default: HIPBLAS_OP_N"]
    HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB = 3,
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipblasLtMatmulDescOpaque_t {
    pub data: [u64; 4usize],
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipblasLtMatrixLayoutOpaque_t {
    pub data: [u64; 4usize],
}
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipblasLtMatmulPreferenceOpaque_t {
    pub data: [u64; 5usize],
}
/// Semi-opaque descriptor for hipblasLtMatrixTransform() operation details"]
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct hipblasLtMatrixTransformDescOpaque_t {
    pub data: [u64; 8usize],
}
/// Opaque descriptor for hipblasLtMatrixTransform() operation details.
///
/// The ``hipblasLtMatrixTransformDesc_t`` is a pointer to an opaque structure holding the description of a matrix transformation operation.
///
/// hipblasLtMatrixTransformDescCreate():
///
/// To create one instance of the descriptor.
///
/// hipblasLtMatrixTransformDescDestroy():
///
/// To destroy a previously created descriptor and release the resources."]
pub type hipblasLtMatrixTransformDesc_t = *mut hipblasLtMatrixTransformDescOpaque_t;
/// Handle to the hipBLASLt library context queue.
///
/// The ``hipblasLtHandle_t`` type is a pointer type to an opaque structure holding the hipBLASLt library context.
///
/// A handle encapsulates the execution state and manages device-side resources associated with the submitted operations.
///
/// A hipBLASLt handle is not safe for concurrent use across multiple HIP streams. Applications must ensure any previously submitted work associated with a handle has completed
///
/// before reusing that handle on a different stream. For multi-stream execution, create one handle per stream.
///
/// Use the following functions to manipulate this library context:
///
/// hipblasLtCreate():
///
/// To initialize the hipBLASLt library context and return a handle to an opaque structure holding the hipBLASLt library context.
///
/// hipblasLtDestroy():
///
/// To destroy a previously created hipBLASLt library context descriptor and release the resources."]
pub type hipblasLtHandle_t = *mut ::core::ffi::c_void;
/// Descriptor of the matrix multiplication operation
///
/// This is a pointer to an opaque structure holding the description of the matrix multiplication operation hipblasLtMatmul().
///
/// Use the following functions to manipulate this descriptor:
///
/// hipblasLtMatmulDescCreate(): To create one instance of the descriptor.
///
/// hipblasLtMatmulDescDestroy(): To destroy a previously created descriptor and release the resources."]
pub type hipblasLtMatmulDesc_t = *mut hipblasLtMatmulDescOpaque_t;
/// Descriptor of the matrix layout.
///
/// This is a pointer to an opaque structure holding the description of a matrix layout.
///
/// Use the following functions to manipulate this descriptor:
///
/// hipblasLtMatrixLayoutCreate(): To create one instance of the descriptor.
///
/// hipblasLtMatrixLayoutDestroy(): To destroy a previously created descriptor and release the resources."]
pub type hipblasLtMatrixLayout_t = *mut hipblasLtMatrixLayoutOpaque_t;
/// Descriptor of the matrix multiplication preference.
///
/// This is a pointer to an opaque structure holding the description of the preferences for hipblasLtMatmulAlgoGetHeuristic() configuration.
///
/// Use the following functions to manipulate this descriptor:
///
/// hipblasLtMatmulPreferenceCreate(): To create one instance of the descriptor.
///
/// hipblasLtMatmulPreferenceDestroy(): To destroy a previously created descriptor and release the resources."]
pub type hipblasLtMatmulPreference_t = *mut hipblasLtMatmulPreferenceOpaque_t;
/// Description of the matrix multiplication algorithm.
///
/// This is an opaque structure holding the description of the matrix multiplication algorithm.
///
/// This structure can be trivially serialized and later restored for use with the same version of the hipBLASLt library to save compute time when selecting the right configuration again."]
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct _hipblasLtMatmulAlgo_t {
    pub data: [u8; 16usize],
    pub max_workspace_bytes: usize,
}
/// Description of the matrix multiplication algorithm.
///
/// This is an opaque structure holding the description of the matrix multiplication algorithm.
///
/// This structure can be trivially serialized and later restored for use with the same version of the hipBLASLt library to save compute time when selecting the right configuration again."]
pub type hipblasLtMatmulAlgo_t = _hipblasLtMatmulAlgo_t;
/// Description of the matrix multiplication algorithm.
///
/// This is a descriptor that holds the configured matrix multiplication algorithm descriptor and its runtime properties.
///
/// This structure can be trivially serialized and later restored for use with the same version of the hipBLASLt library to save compute time when selecting the right configuration again.
///
/// @param algo hipblasLtMatmulAlgo_t struct.
///
/// @param workspaceSize Actual size of workspace memory required.
///
/// @param state Result status. The other fields are valid only if, after a call to hipblasLtMatmulAlgoGetHeuristic(), this member is set to ``HIPBLAS_STATUS_SUCCESS``."]
#[repr(C)]
#[cfg_attr(feature = "stable_auto", derive(stable_auto::Auto))]
#[derive(Debug, Copy, Clone)]
pub struct _hipblasLtMatmulHeuristicResult_t {
    ///<Algo struct"]
    pub algo: hipblasLtMatmulAlgo_t,
    ///<Actual size of workspace memory required."]
    pub workspaceSize: usize,
    ///<Result status. Other fields are valid only if, after call to hipblasLtMatmulAlgoGetHeuristic(), this member is set to HIPBLAS_STATUS_SUCCESS.."]
    pub state: crate::hipblas::hipblasStatus_t,
    ///<Waves count is a device utilization metric. A wavesCount value of 1.0f suggests that when the kernel is launched it will fully occupy the GPU."]
    pub wavesCount: f32,
    ///<Reserved."]
    pub reserved: [::core::ffi::c_int; 4usize],
}
/// Description of the matrix multiplication algorithm.
///
/// This is a descriptor that holds the configured matrix multiplication algorithm descriptor and its runtime properties.
///
/// This structure can be trivially serialized and later restored for use with the same version of the hipBLASLt library to save compute time when selecting the right configuration again.
///
/// @param algo hipblasLtMatmulAlgo_t struct.
///
/// @param workspaceSize Actual size of workspace memory required.
///
/// @param state Result status. The other fields are valid only if, after a call to hipblasLtMatmulAlgoGetHeuristic(), this member is set to ``HIPBLAS_STATUS_SUCCESS``."]
pub type hipblasLtMatmulHeuristicResult_t = _hipblasLtMatmulHeuristicResult_t;

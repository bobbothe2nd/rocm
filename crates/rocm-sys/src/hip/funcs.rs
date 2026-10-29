use super::types::*;

use core::ffi::*;

#[cfg(feature = "dynamic-loading")]
use crate::version::{ROCM_MAJOR as MAJOR, ROCM_MINOR as MINOR};

link! {
    amdhip64 : [7, 8, 9, 10] : rocm_ver;

    hipError_t hipInit(c_uint flags);

    hipError_t hipDriverGetVersion(*mut c_int driverVersion);

    hipError_t hipRuntimeGetVersion(*mut c_int runtimeVersion);

    hipError_t hipDeviceGet(
        *mut hipDevice_t device,
        c_int ordinal,
    );

    hipError_t hipDeviceComputeCapability(
        *mut c_int major,
        *mut c_int minor,
        hipDevice_t device,
    );

    hipError_t hipDeviceGetName(
        *mut c_char name,
        *mut c_int len,
        hipDevice_t device,
    );

    #[since = 5 .1, "5.1"]
    hipError_t hipDeviceGetUuid(
        *mut hipUUID uuid,
        hipDevice_t device,
    );

    #[since = 5 .1, "5.1"]
    hipError_t hipDeviceGetLuid(
        *mut c_char luid,
        *mut c_uint deviceNodeMask,
        hipDevice_t device,
    );

    hipError_t hipDeviceGetP2PAttribute(
        *mut c_int value,
        hipDeviceP2PAttr attr,
        c_int srcDevice,
        c_int dstDevice,
    );

    hipError_t hipDeviceGetPCIBusId(
        *mut c_char pciBusId,
        c_int len,
        c_int device,
    );

    hipError_t hipDeviceGetByPCIBusId(
        *mut c_int device,
        *const c_char pciBusId
    );

    hipError_t hipDeviceTotalMem(
        *mut usize bytes,
        hipDevice_t device,
    );

    hipError_t hipDeviceSynchronize();

    hipError_t hipDeviceReset();

    hipError_t hipSetDevice(c_int deviceId);

    #[since = 6 .0, "6.0"]
    hipError_t hipSetValidDevices(
        *mut c_int device_arr,
        c_int len,
    );

    hipError_t hipGetDevice(*mut c_int device);

    hipError_t hipGetDeviceCount(*mut c_int count);

    hipError_t hipDeviceGetAttribute(
        *mut c_int pi,
        hipDeviceAttribute_t attr,
        c_int deviceId,
    );

    #[since = 5 .1, "5.1"]
    hipError_t hipDeviceGetDefaultMemPool(
        *mut hipMemPool_t mem_pool,
        c_int device,
    );

    #[since = 5 .1, "5.1"]
    hipError_t hipDeviceSetMemPool(
        c_int device,
        hipMemPool_t mem_pool,
    );

    #[since = 5 .1, "5.1"]
    hipError_t hipDeviceGetMemPool(
        c_int device,
        *mut hipMemPool_t mem_pool,
    );

    /// assumes hipDeviceProp_tR0600=hipDeviceProp_t
    hipError_t hipGetDeviceProperties(
        *mut hipDeviceProp_tR0600 prop,
        c_int deviceId,
    );

    #[since = 5 .0, "5.0"]
    hipError_t hipGetDevicePropertiesR0600(
        *mut hipDeviceProp_tR0600 prop,
        c_int deviceId,
    );

    hipError_t hipDeviceGetTexture1DLinearMaxWidth(
        *mut usize max_width,
        *const hipChannelFormatDesc dex,
        c_int device,
    );

    hipError_t hipDeviceSetCacheConfig(
        hipFuncCache_t cacheConfig,
    );

    hipError_t hipDeviceGetCacheConfig(
        *mut hipFuncCache_t cacheConfig,
    );

    hipError_t hipDeviceGetLimit(
        *mut usize pValue,
        hipLimit_t limit,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipDeviceSetLimit(
        hipLimit_t limit,
        usize pValue,
    );

    hipError_t hipDeviceGetSharedMemConfig(
        *mut hipSharedMemConfig pConfig,
    );

    hipError_t hipGetDeviceFlags(
        *mut c_uint flags,
    );

    hipError_t hipDeviceSetSharedMemConfig(
        hipSharedMemConfig config,
    );

    hipError_t hipSetDeviceFlags(
        c_uint flags,
    );

    hipError_t hipInitDevice(
        c_int device,
        c_uint deviceFlags,
        c_uint flags,
    );

    /// assumes hipDeviceProp_tR0600=hipDeviceProp_t
    hipError_t hipChooseDevice(
        *mut c_int device,
        *const hipDeviceProp_tR0600 prop,
    );

    hipError_t hipExtGetLinkTypeAndHopCount(
        c_int device1,
        c_int device2,
        *mut u32 linktype,
        u32 hopcount,
    );

    hipError_t hipIpcGetMemHandle(
        *mut hipIpcMemHandle_t handle,
        *mut c_void devPtr,
    );

    hipError_t hipIpcOpenMemHandle(
        *mut *mut c_void devPtr,
        hipIpcMemHandle_t handle,
        c_uint flags,
    );

    hipError_t hipIpcCloseMemHandle(
        *mut c_void devPtr,
    );

    hipError_t hipIpcGetEventHandle(
        *mut hipIpcEventHandle_t handle,
        hipEvent_t event,
    );

    hipError_t hipIpcOpenEventHandle(
        *mut hipEvent_t event,
        hipIpcEventHandle_t handle,
    );

    hipError_t hipExtModuleLaunchKernel(
        hipFunction_t f,
        u32 globalWorkSizeX,
        u32 globalWorkSizeY,
        u32 globalWorkSizeZ,
        u32 localWorkSizeX,
        u32 localWorkSizeY,
        u32 localWorkSizeZ,
        usize sharedMemBytes,
        hipStream_t hStream,
        *mut *mut c_void kernelParams,
        *mut *mut c_void extra,
        hipEvent_t startEvent,
        hipEvent_t stopEvent,
        u32 flags,
    );

    hipError_t hipHccModuleLaunchKernel(
        hipFunction_t f,
        u32 globalWorkSizeX,
        u32 globalWorkSizeY,
        u32 globalWorkSizeZ,
        u32 localWorkSizeX,
        u32 localWorkSizeY,
        u32 localWorkSizeZ,
        usize sharedMemBytes,
        hipStream_t hStream,
        *mut *mut c_void kernelParams,
        *mut *mut c_void extra,
        hipEvent_t startEvent,
        hipEvent_t stopEvent,
    );

    hipError_t hipFuncSetAttribute(
        *const c_void func,
        hipFuncAttribute attr,
        c_int value,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipKernelSetAttribute(
        hipFunction_attribute attrib,
        c_int value,
        hipKernel_t kernel,
        hipDevice_t dev,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipKernelGetFunction(
        *mut hipFunction_t pFunc,
        hipKernel_t kernel,
    );

    hipError_t hipFuncSetCacheConfig(
        *const c_void func,
        hipFuncCache_t config,
    );

    hipError_t hipFuncSetSharedMemConfig(
        *const c_void func,
        hipSharedMemConfig config,
    );

    hipError_t hipFuncGetAttributes(
        *mut hipFuncAttributes attr,
        *const c_void func,
    );

    hipError_t hipFuncGetAttribute(
        *mut c_int value,
        hipFunction_attribute attrib,
        hipFunction_t hfunc,
    );

    hipError_t hipModuleLaunchKernel(
        hipFunction_t f,
        c_uint gridDimX,
        c_uint gridDimY,
        c_uint gridDimZ,
        c_uint blockDimX,
        c_uint blockDimY,
        c_uint blockDimZ,
        c_uint sharedMemBytes,
        hipStream_t stream,
        *mut *mut c_void kernelParams,
        *mut *mut c_void extra,
    );

    hipError_t hipExtLaunchMultiKernelMultiDevice(
        *mut hipLaunchParams launchParamsList,
        c_int numDevices,
        c_uint flags,
    );

    #[since = 6 .5, "6.5"]
    hipError_t hipLaunchKernelExC(
        *const hipLaunchConfig_t config,
        *const c_void fPtr,
        *mut *mut c_void args,
    );

    #[since = 6 .5, "6.5"]
    hipError_t hipDrvLaunchKernelEx(
        *const HIP_LAUNCH_CONFIG config,
        hipFunction_t f,
        *mut *mut c_void params,
        *mut *mut c_void extra,
    );

    hipError_t hipGetLastError();

    #[since = 6 .0, "6.0"]
    hipError_t hipExtGetLastError();

    *const c_char hipGetErrorName(hipError_t hip_error);

    *const c_char hipGetErrorString(hipError_t hip_error);

    #[since = 5 .3, "5.3"]
    hipError_t hipDrvGetErrorName(
        hipError_t hip_Error,
        *mut *const c_char errorString,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipDrvGetErrorString(
        hipError_t hip_Error,
        *mut *const c_char errorString,
    );

    hipError_t hipStreamCreate(
        *mut hipStream_t stream,
    );

    hipError_t hipStreamCreateWithFlags(
        *mut hipStream_t stream,
        c_uint flags,
    );

    hipError_t hipStreamCreateWithPriority(
        *mut hipStream_t stream,
        c_uint flags,
        c_int priority,
    );

    hipError_t hipDeviceGetStreamPriorityRange(
        *mut c_int leastPriority,
        *mut c_int greatestPriority,
    );

    hipError_t hipStreamDestroy(hipStream_t stream);

    hipError_t hipStreamQuery(hipStream_t stream);

    hipError_t hipStreamSynchronize(hipStream_t stream);

    hipError_t hipStreamWaitEvent(
        hipStream_t stream,
        hipEvent_t event,
        c_uint flags,
    );

    #[since = 5 .2, "5.2"]
    hipError_t hipStreamGetFlags(
        hipStream_t stream,
        *mut c_uint flags,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipStreamGetId(
        hipStream_t stream,
        *mut c_ulonglong streamId,
    );

    hipError_t hipStreamGetPriority(
        hipStream_t stream,
        *mut c_int priority,
    );

    hipError_t hipStreamGetDevice(
        hipStream_t stream,
        *mut hipDevice_t device,
    );

    hipError_t hipExtStreamCreateWithCUMask(
        *mut hipStream_t stream,
        u32 cuMaskSize,
        *const u32 cuMask,
    );

    hipError_t hipExtStreamGetCUMask(
        hipStream_t stream,
        u32 cuMaskSize,
        *mut u32 cuMask,
    );

    hipError_t hipStreamAddCallback(
        hipStream_t stream,
        hipStreamCallback_t callback,
        *mut c_void userData,
        c_uint flags,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipStreamSetAttribute(
        hipStream_t stream,
        hipLaunchAttributeID attr,
        *const hipLaunchAttributeValue value,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipStreamGetAttribute(
        hipStream_t stream,
        hipLaunchAttributeID attr,
        *mut hipLaunchAttributeValue value_out,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipStreamCopyAttributes(
        hipStream_t dst,
        hipStream_t src,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipStreamWaitValue32(
        hipStream_t stream,
        *mut c_void ptr,
        u32 value,
        c_uint flags,
        u32 mask,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipStreamWaitValue64(
        hipStream_t stream,
        *mut c_void ptr,
        u64 value,
        c_uint flags,
        u64 mask,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipStreamWriteValue32(
        hipStream_t stream,
        *mut c_void ptr,
        u32 value,
        c_uint flags,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipStreamWriteValue64(
        hipStream_t stream,
        *mut c_void ptr,
        u64 value,
        c_int flags,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipStreamBatchMemOp(
        hipStream_t stream,
        c_int count,
        *mut hipStreamBatchMemOpParams paramArray,
        c_int flags,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipGraphAddBatchMemOpNode(
        *mut hipGraphNode_t phGraphNode,
        hipGraph_t hGraph,
        *const hipGraphNode_t dependencies,
        usize numDependencies,
        *const hipBatchMemOpNodeParams nodeParams,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipGraphBatchMemOpNodeGetParams(
        hipGraphNode_t hNode,
        *mut hipBatchMemOpNodeParams nodeParams_out,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipGraphBatchMemOpNodeSetParams(
        hipGraphNode_t hNode,
        *mut hipBatchMemOpNodeParams nodeParams,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipGraphExecBatchMemOpNodeSetParams(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t hNode,
        *const hipBatchMemOpNodeParams nodeParams,
    );

    hipError_t hipEventCreateWithFlags(
        *mut hipEvent_t event,
        c_uint flags,
    );

    hipError_t hipEventCreate(*mut hipEvent_t event);

    #[since = 6 .4, "6.4"]
    hipError_t hipEventRecordWithFlags(
        hipEvent_t event,
        hipStream_t stream,
        c_uint flags,
    );

    hipError_t hipEventRecord(
        hipEvent_t event,
        hipStream_t stream,
    );

    hipError_t hipEventDestroy(hipEvent_t event);

    hipError_t hipEventSynchronize(hipEvent_t event);

    hipError_t hipEventElapsedTime(
        *mut c_float ms,
        hipEvent_t start,
        hipEvent_t stop,
    );

    hipError_t hipEventQuery(hipEvent_t event);

    hipError_t hipPointerSetAttribute(
        *const c_void value,
        hipPointer_attribute attribute,
        hipDeviceptr_t ptr,
    );

    #[since = 5 .0, "5.0"]
    hipError_t hipPointerGetAttributes(
        *mut hipPointerAttribute_t attributes,
        *const c_void ptr,
    );

    hipError_t hipPointerGetAttribute(
        *mut c_void data,
        hipPointer_attribute attribute,
        hipDeviceptr_t ptr,
    );

    #[since = 5 .0, "5.0"]
    hipError_t hipDrvPointerGetAttributes(
        c_uint numAttributes,
        *mut hipPointer_attribute attributes,
        *mut *mut c_void data,
        hipDeviceptr_t ptr,
    );

    hipError_t hipMalloc(
        *mut *mut c_void ptr,
        usize size,
    );

    hipError_t hipExtMallocWithFlags(
        *mut *mut c_void ptr,
        usize sizeBytes,
        c_uint flags,
    );

    hipError_t hipHostMalloc(
        *mut *mut c_void ptr,
        usize size,
        c_uint flags,
    );

    hipError_t hipHostAlloc(
        *mut *mut c_void ptr,
        usize size,
        c_uint flags,
    );

    hipError_t hipHostGetDevicePointer(
        *mut *mut c_void devPtr,
        *mut c_void hstPtr,
        c_uint flags,
    );

    hipError_t hipHostGetFlags(
         *mut c_uint flagsPtr,
        *mut c_void hostPtr,
    );

    hipError_t hipHostRegister(
        *mut c_void hostPtr,
        usize sizeBytes,
        c_uint flags,
    );

    hipError_t hipHostUnregister(*mut c_void hostPtr);

    hipError_t hipMallocPitch(
        *mut *mut c_void ptr,
        *mut usize pitch,
        usize width,
        usize height,
    );

    hipError_t hipMemAllocPitch(
        *mut hipDeviceptr_t dptr,
        *mut usize pitch,
        usize widthInBytes,
        usize height,
        c_uint elementSizeBytes,
    );

    hipError_t hipFree(
        *mut c_void ptr,
    );

    hipError_t hipFreeHost(
        *mut c_void ptr,
    );

    hipError_t hipMemcpy(
        *mut c_void dst,
        *const c_void src,
        usize size,
        hipMemcpyKind kind,
    );

    hipError_t hipMemcpyWithStream(
        *mut c_void dst,
        *const c_void src,
        usize sizeBytes,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    hipError_t hipMemcpyHtoD(
        hipDeviceptr_t dst,
        *const c_void src,
        usize sizeBytes,
    );

    hipError_t hipMemcpyDtoH(
        *mut c_void dst,
        hipDeviceptr_t src,
        usize sizeBytes,
    );

    hipError_t hipMemcpyDtoD(
        hipDeviceptr_t dst,
        hipDeviceptr_t src,
        usize sizeBytes,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipMemcpyAtoD(
        hipDeviceptr_t dstDevice,
        hipArray_t srcArray,
        usize srcOffset,
        usize ByteCount,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipMemcpyDtoA(
        hipArray_t dstArray,
        usize dstOffset,
        hipDeviceptr_t srcDevice,
        usize ByteCount,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipMemcpyAtoA(
        hipArray_t dstArray,
        usize dstOffset,
        hipArray_t srcArray,
        usize srcOffset,
        usize ByteCount,
    );

    hipError_t hipMemcpyHtoDAsync(
        hipDeviceptr_t dst,
        *const c_void src,
        usize sizeBytes,
        hipStream_t stream,
    );

    hipError_t hipMemcpyDtoHAsync(
        *mut c_void dst,
        hipDeviceptr_t src,
        usize sizeBytes,
        hipStream_t stream,
    );

    hipError_t hipMemcpyDtoDAsync(
        hipDeviceptr_t dst,
        hipDeviceptr_t src,
        usize sizeBytes,
        hipStream_t stream,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipMemcpyAtoHAsync(
        *mut c_void dstHost,
        hipArray_t srcArray,
        usize srcOffset,
        usize ByteCount,
        hipStream_t stream,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipMemcpyHtoAAsync(
        hipArray_t dstArray,
        usize dstOffset,
        *const c_void srcHost,
        usize ByteCount,
        hipStream_t stream,
    );

    hipError_t hipGetSymbolAddress(
        *mut *mut c_void devPtr,
        *const c_void symbol,
    );

    hipError_t hipGetSymbolSize(
        *mut usize size,
        *const c_void symbol,
    );

    #[since = 6 .1, "6.1"]
    hipError_t hipGetProcAddress(
        *const char symbol,
        *mut *mut c_void pfn,
        c_int hipVersion,
        u64 flags,
        hipDriverProcAddressQueryResult symbolStatus,
    );

    hipError_t hipMemcpyToSymbol(
        *const c_void symbol,
        *const c_void src,
        usize sizeBytes,
        usize offset,
        hipMemcpyKind kind,
    );

    hipError_t hipMemcpyToSymbolAsync(
        *const c_void symbol,
        *const c_void src,
        usize sizeBytes,
        usize offset,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    hipError_t hipMemcpyFromSymbol(
        *mut c_void dst,
        *const c_void symbol,
        usize sizeBytes,
        usize offset,
        hipMemcpyKind kind,
    );

    hipError_t hipMemcpyFromSymbolAsync(
        *mut c_void dst,
        *const c_void symbol,
        usize sizeBytes,
        usize offset,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    hipError_t hipMemcpyAsync(
        *mut c_void dst,
        *const c_void src,
        usize sizeBytes,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    hipError_t hipMemset(
        *mut c_void dst,
        c_int value,
        usize sizeBytes,
    );

    hipError_t hipMemsetAsync(
        *mut c_void dst,
        c_int value,
        usize sizeBytes,
        hipStream_t stream,
    );

    hipError_t hipMemsetD8(
        hipDeviceptr_t dest,
        c_uchar value,
        usize count,
    );

    hipError_t hipMemsetD8Async(
        hipDeviceptr_t dest,
        c_uchar value,
        usize count,
        hipStream_t stream,
    );

    hipError_t hipMemsetD16(
        hipDeviceptr_t dest,
        c_ushort value,
        usize count,
    );

    hipError_t hipMemsetD16Async(
        hipDeviceptr_t dest,
        c_ushort value,
        usize count,
        hipStream_t stream,
    );

    hipError_t hipMemsetD32(
        hipDeviceptr_t dest,
        c_int value,
        usize count,
    );

    hipError_t hipMemsetD32Async(
        hipDeviceptr_t dest,
        c_int value,
        usize count,
    );

    hipError_t hipMemset2D(
        *mut c_void dst,
        usize pitch,
        c_int value,
        usize width,
        usize height,
    );

    hipError_t hipMemset2DAsync(
        *mut c_void dst,
        usize pitch,
        c_int value,
        usize width,
        usize height,
        hipStream_t stream,
    );

    hipError_t hipMemset3D(
        hipPitchedPtr pitchedDevPtr,
        c_int value,
        hipExtent extent,
    );

    hipError_t hipMemset3DAsync(
        hipPitchedPtr pitchedDevPtr,
        c_int value,
        hipExtent extent,
        hipStream_t stream,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemsetD2D8(
        hipDeviceptr_t dst,
        usize dstPitch,
        c_uchar value,
        usize width,
        usize height,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemsetD2D8Async(
        hipDeviceptr_t dst,
        usize dstPitch,
        c_uchar value,
        usize width,
        usize height,
        hipStream_t stream,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemsetD2D16(
        hipDeviceptr_t dst,
        usize dstPitch,
        c_ushort value,
        usize width,
        usize height,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemsetD2D16Async(
        hipDeviceptr_t dst,
        usize dstPitch,
        c_ushort value,
        usize width,
        usize height,
        hipStream_t stream,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemsetD2D32(
        hipDeviceptr_t dst,
        usize dstPitch,
        c_uint value,
        usize width,
        usize height,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemsetD2D32Async(
        hipDeviceptr_t dst,
        usize dstPitch,
        c_uint value,
        usize width,
        usize height,
        hipStream_t stream,
    );

    hipError_t hipMemGetInfo(
        *mut usize free,
        *mut usize total,
    );

    hipError_t hipMemPtrGetInfo(
        *mut c_void ptr,
        *mut usize size,
    );

    hipError_t hipMallocArray(
        *mut hipArray_t array,
        *const hipChannelFormatDesc desc,
        usize width,
        usize height,
        c_uint flags,
    );

    hipError_t hipArrayCreate(
        *mut hipArray_t pHandle,
        *const HIP_ARRAY_DESCRIPTOR pAllocateArray,
    );

    hipError_t hipArrayDestroy(hipArray_t array);

    hipError_t hipArray3DCreate(
        *mut hipArray_t array,
        *const HIP_ARRAY3D_DESCRIPTOR pAllocateArray,
    );

    hipError_t hipMalloc3D(
        *const hipPitchedPtr pitchedDevPtr,
        hipExtent extent,
    );

    hipError_t hipFreeArray(hipArray_t array);

    hipError_t hipMalloc3DArray(
        *mut hipArray_t array,
        *const hipChannelFormatDesc desc,
        hipExtent extent,
        c_uint flags,
    );

    #[since = 5 .6, "5.6"]
    hipError_t hipArrayGetInfo(
        *mut hipChannelFormatDesc desc,
        *mut hipExtent extent,
        *mut c_uint flags,
        hipArray_t array,
    );

    #[since = 5 .6, "5.6"]
    hipError_t hipArrayGetDescriptor(
        *mut HIP_ARRAY_DESCRIPTOR pArrayDescriptor,
        hipArray_t array,
    );

    #[since = 5 .6, "5.6"]
    hipError_t hipArray3DGetDescriptor(
        *mut HIP_ARRAY3D_DESCRIPTOR pArrayDescriptor,
        hipArray_t array,
    );

    hipError_t hipMemcpy2D(
        *mut c_void dst,
        usize dpitch,
        *const c_void src,
        usize spitch,
        usize width,
        usize height,
        hipMemcpyKind kind,
    );

    hipError_t hipMemcpyParam2D(*const hip_Memcpy2D pCopy);

    hipError_t hipMemcpyParam2DAsync(
        *const hip_Memcpy2D pCopy,
        hipStream_t stream,
    );

    hipError_t hipMemcpy2DAsync(
        *mut c_void dst,
        usize dpitch,
        *const c_void src,
        usize spitch,
        usize width,
        usize height,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    hipError_t hipMemcpy2DToArray(
        hipArray_t dst,
        usize wOffset,
        usize hOffset,
        *const c_void src,
        usize spitch,
        usize width,
        usize height,
        hipMemcpyKind kind,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipMemcpy2DToArrayAsync(
        hipArray_t dst,
        usize wOffset,
        usize hOffset,
        *const c_void src,
        usize spitch,
        usize width,
        usize height,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipMemcpy2DArrayToArray(
        hipArray_t dst,
        usize wOffsetDst,
        usize hOffsetDst,
        hipArray_const_t src,
        usize wOffsetSrc,
        usize hOffsetSrc,
        usize width,
        usize height,
        hipMemcpyKind kind,
    );

    hipError_t hipMemcpy2DFromArray(
        *mut c_void dst,
        usize dpitch,
        hipArray_const_t src,
        usize wOffset,
        usize hOffset,
        usize width,
        usize height,
        hipMemcpyKind kind,
    );

    hipError_t hipMemcpy2DFromArrayAsync(
        *mut c_void dst,
        usize dpitch,
        hipArray_const_t src,
        usize wOffset,
        usize hOffset,
        usize width,
        usize height,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    hipError_t hipMemcpyAtoH(
        *mut c_void dst,
        hipArray_t srcArray,
        usize srcOffset,
        usize count,
    );

    hipError_t hipMemcpyHtoA(
        hipArray_t dstArray,
        usize dstOffset,
        *const c_void srcHost,
        usize count,
    );

    hipError_t hipMemcpy3D(*const hipMemcpy3DParms p);

    hipError_t hipMemcpy3DAsync(
        *const hipMemcpy3DParms p,
        hipStream_t stream,
    );

    hipError_t hipDrvMemcpy3D(*const HIP_MEMCPY3D pCopy);

    hipError_t hipDrvMemcpy3DAsync(
        *const HIP_MEMCPY3D pCopy,
        hipStream_t stream,
    );

    hipError_t hipMemGetAddressRange(
        *mut hipDeviceptr_t pbase,
        *mut usize psize,
        hipDeviceptr_t dptr,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemcpyBatchAsync(
        *mut *mut c_void dsts,
        *mut *mut c_void srcs,
        *mut usize sizes,
        usize count,
        *mut hipMemcpyAttributes atttrs,
        *mut usize attrsIdxs,
        usize numAttrs,
        *mut usize failIdx,
        hipStream_t stream,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemcpy3DBatchAsync(
        usize numOps,
        *mut hipMemcpy3DBatchOp opList,
        *mut usize failIdx,
        c_ulonglong flags,
        hipStream_t stream,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemcpy3DPeer(*mut hipMemcpy3DPeerParms p);

    #[since = 7 .1, "7.1"]
    hipError_t hipMemcpy3DPeerAsync(
        *mut hipMemcpy3DPeerParms p,
        hipStream_t stream,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipMipmappedArrayGetMemoryRequirements(
        *mut hipArrayMemoryRequirements memoryRequirements,
        hipMipmappedArray_t mipmap,
        hipDevice_t device,
    );

    #[deprecated = "This API is deprecated, use hipHostMalloc() instead", cfg]
    hipError_t hipMallocHost(
        *mut *mut c_void ptr,
        usize size,
    );

    #[deprecated = "This API is deprecated, use hipHostMalloc() instead", cfg]
    hipError_t hipMemAllocHost(
        *mut *mut c_void ptr,
        usize size,
    );

    hipError_t hipHostFree(*mut c_void ptr);

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipMemcpyToArray(
        hipArray_t dst,
        usize wOffset,
        usize hOffset,
        *const c_void src,
        usize count,
        hipMemcpyKind kind,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipMemcpyFromArray(
        *mut c_void dst,
        hipArray_const_t srcArray,
        usize wOffset,
        usize hOffset,
        usize count,
        hipMemcpyKind kind,
    );

    /// This API is currently not supported on Linux.
    #[since = 4 .3, "4.3"]
    hipError_t hipImportExternalSemaphore(
        *mut hipExternalSemaphore_t extSem_out,
        *const hipExternalSemaphoreHandleDesc semHandleDesc,
    );

    /// This API is currently not supported on Linux.
    #[since = 4 .3, "4.3"]
    hipError_t hipSignalExternalSemaphoresAsync(
        *const hipExternalSemaphore_t extSemArray,
        *const hipExternalSemaphoreSignalParams paramsArray,
        c_uint numExtSems,
        hipStream_t stream,
    );

    /// This API is currently not supported on Linux.
    hipError_t hipWaitExternalSemaphoresAsync(
        *const hipExternalSemaphore_t extSemArray,
        *const hipExternalSemaphoreWaitParams paramsArray,
        c_uint numExtSems,
        hipStream_t stream,
    );

    /// This API is currently not supported on Linux.
    #[since = 4 .3, "4.3"]
    hipError_t hipDestroyExternalSemaphore(hipExternalSemaphore_t extSem);

    #[since = 4 .3, "4.3"]
    hipError_t hipImportExternalMemory(
        *mut hipExternalMemory_t extMem_out,
        *const hipExternalMemoryHandleDesc memHandleDesc,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipExternalMemoryGetMappedBuffer(
        *mut *mut c_void devPtr,
        hipExternalMemory_t extMem,
        *const hipExternalMemoryBufferDesc bufferDesc,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipDestroyExternalMemory(hipExternalMemory_t extMem);

    hipError_t hipExternalMemoryGetMappedMipmappedArray(
        *mut hipMipmappedArray_t mipmap,
        hipExternalMemory_t extMem,
        *const hipExternalMemoryMipmappedArrayDesc mipmapDesc,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMallocAsync(
        *mut *mut c_void dev_ptr,
        usize size,
        hipMemPool_t mem_pool,
        hipStream_t stream,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipFreeAsync(
        *mut c_void dev_ptr,
        hipStream_t stream,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolTrimTo(
        hipMemPool_t mem_pool,
        usize min_bytes_to_hold,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolSetAttribute(
        hipMemPool_t mem_pool,
        hipMemPoolAttr attr,
        *mut c_void value,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolGetAttribute(
        hipMemPool_t mem_pool,
        hipMemPoolAttr attr,
        *mut c_void value,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolSetAccess(
        hipMemPool_t mem_pool,
        *const hipMemAccessDesc desc_list,
        usize count,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolGetAccess(
        *mut hipMemAccessFlags flags,
        hipMemPool_t mem_pool,
        *mut hipMemLocation location,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolCreate(
        *mut hipMemPool_t mem_pool,
        *const hipMemPoolProps pool_props,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolDestroy(hipMemPool_t mem_pool);

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMallocFromPoolAsync(
        *mut *mut c_void dev_ptr,
        usize size,
        hipMemPool_t mem_pool,
        hipStream_t stream,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolExportToShareableHandle(
        *mut c_void shared_handle,
        hipMemPool_t mem_pool,
        hipMemAllocationHandleType handle_type,
        c_uint flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolImportFromShareableHandle(
        *mut hipMemPool_t mem_pool,
        *mut c_void shared_handle,
        hipMemAllocationHandleType handle_type,
        c_uint flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolExportPointer(
        *mut hipMemPoolPtrExportData export_data,
        *mut c_void dev_ptr,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolImportPointer(
        *mut *mut c_void dev_ptr,
        hipMemPool_t mem_pool,
        *mut hipMemPoolPtrExportData export_data,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipMemSetMemPool(
        *mut hipMemLocation location,
        hipMemAllocationType ty,
        hipMemPool_t pool,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipMemGetMemPool(
        *mut hipMemPool_t pool,
        *mut hipMemLocation location,
        hipMemAllocationType ty,
    );

    #[since = 7 .15, "7.15"]
    hipError_t hipMemGetDefaultMemPool(
        *mut hipMemPool_t memPool,
        *mut hipMemLocation location,
        hipMemAllocationType ty,
    );

    /// It is recommend to do the capability check before call this API.
    hipError_t hipMallocManaged(
        *mut *mut c_void dev_ptr,
        usize size,
        c_uint flags,
    );

    /// It is recommend to do the capability check before call this API.
    hipError_t hipMemPrefetchAsync(
        *const c_void dev_ptr,
        usize count,
        c_int device,
        hipStream_t stream,
    );

    /// It is recommend to do the capability check before call this API.
    #[since = 7 .1, "7.1"]
    hipError_t hipMemPrefetchAsync_v2(
        *const c_void dev_ptr,
        usize count,
        hipMemLocation location,
        c_uint flags,
        hipStream_t stream,
    );

    /// It is recommend to do the capability check before call this API.
    #[since = 7 .2, "7.2"]
    hipError_t hipMemPrefetchBatchAsync(
        *mut *mut c_void dev_ptrs,
        *mut usize sizes,
        usize count,
        *mut hipMemLocation prefetch_locs,
        *mut usize prefetch_loc_idxs,
        usize num_prefetch_locs,
        c_ulonglong flags,
        hipStream_t stream,
    );

    /// All memory ranges must be managed memory allocated via hipMallocManaged or system-allocated memory (if device supports pageable memory access).
    ///
    /// This API is implemented on Linux and requires XNACK to be enabled.
    ///
    /// This API is marked as beta, meaning, while this is feature complete, it is still open to changes and may have outstanding issues.
    ///
    /// Reading from a discarded range without first writing or prefetching to it will return an indeterminate value.
    ///
    /// Concurrent reads, writes, or prefetches to discarded ranges result in undefined behavior.
    #[since = 7 .2, "7.2"]
    hipError_t hipMemDiscardBatchAsync(
        *mut *mut c_void dev_ptrs,
        *mut usize sizes,
        usize count,
        c_ulonglong flags,
        hipStream_t stream,
    );

    /// This is the driver API variant that uses hipDeviceptr_t instead of c_void*. Both hipMemDiscardBatchAsync and hipDrvMemDiscardBatchAsync use the same internal implementation.
    ///
    /// Reading from a discarded range without first writing or prefetching to it will return an indeterminate value.
    #[since = 7 .2, "7.2"]
    hipError_t hipDrvMemDiscardBatchAsync(
        *mut hipDeviceptr_t dptrs,
        *mut usize sizes,
        usize count,
        c_ulonglong flags,
        hipStream_t stream,
    );

    /// All memory ranges must be managed memory allocated via hipMallocManaged or system-allocated memory (if device supports pageable memory access).
    ///
    /// This API is implemented on Linux and requires XNACK to be enabled.
    ///
    /// This API is marked as beta, meaning, while this is feature complete, it is still open to changes and may have outstanding issues.
    ///
    /// Reading from a discarded range without first writing or prefetching to it will return an indeterminate value.
    #[since = 7 .2, "7.2"]
    hipError_t hipMemDiscardAndPrefetchBatchAsync(
        *mut *mut c_void dptrs,
        *mut usize sizes,
        usize count,
        *mut hipMemLocation prefetchLocs,
        *mut usize prefetchLocIdxs,
        usize numPrefetchLocs,
        c_ulonglong flags,
        hipStream_t stream,
    );

    /// This is the driver API variant that uses hipDeviceptr_t instead of c_void*.
    #[since = 7 .2, "7.2"]
    hipError_t hipDrvMemDiscardAndPrefetchBatchAsync(
        *mut hipDeviceptr_t dptrs,
        *mut usize sizes,
        usize count,
        *mut hipMemLocation prefetchLocs,
        *mut usize prefetchLocIdxs,
        usize numPrefetchLocs,
        c_ulonglong flags,
        hipStream_t stream,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipMemAdvise(
        *const c_void dev_ptr,
        usize count,
        hipMemoryAdvise advice,
        c_int device,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 7 .1, "7.1"]
    hipError_t hipMemAdvise_v2(
        *const c_void dev_ptr,
        usize count,
        hipMemoryAdvise advice,
        hipMemLocation location,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipMemRangeGetAttributes(
        *mut *mut c_void data,
        *mut usize data_sizes,
        *mut hipMemRangeAttribute attributes,
        usize num_attributes,
        *const c_void dev_ptr,
        usize count,
    );

    /// This API is under development. Currently it is a no-operation (NOP) function on AMD GPUs and returns hipSuccess.
    hipError_t hipStreamAttachMemAsync(
        hipStream_t stream,
        *mut c_void dev_ptr,
        usize length,
        c_int flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemAddressFree(
        *mut c_void devPtr,
        usize size,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemAddressReserve(
        *mut *mut c_void ptr,
        usize size,
        usize alignment,
        *mut c_void addr,
        c_ulonglong flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemCreate(
        *mut hipMemGenericAllocationHandle_t handle,
        usize size,
        *const hipMemAllocationProp prop,
        c_ulonglong flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemExportToShareableHandle(
        *mut c_void shareableHandle,
        hipMemGenericAllocationHandle_t handle,
        hipMemAllocationHandleType handleType,
        c_ulonglong flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemGetAccess(
        *mut c_ulonglong flags,
        *const hipMemLocation location,
        *mut c_void ptr,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemGetAllocationGranularity(
        *mut usize granularity,
        *const hipMemAllocationProp prop,
        hipMemAllocationGranularity_flags option,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemGetAllocationPropertiesFromHandle(
        *mut hipMemAllocationProp prop,
        hipMemGenericAllocationHandle_t handle,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemImportFromShareableHandle(
        *mut hipMemGenericAllocationHandle_t handle,
        *mut c_void osHandle,
        hipMemAllocationHandleType shHandleType,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemMap(
        *mut c_void ptr,
        usize size,
        usize offset,
        hipMemGenericAllocationHandle_t handle,
        c_ulonglong flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemMapArrayAsync(
        *mut hipArrayMapInfo mapInfoList,
        c_uint count,
        hipStream_t stream,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle);

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemRetainAllocationHandle(
        *mut hipMemGenericAllocationHandle_t handle,
        *mut c_void addr,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemSetAccess(
        *mut c_void ptr,
        usize size,
        *const hipMemAccessDesc desc,
        usize count,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemUnmap(
        *mut c_void ptr,
        usize size,
    );

    /// 3D linear filter isn’t supported on GFX90A boards, on which the API hipCreateTextureObject will return hipErrorNotSupported.
    hipError_t hipCreateTextureObject(
        *mut hipTextureObject_t pTextObject,
        *const hipResourceDesc pRescDesc,
        *const hipTextureDesc pTexDesc,
        *const hipResourceViewDesc pResViewDesc,
    );

    hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject);

    hipError_t hipGetChannelDesc(
        *mut hipChannelFormatDesc desc,
        hipArray_const_t array,
    );

    hipError_t hipGetTextureObjectResourceDesc(
        *mut hipResourceDesc pResDesc,
        hipTextureObject_t textureObject,
    );

    hipError_t hipGetTextureObjectResourceViewDesc(
        *mut hipResourceViewDesc pResViewDesc,
        hipTextureObject_t textureObject,
    );

    hipError_t hipGetTextureObjectTextureDesc(
        *mut hipTextureDesc pTexDesc,
        hipTextureObject_t textureObject,
    );

    hipError_t hipTexObjectCreate(
        *mut hipTextureObject_t pTexObject,
        *const HIP_RESOURCE_DESC pResResc,
        *const HIP_TEXTURE_DESC pTexDesc,
        *const HIP_RESOURCE_VIEW_DESC pResViewDesc,
    );

    hipError_t hipTexObjectDestroy(hipTextureObject_t texObject);

    hipError_t hipTexObjectGetResourceDesc(
        *mut HIP_RESOURCE_DESC pResDesc,
        hipTextureObject_t texObject,
    );

    hipError_t hipTexObjectGetResourceViewDesc(
        *mut HIP_RESOURCE_VIEW_DESC pResViewDesc,
        hipTextureObject_t texObject,
    );

    hipError_t hipTexObjectGetTextureDesc(
        *mut HIP_TEXTURE_DESC pTexDesc,
        hipTextureObject_t texObject,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipMallocMipmappedArray(
        *mut hipMipmappedArray_t mipmappedArray,
        *const hipChannelFormatDesc desc,
        hipExtent extent,
        c_uint numLevels,
        c_uint flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray);

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipGetMipmappedArrayLevel(
        *mut hipArray_t levelArray,
        hipMipmappedArray_const_t mipmappedArray,
        c_uint level,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipMipmappedArrayCreate(
        *mut hipMipmappedArray_t pHandle,
        *mut HIP_ARRAY3D_DESCRIPTOR pMipmappedArrayDesc,
        c_uint numMipmapLevels,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray);

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipMipmappedArrayGetLevel(
        *mut hipArray_t pLevelArray,
        hipMipmappedArray_t hMipMappedArray,
        c_uint level,
    );

    hipError_t hipBindTextureToMipmappedArray(
        *const textureReference tex,
        hipMipmappedArray_const_t mipmappedArray,
        *const hipChannelFormatDesc desc,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipGetTextureReference(
        *mut *const texref textureReference,
        *const c_void symbol,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetBorderColor(
        *mut c_float pBorderColor,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetArray(
        *mut hipArray_t pArray,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetAddressMode(
        *mut textureReference texRef,
        c_int dim,
        hipTextureAddressMode am,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetArray(
        *mut textureReference tex,
        hipArray_const_t array,
        c_uint flags,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetFilterMode(
        *mut textureReference texRef,
        hipTextureFilterMode fm,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetFlags(
        *mut textureReference texRef,
        c_uint Flags,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetFormat(
        *mut textureReference texRef,
        hipArray_Format fmt,
        c_int NumPackedComponents,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipBindTexture(
        *mut usize offset,
        *const textureReference tex,
        *const c_void devPtr,
        *const hipChannelFormatDesc desc,
        usize size,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipBindTexture2D(
        *mut usize offset,
        *const textureReference tex,
        *const c_void devPtr,
        *const hipChannelFormatDesc desc,
        usize width,
        usize height,
        usize pitch,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipBindTextureToArray(
        *const textureReference tex,
        hipArray_const_t array,
        *const hipChannelFormatDesc desc,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipGetTextureAlignmentOffset(
        *mut usize offset,
        *const textureReference texref,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipUnbindTexture(*const textureReference tex);

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetAddress(
        *mut hipDeviceptr_t dev_ptr,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetAddressMode(
        *mut hipTextureAddressMode pam,
        *const textureReference texRef,
        c_int dim,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetFilterMode(
        *mut hipTextureFilterMode pfm,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetFlags(
        *mut c_uint pFlags,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetFormat(
        *mut hipArray_Format pFormat,
        *mut c_int pNumChannels,
        *const textureReference texBuf,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetMaxAnisotropy(
        *mut c_int pmaxAnsio,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetMipmapFilterMode(
        *mut hipTextureFilterMode pfm,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetMipmapLevelBias(
        *mut c_float pbias,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetMipmapLevelClamp(
        *mut c_float pminMipmapLevelClamp,
        *mut c_float pmaxMipmapLevelClamp,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetMipMappedArray(
        *mut hipMipmappedArray_t pArray,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetAddress(
        *mut usize ByteOffset,
        *mut textureReference texRef,
        hipDeviceptr_t dptr,
        usize bytes,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetAddress2D(
        textureReference texRef,
        *const HIP_ARRAY_DESCRIPTOR desc,
        hipDeviceptr_t dptr,
        usize Pitch,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetMaxAnisotropy(
        *mut textureReference texRef,
        c_uint maxAniso,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetBorderColor(
        *mut textureReference texRef,
        *mut c_float pBorderColor,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetMipmapFilterMode(
        *mut textureReference texRef,
        hipTextureFilterMode fm,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetMipmapLevelBias(
        *mut textureReference texRef,
        c_float bias,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetMipmapLevelClamp(
        *mut textureReference texRef,
        c_float minMipMapLevelClamp,
        c_float maxMipMapLevelClamp,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetMipmappedArray(
        *mut textureReference texRef,
        *mut hipMipmappedArray mipmappedArray,
        c_uint Flags,
    );

    hipError_t hipCreateSurfaceObject(
        *mut hipSurfaceObject_t pSurfObject,
        *const hipResourceDesc pResDesc,
    );

    hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject);

    #[since = 7 .2, "7.2"]
    hipError_t hipExtEnableLogging();

    #[since = 7 .2, "7.2"]
    hipError_t hipExtDisableLogging();

    #[since = 7 .2, "7.2"]
    hipError_t hipExtSetLoggingParams(
        usize log_level,
        usize log_size,
        usize log_mask,
    );

    #[internal(cfg)]
    c_uint __hipGetPixelAddr(
        c_int x,
        c_int format,
        c_int order,
    );

    hipError_t hipDeviceCanAccessPeer(
        *mut c_int canAccessPeer,
        c_int deviceId,
        c_int peerDeviceId,
    );

    hipError_t hipDeviceEnablePeerAccess(
        c_int peerDeviceId,
        c_uint flags,
    );

    hipError_t hipDeviceDisablePeerAccess(c_int peerDeviceId);

    hipError_t hipMemcpyPeer(
        *mut c_void dst,
        c_int dstDeviceId,
        *const c_void src,
        c_int srcDeviceId,
        usize sizeBytes,
    );

    hipError_t hipMemcpyPeerAsync(
        *mut c_void dst,
        c_int dstDeviceId,
        *const c_void src,
        c_int srcDevice,
        usize sizeBytes,
        hipStream_t stream,
    );

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxCreate(
        *mut hipCtx_t ctx,
        c_uint flags,
        hipDevice_t device,
    );

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxDestroy(hipCtx_t ctx);

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxPopCurrent(*mut hipCtx_t ctx);

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxPushCurrent(hipCtx_t ctx);

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxSetCurrent(hipCtx_t ctx);

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxGetCurrent(hipCtx_t ctx);

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxGetDevice(*mut hipDevice_t device);

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxGetApiVersion(
        hipCtx_t ctx,
        *mut c_uint apiVersion,
    );

    /// AMD devices and some Nvidia GPUs do not support reconfigurable cache. This hint is ignored on those architectures.
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxGetCacheConfig(*mut hipFuncCache_t cacheConfig);

    /// AMD devices and some Nvidia GPUs do not support reconfigurable cache. This hint is ignored on those architectures.
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig);

    /// AMD devices and some Nvidia GPUs do not support shared cache banking, and the hint is ignored on those architectures.
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config);

    /// AMD devices and some Nvidia GPUs do not support shared cache banking, and the hint is ignored on those architectures.
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxGetSharedMemConfig(*mut hipSharedMemConfig pConfig);

    /// This function waits for all streams on the default context to complete execution, and then returns.
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxSynchronize();

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxGetFlags(*mut c_uint flags);

    /// PeerToPeer support is experimental.
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxEnablePeerAccess(
        hipCtx_t peerCtx,
        c_uint flags,
    );

    /// PeerToPeer support is experimental.
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx);

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipDevicePrimaryCtxGetState(
        hipDevice_t dev,
        *mut c_uint flags,
        *mut c_int active,
    );

    /// This function return hipSuccess though doesn’t release the primaryCtx by design on HIP/HIP-CLANG path.
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipDevicePrimaryCtxRetain(
        *mut hipCtx_t pctx,
        hipDevice_t dev,
    );

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipDevicePrimaryCtxSetFlags(
        hipDevice_t dev,
        c_uint flags,
    );

    hipError_t hipModuleGetGlobal(
        *mut hipDeviceptr_t dptr,
        *mut usize bytes,
        hipModule_t hmod,
        *const c_char name,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipModuleLoadFatBinary(
        *mut hipModule_t module,
        *const c_void fatbin,
    );

    hipError_t hipModuleLoad(
        *mut hipModule_t module,
        *const c_char fname,
    );

    hipError_t hipModuleUnload(hipModule_t module);

    hipError_t hipModuleGetFunction(
        *mut hipFunction_t function,
        hipModule_t module,
        *const c_char kname,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipModuleGetFunctionCount(
        *mut c_int count,
        hipModule_t module,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipKernelGetAttribute(
        *mut c_int pi,
        hipFunction_attribute attrib,
        hipKernel_t kernel,
        hipDevice_t dev,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryLoadData(
        *mut hipLibrary_t library,
        *const c_void code,
        *mut hipJitOption jitOptions,
        *mut *mut c_void jitOptionsValues,
        c_uint numJitOptions,
        *mut hipLibraryOption libraryOptions,
        *mut *mut c_void libraryOptionValues,
        c_uint numLibraryOptions,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryLoadFromFile(
        *mut hipLibrary_t library,
        *const c_char fileName,
        *mut hipJitOption jitOptions,
        *mut *mut c_void jitOptionsValues,
        c_uint numJitOptions,
        *mut hipLibraryOption libraryOptions,
        *mut *mut c_void libraryOptionValues,
        c_uint numLibraryOptions,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryUnload(hipLibrary_t library);

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryGetKernel(
        *mut hipKernel_t pKernel,
        hipLibrary_t library,
        *const c_char name,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryGetKernelCount(
        *mut c_uint count,
        hipLibrary_t library,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryGetGlobal(
        *mut *mut c_void dptr,
        *mut usize bytes,
        hipLibrary_t library,
        *const c_char name,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryGetManaged(
        *mut *mut c_void dptr,
        *mut usize bytes,
        hipLibrary_t library,
        *const c_char name,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryEnumerateKernels(
        *mut hipKernel_t kernels,
        c_uint numKernels,
        hipLibrary_t library,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipKernelGetLibrary(
        *mut hipLibrary_t library,
        hipKernel_t kernel,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipKernelGetName(
        *const c_char name,
        hipKernel_t kernel,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipKernelGetParamInfo(
        hipKernel_t kernel,
        usize paramIndex,
        *mut usize paramOffset,
        *mut usize paramSize,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipGetFuncBySymbol(
        *mut hipFunction_t functionPtr,
        *const c_void symbolPtr,
    );

    #[since = 6 .1, "6.1"]
    hipError_t hipGetDriverEntryPoint(
        *const c_char symbol,
        *mut *mut c_void funcPtr,
        c_ulonglong flags,
        *mut hipDriverEntryPointQueryResult driverStatus,
    );

    hipError_t hipModuleGetTexRef(
        *mut *mut textureReference texRef,
        hipModule_t hmod,
        *const c_char name,
    );

    hipError_t hipModuleLoadData(
        *mut hipModule_t module,
        *const c_void image,
    );

    hipError_t hipModuleLoadDataEx(
        *mut hipModule_t module,
        *const c_void image,
        c_uint numOptions,
        *mut hipJitOption options,
        *mut *mut c_void optionValues,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipLinkAddData(
        hipLinkState_t state,
        hipJitInputType ty,
        *mut c_void data,
        usize size,
        *const c_char name,
        c_uint numOptions,
        *mut hipJitOption options,
        *mut *mut c_void optionValues,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipLinkComplete(
        hipLinkState_t state,
        *mut *mut c_void hipBinOut,
        *mut usize sizeOut,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipLinkCreate(
        c_uint numOptions,
        *mut hipJitOption options,
        *mut *mut c_void optionValues,
        *mut hipLinkState_t stateOut,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipLinkDestroy(hipLinkState_t state);

    hipError_t hipModuleOccupancyMaxPotentialBlockSize(
        *mut c_int gridSize,
        *mut c_int blockSize,
        hipFunction_t f,
        usize dynSharedMemPerBlk,
        c_int blockSizeLimit,
    );

    hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
        *mut c_int gridSize,
        *mut c_int blockSize,
        hipFunction_t f,
        usize dynSharedMemPerBlk,
        c_int blockSizeLimit,
        c_uint flags,
    );

    hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        *mut c_int numBlocks,
        hipFunction_t f,
        c_int blockSize,
        usize dynSharedMemPerBlk,
    );

    hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        *mut c_int numBlocks,
        hipFunction_t f,
        c_int blockSize,
        usize dynSharedMemPerBlk,
        c_uint flags,
    );

    hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
        *mut c_int numBlocks,
        *const c_void f,
        c_int blockSize,
        usize dynSharedMemPerBlk,
    );

    hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        *mut c_int numBlocks,
        *const c_void f,
        c_int blockSize,
        usize dynSharedMemPerBlk,
        c_uint flags,
    );

    hipError_t hipOccupancyMaxPotentialBlockSize(
        *mut c_int gridSize,
        *mut c_int blockSize,
        *const c_void f,
        usize dynSharedMemPerBlk,
        c_int blockSizeLimit,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipOccupancyAvailableDynamicSMemPerBlock(
        *mut usize dynamicSmemSize,
        *const c_void f,
        c_int numBlocks,
        c_int blockSize,
    );

    #[since = 6 .5, "6.5"]
    hipError_t hipOccupancyMaxActiveClusters(
        *mut c_int numClusters,
        *const c_void f,
        *const hipLaunchConfig_t config,
    );

    #[since = 6 .5, "6.5"]
    hipError_t hipOccupancyMaxPotentialClusterSize(
        *mut c_int clusterSize,
        *const c_void f,
        *const hipLaunchConfig_t config,
    );

    #[deprecated = "hipProfilerStart API is deprecated, use roctracer/rocTX instead."]
    hipError_t hipProfilerStart();

    #[deprecated = "hipProfilerStop API is deprecated, use roctracer/rocTX instead."]
    hipError_t hipProfilerStop();

    hipError_t hipConfigureCall(
        dim3 gridDim,
        dim3 blockDim,
        usize sharedMem,
        hipStream_t stream,
    );

    hipError_t hipSetupArgument(
        *const c_void arg,
        usize size,
        usize offset,
    );

    hipError_t hipLaunchByPtr(*const c_void func);

    #[internal(cfg)]
    hipError_t __hipPushCallConfiguration(
        dim3 gridDim,
        dim3 blockDim,
        usize sharedMem,
        hipStream_t stream,
    );

    #[internal(cfg)]
    hipError_t __hipPopCallConfiguration(
        *mut dim3 gridDim,
        *mut dim3 blockDim,
        *mut usize sharedMem,
        *mut hipStream_t stream,
    );

    hipError_t hipLaunchKernel(
        *const c_void function_address,
        dim3 numBlocks,
        dim3 dimBlocks,
        *mut *mut c_void args,
        usize sharedMemBytes,
        hipStream_t stream,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipLaunchHostFunc(
        hipStream_t stream,
        hipHostFn_t func,
        *mut c_void userData,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipDrvMemcpy2DUnaligned(*const hip_Memcpy2D pCopy);

    hipError_t hipExtLaunchKernel(
        *const c_void function_address,
        dim3 numBlocks,
        dim3 dimBlocks,
        *mut *mut c_void args,
        usize sharedMemBytes,
        hipStream_t stream,
        hipEvent_t startEvent,
        hipEvent_t stopEvent,
        c_int flags,
    );

    *const c_char hipApiName(u32 id);

    *const c_char hipKernelNameRef(hipFunction_t f);

    *const c_char hipKernelNameRefByPtr(
        *const c_void hostFunction,
        hipStream_t stream,
    );

    c_int hipGetStreamDeviceId(hipStream_t stream);

    #[since = 4 .3, "4.3"]
    hipError_t hipStreamBeginCapture(
        hipStream_t stream,
        hipStreamCaptureMode mode,
    );

    /// param “const hipGraphEdgeData* dependencyData” is currently not supported and
    /// has to be passed as nullptr. This API is marked as beta, meaning, while this is
    /// feature complete, it is still open to changes and may have outstanding issues.
    #[since = 6 .1, "6.1"]
    hipError_t hipStreamBeginCaptureToGraph(
        hipStream_t stream,
        hipGraph_t graph,
        *const hipGraphNode_t dependencies,
        *const hipGraphEdgeData dependenciesData,
        usize numDependencies,
        hipStreamCaptureMode mode,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipStreamEndCapture(
        hipStream_t stream,
        *mut hipGraph_t pGraph,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipStreamGetCaptureInfo(
        hipStream_t stream,
        *mut hipStreamCaptureStatus pCaptureStatus,
        *mut c_ulonglong pId,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipStreamGetCaptureInfo_v2(
        hipStream_t stream,
        *mut hipStreamCaptureStatus captureStatus_out,
        *mut c_ulonglong id_out,
        *mut hipGraph_t graph_out,
        *mut *const hipGraphNode_t dependencies_out ,
        *mut usize numDependencies_out,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipStreamIsCapturing(
        hipStream_t stream,
        *mut hipStreamCaptureStatus pCaptureStatus,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipStreamUpdateCaptureDependencies(
        hipStream_t stream,
        *mut hipGraphNode_t dependencies,
        usize numDependencies,
        c_uint flags,
    );

    #[since = 5 .0, "5.0"]
    hipError_t hipThreadExchangeStreamCaptureMode(*mut hipStreamCaptureMode mode);

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphCreate(
        *mut hipGraph_t pGraph,
        c_uint flags,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphDestroy(hipGraph_t graph);

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphAddDependencies(
        hipGraph_t graph,
        *const hipGraphNode_t from,
        *const hipGraphNode_t to,
        usize numDependencies,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphRemoveDependencies(
        hipGraph_t graph,
        *const hipGraphNode_t from,
        *const hipGraphNode_t to,
        usize numDependencies,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphGetEdges(
        hipGraph_t graph,
        *mut hipGraphNode_t from,
        *mut hipGraphNode_t to,
        *mut usize numEdges,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphGetNodes(
        hipGraph_t graph,
        *mut hipGraphNode_t nodes,
        *mut usize numNodes,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphGetRootNodes(
        hipGraph_t graph,
        *mut hipGraphNode_t pRootNodes,
        *mut usize pNumRootNodes,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphNodeGetDependencies(
        hipGraphNode_t node,
        *mut hipGraphNode_t pDependencies,
        *mut usize pNumDependencies,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphNodeGetDependentNodes(
        hipGraphNode_t node,
        *mut hipGraphNode_t pDependentNodes,
        *mut usize pNumDependentNodes,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphNodeGetType(
        hipGraphNode_t node,
        *mut hipGraphNodeType pType,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphDestroyNode(hipGraphNode_t node);

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphClone(
        *mut hipGraph_t pGraphClone,
        hipGraph_t originalGraph,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphNodeFindInClone(
        *mut hipGraphNode_t pNode,
        hipGraphNode_t originalNode,
        hipGraph_t clonedGraph,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphInstantiate(
        *mut hipGraphExec_t pGraphExec,
        hipGraph_t graph,
        *mut hipGraphNode_t pErrorNode,
        *mut c_char pLogBuffer,
        usize bufferSize,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphInstantiateWithFlags(
        *mut hipGraphExec_t pGraphExec,
        hipGraph_t graph,
        c_ulonglong flags,
    );

    #[since = 6 .1, "6.1"]
    hipError_t hipGraphInstantiateWithParams(
        *mut hipGraphExec_t pGraphExec,
        hipGraph_t graph,
        *mut hipGraphInstantiateParams instantiateParams,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphLaunch(
        hipGraphExec_t graphExec,
        hipStream_t stream,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphUpload(
        hipGraphExec_t graphExec,
        hipStream_t stream,
    );

    #[since = 5 .5, "5.5"]
    hipError_t hipGraphAddNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        *mut hipGraphNodeParams nodeParams,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipGraphExecGetFlags(
        hipGraphExec_t graphExec,
        *mut c_ulonglong flags,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipGraphNodeSetParams(
        hipGraphNode_t node,
        *mut hipGraphNodeParams nodeParams,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipGraphExecNodeSetParams(
        hipGraphExec_t graphExec,
        hipGraphNode_t node,
        *mut hipGraphNodeParams nodeParams,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec);

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecUpdate(
        hipGraphExec_t hGraphExec,
        hipGraph_t hGraph,
        *mut hipGraphNode_t hErrorNode_out,
        *mut hipGraphExecUpdateResult updateResult_out,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphAddKernelNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        *const hipKernelNodeParams pNodeParams,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphKernelNodeGetParams(
        hipGraphNode_t node,
        *mut hipKernelNodeParams pNodeParams,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphKernelNodeSetParams(
        hipGraphNode_t node,
        *const hipKernelNodeParams pNodeParams,
    );

    #[since = 5 .6, "5.6"]
    hipError_t hipDrvGraphAddMemcpyNode(
        *mut hipGraphNode_t phGraphNode,
        hipGraph_t hGraph,
        *const hipGraphNode_t dependencies,
        usize numDependencies,
        *const HIP_MEMCPY3D copyParams,
        hipCtx_t ctx,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphAddMemcpyNode(
        *mut hipGraphNode_t pGraphNode_t,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        *const hipMemcpy3DParms pCopyParams,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphMemcpyNodeGetParams(
        hipGraphNode_t node,
        *mut hipMemcpy3DParms pNodeParams,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphMemcpyNodeSetParams(
        hipGraphNode_t node,
        *const hipMemcpy3DParms pNodeParams,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecMemcpyNodeSetParams(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t node,
        *mut hipMemcpy3DParms pNodeParams,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphAddMemcpyNode1D(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        *mut c_void dst,
        *const c_void src,
        usize count,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphMemcpyNodeSetParams1D(
        hipGraphNode_t node,
        *mut c_void dst,
        *const c_void src,
        usize count,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecMemcpyNodeSetParams1D(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t node,
        *mut c_void dst,
        *const c_void src,
        usize count,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphAddMemcpyNodeFromSymbol(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        *mut c_void dst,
        *const c_void symbol,
        usize count,
        usize offset,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(
        hipGraphNode_t node,
        *mut c_void dst,
        *const c_void symbol,
        usize count,
        usize offset,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t node,
        *mut c_void dst,
        *const c_void symbol,
        usize count,
        usize offset,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphAddMemcpyNodeToSymbol(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        *const c_void symbol,
        *const c_void src,
        usize count,
        usize offset,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphMemcpyNodeSetParamsToSymbol(
        hipGraphNode_t node,
        *const c_void symbol,
        *const c_void src,
        usize count,
        usize offset,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t node,
        *const c_void symbol,
        *const c_void src,
        usize count,
        usize offset,
        hipMemcpyKind kind,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphAddMemsetNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        *const hipMemsetParams pMemsetParams,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphMemsetNodeGetParams(
        hipGraphNode_t node,
        *mut hipMemsetParams pNodeParams,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphMemsetNodeSetParams(
        hipGraphNode_t node,
        *const hipMemsetParams pNodeParams,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecMemsetNodeSetParams(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t node,
        *const hipMemsetParams pNodeParams,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphAddHostNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        *const hipHostNodeParams pNodeParams,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphHostNodeGetParams(
        hipGraphNode_t node,
        *mut hipHostNodeParams pNodeParams,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphHostNodeSetParams(
        hipGraphNode_t node,
        *const hipHostNodeParams pNodeParams,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecHostNodeSetParams(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t node,
        *const hipHostNodeParams pNodeParams,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphAddChildGraphNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        hipGraph_t childGraph,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphChildGraphNodeGetGraph(
        hipGraphNode_t node,
        *mut hipGraph_t pGraph,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecChildGraphNodeSetParams(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t node,
        hipGraph_t childGraph,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphAddEmptyNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphAddEventRecordNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        hipEvent_t event,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphEventRecordNodeGetEvent(
        hipGraphNode_t node,
        *mut hipEvent_t event_out,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphEventRecordNodeSetEvent(
        hipGraphNode_t node,
        hipEvent_t event,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecEventRecordNodeSetEvent(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t hNode,
        hipEvent_t event,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphAddEventWaitNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        hipEvent_t event,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphEventWaitNodeGetEvent(
        hipGraphNode_t node,
        *mut hipEvent_t event_out,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphEventWaitNodeSetEvent(
        hipGraphNode_t node,
        hipEvent_t event,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecEventWaitNodeSetEvent(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t hNode,
        hipEvent_t event,
    );

    #[since = 5 .5, "5.5"]
    hipError_t hipGraphAddMemAllocNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        *mut hipMemAllocNodeParams pNodeParams,
    );

    #[since = 5 .5, "5.5"]
    hipError_t hipGraphMemAllocNodeGetParams(
        hipGraphNode_t node,
        *mut hipMemAllocNodeParams pNodeParams,
    );

    #[since = 5 .5, "5.5"]
    hipError_t hipGraphAddMemFreeNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        *mut c_void dev_ptr,
    );

    #[since = 5 .5, "5.5"]
    hipError_t hipGraphMemFreeNodeGetParams(
        hipGraphNode_t node,
        *mut c_void dev_ptr,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipDeviceGetGraphMemAttribute(
        c_int device,
        hipGraphMemAttributeType attr,
        *mut c_void value,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipDeviceGraphMemTrim(c_int device);

    #[since = 5 .3, "5.3"]
    hipError_t hipUserObjectCreate(
        *mut hipUserObject_t object_out,
        *mut c_void ptr,
        hipHostFn_t destroy,
        c_uint initialRefcount,
        c_uint flags,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipUserObjectRelease(
        hipUserObject_t object,
        c_uint count,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipUserObjectRetain(
        hipUserObject_t object,
        c_uint count,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphRetainUserObject(
        hipGraph_t graph,
        hipUserObject_t object,
        c_uint count,
        c_uint flags,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphReleaseUserObject(
        hipGraph_t graph,
        hipUserObject_t object,
        c_uint count,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphDebugDotPrint(
        hipGraph_t graph,
        *const c_char path,
        c_uint flags,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphKernelNodeCopyAttributes(
        hipGraphNode_t hSrc,
        hipGraphNode_t hDst,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphNodeSetEnabled(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t hNode,
        c_uint isEnabled,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphNodeGetEnabled(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t hNode,
        *mut c_uint isEnabled,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphAddExternalSemaphoresWaitNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        *const hipExternalSemaphoreWaitNodeParams nodeParams,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphAddExternalSemaphoresSignalNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        usize numDependencies,
        *const hipExternalSemaphoreSignalNodeParams nodeParams,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphExternalSemaphoresSignalNodeSetParams(
        hipGraphNode_t hNode,
        *const hipExternalSemaphoreSignalNodeParams nodeParams,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphExternalSemaphoresWaitNodeSetParams(
        hipGraphNode_t hNode,
        *const hipExternalSemaphoreWaitNodeParams nodeParams,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphExternalSemaphoresSignalNodeGetParams(
        hipGraphNode_t hNode,
        *mut hipExternalSemaphoreSignalNodeParams params_out,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphExternalSemaphoresWaitNodeGetParams(
        hipGraphNode_t hNode,
        *mut hipExternalSemaphoreWaitNodeParams params_out,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphExecExternalSemaphoresSignalNodeSetParams(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t hNode,
        *const hipExternalSemaphoreSignalNodeParams nodeParams,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphExecExternalSemaphoresWaitNodeSetParams(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t hNode,
        *const hipExternalSemaphoreWaitNodeParams nodeParams,
    );

    #[since = 6 .0, "6.0"]
    hipError_t hipDrvGraphMemcpyNodeGetParams(
        hipGraphNode_t hNode,
        *mut HIP_MEMCPY3D nodeParams,
    );

    #[since = 6 .0, "6.0"]
    hipError_t hipDrvGraphMemcpyNodeSetParams(
        hipGraphNode_t hNode,
        *const HIP_MEMCPY3D nodeParams,
    );

    #[since = 5 .6, "5.6"]
    hipError_t hipDrvGraphAddMemsetNode(
        *mut hipGraphNode_t phGraphNode,
        hipGraph_t hGraph,
        *const hipGraphNode_t dependencies,
        usize numDependencies,
        *const hipMemsetParams memsetParams,
        hipCtx_t ctx,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipDrvGraphAddMemFreeNode(
        *mut hipGraphNode_t phGraphNode,
        hipGraph_t hGraph,
        *const hipGraphNode_t dependencies,
        usize numDependencies,
        hipDeviceptr_t dptr,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipDrvGraphExecMemcpyNodeSetParams(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t hNode,
        *const HIP_MEMCPY3D copyParams,
        hipCtx_t ctx,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipDrvGraphExecMemsetNodeSetParams(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t hNode,
        *const hipMemsetParams memsetParams,
        hipCtx_t ctx,
    );

    #[since = 6 .5, "6.5"]
    hipError_t hipMemGetHandleForAddressRange(
        *mut c_void handle,
        hipDeviceptr_t dptr,
        usize size,
        hipMemRangeHandleType handleType,
        c_ulonglong flags,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphicsMapResources(
        c_int count,
        *mut hipGraphicsResource_t resources,
        hipStream_t stream,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphicsSubResourceGetMappedArray(
        *mut hipArray_t array,
        hipGraphicsResource_t resource,
        c_uint arrayIndex,
        c_uint mipLevel,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphicsResourceGetMappedPointer(
        *mut *mut c_void devPtr,
        *mut usize size,
        hipGraphicsResource_t resource,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphicsUnmapResources(
        c_int count,
        *mut hipGraphicsResource_t resrources,
        hipStream_t stream,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource);

    #[since = 5 .5, "5.5"]
    hipError_t hipModuleLaunchCooperativeKernel(
        hipFunction_t f,
        c_uint gridDimX,
        c_uint gridDimY,
        c_uint gridDimZ,
        c_uint blockDimX,
        c_uint blockDimY,
        c_uint blockDimZ,
        c_uint sharedMemBytes,
        hipStream_t stream,
        *mut *mut c_void kernelParams,
    );

    #[since = 5 .5, "5.5"]
    hipError_t hipModuleLaunchCooperativeKernelMultiDevice(
        *mut hipFunctionLaunchParams launchParamsList,
        c_uint numDevices,
        c_uint flags,
    );

    hipError_t hipLaunchCooperativeKernel(
        *const c_void f,
        dim3 gridDim,
        dim3 blockDim,
        *mut *mut c_void kernelParams,
        c_uint sharedMemBytes,
        hipStream_t stream,
    );

    #[since = 5 .2, "5.2"]
    hipError_t hipLaunchCooperativeKernelMultiDevice(
        hipLaunchParams launchParamsList,
        c_int numDevices,
        c_uint flags,
    );
}

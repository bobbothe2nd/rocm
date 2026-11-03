use super::types::*;
use crate::internal::*;

#[cfg(feature = "dynamic-loading")]
use crate::version::{ROCM_MAJOR as MAJOR, ROCM_MINOR as MINOR};

link! {
    amdhip64 : [7, 8, 9, 10] : rocm_ver;

    hipError_t hipInit(uint flags);

    hipError_t hipDriverGetVersion(*mut int driverVersion);

    hipError_t hipRuntimeGetVersion(*mut int runtimeVersion);

    hipError_t hipDeviceGet(
        *mut hipDevice_t device,
        int ordinal,
    );

    hipError_t hipDeviceComputeCapability(
        *mut int major,
        *mut int minor,
        hipDevice_t device,
    );

    hipError_t hipDeviceGetName(
        *mut char name,
        *mut int len,
        hipDevice_t device,
    );

    #[since = 5 .1, "5.1"]
    hipError_t hipDeviceGetUuid(
        *mut hipUUID uuid,
        hipDevice_t device,
    );

    #[since = 5 .1, "5.1"]
    hipError_t hipDeviceGetLuid(
        *mut char luid,
        *mut uint deviceNodeMask,
        hipDevice_t device,
    );

    hipError_t hipDeviceGetP2PAttribute(
        *mut int value,
        hipDeviceP2PAttr attr,
        int srcDevice,
        int dstDevice,
    );

    hipError_t hipDeviceGetPCIBusId(
        *mut char pciBusId,
        int len,
        int device,
    );

    hipError_t hipDeviceGetByPCIBusId(
        *mut int device,
        *const char pciBusId
    );

    hipError_t hipDeviceTotalMem(
        *mut size_t bytes,
        hipDevice_t device,
    );

    hipError_t hipDeviceSynchronize();

    hipError_t hipDeviceReset();

    hipError_t hipSetDevice(int deviceId);

    #[since = 6 .0, "6.0"]
    hipError_t hipSetValidDevices(
        *mut int device_arr,
        int len,
    );

    hipError_t hipGetDevice(*mut int device);

    hipError_t hipGetDeviceCount(*mut int count);

    hipError_t hipDeviceGetAttribute(
        *mut int pi,
        hipDeviceAttribute_t attr,
        int deviceId,
    );

    #[since = 5 .1, "5.1"]
    hipError_t hipDeviceGetDefaultMemPool(
        *mut hipMemPool_t mem_pool,
        int device,
    );

    #[since = 5 .1, "5.1"]
    hipError_t hipDeviceSetMemPool(
        int device,
        hipMemPool_t mem_pool,
    );

    #[since = 5 .1, "5.1"]
    hipError_t hipDeviceGetMemPool(
        int device,
        *mut hipMemPool_t mem_pool,
    );

    /// assumes hipDeviceProp_tR0600=hipDeviceProp_t
    hipError_t hipGetDeviceProperties(
        *mut hipDeviceProp_tR0600 prop,
        int deviceId,
    );

    #[since = 5 .0, "5.0"]
    hipError_t hipGetDevicePropertiesR0600(
        *mut hipDeviceProp_tR0600 prop,
        int deviceId,
    );

    hipError_t hipDeviceGetTexture1DLinearMaxWidth(
        *mut size_t max_width,
        *const hipChannelFormatDesc dex,
        int device,
    );

    hipError_t hipDeviceSetCacheConfig(
        hipFuncCache_t cacheConfig,
    );

    hipError_t hipDeviceGetCacheConfig(
        *mut hipFuncCache_t cacheConfig,
    );

    hipError_t hipDeviceGetLimit(
        *mut size_t pValue,
        hipLimit_t limit,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipDeviceSetLimit(
        hipLimit_t limit,
        size_t pValue,
    );

    hipError_t hipDeviceGetSharedMemConfig(
        *mut hipSharedMemConfig pConfig,
    );

    hipError_t hipGetDeviceFlags(
        *mut uint flags,
    );

    hipError_t hipDeviceSetSharedMemConfig(
        hipSharedMemConfig config,
    );

    hipError_t hipSetDeviceFlags(
        uint flags,
    );

    hipError_t hipInitDevice(
        int device,
        uint deviceFlags,
        uint flags,
    );

    /// assumes hipDeviceProp_tR0600=hipDeviceProp_t
    hipError_t hipChooseDevice(
        *mut int device,
        *const hipDeviceProp_tR0600 prop,
    );

    hipError_t hipExtGetLinkTypeAndHopCount(
        int device1,
        int device2,
        *mut uint32_t linktype,
        uint32_t hopcount,
    );

    hipError_t hipIpcGetMemHandle(
        *mut hipIpcMemHandle_t handle,
        *mut void devPtr,
    );

    hipError_t hipIpcOpenMemHandle(
        *mut *mut void devPtr,
        hipIpcMemHandle_t handle,
        uint flags,
    );

    hipError_t hipIpcCloseMemHandle(
        *mut void devPtr,
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
        uint32_t globalWorkSizeX,
        uint32_t globalWorkSizeY,
        uint32_t globalWorkSizeZ,
        uint32_t localWorkSizeX,
        uint32_t localWorkSizeY,
        uint32_t localWorkSizeZ,
        size_t sharedMemBytes,
        hipStream_t hStream,
        *mut *mut void kernelParams,
        *mut *mut void extra,
        hipEvent_t startEvent,
        hipEvent_t stopEvent,
        uint32_t flags,
    );

    hipError_t hipHccModuleLaunchKernel(
        hipFunction_t f,
        uint32_t globalWorkSizeX,
        uint32_t globalWorkSizeY,
        uint32_t globalWorkSizeZ,
        uint32_t localWorkSizeX,
        uint32_t localWorkSizeY,
        uint32_t localWorkSizeZ,
        size_t sharedMemBytes,
        hipStream_t hStream,
        *mut *mut void kernelParams,
        *mut *mut void extra,
        hipEvent_t startEvent,
        hipEvent_t stopEvent,
    );

    hipError_t hipFuncSetAttribute(
        *const void func,
        hipFuncAttribute attr,
        int value,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipKernelSetAttribute(
        hipFunction_attribute attrib,
        int value,
        hipKernel_t kernel,
        hipDevice_t dev,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipKernelGetFunction(
        *mut hipFunction_t pFunc,
        hipKernel_t kernel,
    );

    hipError_t hipFuncSetCacheConfig(
        *const void func,
        hipFuncCache_t config,
    );

    hipError_t hipFuncSetSharedMemConfig(
        *const void func,
        hipSharedMemConfig config,
    );

    hipError_t hipFuncGetAttributes(
        *mut hipFuncAttributes attr,
        *const void func,
    );

    hipError_t hipFuncGetAttribute(
        *mut int value,
        hipFunction_attribute attrib,
        hipFunction_t hfunc,
    );

    hipError_t hipModuleLaunchKernel(
        hipFunction_t f,
        uint gridDimX,
        uint gridDimY,
        uint gridDimZ,
        uint blockDimX,
        uint blockDimY,
        uint blockDimZ,
        uint sharedMemBytes,
        hipStream_t stream,
        *mut *mut void kernelParams,
        *mut *mut void extra,
    );

    hipError_t hipExtLaunchMultiKernelMultiDevice(
        *mut hipLaunchParams launchParamsList,
        int numDevices,
        uint flags,
    );

    #[since = 6 .5, "6.5"]
    hipError_t hipLaunchKernelExC(
        *const hipLaunchConfig_t config,
        *const void fPtr,
        *mut *mut void args,
    );

    #[since = 6 .5, "6.5"]
    hipError_t hipDrvLaunchKernelEx(
        *const HIP_LAUNCH_CONFIG config,
        hipFunction_t f,
        *mut *mut void params,
        *mut *mut void extra,
    );

    hipError_t hipGetLastError();

    #[since = 6 .0, "6.0"]
    hipError_t hipExtGetLastError();

    *const char hipGetErrorName(hipError_t hip_error);

    *const char hipGetErrorString(hipError_t hip_error);

    #[since = 5 .3, "5.3"]
    hipError_t hipDrvGetErrorName(
        hipError_t hip_Error,
        *mut *const char errorString,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipDrvGetErrorString(
        hipError_t hip_Error,
        *mut *const char errorString,
    );

    hipError_t hipStreamCreate(
        *mut hipStream_t stream,
    );

    hipError_t hipStreamCreateWithFlags(
        *mut hipStream_t stream,
        uint flags,
    );

    hipError_t hipStreamCreateWithPriority(
        *mut hipStream_t stream,
        uint flags,
        int priority,
    );

    hipError_t hipDeviceGetStreamPriorityRange(
        *mut int leastPriority,
        *mut int greatestPriority,
    );

    hipError_t hipStreamDestroy(hipStream_t stream);

    hipError_t hipStreamQuery(hipStream_t stream);

    hipError_t hipStreamSynchronize(hipStream_t stream);

    hipError_t hipStreamWaitEvent(
        hipStream_t stream,
        hipEvent_t event,
        uint flags,
    );

    #[since = 5 .2, "5.2"]
    hipError_t hipStreamGetFlags(
        hipStream_t stream,
        *mut uint flags,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipStreamGetId(
        hipStream_t stream,
        *mut ulonglong streamId,
    );

    hipError_t hipStreamGetPriority(
        hipStream_t stream,
        *mut int priority,
    );

    hipError_t hipStreamGetDevice(
        hipStream_t stream,
        *mut hipDevice_t device,
    );

    hipError_t hipExtStreamCreateWithCUMask(
        *mut hipStream_t stream,
        uint32_t cuMaskSize,
        *const uint32_t cuMask,
    );

    hipError_t hipExtStreamGetCUMask(
        hipStream_t stream,
        uint32_t cuMaskSize,
        *mut uint32_t cuMask,
    );

    hipError_t hipStreamAddCallback(
        hipStream_t stream,
        hipStreamCallback_t callback,
        *mut void userData,
        uint flags,
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
        *mut void ptr,
        uint32_t value,
        uint flags,
        uint32_t mask,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipStreamWaitValue64(
        hipStream_t stream,
        *mut void ptr,
        uint64_t value,
        uint flags,
        uint64_t mask,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipStreamWriteValue32(
        hipStream_t stream,
        *mut void ptr,
        uint32_t value,
        uint flags,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipStreamWriteValue64(
        hipStream_t stream,
        *mut void ptr,
        uint64_t value,
        int flags,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipStreamBatchMemOp(
        hipStream_t stream,
        int count,
        *mut hipStreamBatchMemOpParams paramArray,
        int flags,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipGraphAddBatchMemOpNode(
        *mut hipGraphNode_t phGraphNode,
        hipGraph_t hGraph,
        *const hipGraphNode_t dependencies,
        size_t numDependencies,
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
        uint flags,
    );

    hipError_t hipEventCreate(*mut hipEvent_t event);

    #[since = 6 .4, "6.4"]
    hipError_t hipEventRecordWithFlags(
        hipEvent_t event,
        hipStream_t stream,
        uint flags,
    );

    hipError_t hipEventRecord(
        hipEvent_t event,
        hipStream_t stream,
    );

    hipError_t hipEventDestroy(hipEvent_t event);

    hipError_t hipEventSynchronize(hipEvent_t event);

    hipError_t hipEventElapsedTime(
        *mut float ms,
        hipEvent_t start,
        hipEvent_t stop,
    );

    hipError_t hipEventQuery(hipEvent_t event);

    hipError_t hipPointerSetAttribute(
        *const void value,
        hipPointer_attribute attribute,
        hipDeviceptr_t ptr,
    );

    #[since = 5 .0, "5.0"]
    hipError_t hipPointerGetAttributes(
        *mut hipPointerAttribute_t attributes,
        *const void ptr,
    );

    hipError_t hipPointerGetAttribute(
        *mut void data,
        hipPointer_attribute attribute,
        hipDeviceptr_t ptr,
    );

    #[since = 5 .0, "5.0"]
    hipError_t hipDrvPointerGetAttributes(
        uint numAttributes,
        *mut hipPointer_attribute attributes,
        *mut *mut void data,
        hipDeviceptr_t ptr,
    );

    hipError_t hipMalloc(
        *mut *mut void ptr,
        size_t size,
    );

    hipError_t hipExtMallocWithFlags(
        *mut *mut void ptr,
        size_t sizeBytes,
        uint flags,
    );

    hipError_t hipHostMalloc(
        *mut *mut void ptr,
        size_t size,
        uint flags,
    );

    hipError_t hipHostAlloc(
        *mut *mut void ptr,
        size_t size,
        uint flags,
    );

    hipError_t hipHostGetDevicePointer(
        *mut *mut void devPtr,
        *mut void hstPtr,
        uint flags,
    );

    hipError_t hipHostGetFlags(
         *mut uint flagsPtr,
        *mut void hostPtr,
    );

    hipError_t hipHostRegister(
        *mut void hostPtr,
        size_t sizeBytes,
        uint flags,
    );

    hipError_t hipHostUnregister(*mut void hostPtr);

    hipError_t hipMallocPitch(
        *mut *mut void ptr,
        *mut size_t pitch,
        size_t width,
        size_t height,
    );

    hipError_t hipMemAllocPitch(
        *mut hipDeviceptr_t dptr,
        *mut size_t pitch,
        size_t widthInBytes,
        size_t height,
        uint elementSizeBytes,
    );

    hipError_t hipFree(
        *mut void ptr,
    );

    hipError_t hipFreeHost(
        *mut void ptr,
    );

    hipError_t hipMemcpy(
        *mut void dst,
        *const void src,
        size_t size,
        hipMemcpyKind kind,
    );

    hipError_t hipMemcpyWithStream(
        *mut void dst,
        *const void src,
        size_t sizeBytes,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    hipError_t hipMemcpyHtoD(
        hipDeviceptr_t dst,
        *const void src,
        size_t sizeBytes,
    );

    hipError_t hipMemcpyDtoH(
        *mut void dst,
        hipDeviceptr_t src,
        size_t sizeBytes,
    );

    hipError_t hipMemcpyDtoD(
        hipDeviceptr_t dst,
        hipDeviceptr_t src,
        size_t sizeBytes,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipMemcpyAtoD(
        hipDeviceptr_t dstDevice,
        hipArray_t srcArray,
        size_t srcOffset,
        size_t ByteCount,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipMemcpyDtoA(
        hipArray_t dstArray,
        size_t dstOffset,
        hipDeviceptr_t srcDevice,
        size_t ByteCount,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipMemcpyAtoA(
        hipArray_t dstArray,
        size_t dstOffset,
        hipArray_t srcArray,
        size_t srcOffset,
        size_t ByteCount,
    );

    hipError_t hipMemcpyHtoDAsync(
        hipDeviceptr_t dst,
        *const void src,
        size_t sizeBytes,
        hipStream_t stream,
    );

    hipError_t hipMemcpyDtoHAsync(
        *mut void dst,
        hipDeviceptr_t src,
        size_t sizeBytes,
        hipStream_t stream,
    );

    hipError_t hipMemcpyDtoDAsync(
        hipDeviceptr_t dst,
        hipDeviceptr_t src,
        size_t sizeBytes,
        hipStream_t stream,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipMemcpyAtoHAsync(
        *mut void dstHost,
        hipArray_t srcArray,
        size_t srcOffset,
        size_t ByteCount,
        hipStream_t stream,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipMemcpyHtoAAsync(
        hipArray_t dstArray,
        size_t dstOffset,
        *const void srcHost,
        size_t ByteCount,
        hipStream_t stream,
    );

    hipError_t hipGetSymbolAddress(
        *mut *mut void devPtr,
        *const void symbol,
    );

    hipError_t hipGetSymbolSize(
        *mut size_t size,
        *const void symbol,
    );

    #[since = 6 .1, "6.1"]
    hipError_t hipGetProcAddress(
        *const char symbol,
        *mut *mut void pfn,
        int hipVersion,
        uint64_t flags,
        hipDriverProcAddressQueryResult symbolStatus,
    );

    hipError_t hipMemcpyToSymbol(
        *const void symbol,
        *const void src,
        size_t sizeBytes,
        size_t offset,
        hipMemcpyKind kind,
    );

    hipError_t hipMemcpyToSymbolAsync(
        *const void symbol,
        *const void src,
        size_t sizeBytes,
        size_t offset,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    hipError_t hipMemcpyFromSymbol(
        *mut void dst,
        *const void symbol,
        size_t sizeBytes,
        size_t offset,
        hipMemcpyKind kind,
    );

    hipError_t hipMemcpyFromSymbolAsync(
        *mut void dst,
        *const void symbol,
        size_t sizeBytes,
        size_t offset,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    hipError_t hipMemcpyAsync(
        *mut void dst,
        *const void src,
        size_t sizeBytes,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    hipError_t hipMemset(
        *mut void dst,
        int value,
        size_t sizeBytes,
    );

    hipError_t hipMemsetAsync(
        *mut void dst,
        int value,
        size_t sizeBytes,
        hipStream_t stream,
    );

    hipError_t hipMemsetD8(
        hipDeviceptr_t dest,
        uchar value,
        size_t count,
    );

    hipError_t hipMemsetD8Async(
        hipDeviceptr_t dest,
        uchar value,
        size_t count,
        hipStream_t stream,
    );

    hipError_t hipMemsetD16(
        hipDeviceptr_t dest,
        ushort value,
        size_t count,
    );

    hipError_t hipMemsetD16Async(
        hipDeviceptr_t dest,
        ushort value,
        size_t count,
        hipStream_t stream,
    );

    hipError_t hipMemsetD32(
        hipDeviceptr_t dest,
        int value,
        size_t count,
    );

    hipError_t hipMemsetD32Async(
        hipDeviceptr_t dest,
        int value,
        size_t count,
    );

    hipError_t hipMemset2D(
        *mut void dst,
        size_t pitch,
        int value,
        size_t width,
        size_t height,
    );

    hipError_t hipMemset2DAsync(
        *mut void dst,
        size_t pitch,
        int value,
        size_t width,
        size_t height,
        hipStream_t stream,
    );

    hipError_t hipMemset3D(
        hipPitchedPtr pitchedDevPtr,
        int value,
        hipExtent extent,
    );

    hipError_t hipMemset3DAsync(
        hipPitchedPtr pitchedDevPtr,
        int value,
        hipExtent extent,
        hipStream_t stream,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemsetD2D8(
        hipDeviceptr_t dst,
        size_t dstPitch,
        uchar value,
        size_t width,
        size_t height,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemsetD2D8Async(
        hipDeviceptr_t dst,
        size_t dstPitch,
        uchar value,
        size_t width,
        size_t height,
        hipStream_t stream,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemsetD2D16(
        hipDeviceptr_t dst,
        size_t dstPitch,
        ushort value,
        size_t width,
        size_t height,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemsetD2D16Async(
        hipDeviceptr_t dst,
        size_t dstPitch,
        ushort value,
        size_t width,
        size_t height,
        hipStream_t stream,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemsetD2D32(
        hipDeviceptr_t dst,
        size_t dstPitch,
        uint value,
        size_t width,
        size_t height,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemsetD2D32Async(
        hipDeviceptr_t dst,
        size_t dstPitch,
        uint value,
        size_t width,
        size_t height,
        hipStream_t stream,
    );

    hipError_t hipMemGetInfo(
        *mut size_t free,
        *mut size_t total,
    );

    hipError_t hipMemPtrGetInfo(
        *mut void ptr,
        *mut size_t size,
    );

    hipError_t hipMallocArray(
        *mut hipArray_t array,
        *const hipChannelFormatDesc desc,
        size_t width,
        size_t height,
        uint flags,
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
        uint flags,
    );

    #[since = 5 .6, "5.6"]
    hipError_t hipArrayGetInfo(
        *mut hipChannelFormatDesc desc,
        *mut hipExtent extent,
        *mut uint flags,
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
        *mut void dst,
        size_t dpitch,
        *const void src,
        size_t spitch,
        size_t width,
        size_t height,
        hipMemcpyKind kind,
    );

    hipError_t hipMemcpyParam2D(*const hip_Memcpy2D pCopy);

    hipError_t hipMemcpyParam2DAsync(
        *const hip_Memcpy2D pCopy,
        hipStream_t stream,
    );

    hipError_t hipMemcpy2DAsync(
        *mut void dst,
        size_t dpitch,
        *const void src,
        size_t spitch,
        size_t width,
        size_t height,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    hipError_t hipMemcpy2DToArray(
        hipArray_t dst,
        size_t wOffset,
        size_t hOffset,
        *const void src,
        size_t spitch,
        size_t width,
        size_t height,
        hipMemcpyKind kind,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipMemcpy2DToArrayAsync(
        hipArray_t dst,
        size_t wOffset,
        size_t hOffset,
        *const void src,
        size_t spitch,
        size_t width,
        size_t height,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipMemcpy2DArrayToArray(
        hipArray_t dst,
        size_t wOffsetDst,
        size_t hOffsetDst,
        hipArray_const_t src,
        size_t wOffsetSrc,
        size_t hOffsetSrc,
        size_t width,
        size_t height,
        hipMemcpyKind kind,
    );

    hipError_t hipMemcpy2DFromArray(
        *mut void dst,
        size_t dpitch,
        hipArray_const_t src,
        size_t wOffset,
        size_t hOffset,
        size_t width,
        size_t height,
        hipMemcpyKind kind,
    );

    hipError_t hipMemcpy2DFromArrayAsync(
        *mut void dst,
        size_t dpitch,
        hipArray_const_t src,
        size_t wOffset,
        size_t hOffset,
        size_t width,
        size_t height,
        hipMemcpyKind kind,
        hipStream_t stream,
    );

    hipError_t hipMemcpyAtoH(
        *mut void dst,
        hipArray_t srcArray,
        size_t srcOffset,
        size_t count,
    );

    hipError_t hipMemcpyHtoA(
        hipArray_t dstArray,
        size_t dstOffset,
        *const void srcHost,
        size_t count,
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
        *mut size_t psize,
        hipDeviceptr_t dptr,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemcpyBatchAsync(
        *mut *mut void dsts,
        *mut *mut void srcs,
        *mut size_t sizes,
        size_t count,
        *mut hipMemcpyAttributes atttrs,
        *mut size_t attrsIdxs,
        size_t numAttrs,
        *mut size_t failIdx,
        hipStream_t stream,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipMemcpy3DBatchAsync(
        size_t numOps,
        *mut hipMemcpy3DBatchOp opList,
        *mut size_t failIdx,
        ulonglong flags,
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
        *mut *mut void ptr,
        size_t size,
    );

    #[deprecated = "This API is deprecated, use hipHostMalloc() instead", cfg]
    hipError_t hipMemAllocHost(
        *mut *mut void ptr,
        size_t size,
    );

    hipError_t hipHostFree(*mut void ptr);

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipMemcpyToArray(
        hipArray_t dst,
        size_t wOffset,
        size_t hOffset,
        *const void src,
        size_t count,
        hipMemcpyKind kind,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipMemcpyFromArray(
        *mut void dst,
        hipArray_const_t srcArray,
        size_t wOffset,
        size_t hOffset,
        size_t count,
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
        uint numExtSems,
        hipStream_t stream,
    );

    /// This API is currently not supported on Linux.
    hipError_t hipWaitExternalSemaphoresAsync(
        *const hipExternalSemaphore_t extSemArray,
        *const hipExternalSemaphoreWaitParams paramsArray,
        uint numExtSems,
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
        *mut *mut void devPtr,
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
        *mut *mut void dev_ptr,
        size_t size,
        hipMemPool_t mem_pool,
        hipStream_t stream,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipFreeAsync(
        *mut void dev_ptr,
        hipStream_t stream,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolTrimTo(
        hipMemPool_t mem_pool,
        size_t min_bytes_to_hold,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolSetAttribute(
        hipMemPool_t mem_pool,
        hipMemPoolAttr attr,
        *mut void value,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolGetAttribute(
        hipMemPool_t mem_pool,
        hipMemPoolAttr attr,
        *mut void value,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolSetAccess(
        hipMemPool_t mem_pool,
        *const hipMemAccessDesc desc_list,
        size_t count,
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
        *mut *mut void dev_ptr,
        size_t size,
        hipMemPool_t mem_pool,
        hipStream_t stream,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolExportToShareableHandle(
        *mut void shared_handle,
        hipMemPool_t mem_pool,
        hipMemAllocationHandleType handle_type,
        uint flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolImportFromShareableHandle(
        *mut hipMemPool_t mem_pool,
        *mut void shared_handle,
        hipMemAllocationHandleType handle_type,
        uint flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolExportPointer(
        *mut hipMemPoolPtrExportData export_data,
        *mut void dev_ptr,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    ///
    /// This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemPoolImportPointer(
        *mut *mut void dev_ptr,
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
        *mut *mut void dev_ptr,
        size_t size,
        uint flags,
    );

    /// It is recommend to do the capability check before call this API.
    hipError_t hipMemPrefetchAsync(
        *const void dev_ptr,
        size_t count,
        int device,
        hipStream_t stream,
    );

    /// It is recommend to do the capability check before call this API.
    #[since = 7 .1, "7.1"]
    hipError_t hipMemPrefetchAsync_v2(
        *const void dev_ptr,
        size_t count,
        hipMemLocation location,
        uint flags,
        hipStream_t stream,
    );

    /// It is recommend to do the capability check before call this API.
    #[since = 7 .2, "7.2"]
    hipError_t hipMemPrefetchBatchAsync(
        *mut *mut void dev_ptrs,
        *mut size_t sizes,
        size_t count,
        *mut hipMemLocation prefetch_locs,
        *mut size_t prefetch_loc_idxs,
        size_t num_prefetch_locs,
        ulonglong flags,
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
        *mut *mut void dev_ptrs,
        *mut size_t sizes,
        size_t count,
        ulonglong flags,
        hipStream_t stream,
    );

    /// This is the driver API variant that uses hipDeviceptr_t instead of void*. Both hipMemDiscardBatchAsync and hipDrvMemDiscardBatchAsync use the same internal implementation.
    ///
    /// Reading from a discarded range without first writing or prefetching to it will return an indeterminate value.
    #[since = 7 .2, "7.2"]
    hipError_t hipDrvMemDiscardBatchAsync(
        *mut hipDeviceptr_t dptrs,
        *mut size_t sizes,
        size_t count,
        ulonglong flags,
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
        *mut *mut void dptrs,
        *mut size_t sizes,
        size_t count,
        *mut hipMemLocation prefetchLocs,
        *mut size_t prefetchLocIdxs,
        size_t numPrefetchLocs,
        ulonglong flags,
        hipStream_t stream,
    );

    /// This is the driver API variant that uses hipDeviceptr_t instead of void*.
    #[since = 7 .2, "7.2"]
    hipError_t hipDrvMemDiscardAndPrefetchBatchAsync(
        *mut hipDeviceptr_t dptrs,
        *mut size_t sizes,
        size_t count,
        *mut hipMemLocation prefetchLocs,
        *mut size_t prefetchLocIdxs,
        size_t numPrefetchLocs,
        ulonglong flags,
        hipStream_t stream,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipMemAdvise(
        *const void dev_ptr,
        size_t count,
        hipMemoryAdvise advice,
        int device,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 7 .1, "7.1"]
    hipError_t hipMemAdvise_v2(
        *const void dev_ptr,
        size_t count,
        hipMemoryAdvise advice,
        hipMemLocation location,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipMemRangeGetAttributes(
        *mut *mut void data,
        *mut size_t data_sizes,
        *mut hipMemRangeAttribute attributes,
        size_t num_attributes,
        *const void dev_ptr,
        size_t count,
    );

    /// This API is under development. Currently it is a no-operation (NOP) function on AMD GPUs and returns hipSuccess.
    hipError_t hipStreamAttachMemAsync(
        hipStream_t stream,
        *mut void dev_ptr,
        size_t length,
        int flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemAddressFree(
        *mut void devPtr,
        size_t size,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemAddressReserve(
        *mut *mut void ptr,
        size_t size,
        size_t alignment,
        *mut void addr,
        ulonglong flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemCreate(
        *mut hipMemGenericAllocationHandle_t handle,
        size_t size,
        *const hipMemAllocationProp prop,
        ulonglong flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemExportToShareableHandle(
        *mut void shareableHandle,
        hipMemGenericAllocationHandle_t handle,
        hipMemAllocationHandleType handleType,
        ulonglong flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemGetAccess(
        *mut ulonglong flags,
        *const hipMemLocation location,
        *mut void ptr,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemGetAllocationGranularity(
        *mut size_t granularity,
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
        *mut void osHandle,
        hipMemAllocationHandleType shHandleType,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemMap(
        *mut void ptr,
        size_t size,
        size_t offset,
        hipMemGenericAllocationHandle_t handle,
        ulonglong flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemMapArrayAsync(
        *mut hipArrayMapInfo mapInfoList,
        uint count,
        hipStream_t stream,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle);

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemRetainAllocationHandle(
        *mut hipMemGenericAllocationHandle_t handle,
        *mut void addr,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemSetAccess(
        *mut void ptr,
        size_t size,
        *const hipMemAccessDesc desc,
        size_t count,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    #[since = 5 .1, "5.1"]
    hipError_t hipMemUnmap(
        *mut void ptr,
        size_t size,
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
        uint numLevels,
        uint flags,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipFreeMipmappedArray(hipMipmappedArray_t mipmappedArray);

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipGetMipmappedArrayLevel(
        *mut hipArray_t levelArray,
        hipMipmappedArray_const_t mipmappedArray,
        uint level,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipMipmappedArrayCreate(
        *mut hipMipmappedArray_t pHandle,
        *mut HIP_ARRAY3D_DESCRIPTOR pMipmappedArrayDesc,
        uint numMipmapLevels,
    );

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipMipmappedArrayDestroy(hipMipmappedArray_t hMipmappedArray);

    /// This API is implemented on Linux and is under development on Microsoft Windows.
    hipError_t hipMipmappedArrayGetLevel(
        *mut hipArray_t pLevelArray,
        hipMipmappedArray_t hMipMappedArray,
        uint level,
    );

    hipError_t hipBindTextureToMipmappedArray(
        *const textureReference tex,
        hipMipmappedArray_const_t mipmappedArray,
        *const hipChannelFormatDesc desc,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipGetTextureReference(
        *mut *const texref textureReference,
        *const void symbol,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetBorderColor(
        *mut float pBorderColor,
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
        int dim,
        hipTextureAddressMode am,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetArray(
        *mut textureReference tex,
        hipArray_const_t array,
        uint flags,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetFilterMode(
        *mut textureReference texRef,
        hipTextureFilterMode fm,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetFlags(
        *mut textureReference texRef,
        uint Flags,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetFormat(
        *mut textureReference texRef,
        hipArray_Format fmt,
        int NumPackedComponents,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipBindTexture(
        *mut size_t offset,
        *const textureReference tex,
        *const void devPtr,
        *const hipChannelFormatDesc desc,
        size_t size,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipBindTexture2D(
        *mut size_t offset,
        *const textureReference tex,
        *const void devPtr,
        *const hipChannelFormatDesc desc,
        size_t width,
        size_t height,
        size_t pitch,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipBindTextureToArray(
        *const textureReference tex,
        hipArray_const_t array,
        *const hipChannelFormatDesc desc,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipGetTextureAlignmentOffset(
        *mut size_t offset,
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
        int dim,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetFilterMode(
        *mut hipTextureFilterMode pfm,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetFlags(
        *mut uint pFlags,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetFormat(
        *mut hipArray_Format pFormat,
        *mut int pNumChannels,
        *const textureReference texBuf,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetMaxAnisotropy(
        *mut int pmaxAnsio,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetMipmapFilterMode(
        *mut hipTextureFilterMode pfm,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetMipmapLevelBias(
        *mut float pbias,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetMipmapLevelClamp(
        *mut float pminMipmapLevelClamp,
        *mut float pmaxMipmapLevelClamp,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefGetMipMappedArray(
        *mut hipMipmappedArray_t pArray,
        *const textureReference texRef,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetAddress(
        *mut size_t ByteOffset,
        *mut textureReference texRef,
        hipDeviceptr_t dptr,
        size_t bytes,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetAddress2D(
        textureReference texRef,
        *const HIP_ARRAY_DESCRIPTOR desc,
        hipDeviceptr_t dptr,
        size_t Pitch,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetMaxAnisotropy(
        *mut textureReference texRef,
        uint maxAniso,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetBorderColor(
        *mut textureReference texRef,
        *mut float pBorderColor,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetMipmapFilterMode(
        *mut textureReference texRef,
        hipTextureFilterMode fm,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetMipmapLevelBias(
        *mut textureReference texRef,
        float bias,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetMipmapLevelClamp(
        *mut textureReference texRef,
        float minMipMapLevelClamp,
        float maxMipMapLevelClamp,
    );

    #[deprecated = "This API is deprecated.", cfg]
    hipError_t hipTexRefSetMipmappedArray(
        *mut textureReference texRef,
        *mut hipMipmappedArray mipmappedArray,
        uint Flags,
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
        size_t log_level,
        size_t log_size,
        size_t log_mask,
    );

    #[internal(cfg)]
    uint __hipGetPixelAddr(
        int x,
        int format,
        int order,
    );

    hipError_t hipDeviceCanAccessPeer(
        *mut int canAccessPeer,
        int deviceId,
        int peerDeviceId,
    );

    hipError_t hipDeviceEnablePeerAccess(
        int peerDeviceId,
        uint flags,
    );

    hipError_t hipDeviceDisablePeerAccess(int peerDeviceId);

    hipError_t hipMemcpyPeer(
        *mut void dst,
        int dstDeviceId,
        *const void src,
        int srcDeviceId,
        size_t sizeBytes,
    );

    hipError_t hipMemcpyPeerAsync(
        *mut void dst,
        int dstDeviceId,
        *const void src,
        int srcDevice,
        size_t sizeBytes,
        hipStream_t stream,
    );

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxCreate(
        *mut hipCtx_t ctx,
        uint flags,
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
        *mut uint apiVersion,
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
    hipError_t hipCtxGetFlags(*mut uint flags);

    /// PeerToPeer support is experimental.
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxEnablePeerAccess(
        hipCtx_t peerCtx,
        uint flags,
    );

    /// PeerToPeer support is experimental.
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx);

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    hipError_t hipDevicePrimaryCtxGetState(
        hipDevice_t dev,
        *mut uint flags,
        *mut int active,
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
        uint flags,
    );

    hipError_t hipModuleGetGlobal(
        *mut hipDeviceptr_t dptr,
        *mut size_t bytes,
        hipModule_t hmod,
        *const char name,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipModuleLoadFatBinary(
        *mut hipModule_t module,
        *const void fatbin,
    );

    hipError_t hipModuleLoad(
        *mut hipModule_t module,
        *const char fname,
    );

    hipError_t hipModuleUnload(hipModule_t module);

    hipError_t hipModuleGetFunction(
        *mut hipFunction_t function,
        hipModule_t module,
        *const char kname,
    );

    #[since = 7 .1, "7.1"]
    hipError_t hipModuleGetFunctionCount(
        *mut int count,
        hipModule_t module,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipKernelGetAttribute(
        *mut int pi,
        hipFunction_attribute attrib,
        hipKernel_t kernel,
        hipDevice_t dev,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryLoadData(
        *mut hipLibrary_t library,
        *const void code,
        *mut hipJitOption jitOptions,
        *mut *mut void jitOptionsValues,
        uint numJitOptions,
        *mut hipLibraryOption libraryOptions,
        *mut *mut void libraryOptionValues,
        uint numLibraryOptions,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryLoadFromFile(
        *mut hipLibrary_t library,
        *const char fileName,
        *mut hipJitOption jitOptions,
        *mut *mut void jitOptionsValues,
        uint numJitOptions,
        *mut hipLibraryOption libraryOptions,
        *mut *mut void libraryOptionValues,
        uint numLibraryOptions,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryUnload(hipLibrary_t library);

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryGetKernel(
        *mut hipKernel_t pKernel,
        hipLibrary_t library,
        *const char name,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryGetKernelCount(
        *mut uint count,
        hipLibrary_t library,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryGetGlobal(
        *mut *mut void dptr,
        *mut size_t bytes,
        hipLibrary_t library,
        *const char name,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryGetManaged(
        *mut *mut void dptr,
        *mut size_t bytes,
        hipLibrary_t library,
        *const char name,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipLibraryEnumerateKernels(
        *mut hipKernel_t kernels,
        uint numKernels,
        hipLibrary_t library,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipKernelGetLibrary(
        *mut hipLibrary_t library,
        hipKernel_t kernel,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipKernelGetName(
        *const char name,
        hipKernel_t kernel,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipKernelGetParamInfo(
        hipKernel_t kernel,
        size_t paramIndex,
        *mut size_t paramOffset,
        *mut size_t paramSize,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipGetFuncBySymbol(
        *mut hipFunction_t functionPtr,
        *const void symbolPtr,
    );

    #[since = 6 .1, "6.1"]
    hipError_t hipGetDriverEntryPoint(
        *const char symbol,
        *mut *mut void funcPtr,
        ulonglong flags,
        *mut hipDriverEntryPointQueryResult driverStatus,
    );

    hipError_t hipModuleGetTexRef(
        *mut *mut textureReference texRef,
        hipModule_t hmod,
        *const char name,
    );

    hipError_t hipModuleLoadData(
        *mut hipModule_t module,
        *const void image,
    );

    hipError_t hipModuleLoadDataEx(
        *mut hipModule_t module,
        *const void image,
        uint numOptions,
        *mut hipJitOption options,
        *mut *mut void optionValues,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipLinkAddData(
        hipLinkState_t state,
        hipJitInputType ty,
        *mut void data,
        size_t size,
        *const char name,
        uint numOptions,
        *mut hipJitOption options,
        *mut *mut void optionValues,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipLinkComplete(
        hipLinkState_t state,
        *mut *mut void hipBinOut,
        *mut size_t sizeOut,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipLinkCreate(
        uint numOptions,
        *mut hipJitOption options,
        *mut *mut void optionValues,
        *mut hipLinkState_t stateOut,
    );

    #[since = 6 .4, "6.4"]
    hipError_t hipLinkDestroy(hipLinkState_t state);

    hipError_t hipModuleOccupancyMaxPotentialBlockSize(
        *mut int gridSize,
        *mut int blockSize,
        hipFunction_t f,
        size_t dynSharedMemPerBlk,
        int blockSizeLimit,
    );

    hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
        *mut int gridSize,
        *mut int blockSize,
        hipFunction_t f,
        size_t dynSharedMemPerBlk,
        int blockSizeLimit,
        uint flags,
    );

    hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        *mut int numBlocks,
        hipFunction_t f,
        int blockSize,
        size_t dynSharedMemPerBlk,
    );

    hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        *mut int numBlocks,
        hipFunction_t f,
        int blockSize,
        size_t dynSharedMemPerBlk,
        uint flags,
    );

    hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(
        *mut int numBlocks,
        *const void f,
        int blockSize,
        size_t dynSharedMemPerBlk,
    );

    hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        *mut int numBlocks,
        *const void f,
        int blockSize,
        size_t dynSharedMemPerBlk,
        uint flags,
    );

    hipError_t hipOccupancyMaxPotentialBlockSize(
        *mut int gridSize,
        *mut int blockSize,
        *const void f,
        size_t dynSharedMemPerBlk,
        int blockSizeLimit,
    );

    #[since = 7 .2, "7.2"]
    hipError_t hipOccupancyAvailableDynamicSMemPerBlock(
        *mut size_t dynamicSmemSize,
        *const void f,
        int numBlocks,
        int blockSize,
    );

    #[since = 6 .5, "6.5"]
    hipError_t hipOccupancyMaxActiveClusters(
        *mut int numClusters,
        *const void f,
        *const hipLaunchConfig_t config,
    );

    #[since = 6 .5, "6.5"]
    hipError_t hipOccupancyMaxPotentialClusterSize(
        *mut int clusterSize,
        *const void f,
        *const hipLaunchConfig_t config,
    );

    #[deprecated = "hipProfilerStart API is deprecated, use roctracer/rocTX instead."]
    hipError_t hipProfilerStart();

    #[deprecated = "hipProfilerStop API is deprecated, use roctracer/rocTX instead."]
    hipError_t hipProfilerStop();

    hipError_t hipConfigureCall(
        dim3 gridDim,
        dim3 blockDim,
        size_t sharedMem,
        hipStream_t stream,
    );

    hipError_t hipSetupArgument(
        *const void arg,
        size_t size,
        size_t offset,
    );

    hipError_t hipLaunchByPtr(*const void func);

    #[internal(cfg)]
    hipError_t __hipPushCallConfiguration(
        dim3 gridDim,
        dim3 blockDim,
        size_t sharedMem,
        hipStream_t stream,
    );

    #[internal(cfg)]
    hipError_t __hipPopCallConfiguration(
        *mut dim3 gridDim,
        *mut dim3 blockDim,
        *mut size_t sharedMem,
        *mut hipStream_t stream,
    );

    hipError_t hipLaunchKernel(
        *const void function_address,
        dim3 numBlocks,
        dim3 dimBlocks,
        *mut *mut void args,
        size_t sharedMemBytes,
        hipStream_t stream,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipLaunchHostFunc(
        hipStream_t stream,
        hipHostFn_t func,
        *mut void userData,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipDrvMemcpy2DUnaligned(*const hip_Memcpy2D pCopy);

    hipError_t hipExtLaunchKernel(
        *const void function_address,
        dim3 numBlocks,
        dim3 dimBlocks,
        *mut *mut void args,
        size_t sharedMemBytes,
        hipStream_t stream,
        hipEvent_t startEvent,
        hipEvent_t stopEvent,
        int flags,
    );

    *const char hipApiName(uint32_t id);

    *const char hipKernelNameRef(hipFunction_t f);

    *const char hipKernelNameRefByPtr(
        *const void hostFunction,
        hipStream_t stream,
    );

    int hipGetStreamDeviceId(hipStream_t stream);

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
        size_t numDependencies,
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
        *mut ulonglong pId,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipStreamGetCaptureInfo_v2(
        hipStream_t stream,
        *mut hipStreamCaptureStatus captureStatus_out,
        *mut ulonglong id_out,
        *mut hipGraph_t graph_out,
        *mut *const hipGraphNode_t dependencies_out ,
        *mut size_t numDependencies_out,
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
        size_t numDependencies,
        uint flags,
    );

    #[since = 5 .0, "5.0"]
    hipError_t hipThreadExchangeStreamCaptureMode(*mut hipStreamCaptureMode mode);

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphCreate(
        *mut hipGraph_t pGraph,
        uint flags,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphDestroy(hipGraph_t graph);

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphAddDependencies(
        hipGraph_t graph,
        *const hipGraphNode_t from,
        *const hipGraphNode_t to,
        size_t numDependencies,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphRemoveDependencies(
        hipGraph_t graph,
        *const hipGraphNode_t from,
        *const hipGraphNode_t to,
        size_t numDependencies,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphGetEdges(
        hipGraph_t graph,
        *mut hipGraphNode_t from,
        *mut hipGraphNode_t to,
        *mut size_t numEdges,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphGetNodes(
        hipGraph_t graph,
        *mut hipGraphNode_t nodes,
        *mut size_t numNodes,
    );

    #[since = 4 .4, "4.4"]
    hipError_t hipGraphGetRootNodes(
        hipGraph_t graph,
        *mut hipGraphNode_t pRootNodes,
        *mut size_t pNumRootNodes,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphNodeGetDependencies(
        hipGraphNode_t node,
        *mut hipGraphNode_t pDependencies,
        *mut size_t pNumDependencies,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphNodeGetDependentNodes(
        hipGraphNode_t node,
        *mut hipGraphNode_t pDependentNodes,
        *mut size_t pNumDependentNodes,
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
        *mut char pLogBuffer,
        size_t bufferSize,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphInstantiateWithFlags(
        *mut hipGraphExec_t pGraphExec,
        hipGraph_t graph,
        ulonglong flags,
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
        size_t numDependencies,
        *mut hipGraphNodeParams nodeParams,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipGraphExecGetFlags(
        hipGraphExec_t graphExec,
        *mut ulonglong flags,
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
        size_t numDependencies,
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
        size_t numDependencies,
        *const HIP_MEMCPY3D copyParams,
        hipCtx_t ctx,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphAddMemcpyNode(
        *mut hipGraphNode_t pGraphNode_t,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        size_t numDependencies,
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
        size_t numDependencies,
        *mut void dst,
        *const void src,
        size_t count,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphMemcpyNodeSetParams1D(
        hipGraphNode_t node,
        *mut void dst,
        *const void src,
        size_t count,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecMemcpyNodeSetParams1D(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t node,
        *mut void dst,
        *const void src,
        size_t count,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphAddMemcpyNodeFromSymbol(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        size_t numDependencies,
        *mut void dst,
        *const void symbol,
        size_t count,
        size_t offset,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphMemcpyNodeSetParamsFromSymbol(
        hipGraphNode_t node,
        *mut void dst,
        *const void symbol,
        size_t count,
        size_t offset,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecMemcpyNodeSetParamsFromSymbol(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t node,
        *mut void dst,
        *const void symbol,
        size_t count,
        size_t offset,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphAddMemcpyNodeToSymbol(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        size_t numDependencies,
        *const void symbol,
        *const void src,
        size_t count,
        size_t offset,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphMemcpyNodeSetParamsToSymbol(
        hipGraphNode_t node,
        *const void symbol,
        *const void src,
        size_t count,
        size_t offset,
        hipMemcpyKind kind,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphExecMemcpyNodeSetParamsToSymbol(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t node,
        *const void symbol,
        *const void src,
        size_t count,
        size_t offset,
        hipMemcpyKind kind,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphAddMemsetNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        size_t numDependencies,
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
        size_t numDependencies,
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
        size_t numDependencies,
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
        size_t numDependencies,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphAddEventRecordNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        size_t numDependencies,
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
        size_t numDependencies,
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
        size_t numDependencies,
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
        size_t numDependencies,
        *mut void dev_ptr,
    );

    #[since = 5 .5, "5.5"]
    hipError_t hipGraphMemFreeNodeGetParams(
        hipGraphNode_t node,
        *mut void dev_ptr,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipDeviceGetGraphMemAttribute(
        int device,
        hipGraphMemAttributeType attr,
        *mut void value,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipDeviceGraphMemTrim(int device);

    #[since = 5 .3, "5.3"]
    hipError_t hipUserObjectCreate(
        *mut hipUserObject_t object_out,
        *mut void ptr,
        hipHostFn_t destroy,
        uint initialRefcount,
        uint flags,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipUserObjectRelease(
        hipUserObject_t object,
        uint count,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipUserObjectRetain(
        hipUserObject_t object,
        uint count,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphRetainUserObject(
        hipGraph_t graph,
        hipUserObject_t object,
        uint count,
        uint flags,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphReleaseUserObject(
        hipGraph_t graph,
        hipUserObject_t object,
        uint count,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphDebugDotPrint(
        hipGraph_t graph,
        *const char path,
        uint flags,
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
        uint isEnabled,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphNodeGetEnabled(
        hipGraphExec_t hGraphExec,
        hipGraphNode_t hNode,
        *mut uint isEnabled,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphAddExternalSemaphoresWaitNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        size_t numDependencies,
        *const hipExternalSemaphoreWaitNodeParams nodeParams,
    );

    #[since = 5 .3, "5.3"]
    hipError_t hipGraphAddExternalSemaphoresSignalNode(
        *mut hipGraphNode_t pGraphNode,
        hipGraph_t graph,
        *const hipGraphNode_t pDependencies,
        size_t numDependencies,
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
        size_t numDependencies,
        *const hipMemsetParams memsetParams,
        hipCtx_t ctx,
    );

    #[since = 6 .2, "6.2"]
    hipError_t hipDrvGraphAddMemFreeNode(
        *mut hipGraphNode_t phGraphNode,
        hipGraph_t hGraph,
        *const hipGraphNode_t dependencies,
        size_t numDependencies,
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
        *mut void handle,
        hipDeviceptr_t dptr,
        size_t size,
        hipMemRangeHandleType handleType,
        ulonglong flags,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphicsMapResources(
        int count,
        *mut hipGraphicsResource_t resources,
        hipStream_t stream,
    );

    #[since = 4 .5, "4.5"]
    hipError_t hipGraphicsSubResourceGetMappedArray(
        *mut hipArray_t array,
        hipGraphicsResource_t resource,
        uint arrayIndex,
        uint mipLevel,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphicsResourceGetMappedPointer(
        *mut *mut void devPtr,
        *mut size_t size,
        hipGraphicsResource_t resource,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphicsUnmapResources(
        int count,
        *mut hipGraphicsResource_t resrources,
        hipStream_t stream,
    );

    #[since = 4 .3, "4.3"]
    hipError_t hipGraphicsUnregisterResource(hipGraphicsResource_t resource);

    #[since = 5 .5, "5.5"]
    hipError_t hipModuleLaunchCooperativeKernel(
        hipFunction_t f,
        uint gridDimX,
        uint gridDimY,
        uint gridDimZ,
        uint blockDimX,
        uint blockDimY,
        uint blockDimZ,
        uint sharedMemBytes,
        hipStream_t stream,
        *mut *mut void kernelParams,
    );

    #[since = 5 .5, "5.5"]
    hipError_t hipModuleLaunchCooperativeKernelMultiDevice(
        *mut hipFunctionLaunchParams launchParamsList,
        uint numDevices,
        uint flags,
    );

    hipError_t hipLaunchCooperativeKernel(
        *const void f,
        dim3 gridDim,
        dim3 blockDim,
        *mut *mut void kernelParams,
        uint sharedMemBytes,
        hipStream_t stream,
    );

    #[since = 5 .2, "5.2"]
    hipError_t hipLaunchCooperativeKernelMultiDevice(
        hipLaunchParams launchParamsList,
        int numDevices,
        uint flags,
    );
}

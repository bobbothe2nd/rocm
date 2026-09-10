//! All HIP/HIPRTC functions and types as of ROCm 7.15.

mod types;

pub use types::*;

use core::ffi::*;

#[cfg(feature = "dynamic-loading")]
use crate::version::{ROCM_MAJOR as MAJOR, ROCM_MINOR as MINOR};

link! {
    amdhip64 : [7, 8, 9, 10] : rocm_ver;

    pub fn hipInit(flags: c_uint) -> hipError_t;

    pub fn hipDriverGetVersion(driverVersion: *mut c_int) -> hipError_t;

    pub fn hipRuntimeGetVersion(runtimeVersion: *mut c_int) -> hipError_t;

    pub fn hipDeviceGet(
        device: *mut hipDevice_t,
        ordinal: c_int,
    ) -> hipError_t;

    pub fn hipDeviceComputeCapability(
        major: *mut c_int,
        minor: *mut c_int,
        device: hipDevice_t,
    ) -> hipError_t;

    pub fn hipDeviceGetName(
        name: *mut c_char,
        len: *mut c_int,
        device: hipDevice_t,
    ) -> hipError_t;

    #[since = 5 .1, "5.1"]
    pub fn hipDeviceGetUuid(
        uuid: *mut hipUUID,
        device: hipDevice_t,
    ) -> hipError_t;

    #[since = 5 .1, "5.1"]
    pub fn hipDeviceGetLuid(
        luid: *mut c_char,
        deviceNodeMask: *mut c_uint,
        device: hipDevice_t,
    ) -> hipError_t;

    pub fn hipDeviceGetP2PAttribute(
        value: *mut c_int,
        attr: hipDeviceP2PAttr,
        srcDevice: c_int,
        dstDevice: c_int,
    ) -> hipError_t;

    pub fn hipDeviceGetPCIBusId(
        pciBusId: *mut c_char,
        len: c_int,
        device: c_int,
    ) -> hipError_t;

    pub fn hipDeviceGetByPCIBusId(
        device: *mut c_int,
        pciBusId: *const c_char
    ) -> hipError_t;

    pub fn hipDeviceTotalMem(
        bytes: *mut usize,
        device: hipDevice_t,
    ) -> hipError_t;

    pub fn hipDeviceSynchronize() -> hipError_t;

    pub fn hipDeviceReset() -> hipError_t;

    pub fn hipSetDevice(deviceId: c_int) -> hipError_t;

    pub fn hipSetValidDevices(
        device_arr: *mut c_int,
        len: c_int,
    ) -> hipError_t;

    pub fn hipGetDevice(device: *mut c_int) -> hipError_t;

    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;

    pub fn hipDeviceGetAttribute(
        pi: *mut c_int,
        attr: hipDeviceAttribute_t,
        deviceId: c_int,
    ) -> hipError_t;

    #[since = 5 .1, "5.1"]
    pub fn hipDeviceGetDefaultMemPool(
        mem_pool: *mut hipMemPool_t,
        device: c_int,
    ) -> hipError_t;

    #[since = 5 .1, "5.1"]
    pub fn hipDeviceSetMemPool(
        device: c_int,
        mem_pool: hipMemPool_t,
    ) -> hipError_t;

    #[since = 5 .1, "5.1"]
    pub fn hipDeviceGetMemPool(
        device: c_int,
        mem_pool: *mut hipMemPool_t,
    ) -> hipError_t;

    #[doc = "assumes hipDeviceProp_tR0600=hipDeviceProp_t"]
    pub fn hipGetDeviceProperties(
        prop: *mut hipDeviceProp_tR0600,
        deviceId: c_int,
    ) -> hipError_t;

    pub fn hipDeviceSetCacheConfig(
        cacheConfig: hipFuncCache_t,
    ) -> hipError_t;

    pub fn hipDeviceGetCacheConfig(
        cacheConfig: *mut hipFuncCache_t,
    ) -> hipError_t;

    pub fn hipDeviceGetLimit(
        pValue: *mut usize,
        limit: hipLimit_t,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipDeviceSetLimit(
        limit: *mut hipLimit_t,
        pValue: usize,
    ) -> hipError_t;

    pub fn hipDeviceGetSharedMemConfig(
        pConfig: *mut hipSharedMemConfig,
    ) -> hipError_t;

    pub fn hipGetDeviceFlags(
        flags: *mut c_uint,
    ) -> hipError_t;

    pub fn hipDeviceSetSharedMemConfig(
        config: hipSharedMemConfig,
    ) -> hipError_t;

    pub fn hipSetDeviceFlags(
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipInitDevice(
        device: c_int,
        deviceFlags: c_uint,
        flags: c_uint,
    ) -> hipError_t;

    #[doc = "assumes hipDeviceProp_tR0600=hipDeviceProp_t"]
    pub fn hipChooseDevice(
        device: *mut c_int,
        prop: *const hipDeviceProp_tR0600,
    ) -> hipError_t;

    pub fn hipExtGetLinkTypeAndHopCount(
        device1: c_int,
        device2: c_int,
        linktype: *mut u32,
        hopcount: u32,
    ) -> hipError_t;

    pub fn hipIpcGetMemHandle(
        handle: *mut hipIpcMemHandle_t,
        devPtr: *mut c_void,
    ) -> hipError_t;

    pub fn hipIpcOpenMemHandle(
        devPtr: *mut *mut c_void,
        handle: hipIpcMemHandle_t,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipIpcCloseMemHandle(
        devPtr: *mut c_void,
    ) -> hipError_t;

    pub fn hipIpcGetEventHandle(
        handle: *mut hipIpcEventHandle_t,
        event: hipEvent_t,
    ) -> hipError_t;

    pub fn hipIpcOpenEventHandle(
        event: *mut hipEvent_t,
        handle: hipIpcEventHandle_t,
    ) -> hipError_t;

    pub fn hipExtModuleLaunchKernel(
        f: hipFunction_t,
        globalWorkSizeX: u32,
        globalWorkSizeY: u32,
        globalWorkSizeZ: u32,
        localWorkSizeX: u32,
        localWorkSizeY: u32,
        localWorkSizeZ: u32,
        sharedMemBytes: usize,
        hStream: hipStream_t,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
        startEvent: hipEvent_t,
        stopEvent: hipEvent_t,
        flags: u32,
    ) -> hipError_t;

    pub fn hipHccModuleLaunchKernel(
        f: hipFunction_t,
        globalWorkSizeX: u32,
        globalWorkSizeY: u32,
        globalWorkSizeZ: u32,
        localWorkSizeX: u32,
        localWorkSizeY: u32,
        localWorkSizeZ: u32,
        sharedMemBytes: usize,
        hStream: hipStream_t,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
        startEvent: hipEvent_t,
        stopEvent: hipEvent_t,
    ) -> hipError_t;

    pub fn hipFuncSetAttribute(
        func: *const c_void,
        attr: hipFuncAttribute,
        value: c_int,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipKernelSetAttribute(
        attrib: hipFunction_attribute,
        value: c_int,
        kernel: hipKernel_t,
        dev: hipDevice_t,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipKernelGetFunction(
        pFunc: *mut hipFunction_t,
        kernel: hipKernel_t,
    ) -> hipError_t;

    pub fn hipFuncSetCacheConfig(
        func: *const c_void,
        config: hipFuncCache_t,
    ) -> hipError_t;

    pub fn hipFuncSetSharedMemConfig(
        func: *const c_void,
        config: hipSharedMemConfig,
    ) -> hipError_t;

    pub fn hipFuncGetAttributes(
        attr: *mut hipFuncAttributes,
        func: *const c_void,
    ) -> hipError_t;

    pub fn hipFuncGetAttribute(
        value: *mut c_int,
        attrib: hipFunction_attribute,
        hfunc: hipFunction_t,
    ) -> hipError_t;

    pub fn hipModuleLaunchKernel(
        f: hipFunction_t,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        stream: hipStream_t,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;

    pub fn hipExtLaunchMultiKernelMultiDevice(
        launchParamsList: *mut hipLaunchParams,
        numDevices: c_int,
        flags: c_uint,
    ) -> hipError_t;

    #[since = 6 .5, "6.5"]
    pub fn hipLaunchKernelExC(
        config: *const hipLaunchConfig_t,
        fPtr: *const c_void,
        args: *mut *mut c_void,
    ) -> hipError_t;

    #[since = 6 .5, "6.5"]
    pub fn hipDrvLaunchKernelEx(
        config: *const HIP_LAUNCH_CONFIG,
        f: hipFunction_t,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> hipError_t;

    pub fn hipGetLastError() -> hipError_t;

    #[since = 6 .0, "6.0"]
    pub fn hipExtGetLastError() -> hipError_t;

    pub fn hipGetErrorName(hip_error: hipError_t) -> *const c_char;

    pub fn hipGetErrorString(hip_error: hipError_t) -> *const c_char;

    #[since = 5 .3, "5.3"]
    pub fn hipDrvGetErrorName(
        hip_Error: hipError_t,
        errorString: *mut *const c_char,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipDrvGetErrorString(
        hip_Error: hipError_t,
        errorString: *mut *const c_char,
    ) -> hipError_t;

    #[since = 7 .14, "7.14"]
    pub fn hipStreamCreate(
        stream: *mut hipStream_t,
    ) -> hipError_t;

    pub fn hipStreamCreateWithFlags(
        stream: *mut hipStream_t,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipStreamCreateWithPriority(
        stream: *mut hipStream_t,
        flags: c_uint,
        priority: c_int,
    ) -> hipError_t;

    pub fn hipDeviceGetStreamPriorityRange(
        leastPriority: *mut c_int,
        greatestPriority: *mut c_int,
    ) -> hipError_t;

    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;

    pub fn hipStreamQuery(stream: hipStream_t) -> hipError_t;

    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;

    pub fn hipStreamWaitEvent(
        stream: hipStream_t,
        event: hipEvent_t,
        flags: c_uint,
    ) -> hipError_t;

    #[since = 5 .2, "5.2"]
    pub fn hipStreamGetFlags(
        stream: hipStream_t,
        flags: *mut c_uint,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipStreamGetId(
        stream: hipStream_t,
        streamId: *mut c_ulonglong,
    ) -> hipError_t;

    pub fn hipStreamGetPriority(
        stream: hipStream_t,
        priority: *mut c_int,
    ) -> hipError_t;

    pub fn hipStreamGetDevice(
        stream: hipStream_t,
        device: *mut hipDevice_t,
    ) -> hipError_t;

    pub fn hipExtStreamCreateWithCUMask(
        stream: *mut hipStream_t,
        cuMaskSize: u32,
        cuMask: *const u32,
    ) -> hipError_t;

    pub fn hipExtStreamGetCUMask(
        stream: hipStream_t,
        cuMaskSize: u32,
        cuMask: *mut u32,
    ) -> hipError_t;

    pub fn hipStreamAddCallback(
        stream: hipStream_t,
        callback: hipStreamCallback_t,
        userData: *mut c_void,
        flags: c_uint,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipStreamSetAttribute(
        stream: hipStream_t,
        attr: hipLaunchAttributeID,
        value: *const hipLaunchAttributeValue,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipStreamGetAttribute(
        stream: hipStream_t,
        attr: hipLaunchAttributeID,
        value_out: *mut hipLaunchAttributeValue,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipStreamCopyAttributes(
        dst: hipStream_t,
        src: hipStream_t,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipStreamWaitValue32(
        stream: hipStream_t,
        ptr: *mut c_void,
        value: u32,
        flags: c_uint,
        mask: u32,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipStreamWaitValue64(
        stream: hipStream_t,
        ptr: *mut c_void,
        value: u64,
        flags: c_uint,
        mask: u64,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipStreamWriteValue32(
        stream: hipStream_t,
        ptr: *mut c_void,
        value: u32,
        flags: c_uint,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipStreamWriteValue64(
        stream: hipStream_t,
        ptr: *mut c_void,
        value: u64,
        flags: c_int,
    ) -> hipError_t;

    #[since = 6 .4, "6.4"]
    pub fn hipStreamBatchMemOp(
        stream: hipStream_t,
        count: c_int,
        paramArray: *mut hipStreamBatchMemOpParams,
        flags: c_int,
    ) -> hipError_t;

    #[since = 6 .4, "6.4"]
    pub fn hipGraphAddBatchMemOpNode(
        phGraphNode: *mut hipGraphNode_t,
        hGraph: hipGraph_t,
        dependencies: *const hipGraphNode_t,
        numDependencies: usize,
        nodeParams: *const hipBatchMemOpNodeParams,
    ) -> hipError_t;

    #[since = 6 .4, "6.4"]
    pub fn hipGraphBatchMemOpNodeGetParams(
        hNode: hipGraphNode_t,
        nodeParams_out: *mut hipBatchMemOpNodeParams,
    ) -> hipError_t;

    #[since = 6 .4, "6.4"]
    pub fn hipGraphBatchMemOpNodeSetParams(
        hNode: hipGraphNode_t,
        nodeParams: *mut hipBatchMemOpNodeParams,
    ) -> hipError_t;

    #[since = 6 .4, "6.4"]
    pub fn hipGraphExecBatchMemOpNodeSetParams(
        hGraphExec: hipGraphExec_t,
        hNode: hipGraphNode_t,
        nodeParams: *const hipBatchMemOpNodeParams,
    ) -> hipError_t;

    pub fn hipEventCreateWithFlags(
        event: *mut hipEvent_t,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipEventCreate(event: *mut hipEvent_t) -> hipError_t;

    #[since = 6 .4, "6.4"]
    pub fn hipEventRecordWithFlags(
        event: hipEvent_t,
        stream: hipStream_t,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipEventRecord(
        event: hipEvent_t,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipEventDestroy(event: hipEvent_t) -> hipError_t;

    pub fn hipEventSynchronize(event: hipEvent_t) -> hipError_t;

    pub fn hipEventElapsedTime(
        ms: *mut f32,
        start: hipEvent_t,
        stop: hipEvent_t,
    ) -> hipError_t;

    pub fn hipEventQuery(event: hipEvent_t) -> hipError_t;

    pub fn hipPointerSetAttribute(
        value: *const c_void,
        attribute: hipPointer_attribute,
        ptr: hipDeviceptr_t,
    ) -> hipError_t;

    pub fn hipPointerGetAttributes(
        attributes: *mut hipPointerAttribute_t,
        ptr: *const c_void,
    ) -> hipError_t;

    pub fn hipPointerGetAttribute(
        data: *mut c_void,
        attribute: hipPointer_attribute,
        ptr: hipDeviceptr_t,
    ) -> hipError_t;

    #[since = 5 .0, "5.0"]
    pub fn hipDrvPointerGetAttributes(
        numAttributes: c_uint,
        attributes: *mut hipPointer_attribute,
        data: *mut *mut c_void,
        ptr: hipDeviceptr_t,
    ) -> hipError_t;

    pub fn hipMalloc(
        ptr: *mut *mut c_void,
        size: usize,
    ) -> hipError_t;

    pub fn hipExtMallocWithFlags(
        ptr: *mut *mut c_void,
        sizeBytes: usize,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipHostMalloc(
        ptr: *mut *mut c_void,
        size: usize,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipHostAlloc(
        ptr: *mut *mut c_void,
        size: usize,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipHostGetDevicePointer(
        devPtr: *mut *mut c_void,
        hstPtr: *mut c_void,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipHostGetFlags(
        flagsPtr: * mut c_uint,
        hostPtr: *mut c_void,
    ) -> hipError_t;

    pub fn hipHostRegister(
        hostPtr: *mut c_void,
        sizeBytes: usize,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipHostUnregister(hostPtr: *mut c_void) -> hipError_t;

    pub fn hipMallocPitch(
        ptr: *mut *mut c_void,
        pitch: *mut usize,
        width: usize,
        height: usize,
    ) -> hipError_t;

    pub fn hipMemAllocPitch(
        dptr: *mut hipDeviceptr_t,
        pitch: *mut usize,
        widthInBytes: usize,
        height: usize,
        elementSizeBytes: c_uint,
    ) -> hipError_t;

    pub fn hipFree(
        ptr: *mut c_void,
    ) -> hipError_t;

    pub fn hipFreeHost(
        ptr: *mut c_void,
    ) -> hipError_t;

    pub fn hipMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    pub fn hipMemcpyWithStream(
        dst: *mut c_void,
        src: *const c_void,
        sizeBytes: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemcpyHtoD(
        dst: hipDeviceptr_t,
        src: *const c_void,
        sizeBytes: usize,
    ) -> hipError_t;

    pub fn hipMemcpyDtoH(
        dst: *mut c_void,
        src: hipDeviceptr_t,
        sizeBytes: usize,
    ) -> hipError_t;

    pub fn hipMemcpyDtoD(
        dst: hipDeviceptr_t,
        src: hipDeviceptr_t,
        sizeBytes: usize,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipMemcpyAtoD(
        dstDevice: hipDeviceptr_t,
        srcArray: hipArray_t,
        srcOffset: usize,
        ByteCount: usize,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipMemcpyDtoA(
        dstArray: hipArray_t,
        dstOffset: usize,
        srcDevice: hipDeviceptr_t,
        ByteCount: usize,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipMemcpyAtoA(
        dstArray: hipArray_t,
        dstOffset: usize,
        srcArray: hipArray_t,
        srcOffset: usize,
        ByteCount: usize,
    ) -> hipError_t;

    pub fn hipMemcpyHtoDAsync(
        dst: hipDeviceptr_t,
        src: *const c_void,
        sizeBytes: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemcpyDtoHAsync(
        dst: *mut c_void,
        src: hipDeviceptr_t,
        sizeBytes: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemcpyDtoDAsync(
        dst: hipDeviceptr_t,
        src: hipDeviceptr_t,
        sizeBytes: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipMemcpyAtoHAsync(
        dstHost: *mut c_void,
        srcArray: hipArray_t,
        srcOffset: usize,
        ByteCount: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipMemcpyHtoAAsync(
        dstArray: hipArray_t,
        dstOffset: usize,
        srcHost: *const c_void,
        ByteCount: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipGetSymbolAddress(
        devPtr: *mut *mut c_void,
        symbol: *const c_void,
    ) -> hipError_t;

    pub fn hipGetSymbolSize(
        size: *mut usize,
        symbol: *const c_void,
    ) -> hipError_t;

    #[since = 6 .1, "6.1"]
    pub fn hipGetProcAddress(
        symbol: *const char,
        pfn: *mut *mut c_void,
        hipVersion: c_int,
        flags: u64,
        symbolStatus: hipDriverProcAddressQueryResult,
    ) -> hipError_t;

    pub fn hipMemcpyToSymbol(
        symbol: *const c_void,
        src: *const c_void,
        sizeBytes: usize,
        offset: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    pub fn hipMemcpyToSymbolAsync(
        symbol: *const c_void,
        src: *const c_void,
        sizeBytes: usize,
        offset: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemcpyFromSymbol(
        dst: *mut c_void,
        symbol: *const c_void,
        sizeBytes: usize,
        offset: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    pub fn hipMemcpyFromSymbolAsync(
        dst: *mut c_void,
        symbol: *const c_void,
        sizeBytes: usize,
        offset: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        sizeBytes: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemset(
        dst: *mut c_void,
        value: c_int,
        sizeBytes: usize,
    ) -> hipError_t;

    pub fn hipMemsetD8(
        dest: hipDeviceptr_t,
        value: c_uchar,
        count: usize,
    ) -> hipError_t;

    pub fn hipMemsetD8Async(
        dest: hipDeviceptr_t,
        value: c_uchar,
        count: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemsetD16(
        dest: hipDeviceptr_t,
        value: c_ushort,
        count: usize,
    ) -> hipError_t;

    pub fn hipMemsetD16Async(
        dest: hipDeviceptr_t,
        value: c_ushort,
        count: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemsetD32(
        dest: hipDeviceptr_t,
        value: c_int,
        count: usize,
    ) -> hipError_t;

    pub fn hipMemsetD32Async(
        dest: hipDeviceptr_t,
        value: c_int,
        count: usize,
    ) -> hipError_t;

    pub fn hipMemset2D(
        dst: *mut c_void,
        pitch: usize,
        value: c_int,
        width: usize,
        height: usize,
    ) -> hipError_t;

    pub fn hipMemset2DAsync(
        dst: *mut c_void,
        pitch: usize,
        value: c_int,
        width: usize,
        height: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemset3D(
        pitchedDevPtr: hipPitchedPtr,
        value: c_int,
        extent: hipExtent,
    ) -> hipError_t;

    pub fn hipMemset3DAsync(
        pitchedDevPtr: hipPitchedPtr,
        value: c_int,
        extent: hipExtent,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipMemsetD2D8(
        dst: hipDeviceptr_t,
        dstPitch: usize,
        value: c_uchar,
        width: usize,
        height: usize,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipMemsetD2D8Async(
        dst: hipDeviceptr_t,
        dstPitch: usize,
        value: c_uchar,
        width: usize,
        height: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipMemsetD2D16(
        dst: hipDeviceptr_t,
        dstPitch: usize,
        value: c_ushort,
        width: usize,
        height: usize,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipMemsetD2D16Async(
        dst: hipDeviceptr_t,
        dstPitch: usize,
        value: c_ushort,
        width: usize,
        height: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipMemsetD2D32(
        dst: hipDeviceptr_t,
        dstPitch: usize,
        value: c_uint,
        width: usize,
        height: usize,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipMemsetD2D32Async(
        dst: hipDeviceptr_t,
        dstPitch: usize,
        value: c_uint,
        width: usize,
        height: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemGetInfo(
        free: *mut usize,
        total: *mut usize,
    ) -> hipError_t;

    pub fn hipMemPtrGetInfo(
        ptr: *mut c_void,
        size: *mut usize,
    ) -> hipError_t;

    pub fn hipMallocArray(
        array: *mut hipArray_t,
        desc: *const hipChannelFormatDesc,
        width: usize,
        height: usize,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipArrayCreate(
        pHandle: *mut hipArray_t,
        pAllocateArray: *const HIP_ARRAY_DESCRIPTOR,
    ) -> hipError_t;

    pub fn hipArrayDestroy(array: hipArray_t) -> hipError_t;

    pub fn hipArray3DCreate(
        array: *mut hipArray_t,
        pAllocateArray: *const HIP_ARRAY3D_DESCRIPTOR,
    ) -> hipError_t;

    pub fn hipMalloc3D(
        pitchedDevPtr: *const hipPitchedPtr,
        extent: hipExtent,
    ) -> hipError_t;

    pub fn hipFreeArray(array: hipArray_t) -> hipError_t;

    pub fn hipMalloc3DArray(
        array: *mut hipArray_t,
        desc: *const hipChannelFormatDesc,
        extent: hipExtent,
        flags: c_uint,
    ) -> hipError_t;

    #[since = 5 .6, "5.6"]
    pub fn hipArrayGetInfo(
        desc: *mut hipChannelFormatDesc,
        extent: *mut hipExtent,
        flags: *mut c_uint,
        array: hipArray_t,
    ) -> hipError_t;

    #[since = 5 .6, "5.6"]
    pub fn hipArrayGetDescriptor(
        pArrayDescriptor: *mut HIP_ARRAY_DESCRIPTOR,
        array: hipArray_t,
    ) -> hipError_t;

    #[since = 5 .6, "5.6"]
    pub fn hipArray3DGetDescriptor(
        pArrayDescriptor: *mut HIP_ARRAY3D_DESCRIPTOR,
        array: hipArray_t,
    ) -> hipError_t;

    pub fn hipMemcpy2D(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    pub fn hipMemcpyParam2D(pCopy: *const hip_Memcpy2D) -> hipError_t;

    pub fn hipMemcpyParam2DAsync(
        pCopy: *const hip_Memcpy2D,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemcpy2DAsync(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemcpy2DToArray(
        dst: hipArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipMemcpy2DToArrayAsync(
        dst: hipArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipMemcpy2DArrayToArray(
        dst: hipArray_t,
        wOffsetDst: usize,
        hOffsetDst: usize,
        src: hipArray_const_t,
        wOffsetSrc: usize,
        hOffsetSrc: usize,
        width: usize,
        height: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    pub fn hipMemcpy2DFromArray(
        dst: *mut c_void,
        dpitch: usize,
        src: hipArray_const_t,
        wOffset: usize,
        hOffset: usize,
        width: usize,
        height: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    pub fn hipMemcpy2DFromArrayAsync(
        dst: *mut c_void,
        dpitch: usize,
        src: hipArray_const_t,
        wOffset: usize,
        hOffset: usize,
        width: usize,
        height: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemcpyAtoH(
        dst: *mut c_void,
        srcArray: hipArray_t,
        srcOffset: usize,
        count: usize,
    ) -> hipError_t;

    pub fn hipMemcpyHtoA(
        dstArray: hipArray_t,
        dstOffset: usize,
        srcHost: *const c_void,
        count: usize,
    ) -> hipError_t;

    pub fn hipMemcpy3D(p: *const hipMemcpy3DParms) -> hipError_t;

    pub fn hipMemcpy3DAsync(
        p: *const hipMemcpy3DParms,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipDrvMemcpy3D(pCopy: *const HIP_MEMCPY3D) -> hipError_t;

    pub fn hipDrvMemcpy3DAsync(
        pCopy: *const HIP_MEMCPY3D,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipMemGetAddressRange(
        pbase: *mut hipDeviceptr_t,
        psize: *mut usize,
        dptr: hipDeviceptr_t,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipMemcpyBatchAsync(
        dsts: *mut *mut c_void,
        srcs: *mut *mut c_void,
        sizes: *mut usize,
        count: usize,
        atttrs: *mut hipMemcpyAttributes,
        attrsIdxs: *mut usize,
        numAttrs: usize,
        failIdx: *mut usize,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipMemcpy3DBatchAsync(
        numOps: usize,
        opList: *mut hipMemcpy3DBatchOp,
        failIdx: *mut usize,
        flags: c_ulonglong,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipMemcpy3DPeer(p: *mut hipMemcpy3DPeerParms) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipMemcpy3DPeerAsync(
        p: *mut hipMemcpy3DPeerParms,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipMipmappedArrayGetMemoryRequirements(
        memoryRequirements: *mut hipArrayMemoryRequirements,
        mipmap: hipMipmappedArray_t,
        device: hipDevice_t,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated, use hipHostMalloc() instead", cfg]
    pub fn hipMallocHost(
        ptr: *mut *mut c_void,
        size: usize,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated, use hipHostMalloc() instead", cfg]
    pub fn hipMemAllocHost(
        ptr: *mut *mut c_void,
        size: usize,
    ) -> hipError_t;

    pub fn hipHostFree(ptr: *mut c_void) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipMemcpyToArray(
        dst: hipArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const c_void,
        count: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipMemcpyFromArray(
        dst: *mut c_void,
        srcArray: hipArray_const_t,
        wOffset: usize,
        hOffset: usize,
        count: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    #[doc = "This API is currently not supported on Linux."]
    pub fn hipImportExternalSemaphore(
        extSem_out: *mut hipExternalSemaphore_t,
        semHandleDesc: *const hipExternalSemaphoreHandleDesc,
    ) -> hipError_t;

    #[doc = "This API is currently not supported on Linux."]
    pub fn hipSignalExternalSemaphoresAsync(
        extSemArray: *const hipExternalSemaphore_t,
        paramsArray: *const hipExternalSemaphoreSignalParams,
        numExtSems: c_uint,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "This API is currently not supported on Linux."]
    pub fn hipWaitExternalSemaphoresAsync(
        extSemArray: *const hipExternalSemaphore_t,
        paramsArray: *const hipExternalSemaphoreWaitParams,
        numExtSems: c_uint,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "This API is currently not supported on Linux."]
    #[since = 4 .3, "4.3"]
    pub fn hipDestroyExternalSemaphore(extSem: hipExternalSemaphore_t) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipImportExternalMemory(
        extMem_out: *mut hipExternalMemory_t,
        memHandleDesc: *const hipExternalMemoryHandleDesc,
    ) -> hipError_t;

    pub fn hipExternalMemoryGetMappedBuffer(
        devPtr: *mut *mut c_void,
        extMem: hipExternalMemory_t,
        bufferDesc: *const hipExternalMemoryBufferDesc,
    ) -> hipError_t;

    pub fn hipDestroyExternalMemory(extMem: hipExternalMemory_t) -> hipError_t;

    pub fn hipExternalMemoryGetMappedMipmappedArray(
        mipmap: *mut hipMipmappedArray_t,
        extMem: hipExternalMemory_t,
        mipmapDesc: *const hipExternalMemoryMipmappedArrayDesc,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMallocAsync(
        dev_ptr: *mut *mut c_void,
        size: usize,
        mem_pool: hipMemPool_t,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipFreeAsync(
        dev_ptr: *mut c_void,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipMemPoolTrimTo(
        mem_pool: hipMemPool_t,
        min_bytes_to_hold: usize,
    ) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipMemPoolSetAttribute(
        mem_pool: hipMemPool_t,
        attr: hipMemPoolAttr,
        value: *mut c_void,
    ) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipMemPoolGetAttribute(
        mem_pool: hipMemPool_t,
        attr: hipMemPoolAttr,
        value: *mut c_void,
    ) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipMemPoolSetAccess(
        mem_pool: hipMemPool_t,
        desc_list: *const hipMemAccessDesc,
        count: usize,
    ) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipMemPoolGetAccess(
        flags: *mut hipMemAccessFlags,
        mem_pool: hipMemPool_t,
        location: *mut hipMemLocation,
    ) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipMemPoolCreate(
        mem_pool: *mut hipMemPool_t,
        pool_props: *const hipMemPoolProps,
    ) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipMemPoolDestroy(mem_pool: hipMemPool_t) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipMallocFromPoolAsync(
        dev_ptr: *mut *mut c_void,
        size: usize,
        mem_pool: hipMemPool_t,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipMemPoolExportToShareableHandle(
        shared_handle: *mut c_void,
        mem_pool: hipMemPool_t,
        handle_type: hipMemAllocationHandleType,
        flags: c_uint,
    ) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipMemPoolImportFromShareableHandle(
        mem_pool: *mut hipMemPool_t,
        shared_handle: *mut c_void,
        handle_type: hipMemAllocationHandleType,
        flags: c_uint,
    ) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipMemPoolExportPointer(
        export_data: *mut hipMemPoolPtrExportData,
        dev_ptr: *mut c_void,
    ) -> hipError_t;

    #[doc = "
This API is implemented on Linux and is under development on Microsoft Windows.

This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
"]
    #[since = 5 .1, "5.1"]
    pub fn hipMemPoolImportPointer(
        dev_ptr: *mut *mut c_void,
        mem_pool: hipMemPool_t,
        export_data: *mut hipMemPoolPtrExportData,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipMemSetMemPool(
        location: *mut hipMemLocation,
        r#type: hipMemAllocationType,
        pool: hipMemPool_t,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipMemGetMemPool(
        pool: *mut hipMemPool_t,
        location: *mut hipMemLocation,
        r#type: hipMemAllocationType,
    ) -> hipError_t;

    #[since = 7 .15, "7.15"]
    pub fn hipMemGetDefaultMemPool(
        memPool: *mut hipMemPool_t,
        location: *mut hipMemLocation,
        r#type: hipMemAllocationType,
    ) -> hipError_t;

    #[doc = "It is recommend to do the capability check before call this API."]
    pub fn hipMallocManaged(
        dev_ptr: *mut *mut c_void,
        size: usize,
        flags: c_uint,
    ) -> hipError_t;

    #[doc = "It is recommend to do the capability check before call this API."]
    pub fn hipMemPrefetchAsync(
        dev_ptr: *const c_void,
        count: usize,
        device: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "It is recommend to do the capability check before call this API."]
    #[since = 7 .1, "7.1"]
    pub fn hipMemPrefetchAsync_v2(
        dev_ptr: *const c_void,
        count: usize,
        location: hipMemLocation,
        flags: c_uint,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "It is recommend to do the capability check before call this API."]
    #[since = 7 .2, "7.2"]
    pub fn hipMemPrefetchBatchAsync(
        dev_ptrs: *mut *mut c_void,
        sizes: *mut usize,
        count: usize,
        prefetch_locs: *mut hipMemLocation,
        prefetch_loc_idxs: *mut usize,
        num_prefetch_locs: usize,
        flags: c_ulonglong,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "
All memory ranges must be managed memory allocated via hipMallocManaged or system-allocated memory (if device supports pageable memory access).

This API is implemented on Linux and requires XNACK to be enabled.

This API is marked as beta, meaning, while this is feature complete, it is still open to changes and may have outstanding issues.

Reading from a discarded range without first writing or prefetching to it will return an indeterminate value.

Concurrent reads, writes, or prefetches to discarded ranges result in undefined behavior.
"]
    #[since = 7 .2, "7.2"]
    pub fn hipMemDiscardBatchAsync(
        dev_ptrs: *mut *mut c_void,
        sizes: *mut usize,
        count: usize,
        flags: c_ulonglong,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "
This is the driver API variant that uses hipDeviceptr_t instead of void*. Both hipMemDiscardBatchAsync and hipDrvMemDiscardBatchAsync use the same internal implementation.

Reading from a discarded range without first writing or prefetching to it will return an indeterminate value.
"]
    #[since = 7 .2, "7.2"]
    pub fn hipDrvMemDiscardBatchAsync(
        dptrs: *mut hipDeviceptr_t,
        sizes: *mut usize,
        count: usize,
        flags: c_ulonglong,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "
All memory ranges must be managed memory allocated via hipMallocManaged or system-allocated memory (if device supports pageable memory access).

This API is implemented on Linux and requires XNACK to be enabled.

This API is marked as beta, meaning, while this is feature complete, it is still open to changes and may have outstanding issues.

Reading from a discarded range without first writing or prefetching to it will return an indeterminate value.
"]
    #[since = 7 .2, "7.2"]
    pub fn hipMemDiscardAndPrefetchBatchAsync(
        dptrs: *mut *mut c_void,
        sizes: *mut usize,
        count: usize,
        prefetchLocs: *mut hipMemLocation,
        prefetchLocIdxs: *mut usize,
        numPrefetchLocs: usize,
        flags: c_ulonglong,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "This is the driver API variant that uses hipDeviceptr_t instead of void*."]
    #[since = 7 .2, "7.2"]
    pub fn hipDrvMemDiscardAndPrefetchBatchAsync(
        dptrs: *mut hipDeviceptr_t,
        sizes: *mut usize,
        count: usize,
        prefetchLocs: *mut hipMemLocation,
        prefetchLocIdxs: *mut usize,
        numPrefetchLocs: usize,
        flags: c_ulonglong,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    pub fn hipMemAdvise(
        dev_ptr: *const c_void,
        count: usize,
        advice: hipMemoryAdvise,
        device: c_int,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 7 .1, "7.1"]
    pub fn hipMemAdvise_v2(
        dev_ptr: *const c_void,
        count: usize,
        advice: hipMemoryAdvise,
        location: hipMemLocation,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    pub fn hipMemRangeGetAttributes(
        data: *mut *mut c_void,
        data_sizes: *mut usize,
        attributes: *mut hipMemRangeAttribute,
        num_attributes: usize,
        dev_ptr: *const c_void,
        count: usize,
    ) -> hipError_t;

    #[doc = "This API is under development. Currently it is a no-operation (NOP) function on AMD GPUs and returns hipSuccess."]
    pub fn hipStreamAttachMemAsync(
        stream: hipStream_t,
        dev_ptr: *mut c_void,
        length: usize,
        flags: c_int,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemAddressFree(
        devPtr: *mut c_void,
        size: usize,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemAddressReserve(
        ptr: *mut *mut c_void,
        size: usize,
        alignment: usize,
        addr: *mut c_void,
        flags: c_ulonglong,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemCreate(
        handle: *mut hipMemGenericAllocationHandle_t,
        size: usize,
        prop: *const hipMemAllocationProp,
        flags: c_ulonglong,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemExportToShareableHandle(
        shareableHandle: *mut c_void,
        handle: hipMemGenericAllocationHandle_t,
        handleType: hipMemAllocationHandleType,
        flags: c_ulonglong,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemGetAccess(
        flags: *mut c_ulonglong,
        location: *const hipMemLocation,
        ptr: *mut c_void,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemGetAllocationGranularity(
        granularity: *mut usize,
        prop: *const hipMemAllocationProp,
        option: hipMemAllocationGranularity_flags,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemGetAllocationPropertiesFromHandle(
        prop: *mut hipMemAllocationProp,
        handle: hipMemGenericAllocationHandle_t,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemImportFromShareableHandle(
        handle: *mut hipMemGenericAllocationHandle_t,
        osHandle: *mut c_void,
        shHandleType: hipMemAllocationHandleType,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemMap(
        ptr: *mut c_void,
        size: usize,
        offset: usize,
        handle: hipMemGenericAllocationHandle_t,
        flags: c_ulonglong,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemMapArrayAsync(
        mapInfoList: *mut hipArrayMapInfo,
        count: c_uint,
        stream: hipStream_t,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemRelease(handle: hipMemGenericAllocationHandle_t) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemRetainAllocationHandle(
        handle: *mut hipMemGenericAllocationHandle_t,
        addr: *mut c_void,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemSetAccess(
        ptr: *mut c_void,
        size: usize,
        desc: *const hipMemAccessDesc,
        count: usize,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    #[since = 5 .1, "5.1"]
    pub fn hipMemUnmap(
        ptr: *mut c_void,
        size: usize,
    ) -> hipError_t;

    #[doc = "3D linear filter isn’t supported on GFX90A boards, on which the API hipCreateTextureObject will return hipErrorNotSupported."]
    pub fn hipCreateTextureObject(
        pTextObject: *mut hipTextureObject_t,
        pRescDesc: *const hipResourceDesc,
        pTexDesc: *const hipTextureDesc,
        pResViewDesc: *const hipResourceViewDesc,
    ) -> hipError_t;

    pub fn hipDestroyTextureObject(textureObject: hipTextureObject_t) -> hipError_t;

    pub fn hipGetChannelDesc(
        desc: *mut hipChannelFormatDesc,
        array: hipArray_const_t,
    ) -> hipError_t;

    pub fn hipGetTextureObjectResourceDesc(
        pResDesc: *mut hipResourceDesc,
        textureObject: hipTextureObject_t,
    ) -> hipError_t;

    pub fn hipGetTextureObjectResourceViewDesc(
        pResViewDesc: *mut hipResourceViewDesc,
        textureObject: hipTextureObject_t,
    ) -> hipError_t;

    pub fn hipGetTextureObjectTextureDesc(
        pTexDesc: *mut hipTextureDesc,
        textureObject: hipTextureObject_t,
    ) -> hipError_t;

    pub fn hipTexObjectCreate(
        pTexObject: *mut hipTextureObject_t,
        pResResc: *const HIP_RESOURCE_DESC,
        pTexDesc: *const HIP_TEXTURE_DESC,
        pResViewDesc: *const HIP_RESOURCE_VIEW_DESC,
    ) -> hipError_t;

    pub fn hipTexObjectDestroy(texObject: hipTextureObject_t) -> hipError_t;

    pub fn hipTexObjectGetResourceDesc(
        pResDesc: *mut HIP_RESOURCE_DESC,
        texObject: hipTextureObject_t,
    ) -> hipError_t;

    pub fn hipTexObjectGetResourceViewDesc(
        pResViewDesc: *mut HIP_RESOURCE_VIEW_DESC,
        texObject: hipTextureObject_t,
    ) -> hipError_t;

    pub fn hipTexObjectGetTextureDesc(
        pTexDesc: *mut HIP_TEXTURE_DESC,
        texObject: hipTextureObject_t,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    pub fn hipMallocMipmappedArray(
        mipmappedArray: *mut hipMipmappedArray_t,
        desc: *const hipChannelFormatDesc,
        extent: hipExtent,
        numLevels: c_uint,
        flags: c_uint,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    pub fn hipFreeMipmappedArray(mipmappedArray: hipMipmappedArray_t) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    pub fn hipGetMipmappedArrayLevel(
        levelArray: *mut hipArray_t,
        mipmappedArray: hipMipmappedArray_const_t,
        level: c_uint,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    pub fn hipMipmappedArrayCreate(
        pHandle: *mut hipMipmappedArray_t,
        pMipmappedArrayDesc: *mut HIP_ARRAY3D_DESCRIPTOR,
        numMipmapLevels: c_uint,
    ) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    pub fn hipMipmappedArrayDestroy(hMipmappedArray: hipMipmappedArray_t) -> hipError_t;

    #[doc = "This API is implemented on Linux and is under development on Microsoft Windows."]
    pub fn hipMipmappedArrayGetLevel(
        pLevelArray: *mut hipArray_t,
        hMipMappedArray: hipMipmappedArray_t,
        level: c_uint,
    ) -> hipError_t;

    pub fn hipBindTextureToMipmappedArray(
        tex: *const textureReference,
        mipmappedArray: hipMipmappedArray_const_t,
        desc: *const hipChannelFormatDesc,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipGetTextureReference(
        texref: *mut *const textureReference,
        symbol: *const c_void,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefGetBorderColor(
        pBorderColor: *mut f32,
        texRef: *const textureReference,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefGetArray(
        pArray: *mut hipArray_t,
        texRef: *const textureReference,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetAddressMode(
        texRef: *mut textureReference,
        dim: c_int,
        am: hipTextureAddressMode,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetArray(
        tex: *mut textureReference,
        array: hipArray_const_t,
        flags: c_uint,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetFilterMode(
        texRef: *mut textureReference,
        fm: hipTextureFilterMode,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetFlags(
        texRef: *mut textureReference,
        Flags: c_uint,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetFormat(
        texRef: *mut textureReference,
        fmt: hipArray_Format,
        NumPackedComponents: c_int,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipBindTexture(
        offset: *mut usize,
        tex: *const textureReference,
        devPtr: *const c_void,
        desc: *const hipChannelFormatDesc,
        size: usize,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipBindTexture2D(
        offset: *mut usize,
        tex: *const textureReference,
        devPtr: *const c_void,
        desc: *const hipChannelFormatDesc,
        width: usize,
        height: usize,
        pitch: usize,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipBindTextureToArray(
        tex: *const textureReference,
        array: hipArray_const_t,
        desc: *const hipChannelFormatDesc,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipGetTextureAlignmentOffset(
        offset: *mut usize,
        texref: *const textureReference,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipUnbindTexture(tex: *const textureReference) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefGetAddress(
        dev_ptr: *mut hipDeviceptr_t,
        texRef: *const textureReference,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefGetAddressMode(
        pam: *mut hipTextureAddressMode,
        texRef: *const textureReference,
        dim: c_int,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefGetFilterMode(
        pfm: *mut hipTextureFilterMode,
        texRef: *const textureReference,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphicsUnregisterResource(resource: hipGraphicsResource_t) -> hipError_t;
}

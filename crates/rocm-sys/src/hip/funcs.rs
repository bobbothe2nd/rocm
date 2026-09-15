use super::types::*;

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

    pub fn hipDeviceGetTexture1DLinearMaxWidth(
        max_width: *mut usize,
        dex: *const hipChannelFormatDesc,
        device: c_int,
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
        limit: hipLimit_t,
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
        ms: *mut c_float,
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
        ty: hipMemAllocationType,
        pool: hipMemPool_t,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipMemGetMemPool(
        pool: *mut hipMemPool_t,
        location: *mut hipMemLocation,
        ty: hipMemAllocationType,
    ) -> hipError_t;

    #[since = 7 .15, "7.15"]
    pub fn hipMemGetDefaultMemPool(
        memPool: *mut hipMemPool_t,
        location: *mut hipMemLocation,
        ty: hipMemAllocationType,
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
This is the driver API variant that uses hipDeviceptr_t instead of c_void*. Both hipMemDiscardBatchAsync and hipDrvMemDiscardBatchAsync use the same internal implementation.

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

    #[doc = "This is the driver API variant that uses hipDeviceptr_t instead of c_void*."]
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
        pBorderColor: *mut c_float,
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

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefGetFlags(
        pFlags: *mut c_uint,
        texRef: *const textureReference,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefGetFormat(
        pFormat: *mut hipArray_Format,
        pNumChannels: *mut c_int,
        texBuf: *const textureReference,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefGetMaxAnisotropy(
        pmaxAnsio: *mut c_int,
        texRef: *const textureReference,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefGetMipmapFilterMode(
        pfm: *mut hipTextureFilterMode,
        texRef: *const textureReference,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefGetMipmapLevelBias(
        pbias: *mut c_float,
        texRef: *const textureReference,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefGetMipmapLevelClamp(
        pminMipmapLevelClamp: *mut c_float,
        pmaxMipmapLevelClamp: *mut c_float,
        texRef: *const textureReference,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefGetMipMappedArray(
        pArray: *mut hipMipmappedArray_t,
        texRef: *const textureReference,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetAddress(
        ByteOffset: *mut usize,
        texRef: *mut textureReference,
        dptr: hipDeviceptr_t,
        bytes: usize,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetAddress2D(
        texRef: textureReference,
        desc: *const HIP_ARRAY_DESCRIPTOR,
        dptr: hipDeviceptr_t,
        Pitch: usize,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetMaxAnisotropy(
        texRef: *mut textureReference,
        maxAniso: c_uint,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetBorderColor(
        texRef: *mut textureReference,
        pBorderColor: *mut c_float,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetMipmapFilterMode(
        texRef: *mut textureReference,
        fm: hipTextureFilterMode,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetMipmapLevelBias(
        texRef: *mut textureReference,
        bias: c_float,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetMipmapLevelClamp(
        texRef: *mut textureReference,
        minMipMapLevelClamp: c_float,
        maxMipMapLevelClamp: c_float,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated.", cfg]
    pub fn hipTexRefSetMipmappedArray(
        texRef: *mut textureReference,
        mipmappedArray: *mut hipMipmappedArray,
        Flags: c_uint,
    ) -> hipError_t;

    pub fn hipCreateSurfaceObject(
        pSurfObject: *mut hipSurfaceObject_t,
        pResDesc: *const hipResourceDesc,
    ) -> hipError_t;

    pub fn hipDestroySurfaceObject(surfaceObject: hipSurfaceObject_t) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipExtEnableLogging() -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipExtDisableLogging() -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipExtSetLoggingParams(
        log_level: usize,
        log_size: usize,
        log_mask: usize,
    ) -> hipError_t;

    #[private(cfg)]
    pub fn __hipGetPixelAddr(
        x: c_int,
        format: c_int,
        order: c_int,
    ) -> c_int;

    pub fn hipDeviceCanAccessPeer(
        canAccessPeer: *mut c_int,
        deviceId: c_int,
        peerDeviceId: c_int,
    ) -> hipError_t;

    pub fn hipDeviceEnablePeerAccess(
        peerDeviceId: c_int,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipDeviceDisablePeerAccess(peerDeviceId: c_int) -> hipError_t;

    pub fn hipMemcpyPeer(
        dst: *mut c_void,
        dstDeviceId: c_int,
        src: *const c_void,
        srcDeviceId: c_int,
        sizeBytes: usize,
    ) -> hipError_t;

    pub fn hipMemcpyPeerAsync(
        dst: *mut c_void,
        dstDeviceId: c_int,
        src: *const c_void,
        srcDevice: c_int,
        sizeBytes: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxCreate(
        ctx: *mut hipCtx_t,
        flags: c_uint,
        device: hipDevice_t,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxDestroy(ctx: hipCtx_t) -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxPopCurrent(ctx: *mut hipCtx_t) -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxPushCurrent(ctx: hipCtx_t) -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxSetCurrent(ctx: hipCtx_t) -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxGetCurrent(ctx: hipCtx_t) -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxGetDevice(device: *mut hipDevice_t) -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxGetApiVersion(
        ctx: hipCtx_t,
        apiVersion: *mut c_uint,
    ) -> hipError_t;

    #[doc = "AMD devices and some Nvidia GPUs do not support reconfigurable cache. This hint is ignored on those architectures."]
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxGetCacheConfig(cacheConfig: *mut hipFuncCache_t) -> hipError_t;

    #[doc = "AMD devices and some Nvidia GPUs do not support reconfigurable cache. This hint is ignored on those architectures."]
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxSetCacheConfig(cacheConfig: hipFuncCache_t) -> hipError_t;

    #[doc = "AMD devices and some Nvidia GPUs do not support shared cache banking, and the hint is ignored on those architectures."]
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxSetSharedMemConfig(config: hipSharedMemConfig) -> hipError_t;

    #[doc = "AMD devices and some Nvidia GPUs do not support shared cache banking, and the hint is ignored on those architectures."]
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxGetSharedMemConfig(pConfig: *mut hipSharedMemConfig) -> hipError_t;

    #[doc = "This function waits for all streams on the default context to complete execution, and then returns."]
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxSynchronize() -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxGetFlags(flags: *mut c_uint) -> hipError_t;

    #[doc = "PeerToPeer support is experimental."]
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxEnablePeerAccess(
        peerCtx: hipCtx_t,
        flags: c_uint,
    ) -> hipError_t;

    #[doc = "PeerToPeer support is experimental."]
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipCtxDisablePeerAccess(peerCtx: hipCtx_t) -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipDevicePrimaryCtxGetState(
        dev: hipDevice_t,
        flags: *mut c_uint,
        active: *mut c_int,
    ) -> hipError_t;

    #[doc = "This function return hipSuccess though doesn’t release the primaryCtx by design on HIP/HIP-CLANG path."]
    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipDevicePrimaryCtxRelease(dev: hipDevice_t) -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipDevicePrimaryCtxRetain(
        pctx: *mut hipCtx_t,
        dev: hipDevice_t,
    ) -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipDevicePrimaryCtxReset(dev: hipDevice_t) -> hipError_t;

    #[deprecated = "This API is deprecated on the AMD platform, only for equivalent cuCtx driver API on the NVIDIA platform."]
    pub fn hipDevicePrimaryCtxSetFlags(
        dev: hipDevice_t,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipModuleGetGlobal(
        dptr: *mut hipDeviceptr_t,
        bytes: *mut usize,
        hmod: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;

    #[since = 7 .1, "7.1"]
    pub fn hipModuleLoadFatBinary(
        module: *mut hipModule_t,
        fatbin: *const c_void,
    ) -> hipError_t;

    pub fn hipModuleLoad(
        module: *mut hipModule_t,
        fname: *const c_char,
    ) -> hipError_t;

    pub fn hipModuleUnload(module: hipModule_t) -> hipError_t;

    pub fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        kname: *const c_char,
    ) -> hipError_t;

    pub fn hipModuleGetFunctionCount(
        count: *mut c_int,
        module: hipModule_t,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipKernelGetAttribute(
        pi: *mut c_int,
        attrib: hipFunction_attribute,
        kernel: hipKernel_t,
        dev: hipDevice_t,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipLibraryLoadData(
        library: *mut hipLibrary_t,
        code: *const c_void,
        jitOptions: *mut hipJitOption,
        jitOptionsValues: *mut *mut c_void,
        numJitOptions: c_uint,
        libraryOptions: *mut hipLibraryOption,
        libraryOptionValues: *mut *mut c_void,
        numLibraryOptions: c_uint,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipLibraryLoadFromFile(
        library: *mut hipLibrary_t,
        fileName: *const c_char,
        jitOptions: *mut hipJitOption,
        jitOptionsValues: *mut *mut c_void,
        numJitOptions: c_uint,
        libraryOptions: *mut hipLibraryOption,
        libraryOptionValues: *mut *mut c_void,
        numLibraryOptions: c_uint,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipLibraryUnload(library: hipLibrary_t) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipLibraryGetKernel(
        pKernel: *mut hipKernel_t,
        library: hipLibrary_t,
        name: *const c_char,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipLibraryGetKernelCount(
        count: *mut c_uint,
        library: hipLibrary_t,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipLibraryGetGlobal(
        dptr: *mut *mut c_void,
        bytes: *mut usize,
        library: hipLibrary_t,
        name: *const c_char,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipLibraryGetManaged(
        dptr: *mut *mut c_void,
        bytes: *mut usize,
        library: hipLibrary_t,
        name: *const c_char,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipLibraryEnumerateKernels(
        kernels: *mut hipKernel_t,
        numKernels: c_uint,
        library: hipLibrary_t,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipKernelGetLibrary(
        library: *mut hipLibrary_t,
        kernel: hipKernel_t,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipKernelGetName(
        name: *const c_char,
        kernel: hipKernel_t,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipKernelGetParamInfo(
        kernel: hipKernel_t,
        paramIndex: usize,
        paramOffset: *mut usize,
        paramSize: *mut usize,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipGetFuncBySymbol(
        functionPtr: *mut hipFunction_t,
        symbolPtr: *const c_void,
    ) -> hipError_t;

    #[since = 6 .1, "6.1"]
    pub fn hipGetDriverEntryPoint(
        symbol: *const c_char,
        funcPtr: *mut *mut c_void,
        flags: c_ulonglong,
        driverStatus: *mut hipDriverEntryPointQueryResult,
    ) -> hipError_t;

    pub fn hipModuleGetTexRef(
        texRef: *mut *mut textureReference,
        hmod: hipModule_t,
        name: *const c_char,
    ) -> hipError_t;

    pub fn hipModuleLoadData(
        module: *mut hipModule_t,
        image: *const c_void,
    ) -> hipError_t;

    pub fn hipModuleLoadDataEx(
        module: *mut hipModule_t,
        image: *const c_void,
        numOptions: c_uint,
        options: *mut hipJitOption,
        optionValues: *mut *mut c_void,
    ) -> hipError_t;

    #[since = 6 .4, "6.4"]
    pub fn hipLinkAddData(
        state: hipLinkState_t,
        ty: hipJitInputType,
        data: *mut c_void,
        size: usize,
        name: *const c_char,
        numOptions: c_uint,
        options: *mut hipJitOption,
        optionValues: *mut *mut c_void,
    ) -> hipError_t;

    #[since = 6 .4, "6.4"]
    pub fn hipLinkComplete(
        state: hipLinkState_t,
        hipBinOut: *mut *mut c_void,
        sizeOut: *mut usize,
    ) -> hipError_t;

    #[since = 6 .4, "6.4"]
    pub fn hipLinkCreate(
        numOptions: c_uint,
        options: *mut hipJitOption,
        optionValues: *mut *mut c_void,
        stateOut: *mut hipLinkState_t,
    ) -> hipError_t;

    #[since = 6 .4, "6.4"]
    pub fn hipLinkDestroy(state: hipLinkState_t) -> hipError_t;

    pub fn hipModuleOccupancyMaxPotentialBlockSize(
        gridSize: *mut c_int,
        blockSize: *mut c_int,
        f: hipFunction_t,
        dynSharedMemPerBlk: usize,
        blockSizeLimit: c_int,
    ) -> hipError_t;

    pub fn hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
        gridSize: *mut c_int,
        blockSize: *mut c_int,
        f: hipFunction_t,
        dynSharedMemPerBlk: usize,
        blockSizeLimit: c_int,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut c_int,
        f: hipFunction_t,
        blockSize: c_int,
        dynSharedMemPerBlk: usize,
    ) -> hipError_t;

    pub fn hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks: *mut c_int,
        f: hipFunction_t,
        blockSize: c_int,
        dynSharedMemPerBlk: usize,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut c_int,
        f: *const c_void,
        blockSize: c_int,
        dynSharedMemPerBlk: usize,
    ) -> hipError_t;

    pub fn hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks: *mut c_int,
        f: *const c_void,
        blockSize: c_int,
        dynSharedMemPerBlk: usize,
        flags: c_uint,
    ) -> hipError_t;

    pub fn hipOccupancyMaxPotentialBlockSize(
        gridSize: *mut c_int,
        blockSize: *mut c_int,
        f: *const c_void,
        dynSharedMemPerBlk: usize,
        blockSizeLimit: c_int,
    ) -> hipError_t;

    #[since = 7 .2, "7.2"]
    pub fn hipOccupancyAvailableDynamicSMemPerBlock(
        dynamicSmemSize: *mut usize,
        f: *const c_void,
        numBlocks: c_int,
        blockSize: c_int,
    ) -> hipError_t;

    #[since = 6 .5, "6.5"]
    pub fn hipOccupancyMaxActiveClusters(
        numClusters: *mut c_int,
        f: *const c_void,
        config: *const hipLaunchConfig_t,
    ) -> hipError_t;

    #[since = 6 .5, "6.5"]
    pub fn hipOccupancyMaxPotentialClusterSize(
        clusterSize: *mut c_int,
        f: *const c_void,
        config: *const hipLaunchConfig_t,
    ) -> hipError_t;

    #[deprecated = "hipProfilerStart API is deprecated, use roctracer/rocTX instead."]
    pub fn hipProfilerStart() -> hipError_t;

    #[deprecated = "hipProfilerStop API is deprecated, use roctracer/rocTX instead."]
    pub fn hipProfilerStop() -> hipError_t;

    pub fn hipConfigureCall(
        gridDim: dim3,
        blockDim: dim3,
        sharedMem: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipSetupArgument(
        arg: *const c_void,
        size: usize,
        offset: usize,
    ) -> hipError_t;

    pub fn hipLaunchByPtr(func: *const c_void) -> hipError_t;

    #[private(cfg)]
    pub fn __hipPushCallConfiguration(
        gridDim: dim3,
        blockDim: dim3,
        sharedMem: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    #[private(cfg)]
    pub fn __hipPopCallConfiguration(
        gridDim: *mut dim3,
        blockDim: *mut dim3,
        sharedMem: *mut usize,
        stream: *mut hipStream_t,
    ) -> hipError_t;

    pub fn hipLaunchKernel(
        function_address: *const c_void,
        numBlocks: dim3,
        dimBlocks: dim3,
        args: *mut *mut c_void,
        sharedMemBytes: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipLaunchHostFunc(
        stream: hipStream_t,
        func: hipHostFn_t,
        userData: *mut c_void,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipDrvMemcpy2DUnaligned(pCopy: *const hip_Memcpy2D) -> hipError_t;

    pub fn hipExtLaunchKernel(
        function_address: *const c_void,
        numBlocks: dim3,
        dimBlocks: dim3,
        args: *mut *mut c_void,
        sharedMemBytes: usize,
        stream: hipStream_t,
        startEvent: hipEvent_t,
        stopEvent: hipEvent_t,
        flags: c_int,
    ) -> hipError_t;

    pub fn hipApiName(id: u32) -> *const c_char;

    pub fn hipKernelNameRef(f: hipFunction_t) -> *const c_char;

    pub fn hipKernelNameRefByPtr(
        hostFunction: *const c_void,
        stream: hipStream_t,
    ) -> *const c_char;

    pub fn hipGetStreamDeviceId(stream: hipStream_t) -> c_int;

    #[since = 4 .3, "4.3"]
    pub fn hipStreamBeginCapture(
        stream: hipStream_t,
        mode: hipStreamCaptureMode,
    ) -> hipError_t;

    #[doc = "
param “const hipGraphEdgeData* dependencyData” is currently not supported and
has to be passed as nullptr. This API is marked as beta, meaning, while this is
feature complete, it is still open to changes and may have outstanding issues.
"]
    #[since = 6 .1, "6.1"]
    pub fn hipStreamBeginCaptureToGraph(
        stream: hipStream_t,
        graph: hipGraph_t,
        dependencies: *const hipGraphNode_t,
        dependenciesData: *const hipGraphEdgeData,
        numDependencies: usize,
        mode: hipStreamCaptureMode,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipStreamEndCapture(
        stream: hipStream_t,
        pGraph: *mut hipGraph_t,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipStreamGetCaptureInfo(
        stream: hipStream_t,
        pCaptureStatus: *mut hipStreamCaptureStatus,
        pId: *mut c_ulonglong,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipStreamGetCaptureInfo_v2(
        stream: hipStream_t,
        captureStatus_out: *mut hipStreamCaptureStatus,
        id_out: *mut c_ulonglong,
        graph_out: *mut hipGraph_t,
        dependencies_out: *mut *const hipGraphNode_t,
        numDependencies_out: *mut usize,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipStreamIsCapturing(
        stream: hipStream_t,
        pCaptureStatus: *mut hipStreamCaptureStatus,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipStreamUpdateCaptureDependencies(
        stream: hipStream_t,
        dependencies: *mut hipGraphNode_t,
        numDependencies: usize,
        flags: c_uint,
    ) -> hipError_t;

    #[since = 5 .0, "5.0"]
    pub fn hipThreadExchangeStreamCaptureMode(mode: *mut hipStreamCaptureMode) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphCreate(
        pGraph: *mut hipGraph_t,
        flags: c_uint,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphDestroy(graph: hipGraph_t) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipGraphAddDependencies(
        graph: hipGraph_t,
        from: *const hipGraphNode_t,
        to: *const hipGraphNode_t,
        numDependencies: usize,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphRemoveDependencies(
        graph: hipGraph_t,
        from: *const hipGraphNode_t,
        to: *const hipGraphNode_t,
        numDependencies: usize,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphGetEdges(
        graph: hipGraph_t,
        from: *mut hipGraphNode_t,
        to: *mut hipGraphNode_t,
        numEdges: *mut usize,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipGraphGetNodes(
        graph: hipGraph_t,
        nodes: *mut hipGraphNode_t,
        numNodes: *mut usize,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipGraphGetRootNodes(
        graph: hipGraph_t,
        pRootNodes: *mut hipGraphNode_t,
        pNumRootNodes: *mut usize,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphNodeGetDependencies(
        node: hipGraphNode_t,
        pDependencies: *mut hipGraphNode_t,
        pNumDependencies: *mut usize,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphNodeGetDependentNodes(
        node: hipGraphNode_t,
        pDependentNodes: *mut hipGraphNode_t,
        pNumDependentNodes: *mut usize,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphNodeGetType(
        node: hipGraphNode_t,
        pType: *mut hipGraphNodeType,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphDestroyNode(node: hipGraphNode_t) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphClone(
        pGraphClone: *mut hipGraph_t,
        originalGraph: hipGraph_t,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphNodeFindInClone(
        pNode: *mut hipGraphNode_t,
        originalNode: hipGraphNode_t,
        clonedGraph: hipGraph_t,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphInstantiate(
        pGraphExec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        pErrorNode: *mut hipGraphNode_t,
        pLogBuffer: *mut c_char,
        bufferSize: usize,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphInstantiateWithFlags(
        pGraphExec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        flags: c_ulonglong,
    ) -> hipError_t;

    #[since = 6 .1, "6.1"]
    pub fn hipGraphInstantiateWithParams(
        pGraphExec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        instantiateParams: *mut hipGraphInstantiateParams,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphLaunch(
        graphExec: hipGraphExec_t,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphUpload(
        graphExec: hipGraphExec_t,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 5 .5, "5.5"]
    pub fn hipGraphAddNode(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        nodeParams: *mut hipGraphNodeParams,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipGraphExecGetFlags(
        graphExec: hipGraphExec_t,
        flags: *mut c_ulonglong,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipGraphNodeSetParams(
        node: hipGraphNode_t,
        nodeParams: *mut hipGraphNodeParams,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipGraphExecNodeSetParams(
        graphExec: hipGraphExec_t,
        node: hipGraphNode_t,
        nodeParams: *mut hipGraphNodeParams,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphExecDestroy(graphExec: hipGraphExec_t) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphExecUpdate(
        hGraphExec: hipGraphExec_t,
        hGraph: hipGraph_t,
        hErrorNode_out: *mut hipGraphNode_t,
        updateResult_out: *mut hipGraphExecUpdateResult,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphAddKernelNode(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        pNodeParams: *const hipKernelNodeParams,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipGraphKernelNodeGetParams(
        node: hipGraphNode_t,
        pNodeParams: *mut hipKernelNodeParams,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipGraphKernelNodeSetParams(
        node: hipGraphNode_t,
        pNodeParams: *const hipKernelNodeParams,
    ) -> hipError_t;

    #[since = 5 .6, "5.6"]
    pub fn hipDrvGraphAddMemcpyNode(
        phGraphNode: *mut hipGraphNode_t,
        hGraph: hipGraph_t,
        dependencies: *const hipGraphNode_t,
        numDependencies: usize,
        copyParams: *const HIP_MEMCPY3D,
        ctx: hipCtx_t,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphAddMemcpyNode(
        pGraphNode_t: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        pCopyParams: *const hipMemcpy3DParms,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipGraphMemcpyNodeGetParams(
        node: hipGraphNode_t,
        pNodeParams: *mut hipMemcpy3DParms,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipGraphMemcpyNodeSetParams(
        node: hipGraphNode_t,
        pNodeParams: *const hipMemcpy3DParms,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphExecMemcpyNodeSetParams(
        hGraphExec: hipGraphExec_t,
        node: hipGraphNode_t,
        pNodeParams: *mut hipMemcpy3DParms,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphAddMemcpyNode1D(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphMemcpyNodeSetParams1D(
        node: hipGraphNode_t,
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphExecMemcpyNodeSetParams1D(
        hGraphExec: hipGraphExec_t,
        node: hipGraphNode_t,
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphAddMemcpyNodeFromSymbol(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        dst: *mut c_void,
        symbol: *const c_void,
        count: usize,
        offset: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphMemcpyNodeSetParamsFromSymbol(
        node: hipGraphNode_t,
        dst: *mut c_void,
        symbol: *const c_void,
        count: usize,
        offset: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphExecMemcpyNodeSetParamsFromSymbol(
        hGraphExec: hipGraphExec_t,
        node: hipGraphNode_t,
        dst: *mut c_void,
        symbol: *const c_void,
        count: usize,
        offset: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphAddMemcpyNodeToSymbol(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        symbol: *const c_void,
        src: *const c_void,
        count: usize,
        offset: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphMemcpyNodeSetParamsToSymbol(
        node: hipGraphNode_t,
        symbol: *const c_void,
        src: *const c_void,
        count: usize,
        offset: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphExecMemcpyNodeSetParamsToSymbol(
        hGraphExec: hipGraphExec_t,
        node: hipGraphNode_t,
        symbol: *const c_void,
        src: *const c_void,
        count: usize,
        offset: usize,
        kind: hipMemcpyKind,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphAddMemsetNode(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        pMemsetParams: *const hipMemsetParams,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipGraphMemsetNodeGetParams(
        node: hipGraphNode_t,
        pNodeParams: *mut hipMemsetParams,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipGraphMemsetNodeSetParams(
        node: hipGraphNode_t,
        pNodeParams: *const hipMemsetParams,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphExecMemsetNodeSetParams(
        hGraphExec: hipGraphExec_t,
        node: hipGraphNode_t,
        pNodeParams: *const hipMemsetParams,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphAddHostNode(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        pNodeParams: *const hipHostNodeParams,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphHostNodeGetParams(
        node: hipGraphNode_t,
        pNodeParams: *mut hipHostNodeParams,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphHostNodeSetParams(
        node: hipGraphNode_t,
        pNodeParams: *const hipHostNodeParams,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphExecHostNodeSetParams(
        hGraphExec: hipGraphExec_t,
        node: hipGraphNode_t,
        pNodeParams: *const hipHostNodeParams,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphAddChildGraphNode(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        childGraph: hipGraph_t,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphChildGraphNodeGetGraph(
        node: hipGraphNode_t,
        pGraph: *mut hipGraph_t,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphExecChildGraphNodeSetParams(
        hGraphExec: hipGraphExec_t,
        node: hipGraphNode_t,
        childGraph: hipGraph_t,
    ) -> hipError_t;

    #[since = 4 .4, "4.4"]
    pub fn hipGraphAddEmptyNode(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphAddEventRecordNode(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        event: hipEvent_t,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphEventRecordNodeGetEvent(
        node: hipGraphNode_t,
        event_out: *mut hipEvent_t,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphEventRecordNodeSetEvent(
        node: hipGraphNode_t,
        event: hipEvent_t,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphExecEventRecordNodeSetEvent(
        hGraphExec: hipGraphExec_t,
        hNode: hipGraphNode_t,
        event: hipEvent_t,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphAddEventWaitNode(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        event: hipEvent_t,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphEventWaitNodeGetEvent(
        node: hipGraphNode_t,
        event_out: *mut hipEvent_t,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphEventWaitNodeSetEvent(
        node: hipGraphNode_t,
        event: hipEvent_t,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphExecEventWaitNodeSetEvent(
        hGraphExec: hipGraphExec_t,
        hNode: hipGraphNode_t,
        event: hipEvent_t,
    ) -> hipError_t;

    #[since = 5 .5, "5.5"]
    pub fn hipGraphAddMemAllocNode(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        pNodeParams: *mut hipMemAllocNodeParams,
    ) -> hipError_t;

    #[since = 5 .5, "5.5"]
    pub fn hipGraphMemAllocNodeGetParams(
        node: hipGraphNode_t,
        pNodeParams: *mut hipMemAllocNodeParams,
    ) -> hipError_t;

    #[since = 5 .5, "5.5"]
    pub fn hipGraphAddMemFreeNode(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        dev_ptr: *mut c_void,
    ) -> hipError_t;

    #[since = 5 .5, "5.5"]
    pub fn hipGraphMemFreeNodeGetParams(
        node: hipGraphNode_t,
        dev_ptr: *mut c_void,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipDeviceGetGraphMemAttribute(
        device: c_int,
        attr: hipGraphMemAttributeType,
        value: *mut c_void,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipDeviceGraphMemTrim(device: c_int) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipUserObjectCreate(
        object_out: *mut hipUserObject_t,
        ptr: *mut c_void,
        destroy: hipHostFn_t,
        initialRefcount: c_uint,
        flags: c_uint,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipUserObjectRelease(
        object: hipUserObject_t,
        count: c_uint,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipUserObjectRetain(
        object: hipUserObject_t,
        count: c_uint,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphRetainUserObject(
        graph: hipGraph_t,
        object: hipUserObject_t,
        count: c_uint,
        flags: c_uint,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphReleaseUserObject(
        graph: hipGraph_t,
        object: hipUserObject_t,
        count: c_uint,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphDebugDotPrint(
        graph: hipGraph_t,
        path: *const c_char,
        flags: c_uint,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphKernelNodeCopyAttributes(
        hSrc: hipGraphNode_t,
        hDst: hipGraphNode_t,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphNodeSetEnabled(
        hGraphExec: hipGraphExec_t,
        hNode: hipGraphNode_t,
        isEnabled: c_uint,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphNodeGetEnabled(
        hGraphExec: hipGraphExec_t,
        hNode: hipGraphNode_t,
        isEnabled: *mut c_uint,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphAddExternalSemaphoresWaitNode(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        nodeParams: *const hipExternalSemaphoreWaitNodeParams,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphAddExternalSemaphoresSignalNode(
        pGraphNode: *mut hipGraphNode_t,
        graph: hipGraph_t,
        pDependencies: *const hipGraphNode_t,
        numDependencies: usize,
        nodeParams: *const hipExternalSemaphoreSignalNodeParams,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphExternalSemaphoresSignalNodeSetParams(
        hNode: hipGraphNode_t,
        nodeParams: *const hipExternalSemaphoreSignalNodeParams,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphExternalSemaphoresWaitNodeSetParams(
        hNode: hipGraphNode_t,
        nodeParams: *const hipExternalSemaphoreWaitNodeParams,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphExternalSemaphoresSignalNodeGetParams(
        hNode: hipGraphNode_t,
        params_out: *mut hipExternalSemaphoreSignalNodeParams,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphExternalSemaphoresWaitNodeGetParams(
        hNode: hipGraphNode_t,
        params_out: *mut hipExternalSemaphoreWaitNodeParams,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphExecExternalSemaphoresSignalNodeSetParams(
        hGraphExec: hipGraphExec_t,
        hNode: hipGraphNode_t,
        nodeParams: *const hipExternalSemaphoreSignalNodeParams,
    ) -> hipError_t;

    #[since = 5 .3, "5.3"]
    pub fn hipGraphExecExternalSemaphoresWaitNodeSetParams(
        hGraphExec: hipGraphExec_t,
        hNode: hipGraphNode_t,
        nodeParams: *const hipExternalSemaphoreWaitNodeParams,
    ) -> hipError_t;

    #[since = 6 .0, "6.0"]
    pub fn hipDrvGraphMemcpyNodeGetParams(
        hNode: hipGraphNode_t,
        nodeParams: *mut HIP_MEMCPY3D,
    ) -> hipError_t;

    #[since = 6 .0, "6.0"]
    pub fn hipDrvGraphMemcpyNodeSetParams(
        hNode: hipGraphNode_t,
        nodeParams: *const HIP_MEMCPY3D,
    ) -> hipError_t;

    #[since = 5 .6, "5.6"]
    pub fn hipDrvGraphAddMemsetNode(
        phGraphNode: *mut hipGraphNode_t,
        hGraph: hipGraph_t,
        dependencies: *const hipGraphNode_t,
        numDependencies: usize,
        memsetParams: *const hipMemsetParams,
        ctx: hipCtx_t,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipDrvGraphAddMemFreeNode(
        phGraphNode: *mut hipGraphNode_t,
        hGraph: hipGraph_t,
        dependencies: *const hipGraphNode_t,
        numDependencies: usize,
        dptr: hipDeviceptr_t,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipDrvGraphExecMemcpyNodeSetParams(
        hGraphExec: hipGraphExec_t,
        hNode: hipGraphNode_t,
        copyParams: *const HIP_MEMCPY3D,
        ctx: hipCtx_t,
    ) -> hipError_t;

    #[since = 6 .2, "6.2"]
    pub fn hipDrvGraphExecMemsetNodeSetParams(
        hGraphExec: hipGraphExec_t,
        hNode: hipGraphNode_t,
        memsetParams: *const hipMemsetParams,
        ctx: hipCtx_t,
    ) -> hipError_t;

    #[since = 6 .5, "6.5"]
    pub fn hipMemGetHandleForAddressRange(
        handle: *mut c_void,
        dptr: hipDeviceptr_t,
        size: usize,
        handleType: hipMemRangeHandleType,
        flags: c_ulonglong,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphicsMapResources(
        count: c_int,
        resources: *mut hipGraphicsResource_t,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 4 .5, "4.5"]
    pub fn hipGraphicsSubResourceGetMappedArray(
        array: *mut hipArray_t,
        resource: hipGraphicsResource_t,
        arrayIndex: c_uint,
        mipLevel: c_uint,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphicsResourceGetMappedPointer(
        devPtr: *mut *mut c_void,
        size: *mut usize,
        resource: hipGraphicsResource_t,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphicsUnmapResources(
        count: c_int,
        resrources: *mut hipGraphicsResource_t,
        stream: hipStream_t,
    ) -> hipError_t;

    #[since = 4 .3, "4.3"]
    pub fn hipGraphicsUnregisterResource(resource: hipGraphicsResource_t) -> hipError_t;

    #[since = 5 .5, "5.5"]
    pub fn hipModuleLaunchCooperativeKernel(
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
    ) -> hipError_t;

    #[since = 5 .5, "5.5"]
    pub fn hipModuleLaunchCooperativeKernelMultiDevice(
        launchParamsList: *mut hipFunctionLaunchParams,
        numDevices: c_uint,
        flags: c_uint,
    ) -> hipError_t;

    #[since = 5 .2, "5.2"]
    pub fn hipLaunchCooperativeKernel(
        f: *const c_void,
        gridDim: dim3,
        blockDim: dim3,
        kernelParams: *mut *mut c_void,
        sharedMemBytes: c_uint,
        stream: hipStream_t,
    ) -> hipError_t;

    pub fn hipLaunchCooperativeKernelMultiDevice(
        launchParamsList: hipLaunchParams,
        numDevices: c_int,
        flags: c_uint,
    ) -> hipError_t;
}

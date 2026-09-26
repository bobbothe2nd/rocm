use core::{
    ffi::{c_uint, c_ulonglong},
    mem::{ManuallyDrop, transmute},
    ptr::{from_ref, null_mut},
};

use rocm_sys::hip::{
    dim3, hipChildGraphNodeParams, hipEventRecordNodeParams, hipEventWaitNodeParams,
    hipExternalSemaphoreSignalNodeParams, hipExternalSemaphoreWaitNodeParams, hipGraph_t,
    hipGraphAddKernelNode, hipGraphCreate, hipGraphDestroy, hipGraphExec_t, hipGraphExecDestroy,
    hipGraphInstantiateWithFlags, hipGraphLaunch, hipGraphNode_t, hipGraphNodeParams,
    hipGraphNodeParams__bindgen_ty_1, hipGraphNodeType, hipHostNodeParams, hipKernelNodeParams,
    hipMemAllocNodeParams, hipMemFreeNodeParams, hipMemcpyNodeParams, hipMemsetParams,
    hipStreamBeginCapture, hipStreamCaptureMode, hipStreamCaptureStatus, hipStreamEndCapture,
    hipStreamIsCapturing,
};

use crate::hip::{
    HipError,
    module::{Func, LaunchConfig},
    stream::Stream,
};

#[repr(transparent)]
pub struct Graph {
    raw: hipGraph_t,
}

impl Graph {
    pub fn new(flags: GraphInstantiateFlags) -> Result<Self, HipError> {
        let raw: Result<hipGraph_t, HipError> = unsafe {
            wrap_sys_res!(|graph| hipGraphCreate((&raw mut graph).cast(), flags.bits() as c_uint))
        };
        let raw = raw?;

        Ok(Self { raw })
    }

    pub fn destroy(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);

        unsafe {
            handle.destroy_unchecked()?;
        }

        Ok(())
    }

    pub unsafe fn destroy_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipGraphDestroy(self.raw), Ok(())) }
    }

    /// Raw `hipGraph_t`. Do not destroy.
    pub fn hip_graph(&self) -> hipGraph_t {
        self.raw
    }

    pub fn add_kernel_node(
        &mut self,
        dep: &[Node],
        params: &KernelParams,
    ) -> Result<Node, HipError> {
        unsafe {
            wrap_sys_res!(|node| hipGraphAddKernelNode(
                (&raw mut node).cast(),
                self.raw,
                dep.as_ptr().cast(),
                dep.len(),
                from_ref(params).cast()
            ))
        }
    }

    pub fn init(self, flags: GraphInstantiateFlags) -> Result<ExecGraph, HipError> {
        let exec: Result<hipGraphExec_t, HipError> = unsafe {
            wrap_sys_res!(|graph_exec| hipGraphInstantiateWithFlags(
                (&raw mut graph_exec).cast(),
                self.raw,
                flags.bits() as c_ulonglong
            ))
        };
        let exec = exec?;

        Ok(ExecGraph { exec })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            let _ = self.destroy_unchecked();
        }
    }
}

#[repr(transparent)]
pub struct ExecGraph {
    exec: hipGraphExec_t,
}

impl ExecGraph {
    pub fn destroy(self) -> Result<(), HipError> {
        let handle = ManuallyDrop::new(self);

        unsafe {
            handle.destroy_unchecked()?;
        }

        Ok(())
    }

    pub unsafe fn destroy_unchecked(&self) -> Result<(), HipError> {
        unsafe { try_err!(hipGraphExecDestroy(self.exec), Ok(())) }
    }

    /// Raw `hipGraphExec_t`. Do not destroy.
    pub fn hip_graph_exec(&self) -> hipGraphExec_t {
        self.exec
    }
}

impl Drop for ExecGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = self.destroy_unchecked();
        }
    }
}

impl Stream {
    pub fn capture<F: FnOnce(&Self) -> Result<(), HipError>>(
        &self,
        mode: StreamCaptureMode,
        f: F,
    ) -> Result<Graph, HipError> {
        self.start_capture(mode)?;
        f(self)?;
        self.end_capture()
    }

    pub fn start_capture(&self, mode: StreamCaptureMode) -> Result<(), HipError> {
        unsafe {
            try_err!(
                hipStreamBeginCapture(
                    self.raw,
                    transmute::<StreamCaptureMode, hipStreamCaptureMode>(mode)
                ),
                Ok(())
            )
        }
    }

    pub fn end_capture(&self) -> Result<Graph, HipError> {
        let raw: Result<hipGraph_t, HipError> = unsafe {
            wrap_sys_res!(|graph| hipStreamEndCapture(self.raw, (&raw mut graph).cast()))
        };
        let raw = raw?;

        Ok(Graph { raw })
    }

    pub fn is_capturing(&self) -> Result<bool, HipError> {
        let status: Result<hipStreamCaptureStatus, HipError> = unsafe {
            wrap_sys_res!(|status| hipStreamIsCapturing(self.raw, (&raw mut status).cast()))
        };

        match status? {
            hipStreamCaptureStatus::hipStreamCaptureStatusActive => Ok(true),
            hipStreamCaptureStatus::hipStreamCaptureStatusNone => Ok(false),
            hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated => {
                Err(HipError::StreamCaptureInvalidated)
            }
        }
    }

    pub unsafe fn launch_graph(&self, graph: &ExecGraph) -> Result<(), HipError> {
        unsafe { try_err!(hipGraphLaunch(graph.exec, self.raw), Ok(())) }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum StreamCaptureMode {
    Global = 0,
    ThreadLocal = 1,
    Relaxed = 2,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct Node {
    raw: hipGraphNode_t,
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct KernelParams {
    raw: hipKernelNodeParams,
}

impl KernelParams {
    pub const fn new(func: &Func, args: &mut [*mut u8], conf: LaunchConfig) -> Self {
        Self {
            raw: hipKernelNodeParams {
                blockDim: unsafe { transmute::<[u32; 3], dim3>(conf.block) },
                gridDim: unsafe { transmute::<[u32; 3], dim3>(conf.grid) },
                func: func.raw.cast(),
                kernelParams: args.as_mut_ptr().cast(),
                sharedMemBytes: 0,
                extra: null_mut(),
            },
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct GraphParams {
    raw: hipGraphNodeParams,
}

impl GraphParams {
    pub const fn new(ty: NodeType, desc: ParamDescriptor) -> Self {
        Self {
            raw: hipGraphNodeParams {
                type_: unsafe { transmute::<NodeType, hipGraphNodeType>(ty) },
                reserved0: [0; 3],
                __bindgen_anon_1: unsafe {
                    transmute::<ParamDescriptor, hipGraphNodeParams__bindgen_ty_1>(desc)
                },
                reserved2: 0,
            },
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union ParamDescriptor {
    pub reserved1: [::core::ffi::c_longlong; 29usize],
    pub kernel: ParamKernel,
    pub memcpy: ParamMemcpy,
    pub memset: ParamMemset,
    pub host: ParamHost,
    pub graph: ParamChildGraph,
    pub event_wait: ParamEventWait,
    pub event_record: ParamEventRecord,
    pub ext_sem_signal: ParamExternalSemaphoreSignal,
    pub ext_sem_wait: ParamExternalSemaphoreWait,
    pub alloc: ParamMemAlloc,
    pub free: ParamMemFree,
}

pub type ParamMemFree = hipMemFreeNodeParams;
pub type ParamMemAlloc = hipMemAllocNodeParams;
pub type ParamExternalSemaphoreWait = hipExternalSemaphoreWaitNodeParams;
pub type ParamExternalSemaphoreSignal = hipExternalSemaphoreSignalNodeParams;
pub type ParamEventRecord = hipEventRecordNodeParams;
pub type ParamEventWait = hipEventWaitNodeParams;
pub type ParamChildGraph = hipChildGraphNodeParams;
pub type ParamHost = hipHostNodeParams;
pub type ParamMemset = hipMemsetParams;
pub type ParamMemcpy = hipMemcpyNodeParams;
pub type ParamKernel = hipKernelNodeParams;

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum NodeType {
    Kernel = 0,
    Memcpy = 1,
    Memset = 2,
    Host = 3,
    Graph = 4,
    Empty = 5,
    WaitEvent = 6,
    EventRecord = 7,
    ExtSemaphoreSignal = 8,
    ExtSemaphoreWait = 9,
    MemAlloc = 10,
    MemFree = 11,
    MemcpyFromSymbol = 12,
    MemcpyToSymbol = 13,
    BatchMemOp = 14,
    Count = 15,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GraphInstantiateFlags: u32 {
        /// This makes two graph launches unsafe.
        const AUTO_FREE_ON_LAUNCH = 1 << 0;

        /// Automatically upload the graph after instantiation.
        const UPLOAD = 1 << 1;

        const DEV_LAUNCH = 1 << 2;

        const USE_NODE_PRIO = 1 << 3;
    }
}

//! Interpreter backend plugin for Hodu
//!
//! This plugin executes Hodu Snapshots by interpreting nodes sequentially,
//! dispatching to CPU kernels from `hodu_cpu_kernels`.

mod executor;

use executor::{
    binary, cast, concat_split, conv, indexing, matrix, memory, reduce, shape, unary, windowing, TensorStorage,
};
use hodu_plugin_sdk::{
    export_backend_plugin, ops, snapshot::SnapshotNode, BackendCapabilities, BackendPlugin, BuildFormat, BuildTarget,
    Device, HoduError, HoduResult, Snapshot, TensorData,
};
use std::collections::HashMap;
use std::path::Path;

/// Interpreter backend plugin
#[derive(Default)]
pub struct InterpBackend;

impl BackendPlugin for InterpBackend {
    fn name(&self) -> &str {
        "interp"
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::runner_only()
    }

    fn supported_devices(&self) -> Vec<Device> {
        vec![Device::CPU]
    }

    fn run(
        &self,
        snapshot: &Snapshot,
        device: Device,
        inputs: &[(&str, TensorData)],
    ) -> HoduResult<HashMap<String, TensorData>> {
        if device != Device::CPU {
            return Err(HoduError::UnsupportedDevice(device));
        }

        let mut interpreter = Interpreter::new(snapshot);
        interpreter.set_inputs(inputs)?;
        interpreter.execute()?;
        interpreter.get_outputs()
    }

    fn supported_targets(&self) -> Vec<BuildTarget> {
        vec![]
    }

    fn supported_formats(&self, _target: &BuildTarget) -> Vec<BuildFormat> {
        vec![]
    }

    fn build(
        &self,
        _snapshot: &Snapshot,
        _target: &BuildTarget,
        _format: BuildFormat,
        _output: &Path,
    ) -> HoduResult<()> {
        Err(HoduError::BackendError(
            "interp backend does not support AOT compilation".into(),
        ))
    }
}

export_backend_plugin!(InterpBackend, "interp", "0.1.0");

/// Snapshot interpreter
struct Interpreter<'a> {
    snapshot: &'a Snapshot,
    tensors: HashMap<usize, TensorStorage>,
}

impl<'a> Interpreter<'a> {
    fn new(snapshot: &'a Snapshot) -> Self {
        Self {
            snapshot,
            tensors: HashMap::new(),
        }
    }

    fn set_inputs(&mut self, inputs: &[(&str, TensorData)]) -> HoduResult<()> {
        let input_map: HashMap<&str, &TensorData> = inputs.iter().map(|(n, t)| (*n, t)).collect();

        for input in &self.snapshot.inputs {
            let tensor_data = input_map
                .get(input.name.as_str())
                .ok_or_else(|| HoduError::BackendError(format!("Missing input: {}", input.name)))?;

            let storage =
                TensorStorage::from_data(tensor_data.data.clone(), tensor_data.shape.clone(), tensor_data.dtype);
            self.tensors.insert(input.id.0, storage);
        }

        // Load constants
        for constant in &self.snapshot.constants {
            let storage =
                TensorStorage::from_data(constant.data.clone(), constant.shape.dims().to_vec(), constant.dtype);
            self.tensors.insert(constant.id.0, storage);
        }

        Ok(())
    }

    fn execute(&mut self) -> HoduResult<()> {
        for node in &self.snapshot.nodes {
            self.execute_node(node)?;
        }

        Ok(())
    }

    fn execute_node(&mut self, node: &SnapshotNode) -> HoduResult<()> {
        use ops::Op;

        match &node.op {
            Op::Binary(op) => binary::execute_binary(&mut self.tensors, *op, node),
            Op::BinaryLogical(op) => binary::execute_binary_logical(&mut self.tensors, *op, node),
            Op::Cmp(op) => binary::execute_cmp(&mut self.tensors, *op, node),
            Op::CmpScalar(op) => unary::execute_cmp_scalar(&mut self.tensors, *op, node),
            Op::Unary(op) => unary::execute_unary(&mut self.tensors, *op, node),
            Op::UnaryLogical(op) => unary::execute_unary_logical(&mut self.tensors, *op, node),
            Op::UnaryScalar(op) => unary::execute_unary_scalar(&mut self.tensors, *op, node),
            Op::Matrix(op) => matrix::execute_matrix(&mut self.tensors, *op, node),
            Op::Reduce(op) => reduce::execute_reduce(&mut self.tensors, *op, node),
            Op::Concat(op) => concat_split::execute_concat(&mut self.tensors, *op, node),
            Op::Split(op) => concat_split::execute_split(&mut self.tensors, *op, node),
            Op::Indexing(op) => indexing::execute_indexing(&mut self.tensors, *op, node),
            Op::Conv(op) => conv::execute_conv(&mut self.tensors, *op, node),
            Op::Windowing(op) => windowing::execute_windowing(&mut self.tensors, *op, node),
            Op::Shape(op) => shape::execute_shape(&mut self.tensors, *op, node),
            Op::ShapeScalars(op) => shape::execute_shape_scalars(&mut self.tensors, *op, node),
            Op::Cast(op) => cast::execute_cast(&mut self.tensors, *op, node),
            Op::Memory(op) => memory::execute_memory(&mut self.tensors, *op, node),
            Op::Dummy => Ok(()),
        }
    }

    fn get_outputs(&self) -> HoduResult<HashMap<String, TensorData>> {
        let mut outputs = HashMap::new();

        for target in &self.snapshot.targets {
            let storage = self
                .tensors
                .get(&target.id.0)
                .ok_or_else(|| HoduError::BackendError(format!("Output tensor not found: {}", target.name)))?;
            outputs.insert(target.name.clone(), storage.to_tensor_data());
        }

        Ok(outputs)
    }
}

# PyTorch and Detectron2 Libraries
import os
import sys
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TensorRTInfer:
    """
    Implements inference for the Model TensorRT engine.
    """

    def __init__(self, engine_path):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """

        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer_frame(self, frame, scales=None, nms_threshold=None):
        # Prepare the output data.
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network.
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(frame))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])
	
        # Process the results.
        nums = outputs[0][0]
        boxes = outputs[1][0]
        scores = outputs[2][0]
        pred_classes = outputs[3][0]
        masks = outputs[4][0]

        detections = []
        for n in range(int(nums)):
            # Select a mask.
            mask = masks[n]

            # Calculate scaling values for bboxes. 
            scale = self.inputs[0]['shape'][2]                 
            scale /= scales
            scale_y = scale
            scale_x = scale
            
            if nms_threshold and scores[n] < nms_threshold:
                continue
            # Append to detections          
            detections.append({
                'ymin': boxes[n][0] * scale_y,
                'xmin': boxes[n][1] * scale_x,
                'ymax': boxes[n][2] * scale_y,
                'xmax': boxes[n][3] * scale_x,
                'score': scores[n],
                'class': int(pred_classes[n]),
                'mask': mask,
            })
        return detections
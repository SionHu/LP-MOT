import ctypes
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class InferenceBackend:
    # initialize TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

        # load plugin if the model requires one
        if self.model.PLUGIN_PATH is not None:
            try:
                ctypes.cdll.LoadLibrary(self.model.PLUGIN_PATH)
            except OSError as err:
                raise RuntimeError('Plugin not found') from err

        # load trt engine or build one if not found
        if not self.model.ENGINE_PATH.exists():
            self.engine = self.model.build_engine(InferenceBackend.TRT_LOGGER, self.batch_size)
        else:
            runtime = trt.Runtime(InferenceBackend.TRT_LOGGER)
            with open(self.model.ENGINE_PATH, 'rb') as engine_file:
                buf = engine_file.read()
                self.engine = runtime.deserialize_cuda_engine(buf)
        if self.engine is None:
            raise RuntimeError('Unable to load the engine file')
        if self.engine.has_implicit_batch_dimension:
            assert self.batch_size <= self.engine.max_batch_size

        # allocate buffers
        self.bindings = []
        self.outputs = []
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            if self.engine.has_implicit_batch_dimension:
                size *= self.batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # append the device buffer to device bindings
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                if not self.engine.has_implicit_batch_dimension:
                    assert self.batch_size == shape[0]
                self.input = HostDeviceMem(host_mem, device_mem)
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    @property
    def input_handle(self):
        return self.input.host

    @input_handle.setter
    def input_handle(self, val):
        self.input.host[:] = val

    def infer(self):
        self.infer_async()
        return self.synchronize()

    def infer_async(self):
        cuda.memcpy_htod_async(self.input.device, self.input.host, self.stream)
        if self.engine.has_implicit_batch_dimension:
            self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings,
                                       stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

    def synchronize(self):
        self.stream.synchronize()
        return [out.host for out in self.outputs]

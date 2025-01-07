try:
    import pynvml
except ImportError:
    # Probably not an Nvidia GPU => Try if it is an AMD GPU:
    try:
        from pyrsmi import rocml
    except ImportError:
        # Neither Nvidia nor AMD library available => Create an empty function.
        def print_gpu_utilization():
            pass
    else:
        # Implementation for AMD GPUs:
        def print_gpu_utilization():
            rocml.smi_initialize()
            device_count = rocml.smi_get_device_count()
            memory_used = []
            for device_index in range(device_count):
                memory_used.append(rocml.smi_get_device_memory_used(device_index)/1024**3)
            print('Memory occupied on GPUs: ' + ' + '.join([f'{mem:.1f}' for mem in memory_used]) + ' GB.')
else:
    # Implementation for NVidia GPUs:
    def print_gpu_utilization():
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        memory_used = []
        for device_index in range(device_count):
            device_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            device_info = pynvml.nvmlDeviceGetMemoryInfo(device_handle)
            memory_used.append(device_info.used/1024**3)
        print('Memory occupied on GPUs: ' + ' + '.join([f'{mem:.1f}' for mem in memory_used]) + ' GB.')

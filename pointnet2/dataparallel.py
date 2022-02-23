from torch.nn import DataParallel
from itertools import chain

class MyDataParallel(DataParallel):
    # This is a customized DataParallel class for our project
    def __init__(self, *inputs, **kwargs):
        super(MyDataParallel, self).__init__(*inputs, **kwargs)
        self._replicas = None
        

    # Overide the forward method
    def forward(self, *inputs, **kwargs):
        disable_multi_gpu = False
        if "disable_multi_gpu" in kwargs:
            disable_multi_gpu = kwargs["disable_multi_gpu"]
            kwargs.pop("disable_multi_gpu")
        
        need_to_replicate = False
        if 'need_to_replicate' in kwargs:
            need_to_replicate = kwargs['need_to_replicate']
            kwargs.pop('need_to_replicate') # this step is to ensure other parts don't get the unexpected keyward 'need_to_replicate'

        if not self.device_ids or disable_multi_gpu: 
            return self.module(*inputs, **kwargs)

        if self._replicas is None or need_to_replicate:
            self._replicas = self.replicate(self.module, self.device_ids)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))
        
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids) 

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        outputs = self.parallel_apply(self._replicas[:len(inputs)], inputs, kwargs)

        return self.gather(outputs, self.output_device)

    def reset_cond_features(self):
        self.module.reset_cond_features()
        if not self._replicas is None:
            for rep in self._replicas:
                rep.reset_cond_features()
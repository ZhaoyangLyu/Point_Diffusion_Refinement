import numpy as np
class QuantityScheduler():
    def __init__(self, init_epoch, final_epoch, init_value, final_value, num_steps_per_epoch):
        # self.schedule_type = schedule_type
        self.init_step = init_epoch * num_steps_per_epoch
        self.final_step = final_epoch * num_steps_per_epoch
        self.init_value = init_value
        self.final_value = final_value
        self.num_steps_per_epoch = num_steps_per_epoch
        assert self.final_step >= self.init_step
        
    def get_quantity(self, global_step):
        return self.linear_schedule(global_step, self.init_step, self.final_step, self.init_value, self.final_value)
    
    def linear_schedule(self, step, init_step, final_step, init_value, final_value):
        """Linear schedule."""
        assert final_step >= init_step
        if init_step == final_step:
            return final_value
        rate = float(step - init_step) / float(final_step - init_step)
        linear_value = rate * (final_value - init_value) + init_value
        return np.clip(linear_value, min(init_value, final_value), max(init_value, final_value))

if __name__ == '__main__':
    import pdb
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    init_epoch = 20 
    final_epoch = 60 
    total_epoch = 100
    init_value = 1e-4 
    final_value = 1e-1 
    num_steps_per_epoch = 128
    schedule = QuantityScheduler(init_epoch, final_epoch, init_value, final_value, num_steps_per_epoch)

    steps = np.arange(total_epoch * num_steps_per_epoch)
    values = []
    for s in steps:
        values.append(schedule.get_quantity(s))
    values = np.array(values)

    plt.plot(steps, values)
    save_file = 'schedule.png'
    plt.savefig(save_file)
    plt.close()
    # pdb.set_trace()
import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import random
import copy
import sys
import warnings
import pickle


sys.path.insert(0, os.path.dirname(__file__))
from mvp_data_utils import augment_cloud

class ShapeNetH5(data.Dataset):
    def __init__(self, data_dir, train=True, npoints=2048, novel_input=True, novel_input_only=False,
                        scale=1, rank=0, world_size=1, random_subsample=False, num_samples=1000,
                        augmentation=False, return_augmentation_params=False,
                        include_generated_samples=False, generated_sample_path=None,
                        randomly_select_generated_samples=False, # randomly select a trial from multi trial generations
                        use_mirrored_partial_input=False, number_partial_points=2048,
                        load_pre_computed_XT=False, T_step=100, XT_folder=None,
                        append_samples_to_last_rank=True,
                        # use_a_random_indices_file=False,
                        # random_indices_file = 'random_indices.pkl',
                        # random_replace_partial_with_complete_prob=0, benchmark=False,
                        ):
        self.return_augmentation_params = return_augmentation_params
        self.use_mirrored_partial_input = use_mirrored_partial_input
        if use_mirrored_partial_input or load_pre_computed_XT:
            assert novel_input and (not novel_input_only)
            # if use_mirrored_partial_input:
            #     assert random_replace_partial_with_complete_prob == 0
        # if benchmark:
        #     assert not train
        #     assert not load_pre_computed_XT
        #     assert novel_input and (not novel_input_only)
        #     assert not random_subsample # we need to evalute all samples in the benchmark dataset
        #     assert not augmentation # benchmark dataset should not have any data augmentation
        #     assert random_replace_partial_with_complete_prob == 0
        #     if use_mirrored_partial_input:
        #         self.mirrored_input_path = ('%s/mirror_and_concated_partial/mvp_benchmark_mirror_and_concat/mvp_benchmark_mirror_and_concat_%dpts.h5' % 
        #                             (data_dir, number_partial_points))
        #     self.input_path = '%s/mvp_benchmark/MVP_ExtraTest_Shuffled_CP.h5' % data_dir
        if train:
            if use_mirrored_partial_input:
                self.mirrored_input_path = ('%s/mirror_and_concated_partial/mvp_train_input_mirror_and_concat_%dpts.h5' % 
                                    (data_dir, number_partial_points))
            self.input_path = '%s/mvp_train_input.h5' % data_dir
            self.gt_path = '%s/mvp_train_gt_%dpts.h5' % (data_dir, npoints)
        else:
            if use_mirrored_partial_input:
                self.mirrored_input_path = ('%s/mirror_and_concated_partial/mvp_test_input_mirror_and_concat_%dpts.h5' % 
                                    (data_dir, number_partial_points))
            self.input_path = '%s/mvp_test_input.h5' % data_dir
            self.gt_path = '%s/mvp_test_gt_%dpts.h5' % (data_dir, npoints)
        self.npoints = npoints
        self.train = train # controls the trainset and testset
        # self.benchmark = benchmark
        self.augmentation = augmentation # augmentation could be a dict or False

        # self.random_replace_partial_with_complete_prob=random_replace_partial_with_complete_prob
        # if random_replace_partial_with_complete_prob>0:
        #     if npoints != 2048:
        #         raise Exception('random_replace_partial_with_complete only supported 2048 points right now')

        # load partial point clouds and their labels
        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        self.labels = np.array((input_file['labels'][()]))
        # if benchmark:
        #     self.labels = self.labels[:,0]
        # else:
        #     # benchmark dataset has no novel input
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        # load gt complete point cloud
        # if not benchmark: # benchmark dataset has no gt
        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()

        # load XT generated from a trained DDPM
        self.load_pre_computed_XT = load_pre_computed_XT
        if load_pre_computed_XT:
            if train:
                XT_folder = os.path.join(XT_folder, 'train')
            else:
                XT_folder = os.path.join(XT_folder, 'test')
            self.T_step = T_step
            XT_file = os.path.join(XT_folder, 'mvp_generated_data_2048pts_T%d.h5' % T_step)
            self.XT_file = XT_file
            generated_XT_file = h5py.File(XT_file, 'r')
            self.generated_XT = np.array(generated_XT_file['data'])
            generated_XT_file.close()

        # load X0 generated from a trained DDPM
        self.include_generated_samples = include_generated_samples
        self.generated_sample_path = generated_sample_path
        self.randomly_select_generated_samples = randomly_select_generated_samples
        if include_generated_samples:
            # generated_samples/T1000_betaT0.02_shape_completion_no_class_condition_scale_1_no_random_replace_partail_with_complete/ckpt_1403999/
            generated_samples_file = os.path.join(data_dir, generated_sample_path)
            if randomly_select_generated_samples:
                files = os.listdir(generated_samples_file)
                files = [f for f in files if f.startswith('trial')]
                files = [os.path.join(generated_samples_file, f) for f in files]
                files = [generated_samples_file] + files
                generated_samples_file = random.choice(files)
                print('Randomly select file %s for generated samples from %d files' % (generated_samples_file, len(files)))

            # if benchmark:
            #     generated_samples_file = os.path.join(generated_samples_file, 'benchmark')
            # else:
            if train:
                generated_samples_file = os.path.join(generated_samples_file, 'train')
            else:
                generated_samples_file = os.path.join(generated_samples_file, 'test')
            generated_samples_file = os.path.join(generated_samples_file, 'mvp_generated_data_2048pts.h5')

            generated_file = h5py.File(generated_samples_file, 'r')
            self.generated_sample = np.array(generated_file['data'])
            generated_file.close()
            # generated_sample should have the same number of shapes as input_data, 
            # because we generate one complete point cloud using the trained DDPM for each partial point cloud
            # however, they may have different number of points for each shape
        
        # combine normal input and novel input
        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            if use_mirrored_partial_input:
                mirrored_file = h5py.File(self.mirrored_input_path, 'r')
                self.input_data = np.array(mirrored_file['data'])
                mirrored_file.close()
            # elif not benchmark: # not use_mirrored_partial_input and not benchmark
            else:
                self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            # if not benchmark:
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        # split the whole dataset evenly
        # number of partial point clouds is 26 times of the number of gt complete point clouds
        # we split the gt complete point clouds evenly
        if world_size > 1:
            # num_gt_shapes = int(np.ceil(self.input_data.shape[0] / 26)) if benchmark else self.gt_data.shape[0]
            num_gt_shapes = self.gt_data.shape[0]
            if not num_gt_shapes % world_size == 0:
                # raise Exception('The dataset (%d samples) can not be distributed evenly on %d gpus' % (self.gt_data.shape[0], world_size))
                print('The dataset (%d samples) can not be distributed evenly on %d gpus' % (num_gt_shapes, world_size))
                
            num_shapes_per_world = int(np.ceil(num_gt_shapes / world_size))
            start = rank * num_shapes_per_world
            end = (rank+1) * num_shapes_per_world

            if rank == world_size-1 and append_samples_to_last_rank:
                missing = end * 26 - self.input_data.shape[0]
                if missing > 0:
                    # append samples to the last rank so that it has the same number of samples as other ranks
                    # assert (not benchmark) and (not train) 
                    assert train
                    # we should only append samples to the last rank when training 
                    missing = end - self.gt_data.shape[0]
                    supp_gt_idx = random.sample(list(range(self.gt_data.shape[0])), missing)
                    supp_gt_idx = np.array(supp_gt_idx)
                    supp_partial_idx_start = supp_gt_idx * 26
                    # pdb.set_trace()
                    supp_partial_idx_start = supp_partial_idx_start[:, np.newaxis] # (missing, 1)
                    supp_partial_idx = copy.deepcopy(supp_partial_idx_start)
                    for i in range(26):
                        supp_partial_idx = np.concatenate([supp_partial_idx, supp_partial_idx_start+i], axis=1)
                    supp_partial_idx = supp_partial_idx[:,1:]
                    supp_partial_idx = np.reshape(supp_partial_idx, (-1))

                    supp_partial = self.input_data[supp_partial_idx]
                    supp_label = self.labels[supp_partial_idx]
                    # if not benchmark:
                    supp_gt = self.gt_data[supp_gt_idx]
                    if self.include_generated_samples:
                        supp_generated = self.generated_sample[supp_partial_idx]
                    if self.load_pre_computed_XT:
                        supp_generated_XT = self.generated_XT[supp_partial_idx]
            
            self.input_data = self.input_data[start*26:end*26]
            # if not benchmark:
            self.gt_data = self.gt_data[start:end]
            self.labels = self.labels[start*26:end*26]
            if self.include_generated_samples:
                self.generated_sample = self.generated_sample[start*26:end*26]
            if self.load_pre_computed_XT:
                self.generated_XT = self.generated_XT[start*26:end*26]

            if rank == world_size-1 and append_samples_to_last_rank:
                if missing>0:
                    self.input_data = np.concatenate([self.input_data, supp_partial], axis=0)
                    self.labels = np.concatenate([self.labels, supp_label], axis=0)
                    # if not benchmark:
                    self.gt_data = np.concatenate([self.gt_data, supp_gt], axis=0)
                    if self.include_generated_samples:
                        self.generated_sample = np.concatenate([self.generated_sample, supp_generated], axis=0)
                    if self.load_pre_computed_XT:
                        self.generated_XT = np.concatenate([self.generated_XT, supp_generated_XT], axis=0)
                    print('%d samples are appended to the the last rank' % missing)
        
        # randomly subsample the datasets, because we may want to only test the trained DDPM on a fraction of the 
        # dataset to save time 
        self.random_subsample = random_subsample
        if random_subsample:
            if num_samples < self.input_data.shape[0]:
                partial_to_complete_index = np.arange(self.gt_data.shape[0])
                partial_to_complete_index = np.repeat(partial_to_complete_index[:,np.newaxis], 26, axis=1)
                partial_to_complete_index = partial_to_complete_index.reshape((self.gt_data.shape[0]*26))

                # if use_a_random_indices_file:
                #     idx_handle = open(random_indices_file, 'rb')
                #     idx_data = pickle.load(idx_handle)
                #     idx = idx_data['idx']
                #     idx_handle.close()
                #     if world_size>1:
                #         assert not append_samples_to_last_rank
                #         num_partial_shapes_per_world = num_shapes_per_world * 26
                #         idx = idx - rank * num_shapes_per_world
                #         this_rank_samples = (idx < num_partial_shapes_per_world) * (idx >= 0)
                #         idx = idx[this_rank_samples]
                #     num_samples = idx.shape[0]
                # else:

                index = list(range(self.input_data.shape[0]))
                idx = random.sample(index, num_samples)
                idx = np.array(idx) 
                self.input_data = self.input_data[idx]
                self.labels = self.labels[idx]
                self.partial_to_complete_index = partial_to_complete_index[idx]
                if self.include_generated_samples:
                    self.generated_sample = self.generated_sample[idx]
                if self.load_pre_computed_XT:
                    self.generated_XT = self.generated_XT[idx]
            else:
                self.random_subsample = False
                warnings.warn("The provided num_samples (%d) is not less than the number of shapes (%d). random_subsample will not be performed"
                                % (num_samples, self.input_data.shape[0]))

        self.scale = scale
        # shapes in mvp dataset range from -0.5 to 0.5
        # we rescale the, to make the, range from -scale to scale 
        if use_mirrored_partial_input:
            # note that in this case self.input_data is of shape B,N,4
            # the last dimension indicates whether the corresponding point is the original one or the mirrowed point
            self.input_data[:,:,0:3] = self.input_data[:,:,0:3] * 2 * scale
        else:
            self.input_data = self.input_data * 2 * scale
        # if not benchmark:
        self.gt_data = self.gt_data * 2 * scale
        if self.include_generated_samples:
            self.generated_sample = self.generated_sample * 2 * scale
        if self.load_pre_computed_XT:
            self.generated_XT = self.generated_XT * 2 * scale

        print('partial point clouds:', self.input_data.shape)
        # if not benchmark:
        print('gt complete point clouds:', self.gt_data.shape)
        print('labels', self.labels.shape)
        if self.include_generated_samples:
            print('DDPM generated complete point clouds:', self.generated_sample.shape)
        if self.load_pre_computed_XT:
            print('DDPM generated intermediate complete point clouds:', self.generated_XT.shape)
        self.labels = self.labels.astype(int)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # deepcopy is necessary here, because we may alter partial and complete for data augmentation
        # it will change the original data if we do not deep copy
        result = {}
        result['partial'] = copy.deepcopy(self.input_data[index])
        # if not self.benchmark:
        if self.random_subsample:
            gt_idx = self.partial_to_complete_index[index]
        else:
            gt_idx = index // 26
        result['complete'] = copy.deepcopy(self.gt_data[gt_idx])
        
        if self.include_generated_samples:
            result['generated'] = copy.deepcopy(self.generated_sample[index])
        if self.load_pre_computed_XT:
            result['XT'] = copy.deepcopy(self.generated_XT[index])

        # augment the point clouds
        if isinstance(self.augmentation, dict):
            result_list = list(result.values())
            if self.return_augmentation_params:
                result_list, augmentation_params = augment_cloud(result_list, self.augmentation,
                                                                return_augmentation_params=True)
            else:
                result_list = augment_cloud(result_list, self.augmentation, return_augmentation_params=False)
            for idx, key in enumerate(result.keys()):
                result[key] = result_list[idx]
            if self.include_generated_samples:
                # add noise to every point in the point cloud generated by a trained DDPM
                # this is used to train the refinement network
                sigma = self.augmentation.get('noise_magnitude_for_generated_samples', 0)
                if sigma > 0:
                    noise = np.random.normal(scale=sigma, size=result['generated'].shape)
                    noise = noise.astype(result['generated'].dtype)
                    result['generated'] = result['generated'] + noise

        if self.return_augmentation_params:
            for key in augmentation_params.keys():
                result[key] = augmentation_params[key]
        for key in result.keys():
            result[key] = torch.from_numpy(result[key])
        result['label'] = self.labels[index]
        # if self.benchmark:
        #     result['complete'] = 0

        # if self.random_replace_partial_with_complete_prob>0:
        #     if random.random() < self.random_replace_partial_with_complete_prob:
        #         result['partial'] = result['complete']
        return result

if __name__ == '__main__':
    import pdb
    aug_args = {'pc_augm_scale':1.5, 'pc_augm_rot':True, 'pc_rot_scale':30.0, 'pc_augm_mirror_prob':0.5, 'pc_augm_jitter':False, 'translation_magnitude': 0.1}
    aug_args =  False
    include_generated_samples=False
    # benchmark = False
    generated_sample_path='generated_samples/T1000_betaT0.02_shape_completion_mirror_rot_60_scale_1.2_translation_0.05/ckpt_623999'
    dataset = ShapeNetH5('./data/mvp_dataset', train=False, npoints=2048, novel_input=True, novel_input_only=False,
                            augmentation=aug_args, scale=1,
                            random_subsample=True, num_samples=1000,
                            include_generated_samples=include_generated_samples, 
                            generated_sample_path=generated_sample_path,
                            use_mirrored_partial_input=True, number_partial_points=3072,
                            rank=0, world_size=1,
                            load_pre_computed_XT=False, T_step=10, 
                            XT_folder='data/mvp_dataset/generated_samples/T1000_betaT0.02_shape_completion_avg_max_pooling_mirror_rot_90_scale_1.2_translation_0.2/ckpt_545999/',
                            append_samples_to_last_rank=False,
                            return_augmentation_params=False)
    # label, partial, complete = dataset.__getitem__(0)
    # pdb.set_trace()
    # for i in range(len(dataset)):
    #     label, partial, complete = dataset.__getitem__(i)
    #     print('index %d label %d partail shape %s [%.3f, %.3f] complete shape %s [%.3f, %.3f]' % (
    #         i, int(label), partial.shape, partial.min(), partial.max(), complete.shape, complete.min(), complete.max(),))
    # pdb.set_trace()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    for i, data in enumerate(dataloader):
        # label, partial, complete = data
        label, partial, complete = data['label'], data['partial'], data['complete']
        # data['M_inv'] is of shape (B,3,3)
        # data['translation'] is of shape (B,1,3)
        # label, partial, complete, generated = data['label'], data['partial'], data['complete'], data['generated']
        # label, partial, complete, generated = data
        print('index %d label %s partail shape %s [%.3f, %.3f] complete shape %s [%.3f, %.3f]' % (
            i, label.shape, partial.shape, partial.min(), partial.max(), complete.shape, complete.min(), complete.max(),))
        # pdb.set_trace()
        # print('index %d label %s partail shape %s [%.3f, %.3f] complete shape %s [%.3f, %.3f] generated shape %s [%.3f, %.3f]' % (
        #     i, label.shape, partial.shape, partial.min(), partial.max(), complete.shape, complete.min(), complete.max(), 
        #     generated.shape, generated.min(), generated.max()))
        # label, partial, complete, XT = data
        # # label, partial, complete, generated = data
        # print('index %d label %s partail shape %s [%.3f, %.3f] complete shape %s [%.3f, %.3f] XT shape %s [%.3f, %.3f]' % (
        #     i, label.shape, partial.shape, partial.min(), partial.max(), complete.shape, complete.min(), complete.max(),
        #     XT.shape, XT.min(), XT.max()))
        # print('index %d label %s partail shape %s [%.3f, %.3f] complete shape %s [%.3f, %.3f] generated shape %s [%.3f, %.3f]' % (
        #     i, label.shape, partial.shape, partial.min(), partial.max(), complete.shape, complete.min(), complete.max(), 
        #     generated.shape, generated.min(), generated.max()))
        # handle = open('mvp_data_sample_rand_mirror.pkl', 'wb')
        # pickle.dump({'label':label.numpy(), 'gt':complete.numpy(), 'condition':partial.numpy()}, handle)
        # # # 
        # handle.close()
        # pdb.set_trace()
        # for k in range(complete.shape[0]):
        #     compare = complete[0]==complete[k]
        #     print(compare)
        #     if False in compare:
        #         pdb.set_trace()
        # pdb.set_trace()
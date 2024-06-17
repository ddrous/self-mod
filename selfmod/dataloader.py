from typing import Tuple
from ._utils import *
from abc import abstractmethod

import pandas as pd
import torch
from torchvision.transforms import transforms
from PIL import Image



class DataLoader:
    """
    A bas class generator of generators for general-purpose meta-learning regression tasks.
    """
    def __init__(self, 
                 data_path, 
                 envs_batch_size=250, 
                 envs_shuffle=True, 
                 shots_batch_size=1,
                 shots_shuffle=False, 
                 data_split="train", 
                 key=None):

        self.data_path = data_path
        self.envs_batch_size = envs_batch_size
        self.envs_shuffle = envs_shuffle

        self.adaptation = data_split in ["adapt", "test"]
        self.shots_batch_size = shots_batch_size
        self.shots_shuffle = shots_shuffle

        if shots_batch_size <= 0 or shots_batch_size <=0 :
            raise ValueError("A batch size must be greater than 0.")

        self.key = key
        if (self.envs_shuffle or self.shots_shuffle) and self.key is None:
            raise ValueError("Shuffling the dataset requires a key.")


    @abstractmethod
    def sample_environments(self, key, batch_id, nb_envs):
        """ Provides a stateless way to sample a batch of environments """
        pass

    @abstractmethod
    def __iter__(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Loads, transforms and yields a batch of environments """
        pass

    @abstractmethod
    def __len__(self):
        """ Total number of environments / envs_batch size. """
        pass

    def __str__(self) -> str:
        return f"Dataloader properties: \n" + \
                f"Total number of environments: {len(self)} \n" + \
                f"Batch size (envs): {self.envs_batch_size} \n" + \
                f"Number of points per environment = batch size (datapoints): {self.shots_batch_size} \n" + \
                f"Input dimension: {self.input_dim} \n" + \
                f"Output dimension: {self.output_dim} \n"





class CelebADataLoader(DataLoader):
    """
    A celeb a dataloader for meta-learning.
    """
    def __init__(self, 
                 data_path="./data/", 
                 envs_batch_size=250, 
                 envs_shuffle=True, 
                 shots_batch_size=100,
                 shots_shuffle=False, 
                 data_split="train",
                 resolution=(32, 32),
                 order_pixels=False,
                 key=None):

        super().__init__(data_path, 
                        envs_batch_size, 
                        envs_shuffle, 
                        shots_batch_size, 
                        shots_shuffle, 
                        data_split, 
                        key)

        self.input_dim = 2
        self.output_dim = 3
        self.img_size = (*resolution, self.output_dim)
        self.order_pixels = order_pixels
        ## Read the partitioning file: train(0), val(1), test(2)

        partitions = pd.read_csv(self.data_path+'list_eval_partition.txt', 
                                 header=None, 
                                 sep=r'\s+', 
                                 names=['filename', 'partition'])
        if data_split in ["train"]:
            self.files = partitions[partitions['partition'] == 0]['filename'].values
        elif data_split in ["val"]:
            self.files = partitions[partitions['partition'] == 1]['filename'].values
        elif data_split in ["test"]:
            self.files = partitions[partitions['partition'] == 2]['filename'].values
        else:
            raise ValueError(f"Invalid data split provided. Got {data_split}")

        ## A list of MVPs images (or the worst during self-modulation) - Useful for active learning
        # self.mvp_files = self.files

        self.total_envs = len(self.files)
        if self.total_envs == 0:
            raise ValueError("No files found for the split.")
        if envs_batch_size > self.total_envs:
            raise ValueError(f"Envs batch size must be less than the total number of environments")

        self.total_pixels = self.img_size[0] * self.img_size[1]
        if shots_batch_size > self.total_pixels:
            raise ValueError(f"Few shots batch size must be less than the total number of pixels")

        ## Ssee CAVIA code: https://github.com/lmzintgraf/cavia)
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                            transforms.Resize((self.img_size[0], self.img_size[1]), Image.LANCZOS),
                                            transforms.ToTensor(),
                                            ])

        ## Batch bookeeping
        self.nb_batches = np.ceil(self.total_envs / self.envs_batch_size).astype(int)
        self.remainder = self.total_envs % self.envs_batch_size
        self.curr_batch_id = 0


    def get_image(self, filename) -> torch.Tensor:
        img_path = os.path.join(self.data_path+"img_align_celeba/", filename)
        img = self.transform(img_path).float()
        img = img.permute(1, 2, 0)
        return img

    def sample_pixels(self, key, img) -> Tuple[np.ndarray, jnp.ndarray]:
        total_pixels = self.img_size[0] * self.img_size[1]

        if self.order_pixels:
            flattened_indices = jnp.arange(self.shots_batch_size)
        else:
            flattened_indices = jax.random.choice(key=key, a=total_pixels, shape=(self.shots_batch_size,), replace=False)

        x, y = np.unravel_index(flattened_indices, (self.img_size[0], self.img_size[1]))
        coordinates = np.vstack((x, y)).T
        coords = torch.from_numpy(coordinates).float()
        normed_coords = (coords / torch.Tensor(self.img_size[:2])).numpy()

        pixel_values = img[coords[:, 0].long(), coords[:, 1].long(), :].numpy()

        return normed_coords, pixel_values


    def sample_environments(self, key, batch_id, nb_envs) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """ Sample a batch of environments """

        X = np.zeros((nb_envs, self.shots_batch_size, self.input_dim))
        Y = np.zeros((nb_envs, self.shots_batch_size, self.output_dim))

        if self.envs_shuffle:
            sample_idx = jax.random.choice(key=key, a=self.total_envs, shape=(nb_envs,))
            sampled_files = self.files[sample_idx]
        else:
            f_start = batch_id*self.envs_batch_size
            f_end = min([(batch_id+1)*self.envs_batch_size, self.total_envs])
            sampled_files = self.files[f_start:f_end]

        pixel_keys = jax.random.split(key, num=nb_envs)
        for e, img_name in enumerate(sampled_files):
            img = self.get_image(img_name)
            normed_coords, pixel_values = self.sample_pixels(pixel_keys[e], img)
            X[e, :, :] = normed_coords
            Y[e, :, :] = pixel_values

        return jnp.array(X), jnp.array(Y)


    def make_batch(self):

        ## Sample a batch of environments
        if self.curr_batch_id == self.nb_batches-1 and self.remainder != 0:
            X, Y = self.sample_environments(self.key, self.curr_batch_id, self.remainder)
        else:
            X, Y = self.sample_environments(self.key, self.curr_batch_id, self.envs_batch_size)

        ##  Usefull when pixels are ordered
        if self.shots_shuffle:
            X = jax.random.permutation(self.key, X, axis=1)
            Y = jax.random.permutation(self.key, Y, axis=1)

        ## Update the state of the dataloader
        self.key, _ = jax.random.split(self.key)
        self.curr_batch_id += 1

        return X, Y


    def __iter__(self):
        self.curr_batch_id = 0
        # return self
        while self.curr_batch_id < self.nb_batches:
            yield self.make_batch()

    def __next__(self):
        if self.curr_batch_id < self.nb_batches:
            yield self.make_batch()
        else:
            raise StopIteration

    def __len__(self):
        return self.total_envs

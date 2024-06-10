from ._utils import *
import warnings

class DataLoader:
    def __init__(self, dataset, t_eval=None, batch_size=-1, int_cutoff=1.0, shuffle=True, adaptation=False, data_id=None, key=None):

        self.data_id = data_id if data_id else get_id_current_time()
        # if data_id is None:
        #     print("WARNING: You did not provide a dataloader id. A new one has been generated:", self.data_id)
        #     print("WARNING: Note that this id used to distuinguish between adaptations to different environments.")

        if isinstance(dataset, str):
            raw_dat = jnp.load(dataset)
            self.dataset, self.t_eval = jnp.asarray(raw_dat['X']), jnp.asarray(raw_dat['t'])
        else:
            self.dataset = dataset
            self.t_eval = t_eval

        self.shuffle = shuffle
        if self.shuffle:
            if key is None:
                print("WARNING: You are demanding a shuffled dataset but did not provide any keys for that.")
                self.key = get_new_key()
            else:
                self.key = key

        assert jnp.ndim(self.dataset) == 4, "Dataset must be of shape (nb_envs, nb_trajs_per_env, nb_steps_per_traj, data_size)"
        assert self.t_eval.shape[0] == self.dataset.shape[2], "t_eval must have the same length as the number of steps in the dataset"

        datashape = self.dataset.shape
        self.nb_envs = datashape[0]
        self.nb_trajs_per_env = datashape[1]
        self.nb_steps_per_traj = datashape[2]
        self.data_size = datashape[3]

        # print("Dataset shape:", datashape)

        self.int_cutoff = int(int_cutoff*self.nb_steps_per_traj)    ## integration cutoff

        if batch_size < 0 or batch_size > self.nb_trajs_per_env:
            # print("WARNING: batch_size must be between 0 and nb_trajs_per_env. Setting batch_size to maximum.")
            self.batch_size = self.nb_trajs_per_env
        else:
            self.batch_size = batch_size

        self.adaptation = adaptation    ## Is this a dataset for adaptation ?

    # def __iter__(self):     ## TODO! Randomise this function
    #     nb_batches = self.nb_trajs_per_env // self.batch_size
    #     for batch_id in range(nb_batches):
    #         traj_start, traj_end = batch_id*self.batch_size, (batch_id+1)*self.batch_size
    #         yield self.dataset[:, traj_start:traj_end, :self.int_cutoff, :], self.t_eval[:self.int_cutoff]

    def __iter__(self):
        nb_batches = self.nb_trajs_per_env // self.batch_size

        if self.shuffle:
            key = get_new_key(self.key)

            ## The strategy below eleviates encountering the same (env1, traj1) - (env2, traj2) pair across all batches

            ## 1) Extract a subset of environments
            e_start = jax.random.randint(key, shape=(1,), minval=0, maxval=self.nb_envs)[0]
            length = jax.random.randint(key, shape=(1,), minval=e_start+1, maxval=self.nb_envs+1)[0] - e_start
            ## 2) Shuffle that subset accross dimension 1 (trajs), then put them back at the same place
            perm_env = jax.random.permutation(key, self.dataset[e_start:e_start+length, ...], axis=1)
            perm_dataset = self.dataset.at[e_start:e_start+length, ...].set(perm_env)
            ## 3) Shuffle the resulting dataset again accross dimension 1 (for extra randomness)
            perm_dataset = jax.random.permutation(key, perm_dataset, axis=1)

            # ## 1) Extract a subset of environments
            # e_start = jax.random.randint(key, shape=(1,), minval=0, maxval=self.nb_envs)[0]
            # length = jax.random.randint(key, shape=(1,), minval=e_start+1, maxval=self.nb_envs+1)[0] - e_start
            # ## 2) Shuffle that subset accross dimension 1 (trajs), then put them back at the same place
            # perm_env = jax.random.permutation(key, perm_dataset[e_start:e_start+length, ...], axis=1)
            # perm_dataset = self.dataset.at[e_start:e_start+length, ...].set(perm_env)
            # ## 3) Shuffle the resulting dataset again accross dimension 1 (for extra randomness)
            # perm_dataset = jax.random.permutation(key, perm_dataset, axis=1)

        else:
            perm_dataset = self.dataset

        ## We are now ready to iterate over the dataset
        for batch_id in range(nb_batches):
            traj_start, traj_end = batch_id*self.batch_size, (batch_id+1)*self.batch_size
            yield perm_dataset[:, traj_start:traj_end, :self.int_cutoff, :], self.t_eval[:self.int_cutoff]

        if self.shuffle:
            self.key = key

    def __len__(self):
        return self.nb_envs * self.nb_trajs_per_env











class RegDataLoader:
    """
    A simple dataloader for general-purpose meta-learning regression tasks.
    """
    def __init__(self, datapath, batch_size, shuffle=True, adaptation=False, key=None):

        raw_dat = jnp.load(datapath)
        self.X, self.Y = jnp.asarray(raw_dat['X']), jnp.asarray(raw_dat['Y'])

        max_nb_envs = 1000      ## TODO: remove this please !
        self.X = self.X[:max_nb_envs, ...]
        self.Y = self.Y[:max_nb_envs, ...]

        self.nb_envs = self.X.shape[0]
        self.nb_points_per_env = self.X.shape[1]
        self.input_dim = self.X.shape[2]
        self.output_dim = self.Y.shape[2]

        self.shuffle = shuffle
        self.key = key
        if self.shuffle and self.key is None:
            raise ValueError("Shuffling the dataset requires a key.")

        if 0 < batch_size and batch_size <= self.nb_points_per_env:
            self.batch_size = batch_size
        else:
            raise ValueError("Invalid batch size provided.")

        if self.nb_points_per_env % self.batch_size != 0:
            raise ValueError("The batch size must evenly divide the number of datapoints per environment.")

        self.adaptation = adaptation


    def __iter__(self):
        ## To reduce the chance of seeing the same (env1, point1) - (env2, point2) across epochs
        if self.shuffle:
            _, self.key = jax.random.split(self.key)
            # 1) Extract a subset of environments
            e_start = jax.random.randint(self.key, shape=(1,), minval=0, maxval=self.nb_envs)[0]
            length = jax.random.randint(self.key, shape=(1,), minval=e_start+1, maxval=self.nb_envs+1)[0] - e_start
            # 2) Shuffle that subset accross dimension 1 (data points), then put them back at the same place
            X_small = jax.random.permutation(self.key, self.X[e_start:e_start+length, ...], axis=1)
            Y_small = jax.random.permutation(self.key, self.Y[e_start:e_start+length, ...], axis=1)
            X_perm = self.X.at[e_start:e_start+length, ...].set(X_small)
            Y_perm = self.Y.at[e_start:e_start+length, ...].set(Y_small)
            # 3) Shuffle the resulting dataset again accross dimension 1 (for extra randomness)
            X_perm = jax.random.permutation(self.key, X_perm, axis=1)
            Y_perm = jax.random.permutation(self.key, Y_perm, axis=1)
            del X_small, Y_small
        else:
            X_perm = self.X
            Y_perm = self.Y


        ## Iterate over the dataset
        nb_batches = self.nb_points_per_env // self.batch_size
        for batch_id in range(nb_batches):
            batch_start, batch_end = batch_id*self.batch_size, (batch_id+1)*self.batch_size
            yield X_perm[:, batch_start:batch_end, :], Y_perm[:, batch_start:batch_end, :]

    def __len__(self):
        return self.nb_envs * self.nb_points_per_env

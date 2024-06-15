import pickle

from selfmod.dataloader import DataLoader
from selfmod.learner import ArrayContextParams, NeuralContextFlow, Learner
from selfmod.visualtester import VisualTester
from ._utils import *

# import gc


















#%%
class Trainer:
    def __init__(self, learner:Learner, optimisers, key=None):
        if key is None:
            raise ValueError("You must provide a key for the trainer")
        self.key = key      ## Default training key

        if not isinstance(learner, Learner):
            raise ValueError("The learner must be an instance of Learner")
        else:
            self.learner = learner
        self.opt_model, self.opt_ctx = optimisers

        self.opt_model_state = self.opt_model.init(eqx.filter(self.learner.model, eqx.is_array))
        # self.opt_ctx_state = self.opt_ctx.init(eqx.filter(self.learner.contexts, eqx.is_array))     ## TODO Not good !

        self.losses_model = []
        self.losses_ctx = []
 
        # self.val_losses = []



    def train_proximal(self, 
                       super_dataloader: DataLoader, 
                       nb_epochs,
                       nb_outer_steps, 
                       nb_inner_steps_model=1, 
                       nb_inner_steps_ctx=10, 
                       inner_tol_model=1e-2, 
                       inner_tol_ctx=1e-2, 
                       proximal_reg=100., 
                       patience=None, 
                       print_error_every=1, 
                       save_path=False, 
                       val_dataloader=None, 
                       val_criterion=None, 
                       key=None):
        """ Train the model using the proximal gradient descent algorithm """

        key = key if key is not None else self.key

        loss_fn = self.learner.loss_fn

        model = self.learner.model
        opt_state_model = self.opt_model_state

        super_contexts = self.learner.contexts
        ## Create a temporary context of size env_batch_size
        blank_contexts = ArrayContextParams(super_dataloader.envs_batch_size, self.learner.context_size)
        # opt_state_ctx = self.opt_ctx_state
        if not hasattr(self, 'opt_ctx_state'):
            opt_state_ctx = self.opt_ctx.init(eqx.filter(blank_contexts, eqx.is_array))


        @eqx.filter_jit
        def train_step_model(model, model_old, contexts, batch, weightings, opt_state, key):
            print('\nCompiling function "train_step" for the model ...')

            def prox_loss_fn(model, contexts, batch, weightings, key):
                loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
                diff_norm = params_diff_norm_squared(model, model_old)
                return loss + proximal_reg * diff_norm / 2., (*aux_data, diff_norm)

            (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(model, contexts, batch, weightings, key)

            updates, opt_state = self.opt_model.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)

            return model, contexts, opt_state, loss, aux_data


        @eqx.filter_jit
        def train_step_ctx(model, contexts, contexts_old, batch, weightings, opt_state, key):
            print('Compiling function "train_step" for the contexts ...')

            def prox_loss_fn(contexts, model, batch, weightings, key):
                loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
                diff_norm = params_diff_norm_squared(contexts, contexts_old)
                return loss + proximal_reg * diff_norm / 2., (*aux_data, diff_norm)

            (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(contexts, model, batch, weightings, key)

            updates, opt_state = self.opt_ctx.update(grads, opt_state)
            contexts = eqx.apply_updates(contexts, updates)

            return model, contexts, opt_state, loss, aux_data


        if not isinstance(super_dataloader, DataLoader):
            raise ValueError("The dataloader must be an instance of DataLoader")
        if val_dataloader is not None:
            tester = VisualTester(self, key=key)

        print(f"\n\n=== Beginning training with proximal alternating minimization ... ===")
        print(f"    Number of examples in a batch along envs: {super_dataloader.envs_batch_size}")
        print(f"    Number of examples in a batch along datapoints: {super_dataloader.points_batch_size}")
        print(f"    Maximum number of steps per inner minimizations: {nb_inner_steps_model, nb_inner_steps_ctx}")
        print(f"    Maximum number of outer minimizations: {nb_outer_steps}")

        start_time = time.time()

        losses_model = []
        losses_ctx = []
        if val_dataloader is not None:
            val_losses = []

        loss_key, _ = jax.random.split(key)
        early_stopping_count = 0

        for epoch in range(nb_epochs):

            for env_batch, (dataloader, env_ids) in enumerate(super_dataloader):

                super_loss_epoch_model = 0.
                super_loss_epoch_ctx = 0.
                super_nb_batches = 0

                ## Create a temporary context of size env_batch_size
                super_ctx_values = super_contexts.params[env_ids]
                contexts = eqx.tree_at(lambda c: c.params, blank_contexts, super_ctx_values)
                contexts = blank_contexts

                weightings = jnp.ones(dataloader.nb_envs) / dataloader.nb_envs

                for out_step in range(nb_outer_steps):

                    model_old = jax.tree_util.tree_map(lambda x: x, model)
                    contexts_old = jax.tree_util.tree_map(lambda x: x, contexts)

                    ## Model proximal innner minimization
                    model_prev = jax.tree_util.tree_map(lambda x: x, model)
                    for in_step_model in range(nb_inner_steps_model):

                        nb_batches_model = 0
                        loss_sum_model = 0.

                        for _, batch in enumerate(dataloader):
                            loss_key, _ = jax.random.split(loss_key)

                            model, contexts, opt_state_model, loss_model, (_, term1, term2, diff_model_) = train_step_model(model, model_old, contexts, batch, weightings, opt_state_model, loss_key)

                            loss_sum_model += loss_model
                            nb_batches_model += 1

                        diff_model = params_diff_norm_squared(model, model_prev) / params_norm_squared(model_prev)
                        if diff_model < inner_tol_model or out_step==0:
                            break
                        model_prev = model

                    loss_epoch_model = loss_sum_model/nb_batches_model

                    ## TODO: To imitate CAVIA, we should set the ctx to zero here !

                    ## Contexts proximal innner minimization
                    contexts_prev = jax.tree_util.tree_map(lambda x: x, contexts)
                    for in_step_ctx in range(nb_inner_steps_ctx):

                        nb_batches_ctx = 0
                        loss_sum_ctx = 0.

                        for _, batch in enumerate(dataloader):
                            loss_key, _ = jax.random.split(loss_key)

                            model, contexts, opt_state_ctx, loss_ctx, (_, term1, term2, diff_ctx_) = train_step_ctx(model, contexts, contexts_old, batch, weightings, opt_state_ctx, loss_key)

                            loss_sum_ctx += loss_ctx
                            nb_batches_ctx += 1

                        diff_ctx = params_diff_norm_squared(contexts, contexts_prev) / params_norm_squared(contexts_prev)
                        if diff_ctx < inner_tol_ctx or out_step==0:
                            break
                        contexts_prev = contexts

                    loss_epoch_ctx = loss_sum_ctx/nb_batches_ctx

                    super_loss_epoch_model += loss_epoch_model
                    super_loss_epoch_ctx += loss_epoch_ctx
                    super_nb_batches += 1

                    # ## TODO remove the following two lines
                    # losses_model.append(loss_epoch_model)
                    # losses_ctx.append(loss_epoch_ctx)

                # ## Delete dataloader to free up memory
                # del dataloader

                if (env_batch%print_error_every==0) and (out_step%print_error_every==0 or out_step<=3 or out_step==nb_outer_steps-1):
                    # print(f"Meta Batch: {env_batch:-5d}     Outer Step: {out_step:-5d}      LossModel: {loss_epoch_model:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}     ValCrit: {ind_crit:-.8f}", flush=True, end="\r")
                    print(f"Epoch: {epoch:-3d}     EnvBatchId: {env_batch:-3d}      LossModel: {loss_epoch_model:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}", flush=True, end="\n")
                    print(f"\t-NbInnerStepsMod: {in_step_model+1:4d}\n\t-NbInnerStepsCxt: {in_step_ctx+1:4d}\n\t-DiffMod:   {diff_model:.2e}\n\t-DiffCxt:   {diff_ctx:.2e}", flush=True, end="\r")

                if in_step_model < 1 and in_step_ctx < 1:
                    early_stopping_count += 1
                else:
                    early_stopping_count = 0

                if (patience is not None) and (early_stopping_count >= patience):
                    print(f"Stopping early after {patience} steps with no improvement in the loss. Consider increasing the tolerances for the inner minimizations.")
                    break

                losses_model.append(super_loss_epoch_model/super_nb_batches)
                losses_ctx.append(super_loss_epoch_ctx/super_nb_batches)

                ## Reassemble the super_context
                super_ctx_new_params = super_contexts.params.at[env_ids].set(contexts.params)
                super_contexts = eqx.tree_at(lambda c: c.params, super_contexts, super_ctx_new_params)


                if val_dataloader is not None:
                    self.learner.model = model
                    self.learner.contexts = super_contexts
                    ind_crit,_ = tester.test(val_dataloader, criterion=val_criterion, verbose=False)
                    print(f"     Validation Criterion: {ind_crit:-.8f}", flush=True)
                    val_losses.append(np.array([out_step, ind_crit]))
                    ## Check if val loss is lowest to save the model
                    if ind_crit <= jnp.stack(val_losses)[:,1].min() and save_path:
                        print(f"        Saving best model so far ...")
                        self.learner.save_learner(save_path)
                    ## Restore the learner at the last evaluation step
                    if out_step == nb_outer_steps-1:
                        self.learner.load_learner(save_path)

                # print("\n")
                # gc.collect()



        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

        self.losses_model.append(jnp.vstack(losses_model))
        self.losses_ctx.append(jnp.vstack(losses_ctx))

        if val_dataloader is not None:
            if not hasattr(self, 'val_losses'):
                self.val_losses = []
            self.val_losses.append(jnp.vstack(val_losses))

        self.opt_model_state = opt_state_model
        self.opt_ctx_state = opt_state_ctx

        if val_dataloader is None:
            self.learner.model = model
            self.learner.contexts = super_contexts

        # Save the model and results
        if save_path:
            self.save_trainer(save_path)



    def save_trainer(self, path):
        assert path[-1] == "/", "ERROR: The path must end with /"
        # print(f"\nSaving model and results into {path} folder ...\n")

        np.savez(path+"train_histories.npz",
                 losses_model=jnp.vstack(self.losses_model), 
                 losses_ctx=jnp.vstack(self.losses_ctx))

        if hasattr(self, 'val_losses'):
            np.save(path+"val_losses.npy", jnp.vstack(self.val_losses))

        pickle.dump(self.opt_model_state, open(path+"opt_state_model.pkl", "wb"))
        pickle.dump(self.opt_ctx_state, open(path+"opt_state_ctx.pkl", "wb"))

        if not hasattr(self, 'val_losses'):
            self.learner.save_learner(path)


    def restore_trainer(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"
        print(f"\nNo training, loading model and results from {path} folder ...\n")

        histories = np.load(path+"train_histories.npz")
        self.losses_model = [histories['losses_model']]
        self.losses_ctx = [histories['losses_ctx']]

        if os.path.exists(path+"val_losses.npy"):
            self.val_losses = [np.load(path+"val_losses.npy")]

        self.opt_state_model = pickle.load(open(path+"opt_state_model.pkl", "rb"))
        self.opt_state_ctx = pickle.load(open(path+"opt_state_ctx.pkl", "rb"))

        self.learner.load_learner(path)






    def adapt_bulk(self, 
                   super_dataloader: DataLoader, 
                   nb_epochs, 
                   taylor_order=0,
                   optimizer=None, 
                   print_error_every=100, 
                   save_path=False, 
                   key=None):
        """Adapt the model to new environments (in bulk) using the provided dataset. """

        key = key if key is not None else self.key

        loss_fn = self.learner.loss_fn

        ## This is useful if we want to disable the taylor expansion
        if taylor_order==self.learner.model.taylor_order:
            model = self.learner.model
        else:
            print(f"Creating a new model with taylor order {taylor_order} ...")
            model = NeuralContextFlow(self.learner.model.neuralnet, taylor_order)

        blank_contexts = ArrayContextParams(super_dataloader.envs_batch_size, self.learner.context_size)

        if optimizer is None:       ## To continue a previous adaptation
            if hasattr(self, 'opt_adapt'):
                print("Using any previrouly defined optimizer for adapation")
                opt = self.opt_adapt
                super_contexts = self.learner.contexts_adapt
                opt_state = self.opt_state_adapt
            else:
                raise ValueError("No optimizer provided for adaptation, and none previously defined")
        else:
            opt = optimizer
            super_contexts = ArrayContextParams(super_dataloader.nb_envs, self.learner.context_size)
            opt_state = opt.init(blank_contexts)
            self.losses_adapt = []

        @eqx.filter_jit
        def train_step(model, contexts, batch, weightings, opt_state, key):
            print('\nCompiling function "train_step" for context ...')

            loss_fn_ = lambda contexts, model, batch, weightings, key: loss_fn(model, contexts, batch, weightings, key)

            (loss, aux_data), grads = eqx.filter_value_and_grad(loss_fn_, has_aux=True)(contexts, model, batch, weightings, key)

            updates, opt_state = opt.update(grads, opt_state)
            contexts = eqx.apply_updates(contexts, updates)

            return model, contexts, opt_state, loss, aux_data

        if not isinstance(super_dataloader, DataLoader):
            raise ValueError("The dataloader must be an instance of DataLoader")

        nb_env_train_steps_per_epoch = np.ceil(super_dataloader.nb_envs / super_dataloader.envs_batch_size).astype(int)
        total_steps = nb_epochs * nb_env_train_steps_per_epoch

        print(f"\n\n=== Beginning adaptation ... ===")
        print(f"    Number of environments in a batch: {super_dataloader.envs_batch_size}")
        print(f"    Number of envs train steps per epoch: {nb_env_train_steps_per_epoch}")
        print(f"    Number of training epochs: {nb_epochs}")
        print(f"    Total number of training steps: {total_steps}")

        start_time = time.time()

        losses = []
        loss_key, _ = jax.random.split(key)

        for epoch in range(nb_epochs):

            super_loss_sum = 0.
            super_nb_batches = 0

            for env_batch, (dataloader, env_ids) in enumerate(super_dataloader):

                weightings = jnp.ones(dataloader.nb_envs) / dataloader.nb_envs

                contexts = eqx.tree_at(lambda c: c.params, blank_contexts, super_contexts.params[env_ids])

                loss_sum = 0.
                nb_batches = 0

                for i, batch in enumerate(dataloader):
                    loss_key, _ = jax.random.split(loss_key)

                    model, contexts, opt_state, loss, (_, term1, term2) = train_step(model, contexts, batch, weightings, opt_state, loss_key)

                    loss_sum += loss
                    nb_batches += 1

                loss_epoch = loss_sum/nb_batches

                super_loss_sum += loss_epoch
                super_nb_batches += 1

            losses.append(super_loss_sum/super_nb_batches)

            super_contexts_new_params = super_contexts.params.at[env_ids].set(contexts.params)
            super_contexts = eqx.tree_at(lambda c: c.params, super_contexts, super_contexts_new_params)

            if epoch%print_error_every==0 or epoch<=3 or epoch==nb_epochs-1:
                print(f"    Epoch: {epoch:-3d}     Loss: {losses[-1]:-.8f}        ModelNorm: {jnp.mean(term1):-.8f}        ContextsNorm: {jnp.mean(term2):-.8f}", flush=True)

        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent adaptation time: %d hours %d mins %d secs" %time_in_hmsecs)

        self.losses_adapt.append(jnp.vstack(losses))

        self.opt_adapt = opt
        self.opt_state_adapt = opt_state

        self.learner.contexts_adapt = super_contexts

        if save_path:
            self.save_adapted_trainer(save_path)



    def save_adapted_trainer(self, path):
        print(f"\nSaving adaptation parameters into {path} folder ...\n")

        np.savez(path+"adapt_histories_.npz", losses_adapt=jnp.vstack(self.losses_adapt))
        pickle.dump(self.opt_state_adapt, open(path+"/opt_state_adapt.pkl", "wb"))
        eqx.tree_serialise_leaves(path+"/adapted_contexts_.eqx", self.learner.contexts_adapt)



    def restore_adapted_trainer(self, path, dataloader=None):

        if dataloader is None:
            ValueError("ERROR: You must provide the dataset on which this system was adapted.")

        print(f"\nNo adaptation, loading adaptation parameters from {path} folder ...\n")

        histories = np.load(path+"adapt_histories_.npz")
        self.losses_adapt = [histories['losses_adapt']]

        self.opt_state_adapt = pickle.load(open(path+"/opt_state_adapt.pkl", "rb"))

        self.learner.contexts_adapt = ArrayContextParams(dataloader.nb_envs, self.learner.context_size)
        self.learner.contexts_adapt = eqx.tree_deserialise_leaves(path+"/adapted_contexts_.eqx", self.learner.contexts_adapt)

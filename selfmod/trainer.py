import pickle
from typing import Any, Tuple

from selfmod.dataloader import DataLoader
from selfmod.learner import Learner
from selfmod.visualtester import VisualTester
from ._utils import *





#%%

class Trainer:
    def __init__(self, learner:Learner, optimisers, key=None):
        """ Base class for training the models"""

        if key is None:
            raise ValueError("You must provide a key for the trainer")
        self.key = key      ## Default training key

        if not isinstance(learner, Learner):
            raise ValueError("The learner must be an instance of Learner")
        else:
            self.learner = learner
        self.opt_model, self.opt_ctx = optimisers

        self.opt_state_model = self.opt_model.init(eqx.filter(self.learner.model, eqx.is_array))

        self.losses_model = []
        self.losses_ctx = []
 
    def save_trainer(self, path):
        assert path[-1] == "/", "ERROR: The path must end with /"
        # print(f"\nSaving model and results into {path} folder ...\n")

        np.savez(path+"train_histories.npz",
                 losses_model=jnp.vstack(self.losses_model), 
                 losses_ctx=jnp.vstack(self.losses_ctx))

        if hasattr(self, 'val_losses'):
            np.save(path+"val_losses.npy", jnp.vstack(self.val_losses))

        pickle.dump(self.opt_state_model, open(path+"opt_state_model.pkl", "wb"))
        # pickle.dump(self.opt_state_ctx, open(path+"opt_state_ctx.pkl", "wb"))

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
        # self.opt_state_ctx = pickle.load(open(path+"opt_state_ctx.pkl", "rb"))

        self.learner.load_learner(path)


    def save_adapted_trainer(self, path):
        print(f"\nSaving adaptation parameters into {path} folder ...\n")

        np.savez(path+"adapt_histories_.npz", losses_adapt=jnp.vstack(self.losses_adapt))
        # pickle.dump(self.opt_state_adapt, open(path+"/opt_state_adapt.pkl", "wb"))
        eqx.tree_serialise_leaves(path+"/adapted_contexts_.eqx", self.learner.contexts_adapt)






















class NCFTrainer(Trainer):
    def __init__(self, learner:Learner, optimisers, key=None):
        """ Trainer class for the proximal gradient descent algorithms (NCF) """
        super().__init__(learner, optimisers, key)

    def meta_train(self,
                    dataloader: DataLoader, 
                    nb_epochs,
                    nb_outer_steps,
                    nb_inner_steps=(1, 10),
                    inner_tols=(1e-12, 1e-12), 
                    proximal_betas=(100., 100.), 
                    max_train_batches=None,
                    patience=None, 
                    print_error_every=1, 
                    validate_every=100,
                    save_path=False, 
                    val_dataloader=None, 
                    val_criterion_id=None, 
                    max_val_batches=None,
                    val_nb_epochs=10,
                    key=None):
        """ Train the model using the proximal gradient descent algorithm """

        key = key if key is not None else self.key

        if isinstance(nb_inner_steps, int):
            nb_inner_steps = (nb_inner_steps, nb_inner_steps)
        nb_inner_steps_model, nb_inner_steps_ctx = nb_inner_steps

        inner_tol_model, inner_tol_ctx = inner_tols
        proximal_reg_model, proximal_reg_ctx = proximal_betas

        loss_fn = self.learner.loss_fn
        model = self.learner.model
        opt_state_model = self.opt_state_model

        @eqx.filter_jit
        def train_step_model(model, model_old, contexts, batch, weightings, opt_state, key):
            print('     ### Compiling function "train_step" for the model ...  ')

            def prox_loss_fn(model, contexts, batch, weightings, key):
                loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
                diff_norm = params_diff_norm_squared(model, model_old)
                return loss + proximal_reg_model * diff_norm / 2., (*aux_data, diff_norm)

            (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(model, contexts, batch, weightings, key)

            updates, opt_state = self.opt_model.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)

            return model, contexts, opt_state, loss, aux_data


        @eqx.filter_jit
        def train_step_ctx(model, contexts, contexts_old, batch, weightings, opt_state, key):
            print('     ### Compiling function "train_step" for the contexts ...  ')

            def prox_loss_fn(contexts, model, batch, weightings, key):
                loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
                diff_norm = params_diff_norm_squared(contexts, contexts_old)
                return loss + proximal_reg_ctx * diff_norm / 2., (*aux_data, diff_norm)

            (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(contexts, model, batch, weightings, key)

            updates, opt_state = self.opt_ctx.update(grads, opt_state)
            contexts = eqx.apply_updates(contexts, updates)

            return model, contexts, opt_state, loss, aux_data


        # if not isinstance(dataloader, DataLoader):
        #     raise ValueError("The dataloader must be an instance of DataLoader")
        if val_dataloader is not None:
            tester = VisualTester(self, key=key)

        print(f"\n\n=== Beginning meta training ... ===")
        print(f"    Number of examples in a batch along envs: {dataloader.batch_size}")
        print(f"    Maximum number of batches (along envs): {dataloader.num_batches}")
        print(f"    Total number of epochs: {nb_epochs}")
        print(f"    Number of outer minimizations: {nb_outer_steps}")
        print(f"    Maximum numbers of inner steps per outer minimizations: {nb_inner_steps_model, nb_inner_steps_ctx}")

        if max_train_batches is None or max_train_batches<1 or max_train_batches>dataloader.num_batches:
            max_train_batches = dataloader.num_batches
        print(f"    Training on {max_train_batches} batches")
        if val_dataloader is not None:
            if max_val_batches is None or max_val_batches<1 or max_val_batches>val_dataloader.num_batches:
                max_val_batches = val_dataloader.num_batches
            print(f"    Validating on {max_val_batches} batches")

        if isinstance(print_error_every, int):
            print_error_every = (print_error_every, print_error_every)
        print_every_batch, print_every_out_step = print_error_every

        start_time = time.time()

        losses_model = []
        losses_ctx = []
        if val_dataloader is not None:
            val_losses = []

        loss_key, _ = jax.random.split(key)
        early_stopping_count = 0

        for epoch in range(nb_epochs):
            # print(f"\nEPOCH {epoch} ... ")

            for env_batch, batch in enumerate(dataloader):
                if env_batch >= max_train_batches:
                    break
                # if env_batch%10==0:
                #     print(f"  Learning on batch {env_batch} ...")

                loss_epochs_model = 0.
                loss_epochs_ctx = 0.
                nb_batches = 0

                nb_envs_in_batch = batch[0].shape[0]
                weightings = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

                contexts = self.learner.reset_contexts(nb_envs_in_batch)
                opt_state_ctx = self.opt_ctx.init(eqx.filter(contexts, eqx.is_array))

                for out_step in range(nb_outer_steps):
                    # print(f"    Staring outer step {out_step} ...")

                    model_old = jax.tree_util.tree_map(lambda x: x, model)
                    contexts_old = jax.tree_util.tree_map(lambda x: x, contexts)

                    ## Model proximal innner minimization
                    model_prev = jax.tree_util.tree_map(lambda x: x, model)
                    for in_step_model in range(nb_inner_steps_model):

                        loss_key, _ = jax.random.split(loss_key)

                        model, contexts, opt_state_model, loss_model, (_, term2, term3, diff_model_) = train_step_model(model, model_old, contexts, batch, weightings, opt_state_model, loss_key)

                        ## TODO Update the weightings based on loss progress

                        diff_model = params_diff_norm_squared(model, model_prev) / params_norm_squared(model_prev)
                        if diff_model < inner_tol_model or out_step==0:
                            break
                        model_prev = model


                    ## Contexts proximal innner minimization
                    contexts_prev = jax.tree_util.tree_map(lambda x: x, contexts)
                    for in_step_ctx in range(nb_inner_steps_ctx):

                        loss_key, _ = jax.random.split(loss_key)

                        model, contexts, opt_state_ctx, loss_ctx, (_, term2, term3, diff_ctx_) = train_step_ctx(model, contexts, contexts_old, batch, weightings, opt_state_ctx, loss_key)

                        diff_ctx = params_diff_norm_squared(contexts, contexts_prev) / params_norm_squared(contexts_prev)
                        if diff_ctx < inner_tol_ctx or out_step==0:
                            break
                        contexts_prev = contexts


                    if in_step_model < 1 and in_step_ctx < 1:
                        early_stopping_count += 1
                    else:
                        early_stopping_count = 0

                    if (patience is not None) and (early_stopping_count >= patience):
                        print(f"Stopping early after {patience} steps with no improvement in the loss. Consider increasing the tolerances for the inner minimizations.")
                        break

                    losses_model.append(loss_model)
                    losses_ctx.append(loss_ctx)

                    if env_batch%print_every_batch==0 or env_batch==max_train_batches-1:
                        if out_step%print_every_out_step==0 or out_step==nb_outer_steps-1:
                            print(f"Epoch: {epoch:-3d}      Batch: {env_batch:-3d}      OuterStep: {out_step:-3d}      LossModel: {losses_model[-1]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}", flush=True, end="\r")
                            print(f"\n\t-NbInnerStepsMod: {in_step_model+1:4d}\n\t-NbInnerStepsCxt: {in_step_ctx+1:4d}\n\t-DiffMod:   {diff_model:.2e}\n\t-DiffCxt:   {diff_ctx:.2e}", flush=True, end="\r")

                    if val_dataloader is not None and (out_step%validate_every==0 or out_step==nb_outer_steps-1):
                        self.learner.model = model
                        self.learner.contexts = contexts
                        # print("Setting contexts in the metatrainer: \n", contexts.params)

                        ind_crit,_ = tester.evaluate(val_dataloader,
                                                    criterion_id=val_criterion_id,
                                                    max_eval_batches=max_val_batches,
                                                    nb_epochs=val_nb_epochs,
                                                    # nb_inner_steps=None,
                                                    taylor_order=0, 
                                                    verbose=False)
                        print(f"        Validation Criterion: {ind_crit:-.8f}", flush=True)
                        val_losses.append(np.array([out_step, ind_crit]))

                        ## Check if val loss is lowest to save the model
                        if ind_crit <= jnp.stack(val_losses)[:,1].min() and save_path:
                            print(f"        Saving best model so far ...")
                            self.learner.save_learner(save_path)
                        ## Restore the learner at the last evaluation step
                        if out_step == nb_outer_steps-1:
                            self.learner.load_learner(save_path)

                        # ## TODO remember to remove this (stop as soon as we get to 1e-4)
                        # if ind_crit <= 1e-4:
                        #     wall_time = time.time() - start_time
                        #     time_in_hmsecs = seconds_to_hours(wall_time)
                        #     print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)
                        #     return

                # print(f"\n\t-NbInnerStepsMod: {in_step_model+1:4d}\n\t-NbInnerStepsCxt: {in_step_ctx+1:4d}\n\t-DiffMod:   {diff_model:.2e}\n\t-DiffCxt:   {diff_ctx:.2e}", flush=True, end="\r")

                loss_epochs_model += loss_model
                loss_epochs_ctx += loss_ctx
                nb_batches += 1

            # losses_model.append(loss_epochs_model/nb_batches)
            # losses_ctx.append(loss_epochs_ctx/nb_batches)


        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

        self.losses_model.append(jnp.vstack(losses_model))
        self.losses_ctx.append(jnp.vstack(losses_ctx))

        if val_dataloader is not None:
            if not hasattr(self, 'val_losses'):
                self.val_losses = []
            self.val_losses.append(jnp.vstack(val_losses))

        self.opt_state_model = opt_state_model
        if val_dataloader is None:
            self.learner.model = model

        ## DO NOT TRUST. Just for visualisation purposes
        self.opt_ctx_state = opt_state_ctx
        self.learner.contexts = contexts

        # Save the model and results
        if save_path:
            self.save_trainer(save_path)








    def meta_test(self, 
                   dataloader: DataLoader, ## Either a full dataloader or a tuple of batches
                   nb_epochs=10, 
                   taylor_order=0,
                   optimizer=None, 
                   print_error_every=(10, 10), 
                   max_adapt_batches=None,
                   val_dataloader=None,
                   verbose=True,
                   save_path=False, 
                   key=None) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        """Adapt the model to new environments (in bulk) using the provided dataset. """

        key = key if key is not None else self.key

        loss_fn = self.learner.loss_fn
        # model = self.learner.model

        if val_dataloader is None:
            val_dataloader = dataloader

        if isinstance(print_error_every, int):
            print_error_every = (print_error_every, print_error_every)
        print_every_epoch, print_every_batch = print_error_every

        ## This is useful if we want to disable the taylor expansion
        model = self.learner.reset_model(taylor_order, verbose=verbose)

        if optimizer is None:       ## To continue a previous adaptation
            if hasattr(self, 'opt_ctx'):
                if verbose:
                    print("Using any previrouly defined optimizer for adapation")
                opt = self.opt_ctx
            else:
                raise ValueError("No optimizer provided for adaptation, and none previously defined")
        else:
            opt = optimizer
            self.losses_adapt = []

        if not hasattr(self, 'losses_adapt'):
            self.losses_adapt = []

        if verbose:
            print(f"\n\n=== Beginning meta testing ... ===")
            print(f"    Number of examples in a batch along envs: {dataloader.batch_size}")
            print(f"    Maximum number of batches (along envs): {dataloader.num_batches}")

        # if dataloader.num_batches != 1:
        #     raise ValueError("The dataloader must be a single batch of environments for meta-testing with NCF")
        # else:
        #     nb_envs_in_batch = dataloader.batch_size
        #     nb_batches = 1

        if isinstance(dataloader, DataLoader):
            nb_batches = dataloader.nb_batches
        else:
            nb_batches = len(dataloader)    ## A tuple of batches

        if max_adapt_batches is None or max_adapt_batches<1 or max_adapt_batches>dataloader.num_batches:
            max_adapt_batches = nb_batches
        else:
            if verbose and not self.learner.reuse_contexts:
                print(f"    Adapting on {max_adapt_batches} batches")

        #################### Shortcut to not recreate contexts (only use this for single batch cases)
        if self.learner.reuse_contexts and not dataloader.dataset.adaptation and dataloader.num_batches==1:
            if verbose:
                print(f"    Reusing contexts for adaptation on the single bach")

            contexts = self.learner.contexts
            batch = next(iter(val_dataloader))
            weightings = jnp.ones(dataloader.batch_size) / dataloader.batch_size

            loss, aux_data = self.learner.loss_fn(model, contexts, batch, weightings, key)
            state_data = self.learner.batch_predict(model, contexts, batch)

            return jnp.stack(aux_data, axis=1), contexts, state_data
        ####################

        def prox_loss_fn(contexts, model, batch, weightings, key):
            loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
            return loss, aux_data

        @eqx.filter_jit
        def adapt_step(model, contexts, batch, weightings, opt_state, key):
            print('     ### Compiling function "adapt_step" for the contexts ...  ')

            (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(contexts, model, batch, weightings, key)

            updates, opt_state = opt.update(grads, opt_state)
            contexts = eqx.apply_updates(contexts, updates)

            return model, contexts, opt_state, loss, aux_data

        start_time = time.time()

        losses = []
        state_data = [[], [], []]
        loss_key, _ = jax.random.split(key)

        torch.manual_seed(loss_key[0])  # Ensure the same shuffling order
        for env_batch, (batch, val_batch) in enumerate(zip(dataloader, val_dataloader)):
            if env_batch >= max_adapt_batches:
                break

            nb_envs_in_batch = batch[0].shape[0]
            weightings = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

            contexts = self.learner.reset_contexts(nb_envs_in_batch)
            opt_state_ctx = opt.init(eqx.filter(contexts, eqx.is_array))

            losses_epoch = []

            for epoch in range(nb_epochs):

                loss_key, _ = jax.random.split(loss_key)

                model, contexts, opt_state_ctx, loss_ctx, (term1, term2, term3) = adapt_step(model, contexts, batch, weightings, opt_state_ctx, loss_key)

                losses.append(loss_ctx)

                mean_loss_terms = [jnp.mean(term) for term in (term1, term2, term3)]
                losses_epoch.append(jnp.stack([loss_ctx]+mean_loss_terms))

                if epoch == nb_epochs-1:
                    ## Use the contexts and the val_batch to predict Y_hat
                    state_data_ = self.learner.batch_predict(model, contexts, val_batch)
                    [state_data[i].append(state_data_[i]) for i in range(3)]

            losses_epochs = jnp.stack(losses_epoch, axis=0)

            if verbose and (epoch%print_every_epoch==0 or epoch<=3 or epoch==max_adapt_batches-1):
                print(f"Epoch: {epoch:-3d}      Batch: {env_batch:-3d}      Loss: {losses[-1]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}", flush=True, end="\n")


        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        if verbose:
            print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

        losses = jnp.vstack(losses)
        self.losses_adapt.append(losses)

        ## DO NOT TRUST. Just for visualisation purposes
        # if isinstance(dataloader, DataLoader) and dataloader.dataset.adaptation: 
        if dataloader.dataset.adaptation: 
            self.learner.contexts_adapt = contexts
        else: 
            self.learner.contexts = contexts

        if save_path:
            self.save_adapted_trainer(save_path)

        state_data = tuple(jnp.concat(state_data[i], axis=0) for i in range(3))

        return losses_epochs, contexts, state_data





























class CAVIATrainer(Trainer):
    def __init__(self, learner:Learner, optimisers, key=None):
        """ Trainer class for the CAVIA algorithm """
        super().__init__(learner, optimisers, key)


    def meta_train(self,
                    dataloader: DataLoader, 
                    nb_outer_steps,
                    nb_inner_steps=10,
                    print_error_every=(1, 1), 
                    save_path=False, 
                    backup_contexts=False,
                    max_train_batches=None,
                    val_dataloader=None, 
                    val_criterion_id=None, 
                    max_val_batches=None,
                    validate_every=1,
                    key=None):
        """ Train the model using the MAML/CAVIA gradient descent algorithm """

        key = key if key is not None else self.key

        model = self.learner.model
        opt_state_model = self.opt_state_model

        ## 
        if backup_contexts:
            backup_ctx_folder = save_path+"contexts/"
            if not os.path.exists(backup_ctx_folder):
                os.makedirs(backup_ctx_folder)

        def inner_train_step(model, contexts, batch, weightings, opt_state, key):
            print(f'     ### (Re)Compiling function: {inner_train_step.__name__} ...  ')

            nb_envs = contexts.params.shape[0]

            env_loss_fn_ = lambda ctx, model, batch, ctxs, key: self.learner.env_loss_fn(model, batch, ctx, ctxs, key)

            ctx_grad_fn = eqx.filter_value_and_grad(env_loss_fn_, has_aux=True)

            # @eqx.filter_jit
            def step(contexts, model, batch, opt_state, key):

                keys = jax.random.split(key, num=nb_envs)

                (loss, aux_data), grads = eqx.filter_vmap(ctx_grad_fn, in_axes=(0, None, 0, None, 0))(contexts.params, model, batch, contexts.params, keys)

                ### ===== Optimizer approach
                grads_pytree = eqx.tree_at(lambda ptree: ptree.params, contexts, grads)
                updates, opt_state = self.opt_ctx.update(eqx.filter(grads_pytree, eqx.is_array), opt_state)
                contexts = eqx.apply_updates(contexts, updates)
                ### =====

                # #### ===== Simple update rule approach
                # new_params = contexts.params - 0.1*grads
                # contexts = eqx.tree_at(lambda ptree: ptree.params, contexts, new_params)
                # #### =====

                return contexts, opt_state, loss, aux_data

            keys = jax.random.split(key, num=nb_inner_steps)

            ####### Use the simple update rule  #######
            # for i in range(nb_inner_steps):
            #     contexts, opt_state, loss, aux_data = step(contexts, model, batch, opt_state, keys[i])
            ##########################################


            # ####### Use the scan algorithm  #######
            def body_func(carry, key):
                contexts, opt_state = carry
                contexts = eqx.combine(contexts, contexts_stat)

                contexts, opt_state, _, aux_data = step(contexts, model, batch, opt_state, key)

                contexts, _ = eqx.partition(contexts, eqx.is_array)
                return (contexts, opt_state), aux_data

            contexts_dyn, contexts_stat = eqx.partition(contexts, eqx.is_array)
            init_carry = (contexts_dyn, opt_state)
            (contexts_dyn, opt_state), aux_datas = jax.lax.scan(body_func, init_carry, keys)
            contexts = eqx.combine(contexts_dyn, contexts_stat)

            aux_data = [jnp.mean(term) for term in aux_datas]
            # ##########################################

            meta_loss = self.learner.loss_fn(model, contexts, batch, weightings, key)[0]

            return meta_loss, (contexts, opt_state, None, aux_data)



        @eqx.filter_jit
        def outer_train_step(model, contexts, batch, weightings, opt_states, key):
            print(f'     ### (Re)Compiling function: {outer_train_step.__name__} ...  ')

            opt_state_model, opt_state_ctx = opt_states

            (loss, aux_data), grads = eqx.filter_value_and_grad(inner_train_step, has_aux=True)(model, contexts, batch, weightings, opt_state_ctx, key)

            updates, opt_state_model = self.opt_model.update(grads, opt_state_model)
            model = eqx.apply_updates(model, updates)

            new_contexts = aux_data[0]
            opt_states = (opt_state_model, aux_data[1])
            other_loss_terms = aux_data[-1]

            return model, new_contexts, opt_states, loss, other_loss_terms


        # if not isinstance(dataloader, DataLoader):
        #     raise ValueError("The dataloader must be an instance of DataLoader")
        if val_dataloader is not None:
            tester = VisualTester(self, key=key)

        print(f"\n\n=== Beginning meta training ... ===")
        print(f"    Number of examples in a batch: {dataloader.batch_size}")
        print(f"    Total number of batches : {dataloader.num_batches}")
        print(f"    Numbers of inner steps : {nb_inner_steps}")

        if max_train_batches is None or max_train_batches<1 or max_train_batches>dataloader.num_batches:
            max_train_batches = dataloader.num_batches
        print(f"    Training on {max_train_batches} batches")
        if val_dataloader is not None:
            if max_val_batches is None or max_val_batches<1 or max_val_batches>val_dataloader.num_batches:
                max_val_batches = val_dataloader.num_batches
            print(f"    Validating on {max_val_batches} batches")

        if isinstance(print_error_every, int):
            print_error_every = (print_error_every, print_error_every)
        print_every_epoch, print_every_batch = print_error_every

        start_time = time.time()

        losses = []

        if val_dataloader is not None:
            val_losses = []

        loss_key, _ = jax.random.split(key)

        step = 0

        nb_epochs = nb_outer_steps
        for epoch in range(nb_epochs):

            loss_epoch = 0.
            nb_batches = 0

            for env_batch, batch in enumerate(dataloader):
                if env_batch >= max_train_batches:
                    break

                nb_envs_in_batch = batch[0].shape[0]
                weightings = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

                ## Reset the context and the optimizer
                contexts = self.learner.reset_contexts(nb_envs_in_batch)
                opt_state_ctx = self.opt_ctx.init(eqx.filter(contexts, eqx.is_array))

                loss_key, _ = jax.random.split(loss_key)
                opt_states = (opt_state_model, opt_state_ctx)
                model, contexts, opt_states, loss, (term1, term2, term3) = outer_train_step(model, contexts, batch, weightings, opt_states, loss_key)

                opt_state_model, _ = opt_states

                loss_epoch += loss
                nb_batches += 1
                step += 1

                losses.append(loss)

                # print("All loss terms: ", term1, term2, term3)

                if epoch%print_every_epoch==0 or epoch==nb_epochs-1:
                    if env_batch%print_every_batch==0 or env_batch==max_train_batches-1:
                        print(f"Epoch: {epoch:-3d}      Batch: {env_batch:-3d}    Loss: {losses[-1]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}", flush=True, end="\n")

                        # alpha = model.taylor_weight[0]
                        # print(f"Current unnormalised weight of the taylor expansion: {alpha:-.8f}       NormalisedWeight: {jax.nn.sigmoid(model.taylor_scale*alpha):-.8f}", flush=True, end="\r")
                        # print()

                        if backup_contexts and epoch==nb_epochs-1:
                            ## Save the context's numpy array with the suffix of the current batch*epoch
                            context_save_path = backup_ctx_folder+f"contexts_epoch{epoch:04d}_batch{env_batch:06d}.npy"
                            np.save(context_save_path, contexts.params)

                            ## Save the model as well
                            eqx.tree_serialise_leaves(backup_ctx_folder+"model.eqx", model)

            if epoch==nb_epochs-1 and hasattr(self.learner.model, 'taylor_weight'):
                alpha = model.taylor_weight[0]
                print(f"Current unnormalised weight of the taylor expansion: {alpha:-.8f}       NormalisedWeight: {jax.nn.sigmoid(model.taylor_scale*alpha):-.8f}", flush=True, end="\n")
                print()

            if val_dataloader is not None and (epoch%validate_every==0 or epoch==nb_epochs-1):
                self.learner.model = model
                self.learner.contexts = contexts

                ind_crit,_ = tester.evaluate(dataloader,
                                            criterion_id=val_criterion_id,
                                            max_eval_batches=max_val_batches,
                                            nb_epochs=nb_inner_steps,
                                            val_dataloader=val_dataloader,
                                            taylor_order=0, 
                                            verbose=False)
                print(f"        Validation Criterion: {ind_crit:-.8f}", flush=True, end="\n")
                val_losses.append(np.array([step, ind_crit]))

                # ## TODO Make a visualisation and save (like Zintgraff)
                # train_XY = dataloader.sample_environments(key, 0, 1)
                # val_XY = val_dataloader.sample_environments(key, 0, 1)
                # batch = (batch for batch in [train_XY, val_XY])
                # tester.visualizeTrainVal(batch, save_path=save_path, key=key)

                ## Check if val loss is lowest to save the model
                if ind_crit <= jnp.stack(val_losses)[:,1].min() and save_path:
                    print(f"        Saving best model so far ...", end="\n")
                    self.learner.save_learner(save_path)
                ## Restore the learner at the last evaluation step
                if epoch == nb_epochs-1:
                    self.learner.load_learner(save_path)

                # ## TODO remember to remove this (stop as soon as we get to 1e-4)
                # if ind_crit <= 1e-4:
                #     wall_time = time.time() - start_time
                #     time_in_hmsecs = seconds_to_hours(wall_time)
                #     print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)
                #     return

            loss_epoch /= nb_batches


        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        print("\nTotal gradient descent training time: %d hours %d mins %d secs" %time_in_hmsecs)

        self.losses_model.append(jnp.vstack(losses))
        self.losses_ctx.append(jnp.vstack(losses))          ## TODO: Wrong, just for quick prototyping !

        if val_dataloader is not None:
            if not hasattr(self, 'val_losses'):
                self.val_losses = []
            self.val_losses.append(jnp.vstack(val_losses))

        self.opt_state_model = opt_state_model
        if val_dataloader is None:
            self.learner.model = model

        ## DO NOT TRUST. Just for visualisation purposes
        self.opt_ctx_state = opt_state_ctx
        self.learner.contexts = contexts

        # Save the model and results
        if save_path:
            self.save_trainer(save_path)


    def meta_test(self, 
                   dataloader: DataLoader, ## Either a full dataloader or a tuple of batches
                   nb_epochs=10,        ## Number of inner gradient update steps
                   taylor_order=0,
                   optimizer=None, 
                   print_error_every=(1, 1), 
                   max_adapt_batches=None,
                   val_dataloader=None,
                   verbose=True,
                   save_path=False, 
                   key=None) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        """Adapt the model to new environments (in bulk) using the provided dataset. """

        key = key if key is not None else self.key

        nb_inner_steps = nb_epochs
        if val_dataloader is None:
            val_dataloader = dataloader

        ## This is useful if we want to disable the taylor expansion
        model = self.learner.reset_model(taylor_order, verbose=verbose)

        if optimizer is None:       ## To continue a previous adaptation
            if hasattr(self, 'opt_ctx'):
                if verbose:
                    print("Using any previrouly defined optimizer for adapation")
                opt = self.opt_ctx
            else:
                raise ValueError("No optimizer provided for adaptation, and none previously defined")
        else:
            opt = optimizer
            self.losses_adapt = []

        @eqx.filter_jit
        def adapt_step_cavia(model, contexts, batch, weightings, opt_state, opt, env_loss_fn, key):
            print(f'     ### (Re)Compiling function: {adapt_step_cavia.__name__} ...  ')

            nb_envs, context_size = contexts.params.shape

            env_loss_fn_ = lambda ctx, model, batch, ctxs, key: env_loss_fn(model, batch, ctx, ctxs, key)

            ctx_grad_fn = eqx.filter_value_and_grad(env_loss_fn_, has_aux=True)
            keys = jax.random.split(key, num=nb_envs)
            (loss, aux_data), grads = eqx.filter_vmap(ctx_grad_fn, in_axes=(0, None, 0, None, 0))(contexts.params, model, batch, contexts.params, keys)

            #### ===== Optimizer approach
            grads_pytree = eqx.tree_at(lambda ptree: ptree.params, contexts, grads)
            updates, opt_state = opt.update(eqx.filter(grads_pytree, eqx.is_array), opt_state)
            contexts = eqx.apply_updates(contexts, updates)
            #### =====

            # #### ===== Simple update rule approach
            # new_params = contexts.params - 0.1*grads
            # contexts = eqx.tree_at(lambda ptree: ptree.params, contexts, new_params)
            # #### =====

            return model, contexts, opt_state, jnp.mean(loss), aux_data


        if isinstance(dataloader, DataLoader):
            nb_batches = dataloader.nb_batches
        else:
            nb_batches = len(dataloader)    ## A tuple of batches

        if verbose:
            print(f"\n\n=== Beginning adaptation ... ===")
            print(f"    Number of environment batches: {nb_batches}")
            print(f"    Number of envs train steps per batch: {nb_inner_steps}")
            print(f"    Total number of training steps: {nb_batches*nb_inner_steps}")
        if max_adapt_batches is None or max_adapt_batches<1 or max_adapt_batches>nb_batches:
            max_adapt_batches = nb_batches
        else:
            if verbose:
                print(f"    Adapting on {max_adapt_batches} batches")


        #################### Shortcut to not recreate contexts (only use this for single batch cases)
        if self.learner.reuse_contexts and not dataloader.dataset.adaptation and dataloader.num_batches==1:
            if verbose:
                print(f"    Reusing contexts for adaptation on the single bach")

            contexts = self.learner.contexts
            batch = next(iter(val_dataloader))
            weightings = jnp.ones(dataloader.batch_size) / dataloader.batch_size

            loss, aux_data = self.learner.loss_fn(model, contexts, batch, weightings, key)
            state_data = self.learner.batch_predict(model, contexts, batch)

            return jnp.stack(aux_data, axis=1), contexts, state_data
        ####################

        if isinstance(print_error_every, int):
            print_error_every = (print_error_every, print_error_every)
        print_every_epoch, print_every_batch = print_error_every

        start_time = time.time()

        losses = []
        loss_key, _ = jax.random.split(key)
        state_data = [[], [], []]
        # all_contexts = []

        torch.manual_seed(key[0])  # Ensure the same shuffling order
        # for env_batch, batch in enumerate(dataloader):
        for env_batch, (batch, val_batch) in enumerate(zip(dataloader, val_dataloader)):
            if env_batch >= max_adapt_batches:
                break

            nb_envs_in_batch = batch[0].shape[0]
            weightings = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

            contexts = self.learner.reset_contexts(nb_envs_in_batch)
            opt_state = opt.init(contexts)

            for inner_step in range(nb_inner_steps):
                loss_key, _ = jax.random.split(loss_key)

                # model, contexts, opt_state, loss, aux_losses = adapt_step_proxi(model, contexts, batch, weightings, opt_state, opt, self.learner.loss_fn, loss_key)

                model, contexts, opt_state, loss, aux_losses = adapt_step_cavia(model, contexts, batch, weightings, opt_state, opt, self.learner.env_loss_fn, loss_key)

                mean_loss_terms = [jnp.mean(term) for term in aux_losses]
                losses.append(jnp.stack([loss]+mean_loss_terms))

            if verbose and (env_batch%print_every_epoch==0 or env_batch<=3 or env_batch==max_adapt_batches-1):
                print(f"    Batch: {env_batch:-3d}     Loss: {loss:-.8f}        OtherNorms: {jnp.stack(mean_loss_terms)}", flush=True, end="\r")

            ## Use the contexts and the val_batch to predict Y_hat
            state_data_ = self.learner.batch_predict(model, contexts, val_batch)
            [state_data[i].append(state_data_[i]) for i in range(3)]
            # all_contexts.append(contexts)

        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        if verbose:
            print("\nTotal gradient descent adaptation time: %d hours %d mins %d secs" %time_in_hmsecs)

        losses = jnp.vstack(losses)
        if not hasattr(self, 'losses_adapt'):
            self.losses_adapt = []
        self.losses_adapt.append(losses)

        ## DO NOT TRUST. Just for visualisation purposes
        if isinstance(dataloader, DataLoader) and dataloader.dataset.adaptation: 
            self.learner.contexts_adapt = contexts
        else: 
            self.learner.contexts = contexts

        if save_path:
            self.save_adapted_trainer(save_path)

        # ## Use the contexts and the batch to predict Y_hat
        # for batch in val_dataloader:
        #     pass        ## Batch is the last batch from val_dataloader
        # state_data = self.learner.batch_predict(model, contexts, batch)

        state_data = tuple(jnp.concat(state_data[i], axis=0) for i in range(3))
        # all_contexts = jnp.concatenate(all_contexts, axis=0)

        return losses, contexts, state_data












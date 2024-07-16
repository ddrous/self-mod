import pickle
from typing import Any, Tuple

from selfmod.dataloader import DataLoader
from selfmod.learner import ArrayContextParams, NeuralContextFlow, Learner, NeuralODE, NonBatchedNeuralContextFlow
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


    # def restore_adapted_trainer(self, path):

    #     print(f"\nNo adaptation, loading adaptation parameters from {path} folder ...\n")

    #     histories = np.load(path+"adapt_histories_.npz")
    #     self.losses_adapt = [histories['losses_adapt']]

    #     self.opt_state_adapt = pickle.load(open(path+"/opt_state_adapt.pkl", "rb"))





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
                    save_path=False, 
                    val_dataloader=None, 
                    val_criterion_id=None, 
                    key=None):
        """ Train the model using the proximal gradient descent algorithm """

        key = key if key is not None else self.key

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
        print(f"    Maximum number of outer minimizations: {nb_outer_steps}")
        print(f"    Maximum numbers of inner steps per outer minimizations: {nb_inner_steps_model, nb_inner_steps_ctx}")

        if max_train_batches<1 or max_train_batches>dataloader.num_batches or max_train_batches is None:
            max_train_batches = dataloader.num_batches
        else:
            print(f"    Training on {max_train_batches} batches")

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

                contexts = ArrayContextParams(nb_envs_in_batch, self.learner.context_size)
                opt_state_ctx = self.opt_ctx.init(eqx.filter(contexts, eqx.is_array))

                for out_step in range(nb_outer_steps):
                    # print(f"    Staring outer step {out_step} ...")

                    model_old = jax.tree_util.tree_map(lambda x: x, model)
                    contexts_old = jax.tree_util.tree_map(lambda x: x, contexts)

                    ## Contexts proximal innner minimization
                    contexts_prev = jax.tree_util.tree_map(lambda x: x, contexts)
                    for in_step_ctx in range(nb_inner_steps_ctx):

                        loss_key, _ = jax.random.split(loss_key)

                        model, contexts, opt_state_ctx, loss_ctx, (_, term1, term2, diff_ctx_) = train_step_ctx(model, contexts, contexts_old, batch, weightings, opt_state_ctx, loss_key)

                        diff_ctx = params_diff_norm_squared(contexts, contexts_prev) / params_norm_squared(contexts_prev)
                        if diff_ctx < inner_tol_ctx or out_step==0:
                            break
                        contexts_prev = contexts


                    ## Model proximal innner minimization
                    model_prev = jax.tree_util.tree_map(lambda x: x, model)
                    for in_step_model in range(nb_inner_steps_model):

                        loss_key, _ = jax.random.split(loss_key)

                        model, contexts, opt_state_model, loss_model, (_, term1, term2, diff_model_) = train_step_model(model, model_old, contexts, batch, weightings, opt_state_model, loss_key)

                        ## TODO Update the weightings based on loss progress

                        diff_model = params_diff_norm_squared(model, model_prev) / params_norm_squared(model_prev)
                        if diff_model < inner_tol_model or out_step==0:
                            break
                        model_prev = model

                    if in_step_model < 1 and in_step_ctx < 1:
                        early_stopping_count += 1
                    else:
                        early_stopping_count = 0

                    if (patience is not None) and (early_stopping_count >= patience):
                        print(f"Stopping early after {patience} steps with no improvement in the loss. Consider increasing the tolerances for the inner minimizations.")
                        break

                loss_epochs_model += loss_model
                loss_epochs_ctx += loss_ctx
                nb_batches += 1

                losses_model.append(loss_model)
                losses_ctx.append(loss_ctx)

                if env_batch%print_error_every==0 or env_batch<=3 or env_batch==dataloader.num_batches-1:
                    print(f"Epoch: {epoch:-3d}      Batch: {env_batch:-3d}      LossModel: {losses_model[-1]:-.8f}     ContextsNorm: {jnp.mean(term2):-.8f}", flush=True, end="\r")
                    # print(f"\t-NbInnerStepsMod: {in_step_model+1:4d}\n\t-NbInnerStepsCxt: {in_step_ctx+1:4d}\n\t-DiffMod:   {diff_model:.2e}\n\t-DiffCxt:   {diff_ctx:.2e}", flush=True, end="\r")

            # losses_model.append(loss_epochs_model/nb_batches)
            # losses_ctx.append(loss_epochs_ctx/nb_batches)

            if val_dataloader is not None:
                self.learner.model = model

                ind_crit,_ = tester.evaluate(val_dataloader,
                                             criterion_id=val_criterion_id,
                                             max_eval_batches=5,
                                             nb_inner_steps=nb_inner_steps_ctx,
                                             taylor_order=0, 
                                             verbose=False)
                print(f"     Validation Criterion: {ind_crit:-.8f}", flush=True)
                val_losses.append(np.array([epoch, ind_crit]))

                # ## TODO Make a visualisation and save (like Zintgraff)
                # train_XY = dataloader.sample_environments(key, 0, 1)
                # val_XY = val_dataloader.sample_environments(key, 0, 1)
                # batch = (batch for batch in [train_XY, val_XY])
                # tester.visualizeTrainVal(batch, save_path=save_path, key=key)

                ## Check if val loss is lowest to save the model
                if ind_crit <= jnp.stack(val_losses)[:,1].min() and save_path:
                    print(f"        Saving best model so far ...")
                    self.learner.save_learner(save_path)
                ## Restore the learner at the last evaluation step
                if epoch == nb_epochs-1:
                    self.learner.load_learner(save_path)


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
                   nb_inner_steps=10, 
                   taylor_order=0,
                   optimizer=None, 
                   print_error_every=(1, 1), 
                   max_adapt_batches=None,
                   verbose=True,
                   save_path=False, 
                   key=None) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        """Adapt the model to new environments (in bulk) using the provided dataset. """

        return NotImplementedError("The meta_test method is not yet implemented for the NCFTrainer class")



class CAVIATrainer(Trainer):
    def __init__(self, learner:Learner, optimisers, key=None):
        """ Trainer class for the CAVIA algorithm """
        super().__init__(learner, optimisers, key)


    def meta_train(self,
                    dataloader: DataLoader, 
                    nb_epochs,
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

        loss_fn = self.learner.loss_fn
        model = self.learner.model
        opt_state_model = self.opt_state_model

        ## 
        if backup_contexts:
            backup_ctx_folder = save_path+"contexts/"
            if not os.path.exists(backup_ctx_folder):
                os.makedirs(backup_ctx_folder)

        # # @eqx.filter_jit
        # def inner_train_step(model, contexts, batch, weightings, opt_state, key):
        #     print(f'     ### (Re)Compiling function: {inner_train_step.__name__} ...  ')

        #     def prox_loss_fn(contexts, model, batch, weightings, key):                  ## TODO this loss function is already defined as a mean across envs. MAML is environment specific !
        #         loss, aux_data = loss_fn(model, contexts, batch, weightings, key)
        #         return loss, aux_data

        #     for _ in range(nb_inner_steps):
        #         (loss, aux_data), grads = eqx.filter_value_and_grad(prox_loss_fn, has_aux=True)(contexts, model, batch, weightings, key)
        #         updates, opt_state = self.opt_ctx.update(grads, opt_state)
        #         contexts = eqx.apply_updates(contexts, updates)

        #     meta_loss = loss_fn(model, contexts, batch, weightings, key)[0]

        #     return meta_loss, (contexts, opt_state, loss, aux_data)


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
                grads_pytree = ArrayContextParams(nb_envs, self.learner.context_size)
                grads_pytree = eqx.tree_at(lambda ptree: ptree.params, grads_pytree, grads)
                updates, opt_state = self.opt_ctx.update(grads_pytree, opt_state)
                contexts = eqx.apply_updates(contexts, updates)
                ### =====

                # #### ===== Simple update rule approach
                # new_params = contexts.params - 0.1*grads
                # contexts = eqx.tree_at(lambda ptree: ptree.params, contexts, new_params)
                # #### =====

                return contexts, opt_state, loss, aux_data

            keys = jax.random.split(key, num=nb_inner_steps)

            # for i in range(nb_inner_steps): ## TODO : use scan for longer inner step counts
            #     contexts, opt_state, loss, aux_data = step(contexts, model, batch, opt_state, keys[i])

            ## Use the scan algorithm
            def body_func(carry, key):
                contexts, opt_state = carry
                contexts, opt_state, _, aux_data = step(contexts, model, batch, opt_state, key)
                return (contexts, opt_state), aux_data

            init_carry = (contexts, opt_state)
            (contexts, opt_state), aux_datas = jax.lax.scan(body_func, init_carry, keys)
            aux_data = [jnp.mean(term) for term in aux_datas]

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
        if max_val_batches is None or max_val_batches<1 or max_val_batches>val_dataloader.num_batches:
            max_val_batches = val_dataloader.num_batches
        print(f"    Validating on {max_val_batches} batches")

        print_every_epoch, print_every_batch = print_error_every


        start_time = time.time()

        losses = []

        if val_dataloader is not None:
            val_losses = []

        loss_key, _ = jax.random.split(key)

        step = 0

        for epoch in range(nb_epochs):

            loss_epoch = 0.
            nb_batches = 0

            for env_batch, batch in enumerate(dataloader):
                if env_batch >= max_train_batches:
                    break

                nb_envs_in_batch = batch[0].shape[0]
                weightings = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

                ## Reset the context and the optimizer
                contexts = ArrayContextParams(nb_envs_in_batch, self.learner.context_size)
                opt_state_ctx = self.opt_ctx.init(eqx.filter(contexts, eqx.is_array))

                loss_key, _ = jax.random.split(loss_key)
                opt_states = (opt_state_model, opt_state_ctx)
                model, contexts, opt_states, loss, (term1, term2, term3) = outer_train_step(model, contexts, batch, weightings, opt_states, loss_key)

                opt_state_model, _ = opt_states

                loss_epoch += loss
                nb_batches += 1
                step += 1

                losses.append(loss)


                if epoch%print_every_epoch==0 or epoch==nb_epochs-1:
                    if env_batch%print_every_batch==0 or env_batch<=3 or env_batch==dataloader.num_batches-1:
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

            if epoch==nb_epochs-1:
                alpha = model.taylor_weight[0]
                print(f"Current unnormalised weight of the taylor expansion: {alpha:-.8f}       NormalisedWeight: {jax.nn.sigmoid(model.taylor_scale*alpha):-.8f}", flush=True, end="\n")
                print()


            if val_dataloader is not None and epoch%validate_every==0:
                self.learner.model = model

                ind_crit,_ = tester.evaluate(val_dataloader,
                                            criterion_id=val_criterion_id,
                                            max_eval_batches=max_val_batches,
                                            nb_inner_steps=nb_inner_steps,
                                            taylor_order=0, 
                                            verbose=False)
                print(f"     Validation Criterion: {ind_crit:-.8f}", flush=True, end="\n")
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
                   nb_inner_steps=10, 
                   taylor_order=0,
                   optimizer=None, 
                   print_error_every=(1, 1), 
                   max_adapt_batches=None,
                   verbose=True,
                   save_path=False, 
                   key=None) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        """Adapt the model to new environments (in bulk) using the provided dataset. """

        key = key if key is not None else self.key

        loss_fn = self.learner.loss_fn

        ## This is useful if we want to disable the taylor expansion
        if taylor_order==self.learner.model.taylor_order:
            model = self.learner.model
        else:
            if verbose:
                print(f"Creating a new model with taylor order {taylor_order} ...")
            if isinstance(self.learner.model, NeuralContextFlow):
                model = NeuralContextFlow(self.learner.model.neuralnet, taylor_order)
            elif isinstance(self.learner.model, NeuralODE):
                model = NeuralODE(self.learner.model.vectorfield.neuralnet, taylor_order, ivp_args=self.learner.model.ivp_args)
            elif isinstance(self.learner.model, NonBatchedNeuralContextFlow):
                model = NonBatchedNeuralContextFlow(self.learner.model.neuralnet, taylor_order)
            else:
                raise ValueError("The model type is not supported")

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

        print_every_epoch, print_every_batch = print_error_every

        start_time = time.time()

        losses = []
        loss_key, _ = jax.random.split(key)

        for env_batch, batch in enumerate(dataloader):
            if env_batch >= max_adapt_batches:
                break

            nb_envs_in_batch = batch[0].shape[0]
            weightings = jnp.ones(nb_envs_in_batch) / nb_envs_in_batch

            contexts = ArrayContextParams(nb_envs_in_batch, self.learner.context_size)
            opt_state = opt.init(contexts)

            for inner_step in range(nb_inner_steps):
                loss_key, _ = jax.random.split(loss_key)

                # model, contexts, opt_state, loss, aux_losses = adapt_step_proxi(model, contexts, batch, weightings, opt_state, opt, self.learner.loss_fn, loss_key)

                model, contexts, opt_state, loss, aux_losses = adapt_step_cavia(model, contexts, batch, weightings, opt_state, opt, self.learner.env_loss_fn, loss_key)

                mean_loss_terms = [jnp.mean(term) for term in aux_losses]
                losses.append(jnp.stack([loss]+mean_loss_terms))

            if verbose and (env_batch%print_every_batch==0 or env_batch<=3 or env_batch==nb_batches-1):
                print(f"    Batch: {env_batch:-3d}     Loss: {loss:-.8f}        OtherNorms: {jnp.stack(mean_loss_terms)}", flush=True, end="\r")

        wall_time = time.time() - start_time
        time_in_hmsecs = seconds_to_hours(wall_time)
        if verbose:
            print("\nTotal gradient descent adaptation time: %d hours %d mins %d secs" %time_in_hmsecs)

        losses = jnp.vstack(losses)
        if not hasattr(self, 'losses_adapt'):
            self.losses_adapt = []
        self.losses_adapt.append(losses)

        ## DO NOT TRUST. Just for visualisation purposes
        # self.opt_adapt = opt
        # self.opt_state_adapt = opt_state
        self.learner.contexts_adapt = contexts

        if save_path:
            self.save_adapted_trainer(save_path)

        ## Use the contexts and the batch to predict Y_hat
        aux_data = predict_step(model, contexts, batch, key)

        return losses, contexts, aux_data





@eqx.filter_jit
def predict_step(model, contexts, batch, key):
    ## Use the contexts and the batch to predict Y_hat
    X, Y = batch
    Y_hat = jax.vmap(model, in_axes=(0, 0, None))(X, contexts.params, contexts.params)
    return X, Y, Y_hat


@eqx.filter_jit
def adapt_step_proxi(model, contexts, batch, weightings, opt_state, opt, loss_fn, key):
    print('     ### (Re)Compiling function "adapt_step" for context ... ')

    loss_fn_ = lambda contexts, model, batch, weightings, key: loss_fn(model, contexts, batch, weightings, key)

    (loss, aux_data), grads = eqx.filter_value_and_grad(loss_fn_, has_aux=True)(contexts, model, batch, weightings, key)

    updates, opt_state = opt.update(grads, opt_state)
    contexts = eqx.apply_updates(contexts, updates)

    return model, contexts, opt_state, loss, aux_data




@eqx.filter_jit
def adapt_step_cavia(model, contexts, batch, weightings, opt_state, opt, env_loss_fn, key):
    print(f'     ### (Re)Compiling function: {adapt_step_cavia.__name__} ...  ')

    nb_envs, context_size = contexts.params.shape

    env_loss_fn_ = lambda ctx, model, batch, ctxs, key: env_loss_fn(model, batch, ctx, ctxs, key)

    ctx_grad_fn = eqx.filter_value_and_grad(env_loss_fn_, has_aux=True)
    keys = jax.random.split(key, num=nb_envs)
    (loss, aux_data), grads = eqx.filter_vmap(ctx_grad_fn, in_axes=(0, None, 0, None, 0))(contexts.params, model, batch, contexts.params, keys)

    #### ===== Optimizer approach
    grads_pytree = ArrayContextParams(nb_envs, context_size)
    grads_pytree = eqx.tree_at(lambda ptree: ptree.params, grads_pytree, grads)
    updates, opt_state = opt.update(grads_pytree, opt_state)
    contexts = eqx.apply_updates(contexts, updates)
    #### =====

    # #### ===== Simple update rule approach
    # new_params = contexts.params - 0.1*grads
    # contexts = eqx.tree_at(lambda ptree: ptree.params, contexts, new_params)
    # #### =====

    return model, contexts, opt_state, jnp.mean(loss), aux_data

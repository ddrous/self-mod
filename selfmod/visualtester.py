
from selfmod.dataloader import DataLoader
from selfmod.learner import ArrayContextParams
from ._utils import *



















#%%

class VisualTester:
    def __init__(self, trainer, key=None):
        if key is None:
            raise ValueError("Key must be provided for reproducibility.")
        self.key = key

        self.trainer = trainer


    def test(self, super_dataloader, criterion=None, verbose=True):
        """ Compute test metrics on the adaptation dataloader  """ ## TODO non-UQ aware testing

        criterion = criterion if criterion else lambda x, x_hat: jnp.mean((x-x_hat)**2)

        if verbose == True:
            if super_dataloader.adaptation == False:
                print("==  Begining in-domain testing ... ==")
            else:
                print("==  Begining out-of-distribution testing ... ==")

        if super_dataloader.adaptation == False:
            super_contexts = self.trainer.learner.contexts
        else:
            super_contexts = self.trainer.learner.contexts_adapt

        # Y_hat, _ = jax.vmap(self.trainer.learner.model, in_axes=(0, 0, 0))(X, contexts, contexts)
        # batched_criterion = jax.vmap(jax.vmap(criterion, in_axes=(0, 0)), in_axes=(0, 0))

        batched_vmap = jax.vmap(self.trainer.learner.model, in_axes=(0, 0, 0))
        batched_criterion = jax.vmap(jax.vmap(criterion, in_axes=(0, 0)), in_axes=(0, 0))

        blank_contexts = ArrayContextParams(super_dataloader.envs_batch_size, self.trainer.learner.context_size)

        super_loss = 0.
        super_nb_batches = 0
        for env_batch, (dataloader, env_ids) in enumerate(super_dataloader):

            ## Create a temporary context of size env_batch_size
            super_ctx_values = super_contexts.params[env_ids]
            contexts = eqx.tree_at(lambda c: c.params, blank_contexts, super_ctx_values)

            X, Y = dataloader.X, dataloader.Y

            Y_hat, _ = batched_vmap(X, contexts.params, contexts.params)
            crit = jnp.mean(batched_criterion(Y, Y_hat))

            super_loss += crit
            super_nb_batches += 1

        crit = super_loss / super_nb_batches


        # crit_all = batched_criterion(Y, Y_hat).mean(axis=1)
        # crit = crit_all.mean(axis=0)

        if verbose == True:
            if dataloader.adaptation == False:
                print("Test Score (In-Domain):", crit)
            else:
                print("Test Score (OOD):", crit)
            print(flush=True)

        return crit, None



    def visualizeCelebA(self, 
                        dataloader:DataLoader, 
                        few_shot_loader:DataLoader, 
                        resolution=32, 
                        save_path=False, 
                        environment=None, 
                        key=None):
        key = key if key != None else self.key
        e = environment if environment is not None else jax.random.randint(key, (1,), 0, dataloader.nb_envs)[0]

        ## Context dimensions to plot: 3 along x, 3 along y
        ctx_x_key, ctx_y_key = jax.random.split(key, num=2)
        ctx_dims_x = jax.random.randint(ctx_x_key, (3,), 1, self.trainer.learner.context_size)-1
        ctx_dims_y = jax.random.randint(ctx_y_key, (3,), 0, self.trainer.learner.context_size-1)+1

        if dataloader.adaptation == False:
            print("==  Begining in-domain CelebA visualisation ... ==")
        else:
            print("==  Begining out-of-distribution CelebA visualisation ... ==")
        print("    Environment id:", e)
        print("    Visualized context dimensions along x:", ctx_dims_x)
        print("    Visualized context dimensions along y:", ctx_dims_y)

        if dataloader.adaptation == False:
            contexts = self.trainer.learner.contexts.params
        else:
            contexts = self.trainer.learner.contexts_adapt.params

        X_hat = dataloader.X[e]
        Y_hat, _ = self.trainer.learner.model(dataloader.X[e], contexts[e], contexts[e])
        Y_true = dataloader.Y[e]

        X_few_shots = few_shot_loader.X[e]
        Y_few_shots = few_shot_loader.Y[e]

        fig, ax = plt.subplot_mosaic('ABC;DDD;EFG', figsize=(4*3, 3.7*3))
        img_size = (resolution, resolution, 3)

        def make_image(xy_coords, rgb_pixels):
            img = np.zeros(img_size)
            x_coords = (xy_coords[:, 0] * img_size[0]).astype(int)
            y_coords = (xy_coords[:, 1] * img_size[1]).astype(int)
            img[x_coords, y_coords, :] = np.clip(rgb_pixels, 0., 1.)
            return img

        true_img = make_image(X_hat, Y_true)
        ax['A'].imshow(true_img)
        ax['A'].set_title('True', fontsize=14)

        few_shoot_img = make_image(X_few_shots, Y_few_shots)
        ax['B'].imshow(few_shoot_img)
        ax['B'].set_title('Few-shots', fontsize=14)

        pred_img = make_image(X_hat, Y_hat)
        ax['C'].imshow(pred_img)
        ax['C'].set_title('Predicted', fontsize=14)


        nb_envs = dataloader.nb_envs
        losses_model = np.vstack(self.trainer.losses_model)
        losses_ctx = np.vstack(self.trainer.losses_ctx)
        xis = self.trainer.learner.contexts.params

        if dataloader.adaptation == True:  ## Overwrite the above if adaptation
            losses_model = np.vstack(self.trainer.losses_adapt)
            losses_ctx = np.vstack(self.trainer.losses_adapt)
            xis = self.trainer.learner.contexts_adapt.params

        mke = np.ceil(losses_model.shape[0]/100).astype(int)
        mks = 2

        label_model = "Model Loss" if dataloader.adaptation == False else "Model Loss Adapt"
        ax['D'].plot(losses_model[:,0], label=label_model, color="grey", linewidth=3, alpha=1.0)
        label_ctx = "Context Loss" if dataloader.adaptation == False else "Context Loss Adapt"
        ax['D'].plot(losses_ctx[:,0], "x-", markevery=mke, markersize=mks, label=label_ctx, color="grey", linewidth=1, alpha=0.5)

        if dataloader.adaptation==False and hasattr(self.trainer, 'val_losses') and len(self.trainer.val_losses)>0:
            val_losses = np.vstack(self.trainer.val_losses)
            ax['D'].plot(val_losses[:,0], val_losses[:,1], "y.", label="Validation Loss", linewidth=3, alpha=0.5)

        ax['D'].set_xlabel("Step")
        ax['D'].set_title("Loss Terms")
        ax['D'].set_yscale('log')
        ax['D'].legend()


        colors = ['dodgerblue', 'crimson', 'darkgreen', 'purple', 'brown']
        ax['E'].scatter(xis[:,ctx_dims_x[0]], xis[:,ctx_dims_y[1]], s=30, c=colors[0], marker='X')
        ax['F'].scatter(xis[:,ctx_dims_x[1]], xis[:,ctx_dims_y[1]], s=50, c=colors[1], marker='o')
        ax['G'].scatter(xis[:,ctx_dims_x[2]], xis[:,ctx_dims_y[2]], s=60, c=colors[2], marker='+')

        ax['E'].set_title(f'dim {ctx_dims_y[0]} vs dim {ctx_dims_x[0]}')
        ax['E'].set_xlabel(f'dim {ctx_dims_x[0]}')
        ax['E'].set_ylabel(f'dim {ctx_dims_y[0]}')

        ax['F'].set_title(f'dim {ctx_dims_y[1]} vs dim {ctx_dims_x[1]}')
        ax['F'].set_xlabel(f'dim {ctx_dims_x[1]}')
        ax['F'].set_ylabel(f'dim {ctx_dims_y[1]}')

        ax['G'].set_title(f'dim {ctx_dims_y[2]} vs dim {ctx_dims_x[2]}')
        ax['G'].set_xlabel(f'dim {ctx_dims_x[2]}')
        ax['G'].set_ylabel(f'dim {ctx_dims_y[2]}')

        plt.suptitle(f"Results for environment {e}", fontsize=20)

        plt.tight_layout()
        # plt.show();
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Testing finished. Figure saved in:", save_path);

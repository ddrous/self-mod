
from abc import abstractmethod
from selfmod.dataloader import DataLoader
from ._utils import *



class VisualTester:
    def __init__(self, trainer, key=None):
        if key is None:
            raise ValueError("Key must be provided for reproducibility.")
        self.key = key

        self.trainer = trainer

    @abstractmethod
    def evaluate(self, 
                 super_dataloader, 
                 nb_inner_steps=100,
                 loss_criterion=None, 
                 criterion_id=0, 
                 max_eval_batches=-1, 
                 taylor_order=0, 
                 verbose=True):
        """
        Adapt and compute test metrics on the adaptation dataloader.
         - loss_criterion if the one used for training is not satisfactory.
         - criterion_id is the index of the desired criterion from the loss auxiliaries
        """

        ## Adapt and extract the losses for each batch of environment
        losses, _, _ = self.trainer.meta_test(super_dataloader, 
                                              nb_inner_steps=nb_inner_steps, max_adapt_batches=max_eval_batches,taylor_order=taylor_order, 
                                              verbose=False)
        losses_means = jnp.mean(losses, axis=0)

        ## TODO Compute the confidence intervals on the losses

        ## TODO Add the environment-wide UQ from NCF aware testing

        mean_loss = losses_means[criterion_id]
        if verbose:
            print("==  Testing finished ... ==")
            print("    Criterion loss value:", mean_loss)

        return mean_loss, None


    @abstractmethod
    def visualizeTrainVal(self, dataloader, few_shot_loader, save_path=False, environment=None, key=None):
        """ Visualize two samples and their predictions: one from training and the other from validation """
        ## The dataloader muct be a generator of length 2. One containing training data and the second validation data.
        pass

    @abstractmethod
    def visualizeArtefacts(self, adaptation=False, save_path=False, key=None):
        """ Visualize the artefacts of the model : loss, and context dimensions """
        key = key if key != None else self.key

        ## Context dimensions to plot: 3 along x, 3 along y
        ctx_x_key, ctx_y_key = jax.random.split(key, num=2)
        ctx_dims_x = jax.random.randint(ctx_x_key, (3,), 1, self.trainer.learner.context_size)-1
        ctx_dims_y = jax.random.randint(ctx_y_key, (3,), 0, self.trainer.learner.context_size-1)+1

        print("==  Begining artefacts visualisation ... ==")
        print("    Visualized context dimensions along x:", ctx_dims_x)
        print("    Visualized context dimensions along y:", ctx_dims_y)


        fig, ax = plt.subplot_mosaic('DDD;EFG', figsize=(4*3, 3.7*2))

        losses_model = np.vstack(self.trainer.losses_model)
        losses_ctx = np.vstack(self.trainer.losses_ctx)

        if hasattr(self.trainer.learner, 'contexts'):
            xis = self.trainer.learner.contexts.params
        else:
            print("No contexts found. Using zeros.")
            xis = jnp.zeros((10, self.learner.context_size))

        if adaptation == True:  ## Overwrite the above if adaptation
            losses_model = np.vstack(self.trainer.losses_adapt)
            losses_ctx = np.vstack(self.trainer.losses_adapt)
            if hasattr(self.trainer.learner, 'contexts_adapt'):
                xis = self.trainer.learner.contexts_adapt.params
            else:
                print("No contexts found. Using zeros.")
                xis = jnp.zeros((10, self.learner.context_size))

        mke = np.ceil(losses_model.shape[0]/100).astype(int)
        mks = 2

        label_model = "Model Loss" if adaptation == False else "Model Loss Adapt"
        ax['D'].plot(losses_model[:,0], label=label_model, color="grey", linewidth=3, alpha=1.0)
        label_ctx = "Context Loss" if adaptation == False else "Context Loss Adapt"
        ax['D'].plot(losses_ctx[:,0], "x-", markevery=mke, markersize=mks, label=label_ctx, color="grey", linewidth=1, alpha=0.5)

        if adaptation==False and hasattr(self.trainer, 'val_losses') and len(self.trainer.val_losses)>0:
            val_losses = np.vstack(self.trainer.val_losses)
            ax['D'].plot(val_losses[:,0], val_losses[:,1], "y.", label="Validation Loss", linewidth=3, alpha=0.5)

        ax['D'].set_xlabel("Iterations")
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

        plt.suptitle(f"Losses and Context Vectors", fontsize=20)

        plt.tight_layout()
        # plt.show();
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Saving artefacts in:", save_path, flush=True);














#%%

class CelebAVisualTester(VisualTester):
    def __init__(self, trainer, key=None):
        super().__init__(trainer, key)

    def visualizeFewShots(self, 
                        few_shots_loader:DataLoader, 
                        all_shots_loader:DataLoader, 
                        nb_inner_steps=100,
                        save_path=False, 
                        key=None):
        key = key if key != None else self.key
        e = jax.random.randint(key, (1,), 0, few_shots_loader.nb_batches)[0]

        print("==  Begining in-domain CelebA visualisation ... ==")
        print("    Environment batch id:", e)

        ## The contexts are not obtained from a quick adaptation process (hidden in meta-test)
        X, Y = all_shots_loader.sample_environments(key, e, 1)
        _, _, (X, Y, Y_hat) = self.trainer.meta_test(dataloader=[(X, Y)], 
                                                     nb_inner_steps=nb_inner_steps, 
                                                     verbose=False)
        X_hat, Y_true, Y_hat = X[0], Y[0], Y_hat[0]

        X_few_shots, Y_few_shots = few_shots_loader.sample_environments(key, e, 1)
        X_few_shots, Y_few_shots = X_few_shots[0], Y_few_shots[0]

        fig, ax = plt.subplot_mosaic('ABC', figsize=(4*3, 3.7*1))
        img_size = few_shots_loader.img_size

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


        plt.suptitle(f"Sample Predictions", fontsize=20)

        plt.tight_layout()
        # plt.show();
        plt.draw();

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print("Saving visualization in:", save_path, flush=True);

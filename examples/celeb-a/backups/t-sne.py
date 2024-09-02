#%%

""" t-SNE visualization of Celeb-A dataset """

from selfmod import *


context_size = 128
envs_batch_size = 162770

contexts = ArrayContextParams(nb_envs=envs_batch_size,
                            context_size=context_size)

## Load context from "240901-174340-LC64"
contexts = eqx.tree_deserialise_leaves("../runs/240901-174340-LC64/contexts.eqx", contexts).params


## Do a t-SNE visualization of the contexts
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(contexts)

plt.figure(figsize=(16,10))
plt.scatter(tsne_results[:,0], tsne_results[:,1])

plt.show()
#%%

# Per-Example gradient norm and inner product

A short notebook showing a nice technique in PyTorch that allows us to efficiently calculate per-example gradient norms and inner products.
This implementation applies to Linear layers only,
but I think it can be extended to convolution layers in a straightforward manner.

## Inspiration

Let's say we have a mini-batch loss
$
    \ell_{\mathcal{B}}(w)
    := \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \ell_i(w),
$
where $\mathcal{B}$ is mini-batch (of indices).
In PyTorch, when we do a backward pass on $\ell_{\mathcal{B}}(w)$,
we store the gradient $\nabla \ell_{\mathcal{B}}(w)$ and not the **per-example** gradients $\nabla \ell_i(w)$.
We can actually compute the gradients per-example, but this is not done by default because it takes so much memory and is not (always) needed for optimization (i.e., the cost is $|\mathcal{B}| \cdot d$, where $d$ is the model size).

However, sometimes it would be nice if we know the norm for each gradient in our dataset, which is just one number and not $d$ numbers.
For example, one outlier loss in our dataset is enough to increase the gradient norm, which might not be reflective of our solution.
Keeping the gradient norms (instead of the whole gradients) in memory is gonna cost us only $|\mathcal{B}_t|$ numbers, so why not store them somewhere, right?
For example, if we know that the norm with respect to one example is 0,
then we know that our solution is optimal on that example.
Another general case is when we want to take an inner product of two set of gradients on the same mini-batch $\mathcal{B}$: one calculated at model $w_1$ and another at model $w_2$. Namely,
$ \nabla \ell_i(w_1) \cdot \nabla \ell_j(w_2)$
where $i,j \in \mathcal{B}$.
We can also do that efficiently because we have everything we need right there in autograd.

## How?

The idea is to intercept the forward / backward passes, save the quantities we need, and then calculate the norms or inner products.
A more memory-efficient way is to implement our own autograd.Function that has a custom forward / backward logic.
Since the norms and inner products are separable wrt coordinates, they are additive per layer, so we can backpropagate them.
One problem is that this requires a reimplementation of your model with modules that use such autograd.Functions.

The notebook `perexample_influence.ipynb` shows both implementations.
(Well, the inner product case is actually a bit different, but you get the point. You can make it the same with some effort.)
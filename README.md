https://github.com/google/jax
https://github.com/DistrictDataLabs/yellowbrick/tree/main

https://www.scikit-yb.org/en/latest/api/features/pca.html

https://www.scikit-yb.org/en/latest/api/classifier/index.html


first, did all the normal stuff

jaxified the kappa scorer, and made the scorer more continuous


compilation was really slow, so need remove a for loop and "jaxify" everything


part I
        cm = jnp.zeros((self.num_classes, self.num_classes))
        for i in range(len(y_pred)):
            x = y_true[i]
            y_p = y_pred[i]
            cm = cm.at[x, :].add(y_p)
        return cm


part II (but can't have dynamic size so didn't work)
        # y_true = jnp.expand_dims(y_true, axis=1)
        # cm = jnp.zeros((self.num_classes, self.num_classes))
        # for i in jnp.unique(y_true):
        #     cm = cm.at[i].set(y_pred[(y_true[:, 0] == i)].sum(axis=0))
        # return cm


part III
Note the "num classes" variable can't be set dynamically if the function is to be jit compiled because it leads to dynamic size arrays which is not allowed. 
    def confusion_matrix_continuous(self, y_true, y_pred):
        """
        A confusion matrix that support continuous class probabilities, i.e.
        the output of a softmax layer.
        It also outputs a continuous valued confusion matrix. Since it's part
        of our loss function, this gives us a continous loss instead of a
        discrete one.
        """

        def f(i, j):
            """
            for each probability column j in y_true, sum the probabilities of
            of getting the expected answer for each expected answer i
            """
            return jnp.where(y_true == i, y_pred[:, j], 0).sum()

        vecs = vmap(vmap(f, in_axes=(0, None)), in_axes=(None, 0), out_axes=1)(
            jnp.arange(self.num_classes),
            jnp.arange(self.num_classes),
        )
        return vecs

While before compilation could take as long as 40 minutes when using full data, it went down to less than a second



At first the perceptron was being beaten by logistic regression, which was odd because they are pretty similar. Looking closer I found I couldn't even overfit all the data very well on the model, so something seemed off. It turned out that was true: I was missing a bias term.

After adding I could overfit to .84 kappa on the entire dataset, but I should be able to overfit nearly perfectly, so I rejiggered the code to add another layer (and easily add even more). At first I tried relu as the hidden activation, but it didn't work. I switch to tanh, and things got better. Not sure why???

I could have switched to something more out of the box like haiku, but I wanted to refresh my memory on the inner workings of NNs and also learn Jax nuts and bolts better

Of course once overfitting, now we need to worry about regularization. Start by looking at a learning curve ![](2021-12-13-15-26-19.png)
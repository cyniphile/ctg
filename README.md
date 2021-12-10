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



  warnings.warn(label_encoder_deprecation_msg, UserWarning)
/Users/luke/projects/old/ctg/.venv/lib/python3.9/site-packages/xgboost/sklearn.py:1224: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
  warnings.warn(label_encoder_deprecation_msg, UserWarning)
[12:54:00] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.

[12:54:00] WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:576: 
Parameters: { "class_weight", "max_features" } might not be used.

  This could be a false alarm, with some parameters getting used by language bindings but
  then being mistakenly passed down to XGBoost core, or some parameter actually being used
  but getting flagged wrongly here. Please open an issue if you find any such cases.

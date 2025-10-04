---
date: 2025-10-04
authors:
  - sem_k
categories:
  - ML
  - classification
---

# On different interpretations of the class weighting and notion of unbalanced classes

*Class weighting* is often referred to as a simple and powerful technique to use when solving classification problems with *unbalanced classes*. But how can it be interpreted in a probabilistic sense? We are going to answer it using binary classification problem with the logistic regression model as an example. However, the more general question of the article is actually *why unbalanced classes is the issue* in the first place? Is it the issue by itself or is it a conseqence of some underlying property of the data? We will figure this out by the course of the narrative.

<!-- more -->

## Logistic regresion with unbalanced classes

Imagine we have binary classification problem. The *logistic regression* model defines probabilty of class $y = 1$ given object's features $x_i$ as

$$
    p(y = 1 | w, x) = \frac{1}{1 + \exp \big(-\langle w, x \rangle \big)} = \sigma \big( \langle w, x \rangle \big),
$$

where $ \langle w, x \rangle $ is the scalar product, $\sigma(\cdot)$ is the sigmoid function. The $w$ is the model's parameter vector. It defines the *separating* hyperplane in the space of $x$. The further you go from this hyperplane to one half-space or another the more probable label $1$ or label $0$ will be for a given object. The label's probabilities on the line are equal. This is illustrated on the figure below where
    $
    x = \begin{pmatrix}
		x_1 \\
		x_2 \\
	    \end{pmatrix} \in \mathbb{R}^2
    $ 
and 
    $
        w = \begin{pmatrix}
		w_1 \\
		w_2 \\
	        \end{pmatrix} = 
        \begin{pmatrix}
		0 \\
		1 \\
	        \end{pmatrix}
    $.
The color intensity is proportional to $ p(y = 1 | w, x) $.

```plotly
{"file_path": "posts/unbalanced_classes/graphs/log_regr.json"}
```

Imagine, we don't know true $w$. Assume we are given the following dataset

```plotly
{"file_path": "posts/unbalanced_classes/graphs/dataset.json"}
```

We can see that all objects are quite far from the (true) separation line. Their classes turned out to be the most probable under our model. We also see that the number of points in the upper plane is much less than in the lower. Consequently, we have the *dataset with unbalanced classes* - class $1$ is significantly under-represented relative to class $0$.

When we know true $w$, we understand that this situation does not arise from the initial model $p(y|w, x)$ by itself but rather from the positions of the objects in the dataset. Imagine the object were chosen closer to the separation line, then the dataset would be far more likely balanced because the classes' probabilities would be much closer to $0.5$.

But now we are in the situation where we don't know anything about true $w$ and how objects were chosen - they were simply given. Should we worry about unbalanced classes? To begin with, let's just find the most probable model for the dataset.

```plotly
{"file_path": "posts/unbalanced_classes/graphs/ml_model.json"}
```

We see our estimation is not very close to the truth, even though the sample size is enough to learn such a simple model. It seems that the unbalanced classes are the cause of the problem.

We can deal with the issue using technique called *class weights*. Before a full explanation, let's just see the magic of it.

```plotly
{"file_path": "posts/unbalanced_classes/graphs/ml_model_balanced.json"}
```

So in general, the technique adds custom class weight to every entry of the corresponding class in the loss function. Then the model is estimated using the modified loss. In the case of logistic regression, the regular loss is the negative log-likelihood:

$$
    L = -\sum_{i=1}^{N} \left[ y_i \log p(y=1|x_i, w) + (1 - y_i) \log p(y=0|x_i, w) \right].
$$

If we weight class 1 with $\omega_1 > 0$ and class 0 with $\omega_0 > 0$, the modified loss is:

$$
    L_{\text{cw}} = - \sum_{i=1}^{N} \left[ \omega_1 \cdot y_i \log p(y=1|x_i, w) + \omega_0 \cdot (1 - y_i) \log p(y=0|x_i, w) \right].
$$

Without loss of generality, we set $\omega_0 = 1$, $\omega_1 = \omega$. Then $\omega$ is the indicator of how many times class 1 is more "important" to us than class 0. Let's give several prespectives to interpret the change in the loss.

1. From the *optimisation* point of view, terms with higher weight now contribute greater to the function increase. Now the optimiser has to give higher probability to the higher weighted class to compensate for this increase through the $\log p$ term.
2. Assume $\omega \ge 2$ is a positive integer. Then, by multiplying objects' losses by $\omega$ we actually *increase the sample size* of the weighted class in $\omega$ times. Therefore, we directly address the issue of the unbalanced classes by balancing their frequencies in the dataset. If $\omega$ is any positive real, then the effect is essentially the same (note that if $\omega < 1$ then we increase the sample size of the counter-class).
3. Change of the loss always leads to the *change of the probability model* of the data. Let's rewrite the new loss function:

$$
    L_{\text{cw}} = - \sum_{i=1}^{N} \left[ y_i \log p(y=1|x_i, w) + \omega \cdot (1 - y_i) \log p(y=0|x_i, w) \right] = \\ = - \sum_{i=1}^{N} \left[ y_i \log \sigma (\langle w, x_i \rangle) + (1 - y_i) \log (1 - \sigma (\langle w, x_i \rangle))^{\omega} \right].
$$

Now the values under the logarithm do not sum up to one; let's norm them:

$$
    L_{\text{cw}} = - \sum_{i=1}^{N} [ y_i \log \frac{\sigma (\langle w, x_i \rangle)}{\sigma (\langle w, x_i \rangle) + (1 - \sigma (\langle w, x_i \rangle))^{\omega}} + \\ + (1 - y_i) \log \frac{(1 - \sigma (\langle w, x_i \rangle))^{\omega}}{\sigma (\langle w, x_i \rangle) + (1 - \sigma (\langle w, x_i \rangle))^{\omega}} + \\ + \log \big( \sigma (\langle w, x_i \rangle) + (1 - \sigma (\langle w, x_i \rangle))^{\omega} \big) ] \ge \\ \ge - \sum_{i=1}^{N} \left[ y_i \log \tilde{p}(y=1|x_i, w) + (1 - y_i) \log \tilde{p}(y=0|x_i, w) \right] = \tilde{L}.
$$

We denoted new class probabilities as $\tilde{p}$. When we normed the former probabilities we also had to add a new term $ \log \big( \sigma (\langle w, x \rangle) + (1 - \sigma (\langle w, x \rangle))^{\omega} \big) $. In our primary case of $\omega > 1$ this term is nonpositive so the last inequality is valid. What we got at the end is the negative log likelihood $\tilde{L}$ of the *new model* with the class probabilities proportional to $p(y=1|x, w)$ and $p(y=0|x, w)^{\omega}$. Hence $L_{\text{cw}}$ is the upper bound on $\tilde{L}$ and minimising $L_{\text{cw}}$ also gives us an estimation of the new model's optimal parameters. The new model is not logistic regression in general.

Overall, we can interpret the situation as follows: we had our initial logistic model but then someone decreased the probability of class $0$ and generated the labels with new probabilities.

The decrease of class 0 probability leads to the change of the separation line. We can find its new equation from

$$
    p(y=0|x, w)^{\omega} = p(y=1|x, w) \\
    \big( 1 - \sigma (\langle w, x \rangle ) \big)^{\omega} = \sigma (\langle w, x \rangle )
$$

This equation has a unique solution for $\langle w, x \rangle$ when $\omega > 0$ (but cannot be solved analytically in general). Therefore, the separation line will still be a line but shifted towards the less weighted class as we saw from the example above.

!!! info
    On practice, the common choice of the weights is to take them proportional to the inverse frequences of the classes. That's what `class_weight='balanced'` does in some of the sklearn's estimators.

## Do class weights always help with unbalanced classes?

Consider the following example under the previous model.

```plotly
{"file_path": "posts/unbalanced_classes/graphs/mixed_dataset.json"}
```

So several points of class 1 appeared in the lower half-plane which is possible under the logistic regression. The classes are still unbalanced. Now let's find maximum likelihood estimation (MLE) with and without class weights.

```plotly
{"file_path": "posts/unbalanced_classes/graphs/models_comparasion.json"}
```

So we see that using class weights doesn't do a good job here.

## Is class imbalance evil?

To be honest, we can always find examples of the dataset where class weights give better or worse estimation of $w$. The point of the demonstration is different - *choose any given method according to the one's assumptions and knowledge about the data*. If you are given the dataset with unbalanced classes and have no idea why they are unbalanced, just use MLE and rely on its asymptotic properties. Using class weights in this situation becomes a gamble. But if you have some prior knowledge about data generation, then its incorporation in the loss will be healthy. We have already figured out what it means in the case of the class weights. But the knowledge can be different. For example, along with the given data I could say that the $\| w \|$ is small and the second component is closer to zero more probably than the second. This knowledge could be easily transformed into regularising terms on $w$ in the loss function. Or I could say that objects $x_i$ were actually chosen randomly but with very high probability that their distance to the separation line is some given constant $d$ (that's what exactly happened in our demonstration). That is equal to the knowing $p(x) = p(x | w, d)$. Then the full log liklehood of the dataset would be 

$$
    \log p(y, X | w, d) = \log p(X | w, d) + \log p(y| X, w)
$$

and again we would come up with some regularisation of the inital loss.

## Conclusion

The issue of unbalanced classes does not exist by itself. Otherwise the balanced classes would be an issue too, wouldn't they? Any pattern in the data should be examined according to some expectations and prior knowledge of the field you're analyzing or anything else connected to the process of obtaining original data. Only then will the incorporation of associated techniques be justified and productive.

The notebook for the article is available at [![Python](https://img.shields.io/badge/Github-8A2BE2?logo=github)](https://github.com/sem-k32/semk-blog/blob/master/docs/posts/unbalanced_classes/unbalanced_classes.ipynb).

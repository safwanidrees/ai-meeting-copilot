# Machine learning notes

## Generative vs discriminative models

A discriminative model learns the decision boundary directly, modelling P(y | x): the probability of the label given the input. Examples: logistic regression, support vector machines, most neural network classifiers. They usually give better classification accuracy when there is plenty of labelled data.

A generative model learns how the data itself is distributed, modelling P(x | y) and P(y), or the joint P(x, y). Because it models the data, it can generate new samples and handle missing data. Examples: Naive Bayes, Gaussian Mixture Models, Hidden Markov Models, GANs, variational autoencoders, and large language models. They can work better with little labelled data, but they make stronger assumptions and are often harder to train.

Real-world example: for spam filtering with very little data, Naive Bayes (generative) trains quickly and works well; with millions of labelled emails, a discriminative model such as logistic regression or a fine-tuned transformer usually wins.

## Bias-variance trade-off

Bias is error from wrong or overly simple assumptions, which causes underfitting. Variance is error from being too sensitive to the training data, which causes overfitting. Increasing model complexity lowers bias but raises variance; the goal is the complexity that minimises total error on unseen data.

## Overfitting and how to prevent it

Overfitting means the model memorises the training data and performs poorly on new data: training error is low but validation error is high. Remedies: more training data, simpler models, regularization (L1 encourages sparse weights, L2 shrinks weights), dropout, early stopping, data augmentation, and cross-validation to detect it.

## Evaluation metrics

Precision = TP / (TP + FP): of the items predicted positive, how many really are. Recall = TP / (TP + FN): of the real positives, how many were found. F1 is the harmonic mean of precision and recall. Accuracy is misleading on imbalanced data, for example fraud detection where 99 percent of transactions are legitimate. ROC-AUC measures ranking quality across all thresholds.

## Gradient descent

Gradient descent minimises a loss function by repeatedly moving the parameters a small step in the direction of the negative gradient. The learning rate controls the step size: too large and training diverges, too small and it is slow. Stochastic gradient descent uses one example or a mini-batch per step, which is faster and adds noise that can help escape poor local minima. Adam adapts the learning rate per parameter using running averages of the gradients.

## Cross-validation

K-fold cross-validation splits the data into k parts, trains on k minus 1 parts and validates on the remaining part, rotating k times and averaging the scores. It gives a more reliable estimate of performance than a single train-test split and helps with hyperparameter tuning.

## Embeddings and vector search

An embedding maps text to a vector so that texts with similar meaning are close together, usually measured with cosine similarity. Vector databases such as Qdrant store these vectors and find nearest neighbours quickly with approximate indexes such as HNSW.

## Retrieval augmented generation (RAG)

RAG retrieves relevant passages from a document collection and gives them to a language model together with the question, so answers are grounded in up-to-date or private data and hallucinations are reduced. Hybrid search combines keyword search (BM25), which is strong on exact terms such as names and codes, with vector search, which is strong on meaning and synonyms. Reranking with a cross-encoder then orders the best candidates more precisely.

## Transformers and attention

Transformers process all tokens in parallel using self-attention: every token computes how much to attend to every other token using queries, keys and values. Multi-head attention learns several kinds of relationships at once. Positional encodings add word order. Transformers replaced RNNs because they train in parallel and handle long-range dependencies better.

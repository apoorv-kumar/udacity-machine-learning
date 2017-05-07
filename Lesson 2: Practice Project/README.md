# Project

Resources from [UCI ML repo](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

Solution [here](https://github.com/udacity/machine-learning/blob/master/projects/practice_projects/naive_bayes_tutorial/Naive_Bayes_tutorial.ipynb)

Keep the data in 'resources' directory under project root.


# Notes:

## Simple bayesian

### Maximum A posteriori (MAP) hypothesis

In Bayesian statistics, a maximum a posteriori probability (MAP) estimate is an estimate of an unknown quantity, that equals the mode of the posterior distribution. The MAP can be used to obtain a point estimate of an unobserved quantity on the basis of empirical data. It is closely related to Fisher's method of maximum likelihood (ML) estimation, but employs an augmented optimization objective which incorporates a prior distribution (that quantifies the additional information available through prior knowledge of a related event) over the quantity one wants to estimate. MAP estimation can therefore be seen as a regularization of ML estimation.

### Mode

The mode is the value that appears most often in a set of data. The mode of a discrete probability distribution is the value x at which its probability mass function takes its maximum value.

### Posterior probability

The posterior probability is the probability of the parameters θ given the evidence X : p ( θ | X ) 
It contrasts with the likelihood function, which is the probability of the evidence given the parameters: p ( X | θ )


> P(h/D)= P(D/h) P(h) / P(D) 

D : 35 year old customer with an income of $50,000 PA
h : Hypothesis that our customer will buy our computer
 
P(h/D) : Probability that customer D will buy our computer given that we know his age and income 
P(h) : Probability that any customer will buy our computer regardless of age (Prior Probability) 
P(D/h) : Probability that the customer is 35 yrs old and earns $50,000, given that he has bought our computer (Posterior Probability) 
P(D) : Probability that a person from our set of customers  is 35 yrs old and earns $50,000 


### Naïve Bayesian Classification

It is based on the Bayesian theorem It is particularly suited when the dimensionality of the inputs is high. Parameter estimation for naive Bayes models uses the method of maximum likelihood. In spite over-simplified assumptions, it often performs better in many complex real-world situations 

Advantage: Small training data required.

On the flip side, although naive Bayes is known as a decent classifier, it is known to be a bad estimator.

### Vectorization
#### Vector Space Model
Vector -> Array of floats -> Represents data

A way to vectorize text- 
> word: str -> frequency: int

Stages in vectorization:
* Sentence segmentation
* Tokenization
* Lemmatization : finding base/stem of words
* Remove stop words
* Vectorization

Typically our array size would be << corpus size, so we need a 'vectorization strategy'. Strategies ...

#### Bag of words
Document is represented as multiset of its words
Simplest vector model and it can create a lot of columns

#### TF IDF
Fixes some issues with BoW.
Some words appear a lot in general - IDF compensates for that.

w[t] = TF[t]*IDF[t]
        |     |
        |     - log(N/DF[t]) where N: total docs, DF[t]: docs containing t
        |
        --- freq of t/words in doc

#### ngrams
Groups of n words in a seq. Bigram, trigram ...
But this leads to explosion of n-grams. We can use log-likelihood test which can determine whether 2 words occur together by chance because they form a significant unit.

### Scikit Learn
#### fit_transform
fit: feature extraction. For count vectorizer, it'd be a tokenized words.
transform: generates a vector per feature (token) giving a metric (for count vectorizer, the frequency of feature) for each string.
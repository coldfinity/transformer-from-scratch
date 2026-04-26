# Architecture
![Model Architecture](model_architecture.png)

# Input Embedding
Convert original sentence into vector of size 512

Original Sentence (Tokens) -> Input IDs (Position in Vocab) -> Embedding (Vector)

In the embedding layers, multiply the weights by $\sqrt{d_{model}}$

## Positional Encoding
Adding another vector of size 512 which stores the relative or absolute position of the tokens in the sequence.

Formula for PE
$$
\begin{align}
\text{Even Positions} \; PE_{pos,2i} = \sin (pos/10000^{2i/d_{model}}) \\
\text{Odd Positions} \; PE_{pos,2i+1} = \cos (pos/10000^{2i/d_{model}})
\end{align}
$$
where $pos$ is the position and $i$ is the dimension

Dropout for model to avoid overfitting

# Encoder
## Layer Normalisation (Add & Norm)
Batch of $n$ items, each item is made up of words (i.e sentence)

Calculate independent mean $(\mu_n)$ and variance $(\sigma^2_n)$ for each item in the batch. 

New value: $$\hat{x}_j = \frac{x_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}$$
Introduce two new parameters, gamma $(\gamma)$ for multiplicative and beta $(\beta)$ for additive.
Allows the network to be amplified using these values, the network will learn to tune these two parameters to introduce
fluctuations when necessary.

## Feed Forward
Let $W_1, W_2$ be matrices and $b_1, b_2$ be bais.
We can create two linear transformations with a ReLu activation in between
$$
FFN(x) = \max(0,xW_1 + b_1)W_2 + b_2
$$
This function is applied to the encoder and decoder, they use different parameters from layer to layer.

## Multi-Head Attention
Takes the input of the encoder applied 3 times (query, key, values)

Transform into matricies $Q, K, V$, 
$$
\begin{aligned}
Q \times W^Q &= Q' \rightarrow [Q_1 \mid Q_2 \mid Q_3 \mid Q_4 \mid \cdots] \\
K \times W^K &= K' \rightarrow [K_1 \mid K_2 \mid K_3 \mid K_4 \mid \cdots] \\
V \times W^V &= V' \rightarrow [V_1 \mid V_2 \mid V_3 \mid V_4 \mid \cdots]
\end{aligned}
$$
Each volumn of the resulting matrix contains linear cobinatinos of all elements from the original embedding through matrix multiplication. When splitting along $d_model$ dimension, each head recieves linear combination of the complete word embedding through the distributive property of matrix multiplication.

Then apply _Attention_ to each of the smaller matrix
$$
\begin{aligned}
\text{Attention}(Q, K, V) &= \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\
\text{head}_i &= \text{Attention}(QW_i^Q,\, KW_i^K,\, VW_i^V)
\end{aligned}
$$
Concatonate all $head_i$
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1 \ldots \text{head}_h)W^O
$$

$d_k = d_{model} / h$

## Combining
Repeated $N$ times, output of the previous is sent to the next one and the output of the last one is sent to the decoder.
Containing one `MultiHeadAttention`, two `Add&Norm` and one `FeedForward`. 

# Decoder
Repeated $N$ times, `Masked Multi-Head Attention` from decoder, `MultiHeadAttention` is a cross attention  where $Q, K$ is from encoder and $V$ is from decoder, one `FeedForward` and 3 `Add&Norm`.
Two masks because the self-attention uses the target mast and the cross attention uses the source mask from the encoder. 

### Sources
Source: "Attention Is All You Need", Google, https://arxiv.org/pdf/1706.03762
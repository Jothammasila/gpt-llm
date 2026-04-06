# gpt-llm

# **Attention Mechanism With Trainable Weights**

> In self-attention mechanisms, the goal is to determine the importance of each element in a sequence relative to all other elements in that sequence. This importance is captured by attention scores (also called "weights"), which can be used to generate a new representation of the sequence where each element is context-aware.



$$\text{Attention}(Q, K,V) = {softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Think of attention as answering the question:

`“For this word, what should I look at, and what information should I take from what I look at?”`

That’s exactly what $Q, K, V$ are for.

> **Query = question being asked by the word**

**Key $(K)$: “What do I contain?”**

Each word also advertises what kind of information it has.

The key vector describes what this word can offer.

Queries are matched against keys.

So when a query meets a key:

```
High dot product → “Yes, this word has what I’m looking for”
Low dot product → “Not relevant”
```
>**Key = label / description of what a word represents.**


**Value $(V)$: “Here is the information itself”**

Once relevance is decided, we extract the actual content.

The value vector is the information to be passed forward.

Attention weights decide how much of each value to take.



**Value = information payload**

*NB:*
```
> seq_len ---> refers to number of tokens provided.

> batch_size ---> the number of examples in each batch.
A sequence may have a length of 128,
and a batch may have a size of 8 such examples such that the output will be (8, 128)

> d_model ---> the dimension of each token in seq_len after embedding e.g 256;so the output of such a matrix is (seq_len, d_model).
Combining with batch_size idea, the final output will be (batch_size, seq_len, d_model) = (8,128,256) for our example

> max_len ---> is the fixed maximum number of tokens a model is allowed to handle in a sequence. It is not the same as seq_len.
Neural networks work with fixed-size tensors, but: sentences have variable lengths

So we choose a maximum length and force all sequences to conform to it. That fixed length is max_len.

seq_len → actual number of tokens provided.
max_len → allowed capacity

If a sentence is shorter than max_len → pad it
If it’s longer → truncate it.

max_len is what you design the model to handle as the maximum number of tokens.
seq_len is what is actually provided by the user for a given input.


During Model design time
You choose:
    max_len = 512   # or 1024, 2048, etc...

        This Affects:
        ~positional encodings
        ~attention matrix size (max_len × max_len)
        ~computational cost

Runtime / user input
User provides text:
    "I love transformers"

  Tokenized:
    seq_len = 3

  Then:

    if seq_len < max_len → pad

    if seq_len > max_len → truncate or reject


EXAMPLE:

Model design:
  max_len = 128
  d_model = 256

User input:
  seq_len = 87

Actual tensor fed into the model:
  (batch_size, max_len, d_model)
  (1, 128, 256)

But:

->only the first 87 positions are real
->the rest are padding
->attention mask ensures padding is ignored


Important nuance (worth knowing)
    ->max_len is a hard architectural limit unless:
    ->you use relative positional encodings, or
    ->you extend positional embeddings after training
In classic transformers, exceeding max_len is not allowed.
```
---

## **Applying a causal attention mask**

> A causal attention mask enforces the arrow of time, ensuring each token only looks backward, never forward.


> Applying a causal attention mask means preventing a token from attending to future tokens in the sequence.

> It's essential in autoregressive models (language models, decoders).

> ## Why do we need a causal mask?

>In tasks like next-token prediction, when the model is predicting token at position `t`, it must not see: tokens at positions` t+1, t+2, …`

>Otherwise, the model would be cheating by looking ahead.

>So we enforce this rule:` A token may only attend to itself and earlier tokens.`


>## Where the mask is applied?

>The mask is applied ***before*** softmax, on the ***attention scores***:
$$\text{scores}=\frac{QK^T}{\sqrt{d_k}}$$

>We modify the scores so that future positions get $-\infty$


>## What the causal mask looks like

`For seq_len = 4, the causal mask is:`


$$\begin{bmatrix} 0 & -\infty & -\infty & -\infty
\\0 & 0 & -\infty & -\infty\\
0 & 0 & 0 & -\infty
\\0 & 0 & 0& 0
\end{bmatrix}
$$
Rows → queries (current token)

Columns → keys (tokens being attended to)

---

**Multi-Head Reshaping and Dimensions**

> The outputs $(Q, K, V)$ are initially in the shape:
> $$(batch,\ seq\_len,\ d_{model})$$

> The model dimension is defined as:
> $$d_{model} = num\_heads \times head\_dim$$

> Therefore:
> $$head\_dim = \frac{d_{model}}{num\_heads}$$

> To enable multi-head attention, we reshape the last dimension:
> $$(batch,\ seq\_len,\ d\_model) \rightarrow (batch,\ seq\_len,\ num\_heads,\ head\_dim)$$

> This operation does **NOT** change the data, but reorganizes it so that each token is now represented across multiple heads.

> Next, we transpose the tensor to rearrange the dimensions:
> $$(batch,\ seq\_len,\ num\_heads,\ head\_dim) \rightarrow (batch,\ num_heads,\ seq\_len,\ head\_dim)$$

> This step is crucial because it allows each head to process the entire sequence independently during attention computation.

> In summary:
>
> * `.view()` splits the embedding dimension into multiple heads
> * `.transpose()` reorders dimensions to enable parallel attention across heads
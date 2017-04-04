# contextual-cnn-tf

A CNN for topic classification based on "Neural Contextual Conversation Learning with Labeled Question-Answering Pairs", https://arxiv.org/pdf/1607.05809v1.pdf, section 3.1

The implementation is based on wildml's http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
The most important addition is the k-max pooling layer, applied naively.

The CNN is comprised of an embedding layer, followed by a convolutional, kmax-pooling, convolutional, maxpooling and softmax layer and produces a probability vector that represents how likely it is for a question to pertain to each topic.

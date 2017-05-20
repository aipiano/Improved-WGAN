# Improved-WGAN
A Tensorflow implementation of weight-penalty-based WGAN.

Use a modified(remove batch_norm layers in the critic) DCGAN model.

Use weights penality instead of weights clipping when training.


## Result
Some anime face results after training 20k iterations only.

![results](https://github.com/aipiano/Improved-WGAN/blob/master/images/examples.jpg)


## References
- [Towards Principled Methods for Training Generative Adversarial Networks](https://arxiv.org/abs/1701.04862)

- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

- [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)

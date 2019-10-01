## net

This is a project to write a simple library capable of building feedforward neural networks using Python and numpy.

I first started building something like this on top of PyTorch but realised if I was make it a learning experience, I should build at a lower abstraction layer. So here it is, with numpy as my base

For now, with the time I'll be able to dedicate to this, these are my outcomes for the next 10 weeks:

- [ ] Feedforward layers
- [ ] Different activation and loss functions
- [ ] Weight initialization(Xavier and Kaiming Init)
- [ ] Dropout and L1, L2 regularization
- [ ] Backpropagation
- [ ] Training loop with **customizations** (More on this later.)

Above, when I wrote customizations, I basically mean callbacks. There's a ton of interesting things to explore when you think about callbacks during training(notifications, model saving, CI/CD triggers etc). This is where I imagine I'll be spending a lot of time after my base goals have been achieved.


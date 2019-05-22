"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        #conv_output, x = self.forward_pass_on_convolutions(x)
        kl = 0
        conv_counter = 0
        conv_output = None
        for layer in self.model.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                if self.target_layer == conv_counter:
                    x.register_hook(self.save_gradient)
                    conv_output = x

                kl += _kl
                conv_counter +=1

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        #logits = x

        #x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        #x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, plots_path, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        print(conv_output.shape)
        #convoutput = conv_output.detach().numpy()
        #plt.imshow(convoutput[0,:,:,:])
        #plt.savefig(plots_path + 'testViz.png', format='png', dpi=300)
        #plt.close()

        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        #self.model.features.zero_grad()
        #self.model.classifier.zero_grad()
        #optimizer = optim.Adam(net.parameters(), lr=cf.dynamic_lr(cf.lr, epoch), weight_decay=cf.weight_decay)
        #optimizer.zero_grad()
        for layer in self.model.layers:
            layer.zero_grad()
        # Backward pass with specified target
        print(model_output)
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        print(model_output)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]

        # Get convolution outputs
        target = conv_output.data.numpy()[0]

        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient

        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices, however, when I moved the repository to PIL, this
        # option is out of the window. So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, send a PR.
        return cam

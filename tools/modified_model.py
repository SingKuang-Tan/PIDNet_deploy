
import torch
import torch.nn as nn
import numpy as np

# Function to extract the output of a specific layer
def extract_layer_output(model, x, target_layer_name1,target_layer_name2,target_layer_name3):
    outputs = {}
    hooks = {}

    def hook_fn1(module, input, output):
        outputs[target_layer_name1] = output

    def hook_fn2(module, input, output):
        outputs[target_layer_name2] = output

    def hook_fn3(module, input, output):
        outputs[target_layer_name3] = output

    for name, layer in model._modules.items():
        if name == target_layer_name1:
            hooks[name] = layer.register_forward_hook(hook_fn1)
        if name == target_layer_name2:
            hooks[name] = layer.register_forward_hook(hook_fn2)
        if name == target_layer_name3:
            hooks[name] = layer.register_forward_hook(hook_fn3)

    model(x)
    
    for name, hook in hooks.items():
        hook.remove()

    return [outputs[target_layer_name1],outputs[target_layer_name2],outputs[target_layer_name3]]

class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.additional_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        #for name, layer in self.original_model._modules.items():
        #    print(name)

        #for name, module in self.original_model.named_modules():
            #print(f"Layer Name: {name}, Module Type: {module.__class__.__name__}")
        #    print(list(self.original_model.children())[2])
        #    exit(0)

        

        # Name of the layer you want to extract output from
        target_layer_name = 'dfm'  # Example: the third convolutional layer
        output = extract_layer_output(self.original_model, x, 'dfm','seghead_p','seghead_d')

        
        #print(self.original_model)

        #x = self.original_model(x)
        #print(np.shape(x[0]))
        #print(np.shape(x[1]))
        #print(np.shape(x[2]))

        #exit(0)
        #print(np.shape(output))
        x2 = self.additional_conv(output[0])
        #print(np.shape(x))

        #y1=self.original_model.seghead_p(x)
        #y2=self.original_model.seghead_d(x)
        y3=self.original_model.final_layer(x2)
        #x = x.view(x.size(0), -1)
        #x = self.original_model.fc(x)
        return [output[1],y3,output[2]]

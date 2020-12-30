import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import adaptive_instance_normalization as adain_layer
from utils import calc_mean_std

class RC(nn.Module):
    """
    A wrapper class for ReflectionPad2d, Conv2d and an optional relu
    """
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, activation_function=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((padding, padding, padding, padding))
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return F.relu(x) if self.activation_function else x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)

    def forward(self, x):

        x = self.rc1(x)
        x = self.upsample1(x)
        x = self.rc2(x)
        x = self.rc3(x)
        x = self.rc4(x)
        x = self.rc5(x)
        x = self.upsample2(x)
        x = self.rc6(x)
        x = self.rc7(x)
        x = self.upsample3(x)
        x = self.rc8(x)
        x = self.rc9(x)

        return x

# Encoder for VGG19
class VGGEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(VGGEncoder, self).__init__()

        # Load the pretrained model
        vgg = models.vgg19(pretrained=pretrained)
        layers = list(vgg.features.children())

        # Use up to relu4_1 to encode
        self.enc_1 = nn.Sequential(*layers[:2])  # conv1_1 to reLu1_1
        self.enc_2 = nn.Sequential(*layers[2:7])  # conv1_2 to 'relu2_1
        self.enc_3 = nn.Sequential(*layers[7:12])  # conv2_2 to 'relu3_1
        self.enc_4 = nn.Sequential(*layers[12:21])  # conv3_2 to 'relu4_1

    def forward(self, x):
        out_1 = self.enc_1(x)
        out_2 = self.enc_2(out_1)
        out_3 = self.enc_3(out_2)
        out_4 = self.enc_4(out_3)

        return out_1, out_2, out_3, out_4


# AdaIN module
class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

        self.encoder = VGGEncoder()
        self.decoder = Decoder()
        self.mse_loss = nn.MSELoss()

        # Freeze all the layers in the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, content, style, alpha=1.0, interpolation_weights=None):
        assert 0 <= alpha <= 1, '"alpha" should be between 0 and 1.'
        assert interpolation_weights is None or not self.training, 'Interpolation is available only in testing.'

        # Obtain the features from the content image and the style image
        f_content = self.encoder(content)[-1]
        f_style = self.encoder(style)

        if interpolation_weights is not None:
            assert not self.training, 'Interpolation is available only in testing.'
            # Combine the features of style images with interpolation weights
            t = adain_layer(f_content.expand_as(f_style[-1]), f_style[-1])
            og_shape = t.shape
            t = torch.reshape(t, (t.shape[0], -1))
            interpolation_weights = interpolation_weights.unsqueeze(1).expand_as(t)
            t = torch.reshape(t * interpolation_weights, og_shape)
            t = torch.sum(t, dim=0, keepdim=True)
        else:
            t = adain_layer(f_content, f_style[-1])  # Adaptive Normalization

        # Adjust the degree of style transformation
        t = (1 - alpha) * f_content + alpha * t

        # Decode the image
        output_img = self.decoder(t)

        # return image if not in training mode
        if not self.training:
            print("testing")
            return output_img

        # Get the features of the output
        f_output = self.encoder(output_img)

        # calculate content loss, which is the Euclidean distance between
        # the content features (AdaIN output t) and the features of the output image
        loss_content = self.calculate_content_loss(f_output[-1], t)

        # calculate style loss
        loss_style = self.calculate_style_loss(f_output, f_style)

        return output_img, loss_content, loss_style

    # Use the features of the AdaIN output and the features of the original content
    def calculate_content_loss(self, output_features, content_features):

        return self.mse_loss(output_features, content_features)

    def calculate_style_loss(self, f_content, f_style):
        loss_style = 0.0
        for c, s in zip(f_content, f_style):
            c_mean, c_std = calc_mean_std(c)
            s_mean, s_std = calc_mean_std(s)
            loss_style += self.mse_loss(c_mean, s_mean) + self.mse_loss(c_std, s_std)
        return loss_style


# Method to save the model
def save_AdaIn(model, path='models/AdaIN_model.pth'):
    state_dict = {}
    # Get the states of decoder
    decoder_dict = model.decoder.state_dict()
    for key in decoder_dict.keys():
        decoder_dict[key] = decoder_dict[key].to(torch.device('cpu'))
    state_dict['decoder'] = decoder_dict
    # Save the model
    torch.save(state_dict, path)


# Method to load the model
def load_AdaIN(path='models/AdaIN_model.pth'):
    # Load the states
    state_dict = torch.load(path)
    model = AdaIN()
    # Load the states of decoder
    model.decoder.load_state_dict(state_dict['decoder'])
    # Return loaded model
    return model
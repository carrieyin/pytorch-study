from torch import nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_x, target_x):
        outputs = self.encoder(input_x)
        enc_hidden = self.decoder.init_state(outputs)
        return self.decoder(target_x, enc_hidden)
        

import sys
import torch

checpoint_path = sys.argv[1]
model = torch.load(checpoint_path)
for i in range(len(model.decoder.layers)):
    print("decoder layer {} , self attn alphas : {}".format(i, model.decoder.layers[i].self_attn.alphas))
    print("decoder layer {} , encoder attn alphas : {}".format(i, model.decoder.layers[i].encoder_attn.alphas))
    print("encoder layer {} , self attn alphas : {}".format(i, model.encoder.layers[i].self_attn.alphas))

import torch
from classifiers.xnor_classifier import *
from classifiers.dorefa_classifier import *
from classifiers.bnn_classifier import *
from config import FLAGS
from models import *
import pandas as pd
import numpy as np
from ensemble import *
from utils import build_ensemble

from matplotlib import pyplot
from tqdm import tqdm
from torch import no_grad

cuda = torch.cuda.is_available() and not (FLAGS.no_cuda)
device = torch.device("cuda" if cuda else "cpu")
torch.manual_seed(0)
if cuda:
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(0)

print(type(FLAGS.ensemble[0].model.model_type))
print(type(FLAGS.ensemble))
ensemble = build_ensemble(FLAGS.ensemble)
for k, v in ensemble.items():
    # print(k)
    # print(v)
    print(type(v[0]))

# model_1 = eval(FLAGS.ensemble[0].model.model_type)()
# print(type(model_1))
# model_2 = eval(FLAGS.model_2)()
# model_1.to(device)
# model_2.to(device)

# # create the ensemble
# ensemble = Ensemble(
#     FLAGS.bin_type_1,
#     FLAGS.bin_type_2,
#     FLAGS.dataset,
#     FLAGS.batch_size_1,
#     FLAGS.batch_size_2,
#     FLAGS.test_batch_size,
#     model_1,
#     model_2,
#     device,
#     FLAGS.optimizer,
#     FLAGS.lr,
#     FLAGS.steps,
#     FLAGS.gamma,
# )
# # train
# preds_0 = ensemble.train(FLAGS.checkpoint, FLAGS.epochs_1, FLAGS.epochs_2)
# # test
# ensemble.test(preds_0)

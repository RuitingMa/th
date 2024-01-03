import torch
from config import FLAGS
from ensemble_algorithms.bagging import Bagging

from utils import build_ensemble
from ensemble_algorithms.algorithm_abc import AlgorithmABC

cuda = torch.cuda.is_available() and not (FLAGS.no_cuda)
device = torch.device("cuda" if cuda else "cpu")
torch.manual_seed(0)
if cuda:
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(0)

# create the ensemble
ensemble = build_ensemble(
    FLAGS.ensemble,
    device,
    FLAGS.dataset,
    cuda,
    FLAGS.download_data,
)

algorithm = Bagging(ensemble=ensemble)
algorithm.train()
algorithm.test()


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

import torch
from config import FLAGS

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

algorithm = AlgorithmABC.from_config(FLAGS.ensemble_algorithm, ensemble)
algorithm.train()
algorithm.test(FLAGS.dataset, cuda, FLAGS.download_data, 100)

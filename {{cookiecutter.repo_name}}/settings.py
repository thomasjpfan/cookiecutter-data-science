from model_linear import exp as linear_exp
from model_simple_nn import exp as simple_nn_exp
from model_text import exp as text_exp
from process import get_train, get_test

EXPERIMENTS = {
    "linear": linear_exp,
    "simple_nn": simple_nn_exp,
    "text": text_exp,
}
PROCESS_FUNCS = {
    "files__proc_train": get_train,
    "files__proc_test": get_test,
}

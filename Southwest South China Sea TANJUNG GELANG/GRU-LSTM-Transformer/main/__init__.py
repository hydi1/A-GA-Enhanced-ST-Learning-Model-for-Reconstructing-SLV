# 将这段代码写入 models/__init__.py
from . import GRU, LSTM, Transformer


def Model(configs):
    model_name = configs['model']  # 获取 args 里的 model 字段

    if model_name == 'LSTM':
        return LSTM.Model(configs)
    elif model_name == 'GRU':
        return GRU.Model(configs)
    elif model_name == 'Transformer':
        return Transformer.Model(configs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
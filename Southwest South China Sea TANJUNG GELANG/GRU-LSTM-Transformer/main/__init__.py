from . import GRU, LSTM, Transformer

def Model(configs):
    model_name = configs['model']

    if model_name == 'LSTM':
        return LSTM.Model(configs)
    elif model_name == 'GRU':
        return GRU.Model(configs)
    elif model_name == 'Transformer':
        return Transformer.Model(configs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

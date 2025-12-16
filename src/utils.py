from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision, use_auth_token=True
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")
    
def count_parameters(model,train_param=False):
    if train_param:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        
    if total_params >= 1e9:
        return f"{total_params / 1e9:.2f}B"
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.2f}M"
    else:
        return str(total_params)
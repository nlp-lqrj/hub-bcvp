import torch
from torch.optim import Adam, SGD
from transformers import BertForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType


def setup_rola_model(model_path, num_labels, lora_config=None):
    # 加载基础模型
    model = BertForTokenClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    if lora_config is None:
        lora_config = set_default_lora_config()

    # 应用LoRA配置
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def set_default_lora_config():
    """默认LoRA配置"""
    return LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=8,  # LoRA秩
        lora_alpha=32,
        lora_dropout=0.1,
        # target_modules=["query", "key", "value"],
        target_modules=["query", "key", "value", "intermediate.dense", "output.dense"],
        bias="none",
    )


def reset_requires_grad(model):
    # lora配置会冻结原始模型中的所有层的权重，不允许其反传梯度
    # 但是事实上我们希望最后一个线性层照常训练，只是bert部分被冻结，所以需要手动设置
    for param in model.get_submodule("model").get_submodule("classifier").parameters():
        param.requires_grad = True


def save_tunable_params(model, model_params_path):
    saved_params = {
        k: v.to("cpu") for k, v in model.named_parameters() if v.requires_grad
    }
    torch.save(saved_params, model_params_path)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

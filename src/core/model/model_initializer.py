import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # type: ignore
from peft import PeftModel # type: ignore
from typing import Optional, Tuple, Union
import os

def initialize_model_and_tokenizer(
    model_path: str = "google/gemma-2-9b",
    cache_dir: str = "/root/autodl-tmp/gemma",
    lora_path: Optional[str] = None,
    use_quantization: bool = True,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    max_memory: Optional[dict] = None,
    max_length: int = 4096,
) -> Tuple[Union[AutoModelForCausalLM, PeftModel], AutoTokenizer]:
    """
    初始化模型和分词器。
    
    Args:
        model_path (str): 模型的路径或huggingface模型名称
        cache_dir (str): 模型缓存目录
        lora_path (Optional[str]): LoRA权重的路径，如果为None则返回基础模型
        use_quantization (bool): 是否使用4bit量化
        device_map (str): 设备映射策略
        torch_dtype (torch.dtype): 模型的torch数据类型
        trust_remote_code (bool): 是否信任远程代码
        local_files_only (bool): 是否只使用本地文件
        max_memory (Optional[dict]): 每个设备的最大内存使用量
        max_length (int): 模型处理的最大序列长度
    
    Returns:
        Tuple[Union[AutoModelForCausalLM, PeftModel], AutoTokenizer]: 返回(模型, 分词器)的元组
    """
    # 设置CUDA内存分配器
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 如果没有指定max_memory，创建默认配置
    if max_memory is None:
        max_memory = {
            0: "12GiB",  # 限制GPU显存使用
            "cpu": "24GiB"  # 允许CPU内存使用
        }
    
    # 创建tokenizer并设置最大长度
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        model_max_length=max_length
    )
    
    # 配置量化参数
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        max_memory=max_memory
    )
    
    # 如果提供了LoRA路径，加载LoRA权重
    if lora_path:
        model = PeftModel.from_pretrained(
            base_model,
            lora_path
        )
        return model, tokenizer
    
    return base_model, tokenizer

# 使用示例
def get_default_model_and_tokenizer():
    """
    获取默认配置的模型和分词器
    """
    # 设置较小的最大长度和内存限制
    max_memory = {
        0: "10GiB",  # GPU显存限制
        "cpu": "20GiB"  # CPU内存限制
    }
    
    return initialize_model_and_tokenizer(
        max_length=2048,  # 设置较小的最大长度
        max_memory=max_memory,
        use_quantization=True  # 确保使用量化
    )



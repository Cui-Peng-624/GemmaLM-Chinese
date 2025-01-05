我们的任务目标是在中文数据集上微调Gemma-2-9b模型，具体请参考：[Gemma-language-tuning](https://www.kaggle.com/competitions/gemma-language-tuning)

# 1. chat template

``` python
SYSTEM_TEMPLATE = "<start_of_turn>system\n{context}\n<end_of_turn><eos>\n"
USER_TEMPLATE = "<start_of_turn>user\n{prompt}\n<end_of_turn><eos>\n"
MODEL_TEMPLATE = "<start_of_turn>model\n{response}\n<end_of_turn><eos>\n"
```

对话格式：

``` python
from model_utils import apply_chat_template, generate_response

message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "model", "content": "Hello! How can I assist you today?"}
]

message_str = apply_chat_template(message)

response = generate_response(model, tokenizer, message_str)
```

具体请参考：[model_utils.py](https://github.com/Cui-Peng-624/GemmaLM-Chinese/tree/main/src/core/utils/model_utils.py)

# 2. 数据集构建

### 构建 system prompt          
 
对于系统提示词的构建，我们采取三种策略：     
- 不适用系统提示词      
- 使用固定的系统提示词 - 15个固定通用的系统提示词       
- 使用动态的系统提示词 - 根据数据的具体情况，与 GPT 进行交互，异步生成特定的系统提示词       

比例为：0.2 : 0.4 : 0.4            

我们的微调分为三个stage：

### 1. stage1（基础阶段）：
混合 [belle](https://huggingface.co/datasets/BelleGroup/train_1M_CN) 和 [alpaca](https://huggingface.co/datasets/silk-road/alpaca-data-gpt4-chinese) 数据集，这两个数据集都是 instruction 指令数据集，比例为 1 : 1。训练集数据量8w条，验证集数据量1000条。

### 2. stage2（特定任务阶段）：
混合 [news_commentary](https://huggingface.co/datasets/Helsinki-NLP/news_commentary) 翻译数据集，[HC3-Chines](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese) QA数据集，[firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) 故事创作数据集，比例为：0.367 : 0.3853 : 0.2476。再将 stage2 得到的数据与 stage1 的数据进行混合，混合比例为 0.7 : 0.3，得到最终的训练集的数据量为8w条，验证集的数据量1000条。在最终得到的数据中，训练集的数据类型分布如下:
- dialogue: 6030 条 (7.54%)
- translation: 9178 条 (11.47%)
- instruction: 56000 条 (70.00%)
- story_generation: 8792 条 (10.99%)

验证集的数据类型分布如下:
- instruction: 700 条 (70.00%)
- translation: 124 条 (12.40%)
- story_generation: 111 条 (11.10%)
- dialogue: 65 条 (6.50%)

### 3. stage3（专业领域阶段）：
混合 [chinese_modern_poetry](https://huggingface.co/datasets/Iess/chinese_modern_poetry) 中国现代诗词数据集，[chinese_poetry](https://huggingface.co/datasets/ddnoodle/chinese_poetry) 中国古代诗词数据集，[HistoryTrans/Dataset](https://huggingface.co/datasets/HistoryTrans/Dataset) 文言文数据集，比例为：1 : 1 : 1。再将 stage3 得到的数据与 stage2 和 stage1 的数据进行混合，混合比例为 0.5 : 0.3 : 0.2，得到最终的训练集的数据量为8w条，验证集的数据量1000条。在最终得到的数据中，训练集的数据类型分布如下:
- instruction: 40000 条 (50.00%)
- ancient_poetry_creation: 5348 条 (6.69%)
- translation: 9219 条 (11.52%)
- classical_translation: 5342 条 (6.68%)
- story_generation: 8861 条 (11.08%)
- modern_poetry_creation: 5310 条 (6.64%)
- dialogue: 5920 条 (7.40%)

验证集的数据类型分布如下:
- ancient_poetry_creation: 70 条 (7.00%)
- story_generation: 110 条 (11.00%)
- translation: 125 条 (12.50%)
- instruction: 500 条 (50.00%)
- classical_translation: 56 条 (5.60%)
- dialogue: 65 条 (6.50%)
- modern_poetry_creation: 74 条 (7.40%)

数据集构建的具体流程请参考：[data_processing/README.md](https://github.com/Cui-Peng-624/GemmaLM-Chinese/blob/main/src/data_processing/README.md)

# 3. 模型微调

我们三个 stage 都采用 AdaLoRA 进行微调，并根据 stage 动态调整训练参数，具体请参考：[data_processing/参数选择.md](https://github.com/Cui-Peng-624/GemmaLM-Chinese/blob/main/src/data_processing/%E5%8F%82%E6%95%B0%E9%80%89%E6%8B%A9.md)

# 4. 推理

实现了 l2m（least to most，让大模型将问题分解为一系列子问题，逐个解决），self-verification（让大模型对问题的回答进行自我验证） 和 enhanced_solver（结合l2m和self-verification） 三种策略，最终采用了 normal，l2m 和 enhanced_solver 三种推理策略，具体请参考：[adaptive_solver.ipynb](https://github.com/Cui-Peng-624/GemmaLM-Chinese/blob/main/src/core/solvers/adaptive_solver.ipynb)    

# 5. 评估与监控

我们在自己构建的[验证集](https://github.com/Cui-Peng-624/GemmaLM-Chinese/blob/main/src/data_processing/stage1/data_final/val_data.json)上评估了ppl，在ceval的[logic](https://huggingface.co/datasets/ceval/ceval-exam/viewer/logic)数据集上也进行了评估，结果如下：

在自己构建的验证集上：  
基础模型PPL: 16.5727        
微调后模型PPL: 3.8577                
PPL改进比例: 76.72%                

在ceval的logic数据集上：         
基础模型的得分: 25            
微调后模型的得分: 35.78431373            
改进比例：43.14%       

我们使用的是stage1微调后的模型，在评估时采用的都是normal推理模式，且由于资源限制，我们的模型效果并未达到最优，具体请参考[stage1_AdaLoRA.ipynb](https://github.com/Cui-Peng-624/GemmaLM-Chinese/blob/main/src/fine-tuning/stage1_AdaLoRA.ipynb)中的 tensorboard 的输出记录。

# 6. DPO

我们的DPO目标特定为提高模型对用户要求只输出ABCD的遵守程度。数据集构建请参考[RLHF/data_preparation/data_preparation.ipynb](https://github.com/Cui-Peng-624/GemmaLM-Chinese/blob/main/src/RLHF/data_preparation/data_preparation.ipynb)，代码请参考[RLHF/DPO/dpo.ipynb](https://github.com/Cui-Peng-624/GemmaLM-Chinese/blob/main/src/RLHF/data_preparation/dpo.ipynb)


# 7. 未来改进方向
- RAG：但是我没想好往向量数据库储存什么数据，graphrag 在诗歌创作上是个不错的选择，但由于任务的限制，我们无法采用
- RLHF：暂未考虑不同的prompt，具体来说，就是用户要求模型只回答ABCD四个选项有很多不同但类似的prompt，我们暂时不考虑，走完整个流程先
# AutoDL官方学术资源加速
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

import sys
import os

# 添加项目根目录到Python路径
project_root = "/home/cuipeng/Gemma"
sys.path.append(project_root)

# 现在可以正常导入src下的模块
from src.core.model.model_initializer import initialize_model_and_tokenizer
from src.core.utils.model_utils import generate_response, apply_chat_template

# self_verification

import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
from transformers import BitsAndBytesConfig # type: ignore
from peft import PeftModel  # type: ignore # 导入PeftModel用于加载微调模型

class SelfVerifier:
    def __init__(self, model, tokenizer):
        """
        初始化验证器
        Args:
            model: 已初始化的模型实例
            tokenizer: 已初始化的tokenizer实例
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # 添加验证示例
        self.verification_examples = """
        问题1：什么是机器学习？
        答案：机器学习是人工智能的一个分支，它使用统计技术让计算机系统能够"学习"而无需明确编程。

        评估结果：
        1. 准确性：答案基本准确，但缺少具体的学习方式
        2. 完整性：答案不够完整，未提及主要类型
        3. 逻辑性：定义清晰，但缺乏展开
        4. 改进建议：
           - 补充学习方式(监督/无监督)
           - 添加实际应用例子
           - 说明与深度学习的关系

        问题2：为什么要保护环境？
        答案：因为环境污染会危害人类健康，破坏生态平衡。

        评估结果：
        1. 准确性：答案正确但过于简单
        2. 完整性：严重不足，未涉及多个方面
        3. 逻辑性：因果关系正确但不够深入
        4. 改进建议：
           - 补充环境与发展的关系
           - 添加具体环境问题示例
           - 说明保护措施
           - 补充长期影响分析
        """

        self.verify_system_prompt = """
        你是一个专业的答案评估专家。你的任务是评估答案的质量并提供改进建议。

        评估维度：
        1. 准确性 - 答案是否有错误
        2. 完整性 - 是否完整回答问题
        3. 逻辑性 - 论述是否合理
        4. 改进建议 - 具体的改进方向

        请严格按照示例格式输出评估结果。
        """

        self.improve_system_prompt = """
        你是一个专业的答案优化专家。你的任务是根据评估结果改进原始答案。

        改进要求：
        1. 修正评估中指出的错误
        2. 补充缺失的信息
        3. 优化答案结构和逻辑
        4. 确保答案清晰完整

        请直接输出改进后的答案，无需其他说明。
        """

    def verify_answer(self, question, answer):
        """验证答案质量"""
        dialogue = [
            {
                "role": "system",
                "content": self.verify_system_prompt
            },
            {
                "role": "user",
                "content": f"""请参考以下示例，评估这个问答：

{self.verification_examples}

现在请评估：
问题：{question}
答案：{answer}

请严格按照示例格式输出评估结果。"""
            }
        ]
        
        prompt = apply_chat_template(dialogue)
        verification = generate_response(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=512,
            temperature=0.3
        )
        return verification

    def improve_answer(self, question, original_answer, verification):
        """根据验证结果改进答案"""
        dialogue = [
            {
                "role": "system",
                "content": self.improve_system_prompt
            },
            {
                "role": "user",
                "content": f"""示例：
原始答案：机器学习是AI的分支。
评估结果：答案过于简单，缺少细节。
改进答案：机器学习是人工智能的重要分支，通过统计算法使计算机能够从数据中学习规律。它包括监督学习、无监督学习等类型，在图像识别、自然语言处理等领域有广泛应用。

现在请改进：
原始问题：{question}
原始答案：{original_answer}
评估结果：{verification}

请直接给出改进后的答案，无需其他说明。"""
            }
        ]
        
        prompt = apply_chat_template(dialogue)
        improved_answer = generate_response(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=1024,
            temperature=0.7
        )
        return improved_answer

    def verify_and_improve(self, question, answer):
        """验证并改进答案"""
        # 1. 验证答案
        verification = self.verify_answer(question, answer)
        
        # 2. 改进答案
        improved_answer = self.improve_answer(question, answer, verification)
        
        # 3. 再次验证改进后的答案
        final_verification = self.verify_answer(question, improved_answer)
        
        return {
            "original_answer": answer,
            "first_verification": verification,
            "improved_answer": improved_answer,
            "final_verification": final_verification
        }
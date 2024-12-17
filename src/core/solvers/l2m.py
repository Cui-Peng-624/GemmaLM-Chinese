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
from src.core.utils.model_utils import generate_response, apply_chat_template, format_user

# Least-to-Most Prompting (L2M)

import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
from transformers import BitsAndBytesConfig # type: ignore
from peft import PeftModel  # type: ignore # 导入PeftModel用于加载微调模型

class L2MSolver:
    def __init__(self, model, tokenizer):
        """
        初始化L2M求解器
        Args:
            model: 已初始化的模型实例
            tokenizer: 已初始化的tokenizer实例
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # 添加 few-shot 示例
        self.decompose_examples = """
        问题1：量子纠缠现象是什么，它有什么应用？

        分解步骤：
        1. 什么是量子态和量子叠加？
        2. 量子纠缠的基本概念是什么？
        3. 量子纠缠现象如何被实验证实？
        4. 量子纠缠在量子通信和量子计算中有什么应用？

        问题2：区块链技术如何确保交易安全？

        分解步骤：
        1. 什么是哈希函数和加密算法？
        2. 区块链的基本结构是什么？
        3. 区块链如何实现去中心化验证？
        4. 为什么区块链的交易记录难以篡改？
        """

        self.decompose_system_prompt = """
        你是一位擅长分析和分解复杂问题的助手。你的主要任务是将复杂问题分解成若干个由简单到复杂的子问题。

        在分解问题时，你应该：
        1. 仔细分析问题的各个组成部分
        2. 识别解决问题所需的关键知识和技能
        3. 确保子问题之间具有逻辑连贯性
        4. 确保每个子问题的答案都能为最终问题的解决提供必要的支持

        你应该遵循以下原则：
        1. 子问题数量应保持在3-5个之间
        2. 子问题应该按照难度递增的顺序排列
        3. 每个子问题都应该明确、具体且可回答
        4. 避免输出无关的解释或评论

        输出格式要求：
        1. 必须严格按照"分解步骤："开头
        2. 每个子问题前标注序号
        3. 除了分解的子问题外，不要包含任何其他内容

        示例输出：
        分解步骤：
        1. [简单的子问题]
        2. [稍复杂的子问题]
        3. [较复杂的子问题]
        4. [最复杂的子问题]

        请记住：你的回答应该简洁明了，只包含问题分解本身，不需要任何额外的解释或评论。
        """

    def decompose_question(self, complex_question):
        dialogue = [
            {
                "role": "system",
                "content": self.decompose_system_prompt
            },
            {
                "role": "user",
                "content": f"""请参考以下示例，将复杂问题分解为子问题：

    {self.decompose_examples}

    现在请分解这个问题：{complex_question}

    请严格按照示例格式输出，以"分解步骤："开头。"""
            }
        ]
        
        prompt = apply_chat_template(dialogue)
        response = generate_response(self.model, self.tokenizer, prompt, max_new_tokens=512, temperature=0.3) # 降低温度，以保持回答的一致性
        
        # 改进子问题提取方法 ########################################################
        def extract_questions(text):
            import re
            
            pattern = r'\d+\.\s*([^\n]+)' # 匹配：1. This is the first line. 捕获组：This is the first line.
            questions = re.findall(pattern, text)
            
            if questions: # 如果question不为空，则表示提取成功
                return questions
            
            # 如果第一次提取失败，则尝试从"分解步骤："开始提取
            if "分解步骤：" in text:
                text = text.split("分解步骤：")[1].strip()
            
            lines = text.split('\n')
            questions = []
            
            for line in lines:
                line = line.strip()
                line = re.sub(r'^\d+\.\s*', '', line)
                if line and not line.startswith('分解步骤') and not line.startswith('要求'):
                    questions.append(line)
            
            return questions
        ########################################################################
        
        sub_questions = extract_questions(response)
        
        # 边缘处理
        if not sub_questions:
            backup_dialogue = [
                {"role": "system", "content": self.decompose_system_prompt},
                {"role": "user", "content": f"""之前的回答格式不正确。请重新分解问题，严格按照以下格式输出：
                 问题：{complex_question}

                 1. [第一个子问题]
                 2. [第二个子问题]
                 3. [第三个子问题]
                 ...

                请只输出编号和问题，不要有其他内容。"""}
                ]
            backup_prompt = apply_chat_template(backup_dialogue)
            
            response = generate_response(self.model, self.tokenizer, backup_prompt)
            sub_questions = extract_questions(response)
        
        # 如果还没有sub_questions，则报错
        if not sub_questions:
            raise ValueError("无法正确分解问题，请检查模型输出格式")
            
        return sub_questions

    def solve_with_context(self, question, context=""):
        dialogue = [
            {
                "role": "system",
                "content": "你是一个专业的问题解答专家。请参考示例风格，基于已知信息回答问题。"
            },
            {
                "role": "user",
                "content": f"""示例：
    问题：什么是量子态？
    已知信息：无
    答案：量子态是描述量子系统状态的数学表达，它可以同时存在于多个基本状态的叠加中。这种特性使得量子系统具有独特的性质。

    现在请回答：
    已知信息：
    {context}

    问题：{question}"""
            }
        ]
        
        prompt = apply_chat_template(dialogue)
        return generate_response(self.model, self.tokenizer, prompt, max_new_tokens=1024, temperature=0.7)

    def solve(self, complex_question):
        # 1. 分解问题
        sub_questions = self.decompose_question(complex_question) # 返回一个list
        
        # 2. 逐步解决
        context = ""
        solutions = []
        
        for q in sub_questions:
            solution = self.solve_with_context(q, context)
            solutions.append({"question": q, "solution": solution})
            context += f"\n问题：{q}\n答案：{solution}\n"
        
        # 3. 最终整合答案
        dialogue = [
            {
                "role": "system",
                "content": "你是一个专业的总结专家。请参考示例风格，整合前面的解答。"
            },
            {
                "role": "user",
                "content": f"""示例：
    解答过程：
    问题：什么是量子计算？
    答案：量子计算利用量子力学原理进行计算。

    问题：量子计算有什么优势？
    答案：可以并行处理大量数据。

    总结：量子计算是一种基于量子力学原理的新型计算方式，其最大优势在于并行处理能力。

    现在请总结：
    解答过程：
    {context}

    原始问题：{complex_question}"""
            }
        ]
        
        prompt = apply_chat_template(dialogue)
        final_answer = generate_response(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=1536,
            temperature=0.5
        )
        
        return {
            "original_question": complex_question,
            "sub_solutions": solutions,
            "final_answer": final_answer
        }
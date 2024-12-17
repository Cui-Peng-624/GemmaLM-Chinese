import torch
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
from transformers import BitsAndBytesConfig # type: ignore
from peft import PeftModel # type: ignore

from l2m import L2MSolver
from self_verification import SelfVerifier
from enhanced_solver import EnhancedSolver
from model_utils import generate_response, format_prompt, USER_TEMPLATE

class AdaptiveSolver:
    def __init__(self, model, tokenizer, l2m_solver, verifier, enhanced_solver):
        """
        初始化自适应求解器
        Args:
            model: 已初始化的模型实例
            tokenizer: 已初始化的tokenizer实例
            l2m_solver: 已初始化的 L2MSolver 实例
            verifier: 已初始化的 SelfVerifier 实例
            enhanced_solver: 已初始化的 EnhancedSolver 实例
        """
        # 检查必要的组件是否都已提供
        if any(x is None for x in [model, tokenizer, l2m_solver, verifier, enhanced_solver]):
            raise ValueError("必须提供所有必要的已初始化组件：model, tokenizer, l2m_solver, verifier, enhanced_solver")
            
        self.model = model
        self.tokenizer = tokenizer
        self.l2m_solver = l2m_solver
        self.verifier = verifier
        self.enhanced_solver = enhanced_solver
        
        # 初始化直接求解器
        self.direct_solver = self._init_direct_solver()
        
    def _init_direct_solver(self):
        """初始化直接回答的求解器"""
        return lambda question: generate_response(
            self.model,
            self.tokenizer,
            format_prompt(question)
        )
    
    def evaluate_complexity(self, question):
        """评估问题复杂度"""
        examples = """示例问题及其复杂度评分：

        问题1：今天的天气怎么样？
        复杂度：0.1
        原因：这是一个简单的事实性问题，可以直接回答。

        问题2：请解释DNA的双螺旋结构是如何形成的？
        复杂度：0.5
        原因：需要分步骤解释生物学概念，包含多个知识点。

        问题3：请分析全球气候变化对世界经济、生态系统和人类社会的长期影响。
        复杂度：0.9
        原因：需要多角度分析，涉及复杂的因果关系，需要详细论证和验证。

        以上是示例，现在请评估下面的问题："""

        prompt = format_prompt(f"""{examples}

        问题：{question}

        评分标准：
        - 0.0-0.3：简单问题，可以直接回答
        - 0.3-0.7：中等复杂度，需要分步骤思考
        - 0.7-1.0：高度复杂，需要详���分析和验证

        请只输出一个0-1之间的数字分数，不需要其他解释。""")
        
        response = generate_response(self.model, self.tokenizer, prompt).strip()
        try:
            score = float(response)
            return min(max(score, 0.0), 1.0)  # 确保分数在0-1之间
        except ValueError:
            # 如果无法解析为浮点数，返回默认中等复杂度
            return 0.5
    
    def solve(self, question):
        """根据问题复杂度选择合适的求解方法"""
        complexity = self.evaluate_complexity(question)
        print(f"问题复杂度评分: {complexity}")
        
        if complexity < 0.3:
            print("使用直接回答方式")
            return {
                "method": "direct",
                "complexity": complexity,
                "answer": self.direct_solver(question)
            }
        elif complexity < 0.7:
            print("使用L2M方式")
            result = self.l2m_solver.solve(question)
            return {
                "method": "l2m",
                "complexity": complexity,
                "answer": result
            }
        else:
            print("使用增强求解方式")
            result = self.enhanced_solver.solve_complex_question(question)
            return {
                "method": "enhanced",
                "complexity": complexity,
                "answer": result
            }

# 使用示例
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # type: ignore
    from peft import PeftModel # type: ignore
    import torch 
    from l2m import L2MSolver 
    from self_verification import SelfVerifier 
    from enhanced_solver import EnhancedSolver 
    
    # 1. 初始化基础组件
    model_path = "google/gemma-2-9b"
    cache_dir = "/root/autodl-tmp/gemma"
    lora_path = "/root/autodl-tmp/models/stage1/gemma-base-zh-final"
    
    # 2. 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    
    # 3. 初始化基础模型
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 4. 初始化L2M求解器和验证器
    l2m_solver = L2MSolver(model, tokenizer)
    verifier = SelfVerifier(model, tokenizer)
    
    # 5. 初始化增强求解器
    enhanced_solver = EnhancedSolver(l2m_solver, verifier)
    
    # 6. 初始化自适应求解器
    solver = AdaptiveSolver(
        model=model,
        tokenizer=tokenizer,
        l2m_solver=l2m_solver,
        verifier=verifier,
        enhanced_solver=enhanced_solver
    )
    
    # 测试不同复杂度的问题
    questions = [
        "今天星期几？",  # 简单问题
        "请解释光合作用的过程。",  # 中等复杂度
        "请分析人工智能在未来20年可能对人类社会产生的影响，包括就业、教育、医疗等多个方面。"  # 高度复杂
    ]
    
    for q in questions:
        print(f"\n处理问题: {q}")
        result = solver.solve(q)
        print(f"使用方法: {result['method']}")
        print(f"复杂度评分: {result['complexity']}")
        print("答案:", result['answer']) 
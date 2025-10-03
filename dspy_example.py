"""
DSPy Example: Building a Simple Question Answering System

This example demonstrates the core concepts of DSPy:
1. Declarative programming with language models
2. Self-improving through optimization
3. Modular and composable components
"""

import dspy
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

# Configure your language model
# You can use OpenAI, Claude, local models, etc.
# For this example, we'll use OpenAI (you'll need an API key)
# turbo = dspy.OpenAI(model='gpt-3.5-turbo', api_key='YOUR_API_KEY')
# dspy.settings.configure(lm=turbo)

class BasicQA(dspy.Signature):
    """Answer questions with short factual answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class SimpleQAModule(dspy.Module):
    """A simple question-answering module using DSPy."""
    
    def __init__(self):
        super().__init__()
        # Declare a ChainOfThought predictor with the BasicQA signature
        self.generate_answer = dspy.ChainOfThought(BasicQA)
    
    def forward(self, question):
        # Use the predictor to generate an answer
        prediction = self.generate_answer(question=question)
        return dspy.Prediction(answer=prediction.answer)

def demonstrate_basic_usage():
    """
    Demonstrates basic DSPy usage without optimization.
    """
    print("=== Basic DSPy Usage ===\n")
    
    # Create an instance of our QA module
    qa = SimpleQAModule()
    
    # Example questions
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What year did World War II end?"
    ]
    
    print("Without optimization (using default prompts):")
    for q in questions:
        # Note: This would need an actual LM configured to work
        # result = qa(question=q)
        # print(f"Q: {q}")
        # print(f"A: {result.answer}\n")
        print(f"Q: {q}")
        print(f"A: [Would be answered by LM]\n")

class AdvancedQA(dspy.Module):
    """
    A more advanced QA module that uses retrieval-augmented generation (RAG).
    """
    
    def __init__(self):
        super().__init__()
        # This module could include:
        # - A retriever to find relevant context
        # - A generator to produce answers based on context
        self.retrieve = dspy.Retrieve(k=3)  # Retrieve top 3 passages
        self.generate_answer = dspy.ChainOfThought(BasicQA)
    
    def forward(self, question):
        # Step 1: Retrieve relevant context
        context = self.retrieve(question).passages
        
        # Step 2: Generate answer using context
        prediction = self.generate_answer(context=context, question=question)
        
        return dspy.Prediction(
            context=context,
            answer=prediction.answer
        )

def demonstrate_optimization():
    """
    Demonstrates how DSPy can optimize prompts and module behavior.
    """
    print("=== DSPy Optimization ===\n")
    print("""
    DSPy's key innovation is automatic prompt optimization.
    Instead of manually crafting prompts, you can:
    
    1. Define your task with signatures (input/output specs)
    2. Create modules that compose these signatures
    3. Provide training examples
    4. Let DSPy optimize the prompts automatically
    
    Example optimization process:
    """)
    
    print("""
    # Load training data
    dataset = HotPotQA(train_seed=1, train_size=20)
    trainset = [x.with_inputs('question') for x in dataset.train]
    
    # Define metric for optimization
    def qa_metric(example, pred, trace=None):
        return example.answer.lower() in pred.answer.lower()
    
    # Create optimizer
    teleprompter = BootstrapFewShot(metric=qa_metric)
    
    # Optimize the module
    optimized_qa = teleprompter.compile(SimpleQAModule(), trainset=trainset)
    
    # Now optimized_qa has better prompts learned from examples!
    """)

def explain_dspy_concepts():
    """
    Explains the core concepts and benefits of DSPy.
    """
    print("\n" + "="*60)
    print("DSPy: PROGRAMMING (NOT PROMPTING) LANGUAGE MODELS")
    print("="*60 + "\n")
    
    print("🎯 WHAT IS DSPy?")
    print("-" * 40)
    print("""
DSPy (Declarative Self-improving Python) is a framework that shifts
the paradigm from "prompt engineering" to "programming" with LMs.

Instead of:
❌ Writing brittle, hand-crafted prompts
❌ Manually tuning prompt templates
❌ Constant prompt iteration and testing

DSPy offers:
✅ Declarative modules that compose into pipelines
✅ Automatic prompt optimization from examples
✅ Self-improving systems that learn from data
✅ Modular, testable, and maintainable AI systems
    """)
    
    print("🔧 KEY COMPONENTS:")
    print("-" * 40)
    print("""
1. SIGNATURES: Define input/output behavior
   - Specify what goes in and what comes out
   - Like function signatures but for LM calls

2. MODULES: Composable building blocks
   - ChainOfThought: Step-by-step reasoning
   - Retrieve: RAG components
   - ProgramOfThought: Code generation
   - Custom modules you define

3. TELEPROMPTERS: Optimization algorithms
   - BootstrapFewShot: Learn from examples
   - COPRO: Coordinate prompt optimization
   - MIPRO: Multi-stage optimization

4. METRICS: Define success
   - Custom evaluation functions
   - Used to guide optimization
    """)
    
    print("💡 REAL-WORLD APPLICATIONS:")
    print("-" * 40)
    print("""
- RAG Systems: Build sophisticated retrieval-augmented pipelines
- Agents: Create self-improving agent loops
- Classification: Train robust classifiers
- Information Extraction: Extract structured data
- Multi-hop Reasoning: Complex reasoning chains
- And much more...
    """)
    
    print("🚀 WHY USE DSPy?")
    print("-" * 40)
    print("""
1. MAINTAINABILITY: Code, not prompt strings
2. OPTIMIZATION: Automatically improve performance
3. MODULARITY: Reusable components
4. ADAPTABILITY: Easy to switch LMs or tasks
5. RELIABILITY: Systematic, not ad-hoc
6. EFFICIENCY: Reduce manual prompt engineering time
    """)

if __name__ == "__main__":
    explain_dspy_concepts()
    print("\n" + "="*60 + "\n")
    demonstrate_basic_usage()
    print("\n" + "="*60 + "\n")
    demonstrate_optimization()
    
    print("\n" + "="*60)
    print("TO GET STARTED:")
    print("="*60)
    print("""
1. Install DSPy: pip install dspy-ai
2. Configure your LM (OpenAI, Anthropic, local models, etc.)
3. Define your signatures and modules
4. Optimize with training examples
5. Deploy your optimized system!

Check out the docs at: https://dspy.ai
    """)

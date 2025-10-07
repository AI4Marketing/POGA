"""
DSPy Practical Example: Building an Optimized Research Assistant

This example demonstrates the key concepts of DSPy through a practical application:
A research assistant that can answer complex questions using retrieval-augmented generation.
"""

import dspy
from typing import List, Optional
from dspy.datasets import HotPotQA
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

# ============================================================================
# STEP 1: Define Signatures (What we want the LM to do)
# ============================================================================

class GenerateQuery(dspy.Signature):
    """Generate search queries to find relevant information."""
    question: str = dspy.InputField(desc="the question to answer")
    query: str = dspy.OutputField(desc="search query to find relevant information")

class AnswerQuestion(dspy.Signature):
    """Answer questions based on provided context."""
    context: str = dspy.InputField(desc="relevant information")
    question: str = dspy.InputField(desc="question to answer")
    answer: str = dspy.OutputField(desc="concise factual answer")

class VerifyAnswer(dspy.Signature):
    """Verify if an answer is supported by the context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    is_valid: bool = dspy.OutputField(desc="whether answer is fully supported")
    explanation: str = dspy.OutputField(desc="why the answer is valid or not")

# ============================================================================
# STEP 2: Build Modules (Composable components)
# ============================================================================

class SimpleRAG(dspy.Module):
    """Basic RAG without optimization."""
    
    def __init__(self, k=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=k)
        self.generate = dspy.Predict(AnswerQuestion)
    
    def forward(self, question):
        # Retrieve relevant passages
        passages = self.retrieve(question).passages
        context = "\n".join(passages)
        
        # Generate answer
        answer = self.generate(context=context, question=question)
        return dspy.Prediction(answer=answer.answer, context=passages)

class AdvancedRAG(dspy.Module):
    """Advanced RAG with query generation and verification."""
    
    def __init__(self, k=3):
        super().__init__()
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.retrieve = dspy.Retrieve(k=k)
        self.generate_answer = dspy.ChainOfThought(AnswerQuestion)
        self.verify = dspy.Predict(VerifyAnswer)
    
    def forward(self, question):
        # Generate optimized search query
        query = self.generate_query(question=question).query
        
        # Retrieve with generated query
        passages = self.retrieve(query).passages
        context = "\n".join(passages)
        
        # Generate answer with reasoning
        answer = self.generate_answer(context=context, question=question)
        
        # Verify the answer
        verification = self.verify(
            context=context,
            question=question,
            answer=answer.answer
        )
        
        return dspy.Prediction(
            answer=answer.answer,
            query=query,
            context=passages,
            is_valid=verification.is_valid,
            explanation=verification.explanation
        )

class MultiHopRAG(dspy.Module):
    """Multi-hop reasoning for complex questions."""
    
    def __init__(self, hops=2, k=3):
        super().__init__()
        self.hops = hops
        self.retrieve = dspy.Retrieve(k=k)
        self.generate_query = dspy.ChainOfThought(GenerateQuery)
        self.generate_answer = dspy.ChainOfThought(AnswerQuestion)
    
    def forward(self, question):
        context = []
        query = question
        
        # Perform multiple retrieval hops
        for hop in range(self.hops):
            # Generate query for this hop
            query_pred = self.generate_query(question=query)
            
            # Retrieve passages
            passages = self.retrieve(query_pred.query).passages
            context.extend(passages)
            
            # Update query for next hop based on what we found
            if hop < self.hops - 1:
                query = f"{question} Given: {passages[0][:200]}..."
        
        # Generate final answer
        full_context = "\n".join(context)
        answer = self.generate_answer(context=full_context, question=question)
        
        return dspy.Prediction(
            answer=answer.answer,
            context=context,
            hops=self.hops
        )

# ============================================================================
# STEP 3: Define Metrics (How we measure success)
# ============================================================================

def exact_match_metric(example, prediction, trace=None):
    """Check if the predicted answer matches the gold answer."""
    return example.answer.lower() == prediction.answer.lower()

def contains_answer_metric(example, prediction, trace=None):
    """Check if the predicted answer contains the gold answer."""
    return example.answer.lower() in prediction.answer.lower()

def comprehensive_metric(example, prediction, trace=None):
    """
    A more sophisticated metric that considers multiple factors.
    """
    # Check exact match
    exact = example.answer.lower() == prediction.answer.lower()
    if exact:
        return 1.0
    
    # Check if answer is contained
    contained = example.answer.lower() in prediction.answer.lower()
    if contained:
        return 0.8
    
    # Check if key terms are present
    gold_terms = set(example.answer.lower().split())
    pred_terms = set(prediction.answer.lower().split())
    overlap = len(gold_terms & pred_terms) / len(gold_terms) if gold_terms else 0
    
    return overlap * 0.5

# ============================================================================
# STEP 4: Optimization with Teleprompters
# ============================================================================

def optimize_rag_system():
    """
    Demonstrates the optimization process for a RAG system.
    """
    
    # Configure DSPy (you would use real credentials here)
    # lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key="YOUR_KEY")
    # rm = dspy.ColBERTv2(url="http://your-colbert-server")
    # dspy.settings.configure(lm=lm, rm=rm)
    
    print("=" * 60)
    print("DSPy RAG Optimization Demo")
    print("=" * 60)
    
    # Load a dataset (using HotPotQA as example)
    print("\n1. Loading dataset...")
    # dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2, dev_size=50)
    # trainset = dataset.train
    # devset = dataset.dev
    
    # For demo purposes, create synthetic examples
    trainset = [
        dspy.Example(
            question="What is the capital of France?",
            answer="Paris"
        ).with_inputs("question"),
        dspy.Example(
            question="Who wrote Romeo and Juliet?",
            answer="William Shakespeare"
        ).with_inputs("question"),
        dspy.Example(
            question="What year did World War II end?",
            answer="1945"
        ).with_inputs("question"),
    ]
    
    # Initialize unoptimized RAG
    print("\n2. Creating unoptimized RAG system...")
    simple_rag = SimpleRAG(k=3)
    
    # Test unoptimized version
    print("\n3. Testing unoptimized RAG:")
    print("   (Would normally show poor performance)")
    
    # Create optimizer
    print("\n4. Setting up BootstrapFewShot optimizer...")
    optimizer = BootstrapFewShot(
        metric=comprehensive_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        max_rounds=1
    )
    
    # Compile/optimize the RAG system
    print("\n5. Optimizing RAG system...")
    print("   - Running teacher on training examples")
    print("   - Collecting successful traces")
    print("   - Creating demonstrations")
    print("   - Compiling optimized student")
    
    # optimized_rag = optimizer.compile(simple_rag, trainset=trainset)
    # In practice, this would run the optimization process
    
    print("\n6. Optimization complete!")
    print("   The optimized RAG now includes:")
    print("   - Learned few-shot examples")
    print("   - Optimized prompts")
    print("   - Better performance on the task")
    
    # Evaluate performance
    print("\n7. Evaluating performance...")
    # evaluator = Evaluate(devset=devset, metric=comprehensive_metric, num_threads=4)
    # results = evaluator(optimized_rag)
    # print(f"   Score: {results.score:.2%}")
    
    # Advanced optimization with different modules
    print("\n8. Advanced optimization options:")
    print("   - AdvancedRAG with query generation")
    print("   - MultiHopRAG for complex reasoning")
    print("   - Custom modules for specific tasks")
    
    advanced_rag = AdvancedRAG(k=3)
    multihop_rag = MultiHopRAG(hops=2, k=3)
    
    # These can also be optimized
    # optimized_advanced = optimizer.compile(advanced_rag, trainset=trainset)
    # optimized_multihop = optimizer.compile(multihop_rag, trainset=trainset)

# ============================================================================
# STEP 5: Advanced Techniques
# ============================================================================

class PipelineRAG(dspy.Module):
    """
    Demonstrates a pipeline approach with multiple specialized modules.
    """
    
    def __init__(self):
        super().__init__()
        # Question understanding
        self.classify = dspy.Predict("question -> type: str, complexity: int")
        
        # Different strategies based on complexity
        self.simple_qa = dspy.Predict("question -> answer")
        self.complex_rag = AdvancedRAG(k=5)
        
        # Post-processing
        self.refine = dspy.ChainOfThought("draft: str, question: str -> final_answer: str")
    
    def forward(self, question):
        # Classify the question
        classification = self.classify(question=question)
        
        # Route to appropriate handler
        if classification.complexity < 3:
            draft = self.simple_qa(question=question).answer
        else:
            draft = self.complex_rag(question=question).answer
        
        # Refine the answer
        final = self.refine(draft=draft, question=question)
        
        return dspy.Prediction(
            answer=final.final_answer,
            complexity=classification.complexity,
            type=classification.type
        )

class SelfCorrectingRAG(dspy.Module):
    """
    RAG with self-correction loop.
    """
    
    def __init__(self, max_iterations=3):
        super().__init__()
        self.rag = SimpleRAG(k=3)
        self.critique = dspy.ChainOfThought(
            "answer: str, question: str -> is_complete: bool, missing: str"
        )
        self.refine = dspy.ChainOfThought(
            "answer: str, missing: str, context: str -> improved_answer: str"
        )
        self.max_iterations = max_iterations
    
    def forward(self, question):
        # Initial attempt
        result = self.rag(question=question)
        answer = result.answer
        context = result.context
        
        # Self-correction loop
        for i in range(self.max_iterations):
            critique = self.critique(answer=answer, question=question)
            
            if critique.is_complete:
                break
            
            # Refine based on critique
            improved = self.refine(
                answer=answer,
                missing=critique.missing,
                context="\n".join(context)
            )
            answer = improved.improved_answer
        
        return dspy.Prediction(
            answer=answer,
            iterations=i + 1,
            context=context
        )

# ============================================================================
# MAIN: Putting it all together
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         DSPy: Advanced RAG System Development           ║
    ╚══════════════════════════════════════════════════════════╝
    
    This example demonstrates:
    
    1. SIGNATURES: Defining input/output contracts
    2. MODULES: Building composable components
    3. METRICS: Measuring success
    4. OPTIMIZATION: Automatic prompt engineering
    5. ADVANCED: Pipeline and self-correcting systems
    
    Key Concepts Illustrated:
    - Declarative programming with LMs
    - Automatic optimization from examples
    - Modular and composable design
    - Multiple RAG strategies
    - Self-improvement through optimization
    """)
    
    # Run the optimization demo
    optimize_rag_system()
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
    1. DSPy separates 'what' (signatures) from 'how' (optimization)
    2. Modules compose naturally for complex systems
    3. Optimization happens automatically from examples
    4. Same code works across different LMs
    5. Performance improves with data, not manual tweaking
    
    Next Steps:
    - Configure with real LM credentials
    - Load your domain-specific data
    - Define task-specific signatures
    - Create custom metrics
    - Optimize and deploy!
    """)

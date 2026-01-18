# Using Skills from the Marketplace

This guide demonstrates how to effectively use skills from the Claude Skills Marketplace through realistic examples.

## How Skills Activate

Skills automatically activate when your conversation context matches their purpose. You don't need to explicitly invoke them - Claude recognizes when a skill is relevant and seamlessly integrates it into the response.

## Example: AI Engineering Skill

The `ai-engineering-skill` provides expert guidance on building production ML systems.

### Example 1: Choosing Between RAG and Finetuning

**User Question:**
```
I have a customer support chatbot that needs to answer questions about our product
documentation. Should I use RAG or finetuning?
```

**How the Skill Helps:**

When you ask this question, Claude automatically activates the `ai-engineering-skill` and provides structured guidance based on the reference materials:

1. **Context Analysis** - Evaluates your use case requirements
2. **Trade-offs** - Explains the pros and cons of each approach
3. **Recommendation** - Suggests RAG for this scenario because:
   - Documentation updates frequently
   - No need to change model behavior
   - Lower cost and faster iteration
4. **Implementation Guidance** - Outlines how to build the RAG system

### Example 2: Model Evaluation Metrics

**User Question:**
```
What metrics should I track for my text classification model in production?
```

**How the Skill Helps:**

The skill activates and provides comprehensive guidance on:
- Offline metrics (accuracy, precision, recall, F1)
- Online metrics (user satisfaction, task completion)
- System metrics (latency, throughput, cost)
- Monitoring strategies for drift detection

### Example 3: Inference Optimization

**User Question:**
```
My LLM inference is too slow and expensive. How can I optimize it?
```

**How the Skill Helps:**

The skill provides a structured optimization approach:
1. **Profiling** - Identify bottlenecks
2. **Model Optimization** - Quantization, pruning, distillation
3. **Serving Optimization** - Batching, caching, load balancing
4. **Cost-Latency Trade-offs** - Making informed decisions

## Example: Explaining Code Skill

The `explaining-code` skill makes code more accessible through visual diagrams and analogies.

### Example 1: Understanding a Complex Algorithm

**User Question:**
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

How does this quicksort implementation work?
```

**How the Skill Helps:**

The skill activates and provides:

1. **Analogy:**
   "Think of quicksort like organizing a deck of cards by picking a middle card (the pivot)
   and sorting all smaller cards to the left, larger cards to the right, then repeating
   the process on each pile."

2. **Visual Diagram:**
   ```
   [3, 6, 8, 10, 1, 2, 1]
           ↓
       pivot = 10
           ↓
   [3,6,8,1,2,1] [10] []
         ↓
      (recurse)
   ```

3. **Step-by-Step Walkthrough:**
   - Base case: arrays with 0-1 elements are already sorted
   - Choose pivot: middle element
   - Partition: split into three groups
   - Recursion: sort left and right partitions
   - Combine: concatenate results

4. **Common Gotcha:**
   "The middle list is crucial! Without it, duplicate values would cause infinite recursion."

### Example 2: Explaining API Architecture

**User Question:**
```
Can you explain how this API authentication middleware works?
```

**How the Skill Helps:**

The skill provides:
- **Analogy:** "Like a bouncer at a club checking IDs before letting people in"
- **Flow Diagram:** ASCII art showing request → middleware → handler flow
- **Step-by-Step:** What happens at each stage
- **Gotchas:** Common security mistakes to avoid

## Combining Multiple Skills

Sometimes multiple skills work together in a conversation.

### Example: Building and Explaining an ML Feature

**Conversation:**
```
User: I need to add a feature extraction pipeline for my ML model

[ai-engineering-skill activates]
Claude: Let me help you design the feature engineering pipeline...
        [provides architecture and best practices]

User: Can you explain how this feature extraction code works?

[explaining-code skill activates]
Claude: Let me break down this feature extraction with an analogy...
        [provides visual explanation]
```

## Tips for Getting the Most from Skills

### 1. Be Specific

**Less Effective:**
```
"How do I make my model better?"
```

**More Effective:**
```
"My model has 85% accuracy but 60% precision on the minority class.
How should I improve precision without hurting recall too much?"
```

### 2. Provide Context

**Less Effective:**
```
"Should I use RAG?"
```

**More Effective:**
```
"I'm building a legal document Q&A system. Documents are 50-100 pages,
updated quarterly, and need accurate citations. Should I use RAG?"
```

### 3. Ask Follow-Up Questions

Skills maintain context across the conversation:
```
User: How do I implement RAG?
Claude: [provides RAG architecture]

User: What about the chunking strategy?
Claude: [dives deeper into chunking approaches]

User: Which embedding model should I use?
Claude: [recommends embedding models with trade-offs]
```

### 4. Request Examples

Ask for concrete examples when you need them:
```
"Can you show me a code example of that?"
"What would this look like with my Python stack?"
"Can you diagram how these components interact?"
```

## Troubleshooting

### Skill Not Activating?

If you expect a skill to activate but it doesn't:

1. **Be more explicit:**
   ```
   "Using best practices for production ML systems, how should I..."
   ```

2. **Use skill-specific terminology:**
   - AI Engineering: "RAG", "finetuning", "MLOps", "inference optimization"
   - Explaining Code: "how does this work", "explain this code", "walk me through"

3. **Check skill installation:**
   ```bash
   /plugin skill list --installed
   ```

### Skill Providing Wrong Guidance?

Skills are based on specific reference materials and best practices. If guidance doesn't fit your use case:

1. **Provide more context** about your constraints
2. **Ask about trade-offs** explicitly
3. **Request alternatives** if the suggested approach doesn't work

## Exploring Skills

Learn what skills can do:

```bash
# List all installed skills
/plugin skill list --installed

# View skill details
/plugin skill info ai-engineering-skill

# Check for skill updates
/plugin skill list --updates
```

## Example Project Walkthrough

### Building a Production RAG System

This example shows how the `ai-engineering-skill` guides you through a complete project:

**Phase 1: Architecture Design**
```
User: I need to build a RAG system for our product docs
Claude: [skill provides architecture overview]
```

**Phase 2: Implementation**
```
User: How should I chunk the documents?
Claude: [skill explains chunking strategies]

User: What embedding model should I use?
Claude: [skill recommends models with trade-offs]
```

**Phase 3: Evaluation**
```
User: How do I measure if my RAG system is working well?
Claude: [skill provides evaluation framework]
```

**Phase 4: Optimization**
```
User: Latency is too high, how can I optimize?
Claude: [skill suggests optimization strategies]
```

Each step benefits from the skill's domain expertise, providing consistent, production-ready guidance throughout the project.

## Next Steps

- Install skills from the marketplace: `/plugin install <skill-name>`
- Try the example questions above in your conversations
- Explore the skill reference materials in each skill directory
- Contribute your own skills to help others (see [CONTRIBUTING.md](../CONTRIBUTING.md))

---

**Remember:** Skills work best when you provide clear context and ask specific questions. The more information you provide, the better Claude can apply skill knowledge to your specific situation.

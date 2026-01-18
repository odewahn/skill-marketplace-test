# Claude Skills Marketplace

A curated collection of Claude Code skills for AI engineering, code explanation, and more. These skills enhance Claude's capabilities by providing domain-specific knowledge and structured approaches to common development tasks.

## Overview

This marketplace provides ready-to-use skills that extend Claude Code's functionality in specific domains. Skills automatically activate when relevant to your conversation, providing expert guidance and best practices.

**Current Skills:**
- **ai-engineering-skill** - Production ML systems, MLOps, RAG, and finetuning
- **explaining-code** - Visual explanations with diagrams and analogies

## Installation

### Add This Marketplace

Add this marketplace to your Claude Code installation:

```bash
/plugin marketplace add odewahn/skills-marketplace-test
```

### List Available Skills

See all skills available in this marketplace:

```bash
/plugin skill list --available
```

### Install Individual Skills

Install specific skills from the marketplace:

```bash
# Install the AI engineering skill
/plugin install ai-engineering-skill

# Install the code explanation skill
/plugin install explaining-code
```

## Available Skills

### 🤖 AI Engineering Skill

**Name:** `ai-engineering-skill`
**Version:** 1.0.0
**Tags:** ai, ml, production, mlops, rag, finetuning

Practical guide for building production ML systems based on Chip Huyen's AI Engineering book. Provides expert guidance on:

- Model evaluation and deployment strategies
- RAG (Retrieval Augmented Generation) systems
- Finetuning approaches and best practices
- Inference optimization for cost and latency
- MLOps and monitoring
- Dataset engineering
- Foundation model selection

**Example Usage:**
```
User: How do I implement a RAG system?
User: What are best practices for model evaluation?
User: Should I use finetuning or RAG for my use case?
User: How do I optimize inference for cost and latency?
```

**References:**
The skill includes comprehensive reference materials covering:
- Foundation models and selection criteria
- Prompt engineering and safety
- RAG and agents implementation
- Finetuning methodology
- Evaluation frameworks
- Inference optimization
- Dataset engineering
- Architecture and feedback systems

### 📚 Explaining Code Skill

**Name:** `explaining-code`
**Version:** 1.0.0
**Tags:** teaching, documentation, code-review, education, learning

Makes code explanations more accessible through visual diagrams and real-world analogies. Perfect for:

- Understanding unfamiliar codebases
- Code reviews and documentation
- Teaching and learning
- Onboarding new team members

**Example Usage:**
```
User: How does this function work?
User: Explain this codebase to me
User: Can you walk me through what this code does?
User: Help me understand how this algorithm works
```

**Features:**
- Starts with relatable analogies
- Uses ASCII art diagrams for visualization
- Step-by-step code walkthroughs
- Highlights common gotchas and misconceptions

## Usage

Once installed, skills automatically activate when relevant to your conversation. You don't need to explicitly invoke them - Claude will use the appropriate skill based on your questions and context.

**Example Workflows:**

1. **Building an AI System:**
   ```
   You: I need to build a document Q&A system
   Claude: [activates ai-engineering-skill]
           Let me help you design a RAG system...
   ```

2. **Understanding Code:**
   ```
   You: How does this authentication flow work?
   Claude: [activates explaining-code skill]
           Let me explain with an analogy...
   ```

## Updating Skills

Check for updates and upgrade installed skills:

```bash
# Check for updates
/plugin skill list --updates

# Update a specific skill
/plugin update ai-engineering-skill

# Update all skills
/plugin update --all
```

## Contributing

Interested in adding your own skills to this marketplace? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on creating and submitting skills.

## Skill Structure

Each skill in this marketplace follows a consistent structure:

```
skills/skill-name/
├── skill.json          # Skill metadata (version, description, tags)
├── SKILL.md            # Main skill documentation and instructions
└── references/         # Optional: Additional reference materials
```

## License

This marketplace and its skills are available under the MIT License unless otherwise specified in individual skill directories.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub: [odewahn/skills-marketplace-test](https://github.com/odewahn/skills-marketplace-test)
- Check the [examples](examples/) directory for usage patterns

## Marketplace Metadata

This marketplace is identified by:
- **Repository:** odewahn/skills-marketplace-test
- **Type:** skills
- **Version:** 1.0.0
- **Skills Directory:** skills/

---

Built with Claude Code - Extend your AI development capabilities with domain-specific skills.

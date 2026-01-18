# Contributing to Claude Skills Marketplace

Thank you for your interest in contributing to this skills marketplace! This guide will help you create, test, and submit new skills.

## Table of Contents

- [Skill Structure](#skill-structure)
- [Creating a New Skill](#creating-a-new-skill)
- [Skill Metadata Format](#skill-metadata-format)
- [SKILL.md Format](#skillmd-format)
- [Testing Your Skill](#testing-your-skill)
- [Submission Process](#submission-process)
- [Best Practices](#best-practices)

## Skill Structure

Every skill in this marketplace follows a consistent directory structure:

```
skills/your-skill-name/
├── skill.json          # Required: Skill metadata
├── SKILL.md            # Required: Skill instructions and documentation
└── references/         # Optional: Additional reference materials
    ├── reference1.md
    ├── reference2.md
    └── ...
```

### Required Files

1. **skill.json** - Contains metadata about your skill (version, description, tags, etc.)
2. **SKILL.md** - Contains the skill's instructions, documentation, and usage guidelines

### Optional Additions

- **references/** - Directory for additional reference materials, examples, or documentation
- **examples/** - Sample code or usage examples
- **tests/** - Test cases for validating skill behavior

## Creating a New Skill

### Step 1: Choose a Skill Name

Skill names should be:
- Lowercase with hyphens (kebab-case)
- Descriptive and specific
- Unique within the marketplace

**Good Examples:**
- `python-testing-best-practices`
- `api-security-review`
- `react-performance-optimization`

**Avoid:**
- Generic names like `helper` or `utils`
- Names with underscores or spaces
- Overly long names (keep it under 30 characters)

### Step 2: Create Skill Directory

```bash
cd skills
mkdir your-skill-name
cd your-skill-name
```

### Step 3: Create skill.json

Create a `skill.json` file with the following structure:

```json
{
  "name": "your-skill-name",
  "version": "1.0.0",
  "description": "Clear, concise description of what your skill does (1-2 sentences)",
  "author": "your-github-username",
  "license": "MIT",
  "tags": ["tag1", "tag2", "tag3"],
  "created": "2026-01-18",
  "updated": "2026-01-18",
  "dependencies": [],
  "files": {
    "main": "SKILL.md",
    "references": "references/"
  },
  "usage_examples": [
    "Example question 1",
    "Example question 2",
    "Example question 3"
  ]
}
```

### Step 4: Create SKILL.md

Create a `SKILL.md` file with frontmatter and instructions:

```markdown
---
name: your-skill-name
description: Brief description that explains when this skill should activate
---

# Your Skill Name

## Overview

Explain what this skill does and what problem it solves.

## When to Use This Skill

- Scenario 1
- Scenario 2
- Scenario 3

## Instructions

Provide clear, actionable instructions for how Claude should use this skill.

## Examples

Include examples of how the skill should be applied in practice.

## References

Link to any external resources, documentation, or references.
```

## Skill Metadata Format

### skill.json Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Skill identifier (must match directory name) |
| `version` | string | Yes | Semantic version (e.g., "1.0.0") |
| `description` | string | Yes | Brief description (1-2 sentences) |
| `author` | string | Yes | GitHub username or author name |
| `license` | string | Yes | License type (e.g., "MIT", "Apache-2.0") |
| `tags` | array | Yes | Relevant tags for discoverability (3-6 tags) |
| `created` | string | Yes | Creation date (ISO format: YYYY-MM-DD) |
| `updated` | string | Yes | Last update date (ISO format: YYYY-MM-DD) |
| `dependencies` | array | No | Other skills this skill depends on |
| `files` | object | Yes | Map of file types to paths |
| `usage_examples` | array | Yes | 3-5 example questions/prompts |

### Choosing Good Tags

Tags help users discover your skill. Include:
- **Primary domain** (e.g., "python", "javascript", "devops")
- **Functionality** (e.g., "testing", "security", "performance")
- **Use case** (e.g., "code-review", "debugging", "refactoring")

**Example tag sets:**
- API skill: `["api", "rest", "security", "documentation"]`
- Testing skill: `["python", "testing", "pytest", "tdd"]`
- Performance skill: `["optimization", "performance", "profiling", "react"]`

## SKILL.md Format

### Frontmatter

The frontmatter is critical for skill activation:

```yaml
---
name: your-skill-name
description: Specific description that helps Claude know when to activate this skill. Include key phrases and use cases.
---
```

**Tips for descriptions:**
- Be specific about when the skill should activate
- Include key phrases users might say
- Mention specific technologies or domains
- Keep it under 200 characters

### Content Structure

Organize your SKILL.md with clear sections:

1. **Overview** - What the skill does
2. **When to Use** - Activation scenarios
3. **Instructions** - Step-by-step guidance
4. **Best Practices** - Recommendations and guidelines
5. **Examples** - Concrete usage examples
6. **Common Pitfalls** - What to avoid
7. **References** - Links to resources

### Writing Style

- Use clear, actionable language
- Write as if instructing Claude directly
- Include specific examples and code snippets
- Use bullet points and numbered lists
- Keep paragraphs short and scannable

## Testing Your Skill

### Local Testing

Test your skill locally before submitting:

1. **Install the skill locally:**
   ```bash
   /plugin install --local /path/to/your-skill-directory
   ```

2. **Test activation scenarios:**
   - Ask questions that should trigger your skill
   - Verify Claude uses the skill appropriately
   - Check that instructions are clear and actionable

3. **Test edge cases:**
   - Questions that shouldn't trigger the skill
   - Related but out-of-scope questions
   - Multiple skills activating together

### Validation Checklist

Before submitting, verify:

- [ ] `skill.json` is valid JSON with all required fields
- [ ] `SKILL.md` has proper frontmatter
- [ ] Skill name matches directory name
- [ ] Version follows semantic versioning (X.Y.Z)
- [ ] Description is clear and concise
- [ ] Tags are relevant and specific
- [ ] Usage examples are realistic
- [ ] References (if any) are accessible
- [ ] No sensitive information or credentials included

## Submission Process

### Step 1: Fork and Clone

1. Fork this repository on GitHub
2. Clone your fork locally
3. Create a new branch for your skill

```bash
git clone https://github.com/YOUR-USERNAME/skills-marketplace-test.git
cd skills-marketplace-test
git checkout -b add-your-skill-name
```

### Step 2: Add Your Skill

1. Create your skill directory under `skills/`
2. Add all required files (`skill.json`, `SKILL.md`)
3. Add any optional reference materials

### Step 3: Update marketplace.json

Add your skill to the marketplace registry:

```json
{
  "name": "your-skill-name",
  "path": "skills/your-skill-name",
  "version": "1.0.0",
  "description": "Brief description",
  "tags": ["tag1", "tag2"]
}
```

Insert your skill entry in the `skills` array in alphabetical order.

### Step 4: Test the Marketplace

Verify the marketplace structure is valid:

```bash
# Check JSON syntax
cat marketplace.json | python -m json.tool

# Verify skill paths exist
ls -la skills/your-skill-name/
```

### Step 5: Commit and Push

```bash
git add skills/your-skill-name/
git add marketplace.json
git commit -m "Add [your-skill-name] skill"
git push origin add-your-skill-name
```

### Step 6: Create Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template with:
   - Skill name and description
   - What problem it solves
   - Testing you performed
   - Any special considerations

## Best Practices

### Skill Design

1. **Single Responsibility** - Each skill should have one clear purpose
2. **Clear Boundaries** - Define what's in-scope and out-of-scope
3. **Composability** - Skills should work well together
4. **Maintainability** - Keep skills focused and easy to update

### Documentation

1. **User-Focused** - Write for skill users, not just skill creators
2. **Examples** - Include concrete, realistic examples
3. **Context** - Explain the "why" not just the "how"
4. **Maintenance** - Include dates and version info

### Quality

1. **Accuracy** - Ensure technical content is correct and current
2. **Completeness** - Cover common scenarios and edge cases
3. **Clarity** - Use plain language and avoid jargon
4. **Testing** - Thoroughly test before submitting

### Maintenance

1. **Updates** - Keep skills current with technology changes
2. **Versioning** - Use semantic versioning for updates
3. **Changelog** - Document changes in skill.json or separate file
4. **Deprecation** - Clearly mark deprecated skills

## Skill Template

Use this template as a starting point:

```bash
skills/your-skill-name/
├── skill.json
├── SKILL.md
└── references/
    └── README.md
```

**skill.json:**
```json
{
  "name": "your-skill-name",
  "version": "1.0.0",
  "description": "What your skill does",
  "author": "your-username",
  "license": "MIT",
  "tags": ["relevant", "tags"],
  "created": "2026-01-18",
  "updated": "2026-01-18",
  "dependencies": [],
  "files": {
    "main": "SKILL.md"
  },
  "usage_examples": [
    "Example usage 1",
    "Example usage 2"
  ]
}
```

**SKILL.md:**
```markdown
---
name: your-skill-name
description: When and why this skill activates
---

# Skill Name

Your skill content here...
```

## Questions or Issues?

- Open an issue on GitHub
- Tag your issue with "question" or "contribution"
- We'll respond as quickly as possible

Thank you for contributing to the Claude Skills Marketplace!

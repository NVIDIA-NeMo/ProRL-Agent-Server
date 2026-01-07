# NeAR RLER Rewards Guide

This guide explains how to use the RLER (Reinforcement Learning with Evolving Rubrics) reward system for evaluating NeAR research agents, replicating the exact methodology from [DR Tulu](https://github.com/rlresearch/dr-tulu).

## Overview

RLER provides automated, multi-dimensional evaluation of research quality through:

1. **Rubric Reward (50-60%)** - LLM-based evaluation against quality criteria
2. **Citation Reward (20%)** - Quality and coverage of citations
3. **Format Reward (20%)** - Response structure compliance
4. **Search Turns Reward (10-20%)** - Research thoroughness via tool usage

## Quick Start

### Basic Usage

```python
instance = {
    'instance_id': 'research_task_1',
    'problem_statement': 'Research the top AI frameworks in 2025',
    'data_source': 'near',
    'use_rler_reward': True,  # Enable RLER evaluation
    'ground_truth': {
        'rubrics': [
            {
                'description': 'Covers all major AI frameworks',
                'weight': 1.0
            },
            {
                'description': 'Includes performance comparisons',
                'weight': 0.8
            },
            {
                'description': 'Cites recent sources (2024-2025)',
                'weight': 0.6
            }
        ]
    }
}
```

### Response Format

Agents should structure responses as:

```xml
<think>
Planning my research approach...
I'll search for recent AI framework benchmarks.
</think>

<tool_call>{"name": "mcp__tavily__search", "arguments": {"query": "AI frameworks 2025 benchmarks"}}</tool_call>

<think>
Found good data. Now analyzing...
</think>

<answer>
Based on recent research [1][2], the top AI frameworks in 2025 are:

1. **PyTorch 2.5** - Leading in research [1]
2. **TensorFlow 3.0** - Strong production use [2]
...

[1] AI Benchmark Report 2025
[2] Framework Performance Study
</answer>
```

## Reward Components

### 1. Rubric Reward (50-60% weight)

Evaluates answer quality against custom rubrics using LLM judges.

#### Three Evaluation Modes:

**A. Custom Rubrics (Default)**

```python
'ground_truth': {
    'rubrics': [
        {
            'description': 'The answer is comprehensive',
            'weight': 1.0
        },
        {
            'description': 'The answer is accurate',
            'weight': 1.0
        }
    ]
}
```

**B. General Rubric**

```python
'ground_truth': {},
'rler_config': {
    'use_general_rubric': True  # Uses default research criteria
}
```

Default criteria:
- Comprehensiveness
- Thoroughness
- Factuality
- Coherence

**C. Likert Scale (1-10)**

```python
'rler_config': {
    'use_likert_rubric': True  # Simple 1-10 rating
}
```

#### Rubric Scoring Process:

1. LLM evaluates answer against each rubric (0-1 score)
2. Scores grouped by rubric key (MD5 hash of question + description)
3. Weighted average computed: `sum(score × weight) / sum(weight)`
4. All rubric evaluations run in parallel (async)

### 2. Citation Reward (20% weight)

Measures citation quality based on:
- Number of citations
- Coverage relative to answer length
- Ideal ratio: ~1 citation per 100 words

**Formula:**
```python
citation_ratio = num_citations / (answer_length / 100)
score = min(1.0, citation_ratio)
```

**Disable if not needed:**
```python
'rler_config': {
    'use_citation_reward': False  # Reweights to 60% rubric, 20% format, 20% search
}
```

### 3. Format Reward (20% weight)

Validates response structure:

| Format | Score |
|--------|-------|
| Both `<think>` and `<answer>` tags | 1.0 |
| Only `<answer>` tag | 0.5 |
| No tags | 0.0 |

### 4. Search Turns Reward (10-20% weight)

Scores research thoroughness via tool usage:

| Tool Calls | Score |
|------------|-------|
| 0 tools | 0.0 |
| 1 tool | 0.2 |
| 2 tools | 0.4 |
| 3 tools | 0.6 |
| 4 tools | 0.8 |
| 5+ tools | 1.0 |

**Counted tools:**
- `mcp__tavily__search`
- `browser`
- `ipython`
- `bash`
- Generic `<tool_call>`

## Complete Example

```python
# Full instance configuration
instance = {
    'instance_id': 'ai_frameworks_2025',
    'problem_statement': '''
        Research and compare the top 5 AI/ML frameworks in 2025.
        Include performance benchmarks, community adoption, and use cases.
        Create a comparison table.
    ''',
    'data_source': 'near',

    # Enable RLER evaluation
    'use_rler_reward': True,

    # Custom rubrics for this task
    'ground_truth': {
        'rubrics': [
            {
                'description': 'Identifies and describes top 5 AI frameworks',
                'weight': 1.5
            },
            {
                'description': 'Includes quantitative performance benchmarks',
                'weight': 1.0
            },
            {
                'description': 'Discusses community adoption metrics (GitHub stars, downloads)',
                'weight': 0.8
            },
            {
                'description': 'Covers diverse use cases for each framework',
                'weight': 0.7
            },
            {
                'description': 'Provides comparison table or visualization',
                'weight': 1.0
            },
            {
                'description': 'Cites recent sources from 2024-2025',
                'weight': 0.5
            }
        ]
    },

    # RLER configuration
    'rler_config': {
        'use_citation_reward': True,
        'use_likert_rubric': False,
        'use_general_rubric': False
    },

    # Optional: require specific output files
    'required_outputs': ['comparison_table.csv', 'report.md']
}
```

## Evaluation Results

The evaluation returns:

```python
{
    'resolved': True,  # True if reward > 0.5
    'reward': 0.78,    # Final weighted score
    'reward_breakdown': {
        'rubric_reward': 0.85,
        'citation_reward': 0.70,
        'format_reward': 1.0,
        'num_search_turns_reward': 0.80
    },
    'rubric_breakdown': {
        # Per-rubric scores
    },
    'outputs': {
        'report.md': '...',
        'comparison_table.csv': '...'
    }
}
```

## LLM Configuration

RLER uses an LLM judge for rubric evaluation. Configure via environment variables:

```bash
# LLM for rubric evaluation
export OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1"
export OPENAI_API_KEY="nvapi-your-key"
export OPENAI_MODEL="meta/llama-3.3-70b-instruct"

# Optional: LLM parameters
export OPENAI_TEMPERATURE="0.0"
export OPENAI_TOP_P="0.7"
export OPENAI_MAX_TOKENS="1024"
```

## Default Rubrics

If no rubrics provided, NeAR uses these defaults:

```python
default_rubrics = [
    {
        'description': 'The answer is comprehensive and covers all aspects',
        'weight': 1.0
    },
    {
        'description': 'The answer provides thorough analysis with depth',
        'weight': 1.0
    },
    {
        'description': 'The answer is factually accurate and well-supported',
        'weight': 1.0
    },
    {
        'description': 'The answer is well-organized and coherent',
        'weight': 0.5
    }
]
```

## Reward Weights

### With Citations (default)

| Component | Weight |
|-----------|--------|
| Rubric | 50% |
| Citation | 20% |
| Format | 20% |
| Search Turns | 10% |

### Without Citations

| Component | Weight |
|-----------|--------|
| Rubric | 60% |
| Format | 20% |
| Search Turns | 20% |

## Advanced: Evolving Rubrics

DR Tulu's original methodology includes "evolving rubrics" that adapt during training. To implement this:

1. **Track rubric performance** over time
2. **Add new rubrics** when model discovers new aspects
3. **Deprecate rubrics** that no longer discriminate
4. **Maintain rubric buffer** with persistent vs adaptive rubrics

**Example workflow:**

```python
# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Run agent
        results = agent.run(batch)

        # Evaluate with current rubrics
        rewards = evaluate_with_rler(results, rubrics)

        # Update rubrics based on performance
        rubrics = evolve_rubrics(
            rubrics=rubrics,
            agent_outputs=results,
            performance=rewards
        )

        # Train policy with rewards
        policy.update(rewards)
```

## Troubleshooting

### Low Rubric Scores

**Problem**: Rubric rewards consistently low

**Solutions:**
- Check if answer extraction is working (`<answer>` tags)
- Review rubric descriptions for clarity
- Ensure LLM judge has sufficient context
- Try `use_general_rubric=True` as baseline

### Citation Reward Always Zero

**Problem**: No citations detected

**Solutions:**
- Ensure agent includes citations like `[1]`, `[2]`
- Check if citation extraction regex works
- Consider disabling: `use_citation_reward=False`

### Format Reward Zero

**Problem**: Format not recognized

**Solutions:**
- Verify response includes `<think>` and `<answer>` tags
- Check system prompt instructs agent to use tags
- Review agent's actual output format

### Search Turns Reward Low

**Problem**: Tool usage not counted

**Solutions:**
- Verify tools are being called (check logs)
- Ensure tool names match patterns (browser, search, ipython, bash)
- Agent may need more iterations to use tools

## Performance Optimization

### Parallel Rubric Evaluation

All rubrics evaluate in parallel via `asyncio.gather()`:

```python
# Automatic parallelization
results = await asyncio.gather(
    evaluate_rubric_1(),
    evaluate_rubric_2(),
    evaluate_rubric_3(),
    ...
)
```

### Caching

For repeated evaluations, consider caching:

```python
# Cache rubric evaluations by (answer, rubric) hash
cache_key = hashlib.md5(f"{answer}||{rubric}".encode()).hexdigest()
if cache_key in rubric_cache:
    return rubric_cache[cache_key]
```

## References

- **DR Tulu Paper**: [arXiv:2511.19399](https://arxiv.org/abs/2511.19399)
- **DR Tulu Code**: [github.com/rlresearch/dr-tulu](https://github.com/rlresearch/dr-tulu)
- **RLER Methodology**: Reinforcement Learning with Evolving Rubrics for Deep Research

## Sources

- [DR Tulu: An open, end-to-end training recipe for long-form deep research | Ai2](https://allenai.org/blog/dr-tulu)
- [DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research](https://dr-tulu.github.io/)
- [DR Tulu Paper (PDF)](https://www.datocms-assets.com/64837/1763496622-dr_tulu_draft.pdf)
- [[2511.19399] DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research](https://arxiv.org/abs/2511.19399)
- [GitHub - rlresearch/dr-tulu](https://github.com/rlresearch/dr-tulu)

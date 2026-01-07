# NeAR Agent (NeMo Agent Runtime)

NeAR (NeMo Agent Runtime) integration for deep research tasks in ProRL-Agent-Server, with RLER (Reinforcement Learning with Evolving Rubrics) reward system from [DR Tulu](https://github.com/rlresearch/dr-tulu).

## Overview

NeAR provides enhanced research capabilities with:

- **5 Base Tools**: Browser (Playwright), IPython (stateful), Tavily search (MCP), bash, file editor
- **Specialized Filesystem**: `/workspace/` with notes.md, outputs/, assets/, mounted/
- **RLER Rewards**: Multi-component evaluation (rubrics, citations, format, search thoroughness)
- **Extensible Skills**: Easy integration of custom tools via 3 methods

## Quick Start

```python
# Create research task
instance = {
    'instance_id': 'research_task_1',
    'problem_statement': 'Research top AI frameworks in 2025',
    'data_source': 'near',  # Routes to NeARHandler
    'use_rler_reward': True,  # Enable RLER evaluation
    'ground_truth': {
        'rubrics': [
            {'description': 'Comprehensive coverage', 'weight': 1.0},
            {'description': 'Accurate information', 'weight': 1.0}
        ]
    }
}

# Run agent (through ProRL-Agent-Server)
# Agent will:
# 1. Use tools to research
# 2. Document findings in /workspace/notes.md
# 3. Generate outputs in /workspace/outputs/
# 4. Get evaluated via RLER rewards
```

## Architecture

### Handler (3-Stage Pipeline)

**`near_agent_handler.py`** - Implements `AgentHandler` interface

- `init()` - Initialize runtime, workspace, tools
- `run()` - Execute agent with CodeActAgent
- `eval()` - Evaluate with RLER rewards

### Core Logic

**`utils.py`** - 3-stage pipeline implementation

1. **`initialize_agents()`**
   - Creates NeAR container runtime
   - Enables all tools (browser, ipython, bash, editor, MCP)
   - Sets up workspace structure

2. **`run_agent()`**
   - Creates CodeActAgent + AgentController
   - Runs research loop
   - Collects outputs from `/workspace/outputs/`

3. **`evaluate_agent()`**
   - 4 evaluation modes:
     - RLER reward (dr-tulu)
     - Custom reward function
     - Output-based checking
     - Manual review

### Workspace Setup

**`workspace_setup.py`** - Filesystem initialization

Creates:
```
/workspace/
  ├── notes.md          # Research notes
  ├── assets/           # Resources
  ├── mounted/          # User mounts
  └── outputs/          # Final deliverables

/skills/                # Custom skills (initially empty)
```

### RLER Rewards

**`rler_reward.py`** - DR Tulu reward calculator

**Reward Components**:
1. **Rubric** (50-60%) - LLM-based quality evaluation
2. **Citation** (20%) - Citation coverage
3. **Format** (20%) - Response structure
4. **Search** (10-20%) - Tool usage thoroughness

See: `docs/near_rler_rewards.md` for detailed guide

## Container

**`containers/near-runtime/Dockerfile`**

- Base: `nikolaik/python-nodejs:python3.12-nodejs22`
- Tools: Playwright (Chromium), IPython, scientific packages
- Empty `/skills/` directory for future extensions

**Build**:
```bash
cd containers/near-runtime
./build.sh
```

## System Prompt

**`openhands/agenthub/codeact_agent/prompts/system_prompt_near.j2`**

Custom prompt for research tasks:
- Workspace structure documentation
- Tool usage guidelines
- Research workflow best practices
- Output organization instructions

## Registration

**`openhands/nvidia/__init__.py`**

```python
register_agent_handler(NeARHandler())

# Aliases
for near_name in ['near', 'research', 'deep_research']:
    add_name_mapping(near_name, 'near')
```

## Files

```
near_agent/
├── __init__.py                 # Package initialization
├── README.md                   # This file
├── near_agent_handler.py       # Handler class
├── utils.py                    # Core 3-stage pipeline
├── workspace_setup.py          # Filesystem initialization
└── rler_reward.py              # RLER reward calculator
```

## Documentation

- **Skills Guide**: `/docs/near_skills_guide.md` - How to add custom skills
- **RLER Rewards**: `/docs/near_rler_rewards.md` - RLER evaluation guide
- **Plan**: `~/.claude/plans/compressed-wobbling-waffle.md` - Implementation plan

## Environment Variables

```bash
# NeAR container image
export EVAL_DOCKER_IMAGE_PREFIX="nvidia/"

# LLM for RLER rubric evaluation
export OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1"
export OPENAI_API_KEY="nvapi-your-key"
export OPENAI_MODEL="meta/llama-3.3-70b-instruct"

# Tavily search (optional)
export TAVILY_API_KEY="tvly-your-key"
```

## Example: Complete Research Task

```python
instance = {
    'instance_id': 'ai_frameworks_comparison',
    'problem_statement': '''
        Research and compare PyTorch, TensorFlow, and JAX.
        Create a comparison table and written analysis.
    ''',
    'data_source': 'near',

    # RLER evaluation
    'use_rler_reward': True,
    'ground_truth': {
        'rubrics': [
            {'description': 'Covers all three frameworks', 'weight': 1.5},
            {'description': 'Includes performance benchmarks', 'weight': 1.0},
            {'description': 'Discusses pros/cons', 'weight': 1.0},
            {'description': 'Creates comparison table', 'weight': 1.0},
            {'description': 'Cites recent sources', 'weight': 0.5}
        ]
    },
    'rler_config': {
        'use_citation_reward': True,
        'use_general_rubric': False
    },

    # Optional output requirements
    'required_outputs': ['comparison_table.csv', 'analysis.md']
}
```

**Agent Process**:
1. Plans research approach in `<think>` tags
2. Uses `mcp__tavily__search` to find information
3. Uses `browser` to visit official docs
4. Uses `ipython` to analyze data
5. Documents findings in `/workspace/notes.md`
6. Creates outputs:
   - `comparison_table.csv` - Feature comparison
   - `analysis.md` - Written analysis
7. Evaluation via RLER:
   - Rubric: LLM evaluates each criterion
   - Citation: Checks coverage
   - Format: Validates `<think>` and `<answer>` tags
   - Search: Counts tool usage

## Extending with Custom Skills

See `/docs/near_skills_guide.md` for complete guide.

**Quick example** (OCR skill):

```python
# 1. Create tool definition
# openhands/nvidia/near_agent/tools/ocr.py
OCRTool = ChatCompletionToolParam(...)

# 2. Add to custom agent
class NeARAgent(CodeActAgent):
    def _get_tools(self):
        tools = super()._get_tools()
        tools.append(OCRTool)
        return tools

# 3. Add dependency to Dockerfile
RUN apt-get install tesseract-ocr
```

## Troubleshooting

### Container Build Issues
```bash
# Verify base image
docker pull nikolaik/python-nodejs:python3.12-nodejs22

# Check build logs
cd containers/near-runtime
docker build -t test . 2>&1 | tee build.log
```

### RLER Rewards Low
- Check agent response format (`<think>`, `<answer>` tags)
- Verify LLM judge is configured (OPENAI_API_KEY)
- Review rubric descriptions for clarity
- Try `use_general_rubric=True` as baseline

### No Outputs Generated
- Check `/workspace/outputs/` in container
- Verify agent used `finish` action
- Review agent logs for errors

## Performance

- **Parallel Tool Execution**: OpenHands supports parallel actions
- **Async RLER Evaluation**: All rubrics evaluated in parallel
- **Resource Allocation**: 2x resource factor for research tasks

## References

- **DR Tulu**: https://github.com/rlresearch/dr-tulu
- **DR Tulu Paper**: https://arxiv.org/abs/2511.19399
- **NeAR POC**: https://github.com/NVIDIA/near (conceptual reference)

## License

Copyright 2025 NVIDIA CORPORATION & AFFILIATES
SPDX-License-Identifier: Apache-2.0

# NeAR Skills Injection Guide

This guide explains how to add custom skills to the NeAR agent for specialized tasks beyond the base tools (browser, ipython, bash, editor, web search).

## Overview

NeAR agents start with a foundational set of research tools. The `/skills/` directory is initially **EMPTY**, providing a clean starting point. You can extend the agent's capabilities by adding custom skills using one of three methods.

## Workspace Structure

```
/workspace/
  ├── notes.md          # Research notes and findings
  ├── assets/           # Resources and data files
  ├── mounted/          # User-provided filesystem mounts (read-only)
  └── outputs/          # Final deliverables

/skills/                # Custom skills directory (initially empty)
```

## Initial State

- **Base Tools**: browser, ipython, bash, str_replace_editor, mcp__tavily__search
- **Skills Directory**: `/skills/` exists but is empty
- **No Dependencies**: No NeAR POC components are required

## Three Methods for Adding Skills

### Method 1: Custom Tool (Recommended for NeAR-specific skills)

**Best for**: Lightweight skills specific to NeAR, minimal setup required

**Process**:

1. **Create tool definition**

```python
# File: openhands/nvidia/near_agent/tools/ocr.py
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

OCRTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='ocr_extract',
        description='Extract text from images using OCR (Tesseract)',
        parameters={
            'type': 'object',
            'properties': {
                'image_path': {
                    'type': 'string',
                    'description': 'Path to image file in workspace',
                },
                'language': {
                    'type': 'string',
                    'description': 'OCR language code (default: eng)',
                    'default': 'eng',
                },
            },
            'required': ['image_path'],
        },
    ),
)
```

2. **Create custom agent class**

```python
# File: openhands/nvidia/near_agent/near_agent.py
from openhands.agenthub.codeact_agent import CodeActAgent
from openhands.nvidia.near_agent.tools.ocr import OCRTool

class NeARAgent(CodeActAgent):
    """Custom NeAR agent with additional tools."""

    def _get_tools(self) -> list['ChatCompletionToolParam']:
        tools = super()._get_tools()  # Get base tools
        tools.append(OCRTool)  # Add OCR tool
        return tools
```

3. **Implement action handler**

```python
# File: openhands/nvidia/near_agent/actions.py
from openhands.events.action import Action
from openhands.events.action import CmdRunAction

class OCRAction(Action):
    """Action to extract text from images using OCR."""

    image_path: str
    language: str = 'eng'

    def run(self, runtime) -> str:
        # Execute OCR using Tesseract in container
        cmd = f"tesseract {self.image_path} stdout -l {self.language}"
        result = runtime.run_action(CmdRunAction(command=cmd))
        return result.content
```

4. **Add dependencies to container**

```dockerfile
# File: containers/near-runtime/Dockerfile
# Add to system packages section:
RUN apt-get update && apt-get install -y tesseract-ocr && \
    pip install pytesseract
```

5. **Update system prompt** (optional)

Add OCR tool description to `openhands/agenthub/codeact_agent/prompts/system_prompt_near.j2`:

```jinja2
6. **ocr_extract** - Extract text from images
   * Supports multiple languages (default: English)
   * Use for document digitization and image text extraction
```

6. **Rebuild container**

```bash
cd containers/near-runtime
./build.sh
```

### Method 2: MCP Server (Recommended for reusable/external skills)

**Best for**: Skills shared across multiple agents, external services, reusable components

**Process**:

1. **Create MCP server implementation**

```python
# File: skills/ocr_mcp_server.py
import asyncio
from mcp.server import MCPServer
from mcp.types import Tool, TextContent

server = MCPServer()

@server.tool()
async def ocr_extract(image_path: str, language: str = 'eng') -> str:
    """Extract text from image using Tesseract OCR."""
    import subprocess
    result = subprocess.run(
        ['tesseract', image_path, 'stdout', '-l', language],
        capture_output=True,
        text=True
    )
    return result.stdout

if __name__ == '__main__':
    asyncio.run(server.run())
```

2. **Add MCP server to configuration**

Update `openhands/nvidia/near_agent/utils.py` in the `initialize_agents()` function:

```python
# In initialize_agents(), after creating runtime
extra_stdio_servers = [
    MCPStdioServerConfig(
        name='ocr',
        command='python',
        args=['/path/to/skills/ocr_mcp_server.py'],
        env={'TESSERACT_PATH': '/usr/bin/tesseract'},
    )
]

# Get updated MCP config
runtime.get_mcp_config(extra_stdio_servers)
```

3. **The skill is automatically available**

Agent can now use `mcp__ocr__ocr_extract` tool:

```python
{
    "name": "mcp__ocr__ocr_extract",
    "arguments": {
        "image_path": "/workspace/document.png",
        "language": "eng"
    }
}
```

### Method 3: Container-based Skills (For complex dependencies)

**Best for**: Skills with heavy system dependencies, complex installations, or large binaries

**Process**:

1. **Add dependencies to Dockerfile**

```dockerfile
# File: containers/near-runtime/Dockerfile
# Add Tesseract OCR with multiple language packs
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-spa && \
    pip install pytesseract pillow
```

2. **Create skill implementation in container**

```python
# File: containers/near-runtime/skills/ocr/ocr_skill.py
import pytesseract
from PIL import Image
import json
import sys

def extract_text(image_path, language='eng'):
    """Extract text from image using Tesseract."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=language)
    return text

if __name__ == '__main__':
    # Read parameters from stdin
    params = json.loads(sys.stdin.read())
    text = extract_text(params['image_path'], params.get('language', 'eng'))
    print(json.dumps({'text': text}))
```

3. **Add skill to container build**

```dockerfile
# File: containers/near-runtime/Dockerfile
# Copy skill implementation
COPY skills/ocr/ /skills/ocr/
```

4. **Create wrapper tool**

```python
# File: openhands/nvidia/near_agent/tools/ocr_container.py
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

OCRContainerTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='ocr_container_extract',
        description='Extract text from images using containerized OCR',
        parameters={
            'type': 'object',
            'properties': {
                'image_path': {'type': 'string', 'description': 'Path to image'},
                'language': {'type': 'string', 'description': 'Language code'},
            },
            'required': ['image_path'],
        },
    ),
)

# Implementation calls the container skill
def execute_ocr_container(runtime, image_path, language='eng'):
    import json
    params = json.dumps({'image_path': image_path, 'language': language})
    cmd = f"echo '{params}' | python /skills/ocr/ocr_skill.py"
    result = runtime.run_action(CmdRunAction(command=cmd))
    return json.loads(result.content)['text']
```

## Comparison of Methods

| Feature | Method 1: Custom Tool | Method 2: MCP Server | Method 3: Container-based |
|---------|----------------------|---------------------|---------------------------|
| **Setup Complexity** | Low | Medium | Medium-High |
| **Reusability** | NeAR-specific | Cross-agent | NeAR-specific |
| **Dependencies** | Simple | Medium | Complex |
| **Container Rebuild** | Required | Not required | Required |
| **Best For** | Quick additions | Shared skills | Heavy dependencies |

## Example: Complete OCR Skill Implementation

Here's a complete example using **Method 1** (recommended for most cases):

### Step 1: Create Tool Definition

```python
# openhands/nvidia/near_agent/tools/ocr.py
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

OCRTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='ocr_extract',
        description='Extract text from images using Tesseract OCR. Supports multiple languages.',
        parameters={
            'type': 'object',
            'properties': {
                'image_path': {
                    'type': 'string',
                    'description': 'Path to image file (relative to /workspace/)',
                },
                'language': {
                    'type': 'string',
                    'description': 'OCR language code (eng, fra, spa, etc.)',
                    'default': 'eng',
                },
                'output_format': {
                    'type': 'string',
                    'enum': ['text', 'json', 'hocr'],
                    'description': 'Output format',
                    'default': 'text',
                },
            },
            'required': ['image_path'],
        },
    ),
)
```

### Step 2: Extend Container

```dockerfile
# containers/near-runtime/Dockerfile
# Add after the existing RUN apt-get install section:
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-spa \
# Add after the uv pip install section:
    pytesseract \
```

### Step 3: Create Agent Extension

```python
# openhands/nvidia/near_agent/near_agent.py
from openhands.agenthub.codeact_agent import CodeActAgent
from openhands.nvidia.near_agent.tools.ocr import OCRTool

class NeARAgent(CodeActAgent):
    """NeAR agent with OCR capabilities."""

    VERSION = '1.0'

    def _get_tools(self) -> list['ChatCompletionToolParam']:
        """Get all tools including OCR."""
        tools = super()._get_tools()
        tools.append(OCRTool)
        return tools
```

### Step 4: Implement Action Handler

```python
# openhands/nvidia/near_agent/actions.py
from openhands.events.action import Action
from openhands.events.action import CmdRunAction
from pydantic import Field

class OCRAction(Action):
    """Extract text from images using OCR."""

    image_path: str = Field(description="Path to image file")
    language: str = Field(default='eng', description="OCR language code")
    output_format: str = Field(default='text', description="Output format")

    async def run(self, runtime) -> str:
        """Execute OCR on the image."""
        format_flag = {
            'text': '',
            'json': '--json',
            'hocr': 'hocr',
        }.get(self.output_format, '')

        cmd = f"tesseract {self.image_path} stdout -l {self.language} {format_flag}"
        result = runtime.run_action(CmdRunAction(command=cmd))

        if result.exit_code != 0:
            return f"Error: {result.content}"

        return result.content
```

### Step 5: Update System Prompt

Edit `openhands/agenthub/codeact_agent/prompts/system_prompt_near.j2`:

```jinja2
6. **ocr_extract** - Optical Character Recognition
   * Extract text from images and documents
   * Supports multiple languages (eng, fra, spa, etc.)
   * Output formats: plain text, JSON, HOCR
   * Usage: Digitize documents, extract text from screenshots
```

### Step 6: Build and Test

```bash
# Build container
cd containers/near-runtime
./build.sh

# Test OCR installation
docker run --rm nvidia/near-runtime:latest tesseract --version

# Test with sample image
docker run --rm -v $(pwd)/test.png:/workspace/test.png nvidia/near-runtime:latest \
  tesseract /workspace/test.png stdout
```

## Best Practices

### Skill Design
- **Single Responsibility**: Each skill should do one thing well
- **Clear Interface**: Use descriptive parameter names and documentation
- **Error Handling**: Return informative error messages
- **Language Support**: Consider internationalization for text-based skills

### Testing
- **Unit Tests**: Test skill logic independently
- **Integration Tests**: Test with actual runtime
- **Container Tests**: Verify dependencies are installed correctly

### Documentation
- **README.md**: Provide usage examples for each skill
- **Parameter Descriptions**: Document all parameters clearly
- **Examples**: Include common use cases

### Security
- **Input Validation**: Sanitize all user inputs
- **Path Safety**: Prevent directory traversal attacks
- **Resource Limits**: Set timeouts for long-running operations
- **Secrets Management**: Use environment variables, not hardcoded secrets

## Troubleshooting

### Skill Not Available
**Problem**: Agent doesn't recognize the new skill

**Solutions**:
1. Verify tool is added to `_get_tools()` method
2. Check tool name matches action handler
3. Rebuild container if dependencies were added
4. Restart runtime to pick up changes

### Container Build Fails
**Problem**: Docker build fails with dependency errors

**Solutions**:
1. Check package names are correct for your base image
2. Verify apt repositories are updated (`apt-get update`)
3. Check for conflicting package versions
4. Review build logs for specific error messages

### Runtime Errors
**Problem**: Skill executes but returns errors

**Solutions**:
1. Check dependencies are installed: `docker run --rm <image> <command> --version`
2. Verify file paths are absolute and accessible
3. Check environment variables are set correctly
4. Review action handler implementation for bugs

## Support and Resources

- **OpenHands Documentation**: Tool creation patterns
- **MCP Protocol**: https://modelcontextprotocol.io/
- **ProRL-Agent-Server**: Review existing handlers for patterns

## Examples in Production

### Image Processing Pipeline
Combining multiple skills for document processing:

```python
# 1. OCR skill to extract text
ocr_extract(image_path="/workspace/document.png")

# 2. NLP skill to analyze extracted text
analyze_text(text=extracted_text)

# 3. Translation skill for multilingual support
translate(text=analyzed_text, target_lang="es")
```

### Data Enrichment Workflow
Using MCP servers for external data:

```python
# 1. Web search for context
mcp__tavily__search(query="company name")

# 2. API lookup skill (custom MCP)
mcp__company_api__get_info(company_id="12345")

# 3. Database query skill (custom MCP)
mcp__postgres__query(sql="SELECT * FROM companies WHERE id=12345")
```

---

For questions or contributions, please file an issue on the ProRL-Agent-Server repository.

# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""
RLER (Reinforcement Learning with Evolving Rubrics) Reward Calculator
Replicates the exact reward calculation from dr-tulu: https://github.com/rlresearch/dr-tulu

Reward Components:
1. Rubric Reward (50% or 60% weight) - LLM-based evaluation against rubrics
2. Citation Reward (20% weight, optional) - Quality of citations
3. Format Reward (20% weight) - Response structure validation
4. Search Turns Reward (10% or 20% weight) - Number of research actions

Reference: DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research
https://arxiv.org/abs/2511.19399
"""

import asyncio
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from openhands.nvidia.logger import nvidia_logger as logger

# Default reward weights (with citations)
REWARD_WEIGHTS = {
    'rubric_reward': 0.5,  # 50%
    'citation_reward': 0.2,  # 20%
    'format_reward': 0.2,  # 20%
    'num_search_turns_reward': 0.1,  # 10%
}

# Alternative weights (without citations)
REWARD_WEIGHTS_WITHOUT_CITATION = {
    'rubric_reward': 0.6,  # 60%
    'format_reward': 0.2,  # 20%
    'num_search_turns_reward': 0.2,  # 20%
}


class RLERRewardCalculator:
    """RLER reward calculator implementing dr-tulu methodology."""

    def __init__(
        self,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        use_citation_reward: bool = True,
        use_likert_rubric: bool = False,
        use_general_rubric: bool = False,
    ):
        """Initialize RLER reward calculator.

        Args:
            llm_base_url: LLM API base URL for rubric evaluation
            llm_api_key: LLM API key
            llm_model: LLM model name for evaluation
            use_citation_reward: Include citation reward component
            use_likert_rubric: Use 1-10 Likert scale instead of rubrics
            use_general_rubric: Use general rubric instead of custom rubrics
        """
        self.llm_base_url = llm_base_url or os.getenv(
            'OPENAI_BASE_URL', 'https://integrate.api.nvidia.com/v1'
        )
        self.llm_api_key = llm_api_key or os.getenv('OPENAI_API_KEY', '')
        self.llm_model = llm_model or os.getenv(
            'OPENAI_MODEL', 'meta/llama-3.3-70b-instruct'
        )
        self.use_citation_reward = use_citation_reward
        self.use_likert_rubric = use_likert_rubric
        self.use_general_rubric = use_general_rubric

        # Initialize async OpenAI client
        self.client = AsyncOpenAI(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )

        # Select weight configuration
        self.weights = (
            REWARD_WEIGHTS if use_citation_reward else REWARD_WEIGHTS_WITHOUT_CITATION
        )

    async def compute_reward(
        self,
        response: str,
        ground_truth: Dict[str, Any],
        question: str,
    ) -> Dict[str, Any]:
        """Compute complete RLER reward for a research response.

        Args:
            response: Agent's full response including think tags, answer, citations
            ground_truth: Dict containing rubrics, expected content, etc.
            question: Original research question

        Returns:
            Dict with reward score and component breakdown
        """
        # Extract answer from response
        answer = self._extract_answer(response)
        if not answer:
            logger.warning('Failed to extract answer from response')
            return self._zero_reward()

        # Compute all reward components in parallel
        tasks = []

        # 1. Rubric reward (async LLM evaluation)
        rubric_task = self._compute_rubric_reward(answer, ground_truth, question)
        tasks.append(('rubric', rubric_task))

        # 2. Citation reward (if enabled)
        if self.use_citation_reward:
            citations = self._extract_citations(response)
            citation_task = self._compute_citation_reward(question, answer, citations)
            tasks.append(('citation', citation_task))

        # 3. Format reward (synchronous, but wrapped in async)
        format_reward = self._compute_format_reward(response)

        # 4. Search turns reward (synchronous)
        search_turns_reward = self._score_num_search_turns(response)

        # Execute async tasks
        results = {}
        if tasks:
            task_results = await asyncio.gather(
                *[task for _, task in tasks], return_exceptions=True
            )
            for (name, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    logger.error(f'Error computing {name} reward: {result}')
                    results[f'{name}_reward'] = 0.0
                else:
                    results[f'{name}_reward'] = result

        # Add synchronous rewards
        results['format_reward'] = format_reward
        results['num_search_turns_reward'] = search_turns_reward

        # Compute weighted final reward
        final_reward = 0.0
        for key, weight in self.weights.items():
            if key in results:
                final_reward += weight * results[key]

        return {
            'reward': final_reward,
            'log_values': results,
            'rubric_breakdown': results.get('rubric_breakdown', {}),
        }

    async def _compute_rubric_reward(
        self, answer: str, ground_truth: Dict[str, Any], question: str
    ) -> float:
        """Compute rubric-based reward using LLM evaluation.

        Args:
            answer: Extracted answer text
            ground_truth: Dict with rubrics or expected content
            question: Original question

        Returns:
            Normalized rubric score (0-1)
        """
        if self.use_likert_rubric:
            # Likert scale mode (1-10 rating)
            return await self._evaluate_likert_scale(answer, question)
        elif self.use_general_rubric:
            # General rubric mode
            return await self._evaluate_general_rubric(answer, question)
        else:
            # Custom rubrics mode (default)
            rubrics = ground_truth.get('rubrics', [])
            if not rubrics:
                logger.warning('No rubrics provided in ground_truth')
                return 0.0
            return await self._evaluate_custom_rubrics(answer, question, rubrics)

    async def _evaluate_likert_scale(self, answer: str, question: str) -> float:
        """Evaluate using 1-10 Likert scale.

        Args:
            answer: Answer text
            question: Question

        Returns:
            Normalized score (0-1)
        """
        prompt = f"""Rate the quality of this research answer on a scale from 1 to 10.

Question: {question}

Answer: {answer}

Provide a single integer rating between 1 and 10, where:
- 1-3: Poor quality, incomplete, inaccurate
- 4-6: Acceptable, some gaps or issues
- 7-8: Good quality, mostly complete and accurate
- 9-10: Excellent, comprehensive and highly accurate

Return only a JSON object with format: {{"rating": <integer>}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            content = response.choices[0].message.content or '{}'
            result = json.loads(content.strip())
            rating = result.get('rating', 1)
            # Normalize to 0-1
            return (rating - 1) / 9.0
        except Exception as e:
            logger.error(f'Likert evaluation failed: {e}')
            return 0.0

    async def _evaluate_general_rubric(self, answer: str, question: str) -> float:
        """Evaluate using general research quality rubric.

        Args:
            answer: Answer text
            question: Question

        Returns:
            Normalized score (0-1)
        """
        general_rubric = """Evaluate the answer on these criteria (each 0-1):
1. Comprehensiveness: Does it cover all aspects of the question?
2. Thoroughness: Is the analysis deep and detailed?
3. Factuality: Are the facts and claims accurate?
4. Coherence: Is the answer well-organized and logical?"""

        prompt = f"""Evaluate this research answer using the following rubric:

{general_rubric}

Question: {question}

Answer: {answer}

Return a JSON object with scores for each criterion:
{{"comprehensiveness": <0-1>, "thoroughness": <0-1>, "factuality": <0-1>, "coherence": <0-1>}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            content = response.choices[0].message.content or '{}'
            scores = json.loads(content.strip())
            # Average all criteria
            avg_score = sum(scores.values()) / len(scores)
            return avg_score
        except Exception as e:
            logger.error(f'General rubric evaluation failed: {e}')
            return 0.0

    async def _evaluate_custom_rubrics(
        self, answer: str, question: str, rubrics: List[Dict[str, Any]]
    ) -> float:
        """Evaluate using custom rubrics with parallel evaluation.

        Args:
            answer: Answer text
            question: Question
            rubrics: List of rubric dicts with 'description' and optional 'weight'

        Returns:
            Weighted average score (0-1)
        """
        # Group rubrics by key (MD5 hash of question + description)
        rubric_groups = {}
        for rubric in rubrics:
            rubric_key = self._create_rubric_key(question, rubric['description'])
            if rubric_key not in rubric_groups:
                rubric_groups[rubric_key] = {'rubrics': [], 'scores': [], 'weights': []}
            rubric_groups[rubric_key]['rubrics'].append(rubric)

        # Evaluate each rubric group in parallel
        tasks = []
        for rubric_key, group in rubric_groups.items():
            # Take first rubric from group (they have same description)
            rubric = group['rubrics'][0]
            task = self._evaluate_single_rubric(answer, question, rubric)
            tasks.append((rubric_key, task))

        # Gather results
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Compute weighted average
        total_score = 0.0
        total_weight = 0.0

        for (rubric_key, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f'Rubric evaluation failed for {rubric_key}: {result}')
                continue

            group = rubric_groups[rubric_key]
            for rubric in group['rubrics']:
                weight = rubric.get('weight', 1.0)
                group['scores'].append(result)
                group['weights'].append(weight)
                total_score += result * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_score / total_weight

    async def _evaluate_single_rubric(
        self, answer: str, question: str, rubric: Dict[str, Any]
    ) -> float:
        """Evaluate answer against a single rubric.

        Args:
            answer: Answer text
            question: Question
            rubric: Rubric dict with 'description'

        Returns:
            Score (0-1)
        """
        rubric_desc = rubric.get('description', '')

        prompt = f"""Evaluate this answer against the following rubric:

Rubric: {rubric_desc}

Question: {question}

Answer: {answer}

Rate how well the answer meets this rubric criterion on a scale from 0 to 1:
- 0.0: Does not meet criterion at all
- 0.5: Partially meets criterion
- 1.0: Fully meets criterion

Return only a JSON object: {{"score": <0-1 float>}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            content = response.choices[0].message.content or '{}'
            result = json.loads(content.strip())
            return float(result.get('score', 0.0))
        except Exception as e:
            logger.error(f'Single rubric evaluation failed: {e}')
            return 0.0

    async def _compute_citation_reward(
        self, question: str, answer: str, citations: List[str]
    ) -> float:
        """Compute citation quality reward.

        Args:
            question: Question
            answer: Answer text
            citations: List of citation identifiers

        Returns:
            Citation score (0-1)
        """
        if not citations:
            return 0.0

        # Simple citation quality heuristic
        # TODO: Can be enhanced with LLM-based citation verification
        num_citations = len(citations)
        answer_length = len(answer.split())

        # Ideal: 1 citation per ~100 words
        ideal_ratio = answer_length / 100.0
        citation_ratio = num_citations / max(ideal_ratio, 1.0)

        # Normalize to 0-1 (clip at 1.0)
        score = min(1.0, citation_ratio)

        return score

    def _compute_format_reward(self, response: str) -> float:
        """Compute format reward based on response structure.

        Args:
            response: Full response

        Returns:
            Format score (0-1)
        """
        # Check for required tags: <think>, <answer>
        has_think = '<think>' in response and '</think>' in response
        has_answer = '<answer>' in response and '</answer>' in response

        # Both required for full score
        if has_think and has_answer:
            return 1.0
        elif has_answer:
            return 0.5  # Partial credit for answer only
        else:
            return 0.0

    def _score_num_search_turns(self, response: str) -> float:
        """Score number of research/search actions taken.

        Args:
            response: Full response

        Returns:
            Search turns score (0-1)
        """
        # Count tool calls (browser, search, ipython)
        tool_patterns = [
            r'<tool_call>',
            r'mcp__tavily__search',
            r'browser',
            r'ipython',
            r'bash',
        ]

        tool_count = 0
        for pattern in tool_patterns:
            tool_count += len(re.findall(pattern, response, re.IGNORECASE))

        # Normalize: 1-5 tools = 0.2-1.0, >5 = 1.0
        if tool_count == 0:
            return 0.0
        elif tool_count <= 5:
            return tool_count * 0.2
        else:
            return 1.0

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response.

        Args:
            response: Full response

        Returns:
            Extracted answer text
        """
        # Extract text between <answer> tags
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: return full response
        return response.strip()

    def _extract_citations(self, response: str) -> List[str]:
        """Extract citations from response.

        Args:
            response: Full response

        Returns:
            List of citation identifiers
        """
        # Extract citation patterns like [1], [2], etc.
        citations = re.findall(r'\[(\d+)\]', response)
        return list(set(citations))  # Unique citations

    def _create_rubric_key(self, question: str, rubric_desc: str) -> str:
        """Create unique key for rubric grouping.

        Args:
            question: Question
            rubric_desc: Rubric description

        Returns:
            MD5 hash key
        """
        combined = f'{question}||{rubric_desc}'
        return hashlib.md5(combined.encode()).hexdigest()

    def _zero_reward(self) -> Dict[str, Any]:
        """Return zero reward result.

        Returns:
            Zero reward dict
        """
        return {
            'reward': 0.0,
            'log_values': {
                'rubric_reward': 0.0,
                'citation_reward': 0.0,
                'format_reward': 0.0,
                'num_search_turns_reward': 0.0,
            },
        }


# Convenience function for integration
async def compute_rler_reward(
    response: str,
    ground_truth: Dict[str, Any],
    question: str,
    use_citation_reward: bool = True,
    use_likert_rubric: bool = False,
    use_general_rubric: bool = False,
) -> Dict[str, Any]:
    """Compute RLER reward for a research response.

    Args:
        response: Agent's full response
        ground_truth: Dict with rubrics and expected content
        question: Original question
        use_citation_reward: Include citation reward
        use_likert_rubric: Use Likert scale
        use_general_rubric: Use general rubric

    Returns:
        Dict with reward score and breakdown
    """
    calculator = RLERRewardCalculator(
        use_citation_reward=use_citation_reward,
        use_likert_rubric=use_likert_rubric,
        use_general_rubric=use_general_rubric,
    )

    return await calculator.compute_reward(response, ground_truth, question)

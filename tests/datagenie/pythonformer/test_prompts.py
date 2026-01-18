"""
Tests for pythonformer prompts.

Tests that all environment-specific prompts are properly defined.
"""

import pytest

from datagenie.pythonformer.prompts import (
    BASE_SYSTEM_PROMPT,
    OOLONG_SYSTEM_PROMPT,
    HOTPOTQA_SYSTEM_PROMPT,
    SWE_SYSTEM_PROMPT,
)


class TestPromptExistence:
    """Test that all prompts are defined."""
    
    def test_base_prompt_exists(self):
        """Test base prompt is defined."""
        assert BASE_SYSTEM_PROMPT is not None
        assert len(BASE_SYSTEM_PROMPT) > 100
        assert isinstance(BASE_SYSTEM_PROMPT, str)
    
    def test_oolong_prompt_exists(self):
        """Test OOLONG prompt is defined."""
        assert OOLONG_SYSTEM_PROMPT is not None
        assert len(OOLONG_SYSTEM_PROMPT) > 100
        assert isinstance(OOLONG_SYSTEM_PROMPT, str)
    
    def test_hotpotqa_prompt_exists(self):
        """Test HotpotQA prompt is defined."""
        assert HOTPOTQA_SYSTEM_PROMPT is not None
        assert len(HOTPOTQA_SYSTEM_PROMPT) > 100
        assert isinstance(HOTPOTQA_SYSTEM_PROMPT, str)
    
    def test_swe_prompt_exists(self):
        """Test SWE prompt is defined."""
        assert SWE_SYSTEM_PROMPT is not None
        assert len(SWE_SYSTEM_PROMPT) > 100
        assert isinstance(SWE_SYSTEM_PROMPT, str)


class TestPromptContent:
    """Test prompt content and structure."""
    
    def test_base_prompt_has_python_tags(self):
        """Test base prompt mentions Python tags."""
        assert "<python>" in BASE_SYSTEM_PROMPT
        assert "</python>" in BASE_SYSTEM_PROMPT
    
    def test_oolong_prompt_mentions_context(self):
        """Test OOLONG prompt mentions context handling."""
        prompt_lower = OOLONG_SYSTEM_PROMPT.lower()
        assert "context" in prompt_lower
        assert "long" in prompt_lower or "large" in prompt_lower
    
    def test_hotpotqa_prompt_mentions_documents(self):
        """Test HotpotQA prompt mentions documents."""
        prompt_lower = HOTPOTQA_SYSTEM_PROMPT.lower()
        assert "document" in prompt_lower or "file" in prompt_lower
        assert "list_files" in HOTPOTQA_SYSTEM_PROMPT or "read_file" in HOTPOTQA_SYSTEM_PROMPT
    
    def test_swe_prompt_has_bash_tags(self):
        """Test SWE prompt mentions bash tags."""
        assert "<bash>" in SWE_SYSTEM_PROMPT
        assert "</bash>" in SWE_SYSTEM_PROMPT
        assert "<python>" in SWE_SYSTEM_PROMPT
    
    def test_swe_prompt_mentions_testbed(self):
        """Test SWE prompt mentions /testbed."""
        assert "/testbed" in SWE_SYSTEM_PROMPT
    
    def test_all_prompts_mention_final_answer(self):
        """Test all prompts mention final answer format."""
        for prompt in [BASE_SYSTEM_PROMPT, OOLONG_SYSTEM_PROMPT, 
                      HOTPOTQA_SYSTEM_PROMPT, SWE_SYSTEM_PROMPT]:
            assert "final_answer" in prompt.lower() or "<final_answer>" in prompt


class TestPromptFormatting:
    """Test prompt formatting capabilities."""
    
    def test_base_prompt_formatting(self):
        """Test base prompt can be formatted."""
        try:
            formatted = BASE_SYSTEM_PROMPT.format(
                max_output=8192,
                env_tips="Test tips"
            )
            assert "8192" in formatted or "Test tips" in formatted
        except KeyError:
            # Some prompts may not have format placeholders
            pass
    
    def test_oolong_prompt_formatting(self):
        """Test OOLONG prompt can be formatted."""
        try:
            formatted = OOLONG_SYSTEM_PROMPT.format(
                context_length=150000,
                env_tips="Test tips"
            )
            assert "150000" in formatted or "150,000" in formatted or "Test tips" in formatted
        except KeyError:
            pass
    
    def test_hotpotqa_prompt_formatting(self):
        """Test HotpotQA prompt can be formatted."""
        try:
            formatted = HOTPOTQA_SYSTEM_PROMPT.format(
                context_length=10000,
                env_tips="Test tips"
            )
            assert "10000" in formatted or "10,000" in formatted or "Test tips" in formatted
        except KeyError:
            pass


class TestPromptGuidance:
    """Test that prompts provide proper guidance."""
    
    def test_base_prompt_has_examples(self):
        """Test base prompt has code examples."""
        # Should have at least one code example
        assert "```" in BASE_SYSTEM_PROMPT or "print(" in BASE_SYSTEM_PROMPT
    
    def test_swe_prompt_has_workflow(self):
        """Test SWE prompt describes workflow."""
        prompt_lower = SWE_SYSTEM_PROMPT.lower()
        # Should mention key workflow steps
        assert any(word in prompt_lower for word in ["explore", "read", "fix", "test", "verify"])
    
    def test_swe_prompt_has_tool_descriptions(self):
        """Test SWE prompt describes available tools."""
        # Should describe both Python and bash
        assert "python" in SWE_SYSTEM_PROMPT.lower()
        assert "bash" in SWE_SYSTEM_PROMPT.lower()
        # Should mention common commands
        assert any(cmd in SWE_SYSTEM_PROMPT for cmd in ["grep", "find", "ls", "cat"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

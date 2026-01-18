from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class Turn(BaseModel):
    """A single turn in the conversation."""
    role: str  # "user", "assistant", "tool"
    content: str
    code: Optional[str] = None

class TrajectoryResult(BaseModel):
    """Result of generating a trajectory."""
    success: bool
    final_answer: str
    turns: List[Turn]
    num_code_blocks: int = 0
    system_prompt: str = ""  # The system prompt used for this trajectory
    # Reward tracking for RL/filtering
    total_reward: float = 0.0  # Cumulative reward across all steps
    step_rewards: List[float] = Field(default_factory=list)  # Per-step rewards
    num_errors: int = 0  # Count of execution errors

class PipelineStats(BaseModel):
    """Track pipeline statistics."""
    total: int = 0
    successful: int = 0
    failed: int = 0
    answers_correct: int = 0
    answers_incorrect: int = 0
    answers_unknown: int = 0  # No expected answer to compare
    avg_turns: float = 0.0
    avg_code_blocks: float = 0.0
    total_turns: int = 0
    total_code_blocks: int = 0
    
    def record(self, result: TrajectoryResult, answer_correct: Optional[bool] = None) -> None:
        """Record a result."""
        self.total += 1
        if result.success:
            self.successful += 1
        else:
            self.failed += 1
        
        # Track answer correctness
        if answer_correct is True:
            self.answers_correct += 1
        elif answer_correct is False:
            self.answers_incorrect += 1
        else:
            self.answers_unknown += 1
        
        self.total_turns += len(result.turns)
        self.total_code_blocks += result.num_code_blocks
        self.avg_turns = self.total_turns / self.total if self.total > 0 else 0.0
        self.avg_code_blocks = self.total_code_blocks / self.total if self.total > 0 else 0.0
    
    def report(self) -> str:
        """Generate statistics report."""
        success_rate = (self.successful / self.total * 100) if self.total > 0 else 0
        validated = self.answers_correct + self.answers_incorrect
        accuracy = (self.answers_correct / validated * 100) if validated > 0 else 0
        return f"""
=== Pipeline Statistics ===
Total processed:    {self.total:,}
Successful:         {self.successful:,}
Failed:             {self.failed:,}
Success rate:       {success_rate:.1f}%
Avg turns:          {self.avg_turns:.1f}
Avg code blocks:    {self.avg_code_blocks:.1f}

=== Answer Validation ===
Correct:            {self.answers_correct:,}
Incorrect:          {self.answers_incorrect:,}
Unknown:            {self.answers_unknown:,}
Accuracy:           {accuracy:.1f}% (of {validated} validated)
"""

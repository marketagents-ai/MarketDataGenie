
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from datagenie.pythonformer.pipeline import PythonformerPipeline
from datagenie.pythonformer.config import PythonformerConfig
from datagenie.pythonformer.models import Turn, TrajectoryResult
from datagenie.pythonformer.utils.db_utils import PythonformerDataInserter

@pytest.fixture
def mock_db():
    return AsyncMock()

@pytest.fixture
def mock_inserter(mock_db):
    inserter = PythonformerDataInserter(mock_db)
    inserter.insert_trajectory = AsyncMock(return_value=uuid4())
    inserter.insert_turn = AsyncMock(return_value=True)
    inserter.insert_request = AsyncMock(return_value=True)
    inserter.insert_sharegpt_dataset = AsyncMock(return_value=True)
    return inserter

@pytest.fixture
def mock_config():
    from datagenie.pythonformer.config import DatasetConfig, EnvironmentType
    # minimalist config for testing
    return PythonformerConfig(
        dataset=DatasetConfig(environment=EnvironmentType.MATH_PYTHON),
        main_model="test-model"
    )

@pytest.mark.asyncio
async def test_pipeline_persists_trajectory_and_turns(mock_inserter, mock_config):
    """
    Verify that the pipeline calls insert_trajectory and insert_turn,
    and crucially DOES NOT call insert_environment_state.
    """
    # Mocking the LLM client to avoid real calls
    with patch("datagenie.pythonformer.pipeline.llm_chat", new_callable=AsyncMock) as mock_llm:
        # returns one mock response
        mock_llm.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Final Answer: \\boxed{42}", tool_calls=None))],
            usage=MagicMock(total_tokens=10)
        )
        
        pipeline = PythonformerPipeline(config=mock_config, data_inserter=mock_inserter)
        
        # Override _execute_agent_turn to avoid intricate environment/tool execution logic
        # We just want to test the persistence logic at the end of run()
        # pipeline._execute_agent_turn = AsyncMock(return_value=...) 
        # Actually better to let it run but mock the dependencies within run
        
        # Let's mock the run method's internal components if needed, or simply trust that
        # passing a "Final Answer" immediately will cause it to finish and save.
        
        result = await pipeline.run(
            task="Calculate 6 * 7",
            task_id="test_task_1",
            metadata={"source": "test"}
        )

        assert result.success is True
        
        # Verify persistence calls
        assert mock_inserter.insert_trajectory.called
        assert mock_inserter.insert_turn.called
        
        # Verify insert_environment_state is NOT called
        # We need to ensure the method doesn't even exist or isn't called if it does
        if hasattr(mock_inserter, "insert_environment_state"):
             assert not mock_inserter.insert_environment_state.called
        
@pytest.mark.asyncio
async def test_persistence_graceful_failure(mock_inserter, mock_config):
    """
    Verify that if DB insertion fails, the pipeline still returns the result 
    and doesn't crash.
    """
    # Make insertion fail
    mock_inserter.insert_trajectory.side_effect = Exception("DB Connection Failed")
    
    with patch("datagenie.pythonformer.pipeline.llm_chat", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = MagicMock(
           choices=[MagicMock(message=MagicMock(content="Final Answer: \\boxed{42}", tool_calls=None))],
           usage=MagicMock(total_tokens=10)
        )
        
        pipeline = PythonformerPipeline(config=mock_config, data_inserter=mock_inserter)
        
        # Should not raise exception
        result = await pipeline.run(task="Test Task", task_id="task_fail_db")
        
        assert result.success is True
        assert mock_inserter.insert_trajectory.called
        

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid

from market_agents.memory.agent_storage.setup_db import AsyncDatabase

from datagenie.pythonformer.models import Turn, TrajectoryResult

class PythonformerDataInserter:
    def __init__(self, db: AsyncDatabase):
        self.db = db
        self.logger = logging.getLogger("pythonformer_data_inserter")

    def _make_serializable(self, obj: Any) -> Any:
        """Helper to make objects JSON serializable (handles Pydantic and UUIDs)."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(i) for i in obj]
        if hasattr(obj, "hex"): # UUID
            return str(obj)
        if isinstance(obj, (datetime, timezone)):
            return obj.isoformat()
        return obj



    async def insert_request(self, req_data: Dict[str, Any]):
        """Insert inference request data into requests table."""
        query = """
            INSERT INTO requests (
                prompt_context_id, start_time, end_time, total_time,
                model, max_tokens, temperature, messages, system,
                raw_response, completion_tokens, prompt_tokens, total_tokens
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        """
        try:
            await self.db.execute(
                query,
                req_data.get("prompt_context_id"),
                req_data.get("start_time"),
                req_data.get("end_time"),
                req_data.get("total_time"),
                req_data.get("model"),
                req_data.get("max_tokens"),
                req_data.get("temperature"),
                json.dumps(self._make_serializable(req_data.get("messages", []))),
                req_data.get("system"),
                json.dumps(self._make_serializable(req_data.get("raw_response", {}))),
                req_data.get("completion_tokens"),
                req_data.get("prompt_tokens"),
                req_data.get("total_tokens")
            )
        except Exception as e:
            self.logger.error(f"Error inserting request: {e}")
            return False
        return True

    async def insert_trajectory(self, result: TrajectoryResult, task_id: str, env_name: str, metadata: Dict[str, Any] = None) -> Optional[uuid.UUID]:
        """Insert full trajectory result into pythonformer_trajectories."""
        query = """
            INSERT INTO pythonformer_trajectories (
                task_id, environment_name, success, final_answer,
                num_code_blocks, num_errors, total_reward, step_rewards,
                system_prompt, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
        """
        try:
            rows = await self.db.fetch(
                query,
                task_id,
                env_name,
                result.success,
                result.final_answer,
                result.num_code_blocks,
                result.num_errors,
                result.total_reward,
                json.dumps(result.step_rewards),
                result.system_prompt,
                json.dumps(self._make_serializable(metadata or {}))
            )
            if not rows:
                return None
            
            trajectory_id = rows[0]['id']
            
            # Insert turns
            for i, turn in enumerate(result.turns):
                # For now, per-turn reward is not explicitly in TrajectoryResult's Turn
                # but we could infer from step_rewards if lengths match
                turn_reward = result.step_rewards[i] if i < len(result.step_rewards) else 0.0
                await self.insert_turn(trajectory_id, i + 1, turn, turn_reward)
                
            return trajectory_id
        except Exception as e:
            self.logger.error(f"Error inserting trajectory: {e}")
            return None

    async def insert_turn(self, trajectory_id: uuid.UUID, round_num: int, turn: Turn, reward: float = 0.0):
        """Insert individual turn into pythonformer_turns."""
        query = """
            INSERT INTO pythonformer_turns (
                trajectory_id, round, role, content, code, reward
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """
        try:
            await self.db.execute(
                query,
                trajectory_id,
                round_num,
                turn.role,
                turn.content,
                turn.code,
                reward
            )
        except Exception as e:
            self.logger.error(f"Error inserting turn: {e}")
            return False
        return True
    
    async def insert_sharegpt_dataset(self, task_id: str, conversations: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None):
        """Insert ShareGPT formatted dataset into pythonformer_sharegpt_datasets."""
        query = """
            INSERT INTO pythonformer_sharegpt_datasets (task_id, conversations, metadata)
            VALUES ($1, $2, $3)
        """
        try:
            await self.db.execute(
                query,
                task_id,
                json.dumps(self._make_serializable(conversations)),
                json.dumps(self._make_serializable(metadata or {}))
            )
        except Exception as e:
            self.logger.error(f"Error inserting sharegpt dataset: {e}")
            return False
        return True

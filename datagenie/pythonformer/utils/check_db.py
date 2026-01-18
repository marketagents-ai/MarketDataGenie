import asyncio
import os
import json
from market_agents.memory.agent_storage.setup_db import AsyncDatabase
from market_agents.memory.config import AgentStorageConfig
from dotenv import load_dotenv

async def check_counts():
    load_dotenv()
    db_args = {}
    if os.getenv("DB_NAME"): db_args["db_name"] = os.getenv("DB_NAME")
    if os.getenv("DB_USER"): db_args["user"] = os.getenv("DB_USER")
    if os.getenv("DB_PASSWORD"): db_args["password"] = os.getenv("DB_PASSWORD")
    if os.getenv("DB_HOST"): db_args["host"] = os.getenv("DB_HOST")
    if os.getenv("DB_PORT"): db_args["port"] = os.getenv("DB_PORT")

    db_config = AgentStorageConfig(**db_args)
    db = AsyncDatabase(db_config)
    await db.initialize()
    
    tables = ['requests', 'pythonformer_trajectories', 'pythonformer_turns', 'pythonformer_sharegpt_datasets']
    print("--- Database Table Counts ---")
    for table in tables:
        try:
            res = await db.fetch(f"SELECT COUNT(*) as count FROM {table}")
            print(f"{table}: {res[0]['count']}")
        except Exception as e:
            print(f"{table}: ERROR ({e})")
    
    print("\n--- Latest Trajectory Result ---")
    try:
        res = await db.fetch("SELECT * FROM pythonformer_trajectories ORDER BY created_at DESC LIMIT 1")
        if res:
            # Drop system_prompt for brevity
            row = dict(res[0])
            if 'system_prompt' in row: row['system_prompt'] = row['system_prompt'][:100] + "..."
            print(json.dumps(row, indent=2, default=str))
        else:
            print("No trajectories found yet.")
    except Exception as e:
        print(f"Error fetching trajectory: {e}")

    print("\n--- Latest ShareGPT Entry ---")
    try:
        res = await db.fetch("SELECT * FROM pythonformer_sharegpt_datasets ORDER BY created_at DESC LIMIT 1")
        if res:
            row = dict(res[0])
            # Truncate conversations for brevity
            if 'conversations' in row:
                conv = json.loads(row['conversations']) if isinstance(row['conversations'], str) else row['conversations']
                row['conversations'] = f"[{len(conv)} messages]"
            print(json.dumps(row, indent=2, default=str))
        else:
            print("No ShareGPT entries found yet.")
    except Exception as e:
        print(f"Error fetching ShareGPT entry: {e}")

if __name__ == "__main__":
    asyncio.run(check_counts())

import asyncio
import logging
import os
from market_agents.memory.agent_storage.setup_db import AsyncDatabase
from market_agents.memory.config import AgentStorageConfig
from dotenv import load_dotenv

async def setup_pythonformer_tables(db: AsyncDatabase):
    """Create specialized tables for Pythonformer trajectories and turns."""
    
    # Trajectories table
    trajectories_query = """
    CREATE TABLE IF NOT EXISTS pythonformer_trajectories (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        task_id TEXT,
        environment_name TEXT,
        success BOOLEAN,
        final_answer TEXT,
        num_code_blocks INTEGER,
        num_errors INTEGER,
        total_reward DOUBLE PRECISION,
        step_rewards JSONB,
        system_prompt TEXT,
        metadata JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    # Turns table
    turns_query = """
    CREATE TABLE IF NOT EXISTS pythonformer_turns (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        trajectory_id UUID REFERENCES pythonformer_trajectories(id) ON DELETE CASCADE,
        round INTEGER,
        role TEXT,
        content TEXT,
        code TEXT,
        reward DOUBLE PRECISION,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """

    # ShareGPT Datasets (for final fine-tuning output)
    sharegpt_query = """
    CREATE TABLE IF NOT EXISTS pythonformer_sharegpt_datasets (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        task_id TEXT NOT NULL,
        conversations JSONB NOT NULL,
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_pythonformer_sharegpt_task ON pythonformer_sharegpt_datasets(task_id);
    """
    
    try:
        await db.execute(trajectories_query)
        await db.execute(turns_query)
        await db.execute(sharegpt_query)
        print("✅ Pythonformer specialized tables created successfully.")
    except Exception as e:
        print(f"❌ Error creating Pythonformer tables: {e}")
        raise

async def purge_pythonformer_data(db: AsyncDatabase):
    """Purge ALL data from the database by dropping the public schema."""
    try:
        print("💥 Initiating FULL database purge...")
        # Drop and recreate public schema (fastest way to nuke everything)
        await db.execute("DROP SCHEMA public CASCADE;")
        await db.execute("CREATE SCHEMA public;")
        await db.execute("GRANT ALL ON SCHEMA public TO public;")
        await db.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✅ Public schema recreated. Database is empty.")
        
        # Re-initialize base tables using the existing script logic
        # We need to run the synchronous setup_database.py logic
        # We'll do this by importing and calling it in a thread or subprocess to avoid blocking async loop
        # But for simplicity in this script, we can just execute the SQL queries if we had them or call the function
        # Better: let's re-run the setup_database.py script as a subprocess
        import subprocess
        import sys
        
        print("🔄 Re-initializing base database schema...")
        base_setup_script = "market_agents/agents/db/setup_database.py"
        try:
            subprocess.run([sys.executable, base_setup_script], check=True)
            print("✅ Base schema restored.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to restore base schema: {e}")
            raise

        # Now setup pythonformer specialized tables
        await setup_pythonformer_tables(db)
        print("✅ Pythonformer tables restored.")
        
        print("✨ Database successfully reset to fresh state.")

    except Exception as e:
        print(f"❌ Error purging data: {e}")
        raise

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--purge", action="store_true", help="Purge existing data")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    
    # Manually map env vars if they exist
    db_args = {}
    if os.getenv("DB_NAME"): db_args["db_name"] = os.getenv("DB_NAME")
    if os.getenv("DB_USER"): db_args["user"] = os.getenv("DB_USER")
    if os.getenv("DB_PASSWORD"): db_args["password"] = os.getenv("DB_PASSWORD")
    if os.getenv("DB_HOST"): db_args["host"] = os.getenv("DB_HOST")
    if os.getenv("DB_PORT"): db_args["port"] = os.getenv("DB_PORT")

    db_config = AgentStorageConfig(**db_args)
    db = AsyncDatabase(db_config)
    
    print(f"Connecting to DB: {db_config.db_name} as {db_config.user}...")
    await db.initialize()
    if args.purge:
        await purge_pythonformer_data(db)
    else:
        await setup_pythonformer_tables(db)

if __name__ == "__main__":
    asyncio.run(main())

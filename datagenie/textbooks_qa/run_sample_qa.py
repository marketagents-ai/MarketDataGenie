"""
Sample Textbook QA Generation

This script runs a sample QA generation on a small set of textbook chunks
to demonstrate the pipeline in action.
"""

import asyncio
import logging
import os
from pathlib import Path
import argparse

# MarketAgents imports
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona
from minference.lite.models import LLMConfig

# Local imports
from datagen_textbooks_qa import TextbookQAGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_market_agent() -> MarketAgent:
    """Setup MarketAgent for sample generation"""
    
    # Create persona for educational content generation
    persona = Persona(
        name="Educational Content Specialist",
        role="Expert in creating high-quality educational content and assessments",
        persona="""
        I am an experienced educational content specialist with expertise in:
        - Curriculum development and assessment design
        - Creating grade-appropriate educational materials
        - Developing clear and effective learning objectives
        - Ensuring educational content is engaging and accessible
        - Maintaining high standards for educational quality
        
        I have extensive experience working with educational materials and understand
        the cultural and linguistic nuances required for effective learning content.
        """,
        objectives=[
            "Create educationally valuable QA conversations",
            "Ensure grade-appropriate content and language",
            "Maintain high quality and accuracy standards",
            "Support effective learning outcomes",
            "Always verify information accuracy",
            "Use age-appropriate language and concepts",
            "Ensure questions test genuine understanding",
            "Provide clear and helpful feedback"
        ]
    )
    
    # Setup LLM configuration
    llm_config = LLMConfig(
        client="litellm",
        model="Hermes-3-Llama-3.1-405B",
        #client="openai",
        #model="gpt-5-nano",
        temperature=0.5,
        max_tokens=4096
    )
    
    # Setup LLM orchestrator
    from minference.lite.inference import InferenceOrchestrator
    llm_orchestrator = InferenceOrchestrator()
    
    # Create MarketAgent with proper configuration
    agent = MarketAgent(
        name="textbook_qa_specialist",
        persona=persona,
        llm_config=llm_config,
        llm_orchestrator=llm_orchestrator,
        task="Generate high-quality educational QA conversations from textbook content",
        memory_enabled=True,
        verbose=True
    )
    
    return agent

async def run_sample_generation():
    """Run sample QA generation on a few chunks"""

    logger.info("Starting Sample Textbook QA Generation")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples to generate from the dataset')
    parser.add_argument('--concurrency', type=int, default=None, help='Parallel batch size override for full-dataset runs')
    parser.add_argument('--per-task-timeout', type=int, default=None, help='Timeout in seconds per chunk (override)')
    args, _ = parser.parse_known_args()

    if args.config:
        config = {"config_path": args.config}
    else:
        config = {
            "output_dir": "outputs/sample_qa",
            "num_samples": args.num_samples or 2,
            "subjects": None,
            "grades": None
        }

    if not args.config and not config.get("dataset_name"):
        raise SystemExit("Provide --config with dataset.name, or set dataset_name in the script.")

    try:
        # Initialize generator
        logger.info("Initializing TextbookQAGenerator...")
        generator = TextbookQAGenerator(**config)
        # Ensure CLI overrides YAML/defaults for sample size
        if args.num_samples is not None:
            generator.num_samples = args.num_samples
        if args.concurrency is not None:
            generator.batch_size = args.concurrency
        if args.per_task_timeout is not None:
            generator.per_task_timeout = args.per_task_timeout

        # First, explore the dataset to find available subjects and grades
        logger.info("Exploring dataset to find available content...")

        # Load a small sample without filters
        if args.config:
            explore_generator = TextbookQAGenerator(config_path=args.config, num_samples=args.num_samples or 20)
        else:
            explore_generator = TextbookQAGenerator(
                dataset_name=config["dataset_name"],
                num_samples=args.num_samples or 20,
                subjects=None,
                grades=None
            )
        # Ensure CLI overrides YAML/defaults for exploration as well
        if args.num_samples is not None:
            explore_generator.num_samples = args.num_samples or explore_generator.num_samples

        all_chunks = explore_generator.load_huggingface_dataset()
        logger.info(f"Found {len(all_chunks)} total chunks")

        if not all_chunks:
            logger.error("No chunks found in dataset!")
            return

        # Show available subjects and grades
        subjects = set(chunk.subject for chunk in all_chunks)
        grades = set(chunk.grade for chunk in all_chunks)

        logger.info(f"Available subjects: {subjects}")
        logger.info(f"Available grades: {grades}")

        # Choose a specific subject and grade for the sample
        # Let's pick the first available subject and grade
        sample_subject = list(subjects)[0]
        sample_grade = list(grades)[0]

        logger.info(f"Selected for sample: Subject={sample_subject}, Grade={sample_grade}")

        # Show some sample chunks
        sample_chunks = [chunk for chunk in all_chunks
                        if chunk.subject == sample_subject and chunk.grade == sample_grade][:3]

        logger.info(f"Sample chunks for {sample_subject} Grade {sample_grade}:")
        for i, chunk in enumerate(sample_chunks):
            logger.info(f"  {i+1}. ID: {chunk.id}")
            logger.info(f"     Chapter: {chunk.chapter_title}")
            logger.info(f"     Text length: {len(chunk.text)} characters")
            logger.info(f"     Text preview: {chunk.text[:150]}...")
            logger.info("")

        # Now run the actual generation with the selected subject and grade
        logger.info("Setting up for QA generation...")

        # Update generator with specific filters only if not using config
        if not args.config:
            generator.subjects = [sample_subject]
            generator.grades = [sample_grade]
            generator.num_samples = args.num_samples or 2  # Small sample

        # Load filtered chunks
        chunks = generator.load_huggingface_dataset()
        # Hard cap to exactly the requested number of samples
        if generator.num_samples is not None:
            chunks = chunks[: generator.num_samples]
        logger.info(f"Loaded {len(chunks)} chunks for QA generation")

        if not chunks:
            logger.error("No chunks found with the specified filters!")
            return

        # Setup agent for actual QA generation
        logger.info("Setting up MarketAgent for QA generation...")
        agent = setup_market_agent()

        # Generate QA
        logger.info("Starting QA generation...")
        results = []

        if args.config:
            logger.info("Running full-dataset generation via parallel workflow rollouts...")
            # Setup agent (template)
            agent = setup_market_agent()
            results = await generator.generate_dataset(agent, max_workers=getattr(generator, 'batch_size', 1), agent_factory=setup_market_agent)
        else:
            # Sample mode: sequential small run
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk.id}")
                agent = setup_market_agent()  # fresh agent per chunk
                try:
                    result = await generator.generate_qa_for_chunk(chunk, agent)
                    results.append(result)
                    logger.info(f"Generated QA for chunk {chunk.id}")
                except Exception as e:
                    logger.error(f"Failed to generate QA for chunk {chunk.id}: {e}")
                    error_result = {
                        "chunk_id": chunk.id,
                        "error": str(e),
                        "timestamp": "2024-01-01T12:00:00"
                    }
                    results.append(error_result)
            generator.results = results
            generator.save_results(results)

        # Print summary
        print("\n" + "="*60)
        print("SAMPLE QA GENERATION COMPLETED")
        print("="*60)
        print(f"Subject: {sample_subject}")
        print(f"Grade: {sample_grade}")
        print(f"Chunks processed: {len(chunks)}")
        print(f"Results saved to: {generator.output_dir}")
        if args.config:
            print(f"Total results: {len(results)}")
        print("\nSample Results:")

        for i, result in enumerate(results):
            print(f"\n{i+1}. Chunk ID: {result['chunk_id']}")

            # Check if we have error or success
            if 'error' in result:
                print(f"   Status: Error")
                print(f"   Error: {result['error']}")
            else:
                # Try to get chapter title safely
                chapter_title = (
                    result.get('chapter_title')
                    or (result.get('qa_conversation', {}) or {}).get('chapter_title')
                    or (result.get('metadata', {}) or {}).get('chapter_title')
                    or 'Unknown'
                )
                print(f"   Chapter: {chapter_title}")

                # Check if we have QA conversation data
                if 'qa_conversation' in result and isinstance(result['qa_conversation'], dict):
                    qa_data = result['qa_conversation']
                    if 'question' in qa_data:
                        print(f"   Question: {qa_data['question']}")
                    if 'answer' in qa_data:
                        print(f"   Answer: {qa_data['answer']}")
                    if 'quality_score' in qa_data:
                        print(f"   Quality Score: {qa_data['quality_score']}")
                    if 'rephrased_text' in qa_data:
                        print(f" Rephrased Text: {qa_data['rephrased_text']}")
                else:
                    print(f"   Status: Completed but no QA data")
                    print(f"   Result: {type(result.get('qa_conversation', 'No data'))}")

        print("="*60)

        logger.info("Sample QA generation completed successfully!")

    except Exception as e:
        logger.error(f"Error during sample generation: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_sample_generation())

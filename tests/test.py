from pathlib import Path
import sys

from google import genai
from langsmith import wrappers

# Allow running this file directly from the tests/ directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.rag import PineconeWrapper


def main() -> None:
    gemini_api_key = Config.get_gemini_api_key(use_test=True)
    if not gemini_api_key:
        raise RuntimeError(
            "No Gemini API key available for test mode. Please set GEMINI_API_KEY_TEST or GEMINI_API_KEY in .env."
        )

    # Ensure runtime env vars are present for Gemini + LangSmith integrations.
    Config.setup_langsmith_env(use_test=True)

    if not Config.langsmith_enabled():
        raise RuntimeError(
            "LangSmith tracing is not enabled. Set LANGSMITH_API_KEY and LANGSMITH_TRACING=true in .env."
        )

    # Initialize Gemini client
    gemini_client = genai.Client(api_key=gemini_api_key)

    # Wrap the Gemini client to enable LangSmith tracing
    client = wrappers.wrap_gemini(
        gemini_client,
        tracing_extra={
            "tags": ["gemini", "python"],
            "metadata": {
                "integration": "google-genai",
            },
        },
    )

    # Make a traced Gemini call
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents="Explain how to design an SysML V2 model for a self-driving car system in 3 steps with reasoning.",
    )

    print(response)

    # # Initialize Pinecore and set up index
    # pinecore = PineconeWrapper()
    # index_name = "ai-prototypingtest"
    #
    # # Create index if it does not exist
    # pinecore.create_index(index_name)
    #
    # # Upsert test records with integrated embedding
    # pinecore.upsert(
    #     index_name=index_name,
    #     records=[
    #         {
    #             "_id": "chunk-0001",
    #             "chunk_text": "SysML v2 blocks represent structural elements in a system model.",
    #             "doc_id": "sysml-v2-intro",
    #             "chunk_id": 1,
    #             "domain": "mbse",
    #             "source": "test-data",
    #             "language": "en",
    #             "access_level": "internal",
    #         }
    #     ],
    #     namespace="mbse-demo",
    # )




if __name__ == "__main__":
    main()
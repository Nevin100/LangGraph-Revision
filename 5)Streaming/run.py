import asyncio
from graph import graph
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# async main function to run the graph with streaming enabled
async def main():
    input_state = {
        "messages": [{"role": "user", "content": "What is recursion?"}]
    }

    print("Streaming tokens:\n")
    async for chunk in graph.astream(input_state, stream_mode="messages"):
        # chunk[0] = message token, chunk[1] = metadata
        token = chunk[0].content
        if token:
            print(token, end="", flush=True)

asyncio.run(main())
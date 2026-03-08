import sys
import os
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autobot.agent.mission_agent import MissionAgent
from autobot.agent.mission import MissionStatus, ObjectiveStatus

async def test_mission_agent():
    print("--- Testing Mission Agent Planning & Coordination ---")
    
    # Mock LLM Client
    mock_llm = MagicMock()
    mock_llm.chat.completions.create = AsyncMock()
    
    # Mock planning response
    mock_llm.chat.completions.create.side_effect = [
        # 1. Mission Planning response
        MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
            "objectives": [
                {"id": "obj1", "description": "Search for Titanic competition"},
                {"id": "obj2", "description": "Select the competition and download data"}
            ]
        })))]),
        # 2. AgentLoop step 1 response (Done)
        MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
            "thinking": "Found the competition.",
            "evaluation_previous_goal": "Success",
            "memory": "",
            "next_goal": "Stop",
            "action": [{"done": {"text": "Competition found", "success": True}}]
        })))]),
        # 3. AgentLoop step 2 response (Done)
        MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
            "thinking": "Downloaded data.",
            "evaluation_previous_goal": "Success",
            "memory": "",
            "next_goal": "Stop",
            "action": [{"done": {"text": "Data downloaded", "success": True}}]
        })))]),
    ]
    
    # Mock Page
    mock_page = MagicMock()
    mock_page.url = "https://kaggle.com"
    mock_page.context.new_page = AsyncMock()
    
    agent = MissionAgent(
        page=mock_page,
        llm_client=mock_llm,
        mission_goal="Automate Kaggle Titanic competition",
        model="gpt-test"
    )
    
    print("Running mission...")
    result = await agent.run()
    
    print(f"\nMission Result: {result}")
    
    # Check mission status
    if agent.mission.status == MissionStatus.COMPLETED:
        print("[OK] Mission marked as COMPLETED.")
    else:
        print(f"[FAIL] Mission status is {agent.mission.status}")

    # Check objectives
    for obj in agent.mission.objectives:
        print(f"Objective '{obj.description}': {obj.status}")
        if obj.status != ObjectiveStatus.COMPLETED:
            print(f"[FAIL] Objective should be COMPLETED.")

if __name__ == "__main__":
    asyncio.run(test_mission_agent())

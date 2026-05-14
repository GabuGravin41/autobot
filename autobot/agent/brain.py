"""
Cognitive Brain — A non-linear, multi-agent reasoning engine designed for deep reasoning and dynamic routing.

Instead of a single monolithic LLM call processing the entire state simultaneously, the CognitiveBrain splits the cognitive load into specialized "brain regions".
These agents do not execute actions themselves; they collaborate to synthesize a decision.

This architecture dramatically reduces token usage while enabling highly flexible, organic intelligence. It features dynamic fallbacks, allowing the system to replan, reroute, or ask the user for help when stuck.

Brain Regions:
1. Perception (Visual Cortex): Analyzes raw DOM + Screenshot -> Dense text summary of actionable state.
2. Meta-Cognition (Anterior Cingulate Cortex): Evaluates progress, detects failures/loops, and routes the flow (PROCEED, REPLAN, ASK_USER).
3. Planner (Prefrontal Cortex): Decides the next logical step in natural language based on the routing strategy.
4. Motor (Motor Cortex): Translates the natural language plan into exact JSON tool calls.
"""
import json
import logging
import asyncio
from typing import Any

from autobot.agent.models import AgentOutput, ActionModel
from autobot.dom.models import BrowserState

logger = logging.getLogger(__name__)

class CognitiveBrain:
    """
    Coordinates specialized cognitive agents to determine the next computer action.
    """
    def __init__(self, llm_client: Any, model: str, fast_model: str | None = None):
        self.llm_client = llm_client
        self.model = model
        # Use a faster/cheaper model for Perception and Motor if available
        self.fast_model = fast_model or model

    async def think_and_decide(
        self,
        goal: str,
        browser_state: BrowserState,
        history_summary: str,
        scratchpad: list[str],
        tool_catalog: str
    ) -> AgentOutput:
        """
        Run the cognitive pipeline: Perception -> Memory -> Planner -> Motor
        """
        logger.debug("🧠 CognitiveBrain: Starting cognitive cycle...")

        # 1. Perception Phase (Compress observation)
        perception_summary = await self._perceive_state(browser_state)
        
        # 2. Meta-Cognitive Phase (Self-Reflection & Routing)
        route, meta_context = await self._meta_cognition(goal, perception_summary, history_summary)
        
        if route == "ASK_USER":
            logger.info(f"⏸️ Brain routing to Human-in-the-Loop: {meta_context}")
            # Output an empty action so the loop pauses, and use the narrative to alert the user
            return AgentOutput(
                thinking=f"Meta-Cognition determined human input is required: {meta_context}",
                next_goal="Wait for user input",
                narrative=f"PAUSED: {meta_context}",
                action=[]
            )
            
        if route == "REPLAN":
            logger.info(f"🔄 Brain routing to new strategy: {meta_context}")
            scratchpad.append(f"[STRATEGY CHANGE] {meta_context}")
            
        # 3. Planner Phase (Decide what to do next in natural language)
        plan_text = await self._plan_next_step(goal, perception_summary, history_summary, scratchpad)
        
        # 4. Motor Phase (Translate plan to exact JSON action)
        motor_output = await self._generate_motor_action(plan_text, browser_state, tool_catalog)
        
        return motor_output

    async def _meta_cognition(self, goal: str, perception: str, history: str) -> tuple[str, str]:
        """
        Meta-Cognition (Anterior Cingulate Cortex): 
        Evaluates the current state against recent history to detect if the agent is stuck,
        looping, or needs human intervention.
        """
        # If there's no history, we just started
        if not history.strip():
            return "PROCEED", "Starting fresh"
            
        prompt = f"""You are the Meta-Cognitive Monitor of an autonomous AI.
Your job is to review the recent history and current state, and decide the high-level routing strategy.

GOAL: {goal}

CURRENT PERCEPTION:
{perception}

RECENT HISTORY:
{history}

Decide the routing state:
1. PROCEED: The agent is making normal progress.
2. REPLAN: The agent is stuck in a loop, repeating actions, or the current approach is failing.
3. ASK_USER: The agent has hit a hard blocker (e.g. CAPTCHA, missing password, ambiguous instructions) and CANNOT proceed without human help.

Respond in EXACTLY this format:
ROUTING: <PROCEED|REPLAN|ASK_USER>
REASON: <One sentence explanation or the new strategy/question>
"""
        resp = await self.llm_client.chat.completions.create(
            model=self.fast_model,
            messages=[{"role": "user", "content": f"SYSTEM: You are the Meta-Cognitive Monitor of an autonomous AI.\n\n{prompt}"}],
            temperature=0.1,
            max_tokens=150
        )
        output = resp.choices[0].message.content.strip()
        
        route = "PROCEED"
        reason = ""
        for line in output.split("\n"):
            if line.startswith("ROUTING:"):
                route = line.replace("ROUTING:", "").strip()
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
        
        if route != "PROCEED":
            logger.warning(f"🧠 Meta-Cognition [{route}]: {reason}")
                
        return route, reason

    async def _perceive_state(self, state: BrowserState) -> str:
        """
        Visual Cortex: Compress the massive DOM and screenshot into a dense text summary.
        Token optimization: This prevents the Planner from seeing thousands of tokens of DOM.
        """
        # If there's no DOM, just return basic info
        if not state.element_tree:
            return f"Page URL: {state.url}\nNo interactive elements found."

        # Truncate DOM tree for perception to avoid blowing up context
        # The motor agent will need the exact indices, but planner just needs to know what's there
        dom_text = state.element_tree.to_text()
        if len(dom_text) > 4000:
            dom_text = dom_text[:4000] + "\n...[truncated]"

        prompt = f"""You are the Perception Module of an AI agent.
Analyze the current screen state and provide a highly compressed, dense summary of what is visible and actionable.
Focus on elements relevant to navigating, reading, or interacting.

URL: {state.url}
Title: {state.title}

DOM Elements:
{dom_text}

Output a short paragraph describing the state of the screen and the most prominent actionable elements (buttons, inputs, links)."""

        messages = [{"role": "user", "content": prompt}]
        
        # Add vision if screenshot is available
        if state.screenshot_b64:
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state.screenshot_b64}"}}
            ]

        resp = await self.llm_client.chat.completions.create(
            model=self.fast_model,
            messages=[{"role": "user", "content": f"SYSTEM: You are the Perception Module of an AI agent. Summarize screen concisely.\n\n{messages[0]['content']}"}],
            temperature=0.1,
            max_tokens=300
        )
        summary = resp.choices[0].message.content.strip()
        logger.debug(f"👁️ Perception: {summary[:100]}...")
        return summary

    async def _plan_next_step(
        self, goal: str, perception: str, history: str, scratchpad: list[str]
    ) -> str:
        """
        Prefrontal Cortex: High-level reasoning and decision making.
        """
        notes = "\n".join(scratchpad[-5:]) if scratchpad else "None"
        
        prompt = f"""You are the Executive Planner of an autonomous AI agent.
Your job is to decide the very next action to take to achieve the goal.

GOAL: {goal}

CURRENT STATE PERCEPTION:
{perception}

RECENT HISTORY:
{history}

SCRATCHPAD NOTES:
{notes}

Based on the goal and current state, what EXACTLY should the agent do next?
Do not output code or JSON. Output a natural language instruction for the Motor Cortex.
Example: "Click the 'Login' button at the top right" or "Type 'machine learning' into the search bar".
If the goal is achieved, output: "We are done. Mission accomplished."
"""
        resp = await self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"SYSTEM: You are the Executive Planner. Decide next step.\n\n{prompt}"}],
            temperature=0.3,
            max_tokens=200
        )
        plan = resp.choices[0].message.content.strip()
        logger.debug(f"🧠 Planner: {plan}")
        return plan

    async def _generate_motor_action(
        self, plan: str, state: BrowserState, tool_catalog: str
    ) -> AgentOutput:
        """
        Motor Cortex: Maps the natural language plan to exact tool schemas using the DOM indices.
        """
        dom_text = state.element_tree.to_text() if state.element_tree else "No DOM"
        
        prompt = f"""You are the Motor Cortex of an AI agent. 
Your job is to translate the Executive Planner's natural language instruction into an EXACT JSON action.

PLANNER INSTRUCTION: {plan}

AVAILABLE TOOLS:
{tool_catalog}

CURRENT DOM:
{dom_text}

Output ONLY valid JSON matching this schema:
{{
  "thinking": "Brief explanation of how I'm mapping the instruction to the tool",
  "evaluation_previous_goal": "N/A",
  "memory": "N/A",
  "next_goal": "{plan}",
  "narrative": "{plan}",
  "confidence": "high",
  "hypotheses": [],
  "action": [
    {{ "computer_call": {{"call": "computer.browser.click_element(3)"}} }}
  ]
}}

Make sure to find the correct DOM index [N] from the CURRENT DOM and use it in your tool call.
If the instruction says we are done, use the "done" action.
"""
        resp = await self.llm_client.chat.completions.create(
            model=self.fast_model,
            messages=[{"role": "user", "content": f"SYSTEM: You are the Motor Cortex. Translate instructions into JSON.\n\n{prompt}"}],
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"}
        )
        raw_json = resp.choices[0].message.content.strip()
        
        try:
            data = json.loads(raw_json)
            out = AgentOutput(**data)
            # Log the chosen actions for diagnostics
            action_names = [a.action_name for a in out.action]
            logger.info(f"⚙️ Motor: Plan '{plan[:50]}...' -> {action_names}")
            return out
        except Exception as e:
            logger.error(f"Motor agent failed to output valid JSON: {e}")
            # Fallback safe output
            return AgentOutput(
                thinking="Fallback due to parsing error",
                next_goal=plan,
                action=[]
            )

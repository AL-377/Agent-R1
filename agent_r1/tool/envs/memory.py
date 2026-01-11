"""
Memory Environment for Medical Dialogue Memory Model
Manages memory operations and provides memory state to the agent
"""

from agent_r1.tool.base import BaseToolEnv, BaseTool
from agent_r1.tool.memory_manager import MemoryManager
from typing import List, Dict, Tuple, Any
import re
import json


class MemoryEnv(BaseToolEnv):
    """
    Memory Environment for managing memory operations
    
    Provides:
    - Memory state at each step (for agent input)
    - Memory operation execution (insert, update, delete, wait)
    - RAG retrieval interface
    """
    
    def __init__(
        self,
        tools: List[BaseTool],
        max_tool_response_length: int,
        memory_manager: MemoryManager = None,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        index_type: str = "Flat",
        device: str = "cpu"
    ):
        """
        Initialize Memory Environment
        
        Args:
            tools: List of memory tools (insert, update, delete, wait)
            max_tool_response_length: Maximum length for tool responses
            memory_manager: MemoryManager instance (if None, will create one)
            embedding_model: Embedding model for RAG
            index_type: FAISS index type
            device: Device for embedding model
        """
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.max_tool_response_length = max_tool_response_length
        
        # Initialize or use provided memory manager
        if memory_manager is None:
            self.memory_manager = MemoryManager(
                embedding_model=embedding_model,
                index_type=index_type,
                device=device
            )
        else:
            self.memory_manager = memory_manager
        
        # Inject memory_manager into tools
        for tool in self.tools:
            if hasattr(tool, 'memory_manager'):
                tool.memory_manager = self.memory_manager
        
        # Tool call format
        self.tool_call_start = "<tool_call>"
        self.tool_call_end = "</tool_call>"
        self.tool_response_start = "<tool_response>"
        self.tool_response_end = "</tool_response>"
        self.eos_token = "<|im_end|>"
        self.parallel_tool_calls = True  # Allow multiple memory operations per step
        
    def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
        """
        Execute memory operations from agent response
        
        Args:
            raw_response: Raw response from LLM containing tool calls
        
        Returns:
            tool_response: Formatted tool response
            success: List of success flags for each tool call
            active: Whether trajectory is still active
        """
        tool_calls = self.extract_tool_calls(raw_response)
        if len(tool_calls) == 0:
            return "", [], False
        
        if not self.parallel_tool_calls:
            tool_calls = [tool_calls[0]]
        
        tool_responses = []
        tool_successes = []
        
        for tool_call in tool_calls:
            if tool_call is None:
                tool_responses.append("Error: JSONDecodeError")
                tool_successes.append(False)
            else:
                if "name" not in tool_call:
                    tool_responses.append("Error: No tool name")
                    tool_successes.append(False)
                else:
                    tool_name = tool_call["name"]
                    if tool_name not in self.tool_map:
                        tool_responses.append(f"Error: ToolNotFoundError: {tool_name}")
                        tool_successes.append(False)
                    else:
                        tool = self.tool_map[tool_name]
                        if "arguments" not in tool_call:
                            tool_responses.append("Error: No tool arguments")
                            tool_successes.append(False)
                        elif not tool.validate_args(tool_call["arguments"]):
                            tool_responses.append("Error: Invalid tool arguments")
                            tool_successes.append(False)
                        else:
                            # Execute tool with memory_manager context
                            tool_result = tool.execute(
                                tool_call["arguments"],
                                memory_manager=self.memory_manager
                            )
                            tool_responses.append(tool_result["content"])
                            tool_successes.append(tool_result["success"])
        
        tool_response = self.format_tool_response(tool_responses)
        return tool_response, tool_successes, True
    
    def batch_step(self, raw_responses: List[str]) -> Tuple[List[str], List[List[bool]], List[bool]]:
        """
        Batch execute memory operations
        
        Args:
            raw_responses: List of raw responses from LLM
        
        Returns:
            batch_tool_responses: List of formatted tool responses
            batch_tool_successes: List of success flags for each response
            batch_active: List of active flags
        """
        batch_tool_responses = [[]] * len(raw_responses)
        batch_tool_successes = [[]] * len(raw_responses)
        batch_active = [True] * len(raw_responses)
        
        # Collect tool calls for batch execution
        success_tool_calls_arguments = {}  # key: tool_name, value: [arguments]
        success_tool_calls_index = {}  # key: tool_name, value: [(i, j)]
        
        for i, raw_response in enumerate(raw_responses):
            tool_calls = self.extract_tool_calls(raw_response)
            if len(tool_calls) == 0:
                batch_tool_successes[i] = []
                batch_active[i] = False
                batch_tool_responses[i] = []
                continue
            
            if not self.parallel_tool_calls:
                tool_calls = [tool_calls[0]]
            
            tool_responses = []
            tool_successes = []
            
            for j, tool_call in enumerate(tool_calls):
                if tool_call is None:
                    tool_responses.append("Error: JSONDecodeError")
                    tool_successes.append(False)
                else:
                    if "name" not in tool_call:
                        tool_responses.append("Error: No tool name")
                        tool_successes.append(False)
                    elif "arguments" not in tool_call:
                        tool_responses.append("Error: No tool arguments")
                        tool_successes.append(False)
                    else:
                        tool_name = tool_call["name"]
                        if tool_name not in self.tool_map:
                            tool_responses.append(f"Error: ToolNotFoundError: {tool_name}")
                            tool_successes.append(False)
                        else:
                            tool = self.tool_map[tool_name]
                            if not tool.validate_args(tool_call["arguments"]):
                                tool_responses.append("Error: Invalid tool arguments")
                                tool_successes.append(False)
                            else:
                                # Prepare for batch execution
                                if tool_name not in success_tool_calls_arguments:
                                    success_tool_calls_arguments[tool_name] = []
                                    success_tool_calls_index[tool_name] = []
                                tool_responses.append("Executing...")
                                tool_successes.append(False)
                                success_tool_calls_arguments[tool_name].append(tool_call["arguments"])
                                success_tool_calls_index[tool_name].append((i, j))
            
            batch_tool_responses[i] = tool_responses
            batch_tool_successes[i] = tool_successes
        
        # Batch execute tools
        for tool_name, args_list in success_tool_calls_arguments.items():
            tool = self.tool_map[tool_name]
            batch_results = tool.batch_execute(
                args_list,
                memory_manager=self.memory_manager
            )
            for batch_result, (i, j) in zip(batch_results, success_tool_calls_index[tool_name]):
                assert batch_tool_responses[i][j] == "Executing..."
                batch_tool_responses[i][j] = batch_result["content"]
                batch_tool_successes[i][j] = batch_result["success"]
        
        # Format responses
        batch_tool_responses_ = []
        for i, tool_responses in enumerate(batch_tool_responses):
            if batch_active[i]:
                assert len(batch_tool_responses[i]) > 0
                batch_tool_responses_.append(self.format_tool_response(tool_responses))
            else:
                batch_tool_responses_.append("")
        
        return batch_tool_responses_, batch_tool_successes, batch_active
    
    def stop(self, raw_response: str) -> bool:
        """
        Check if trajectory should stop
        
        Args:
            raw_response: Raw response from LLM
        
        Returns:
            True if should stop (no tool calls), False otherwise
        """
        tool_calls = self.extract_tool_calls(raw_response)
        return len(tool_calls) == 0
    
    def extract_tool_calls(self, raw_response: str) -> List[Any]:
        """
        Extract tool calls from raw response
        
        Args:
            raw_response: Raw response string
        
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        pattern = re.compile(
            f"{re.escape(self.tool_call_start)}(.*?){re.escape(self.tool_call_end)}",
            re.DOTALL
        )
        for tool_call_str in re.findall(pattern, raw_response):
            try:
                tool_call = json.loads(tool_call_str)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                tool_calls.append(None)
        
        return tool_calls
    
    def format_tool_response(self, tool_responses: List[str]) -> str:
        """
        Format tool responses for agent
        
        Args:
            tool_responses: List of tool response strings
        
        Returns:
            Formatted tool response message
        """
        tool_message = "<|im_end|>\n<|im_start|>user\n"
        for i, tool_response in enumerate(tool_responses):
            if len(tool_response) > self.max_tool_response_length:
                tool_response = tool_response[:self.max_tool_response_length] + "..."
            tool_message += f"<tool_response>\n{tool_response}\n</tool_response>"
            if i < len(tool_responses) - 1:
                tool_message += "\n"
        tool_message += "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        return tool_message
    
    def get_memory_state(self) -> Dict[str, Any]:
        """
        Get current memory state (for providing to agent at each step)
        
        Returns:
            Dictionary containing memory state for each layer
        """
        return self.memory_manager.get_memory_state()
    
    def get_memory_summary(self) -> str:
        """
        Get text summary of current memory state
        
        Returns:
            Formatted string summary
        """
        return self.memory_manager.get_memory_summary()
    
    def retrieve_memories(self, query: str, layers: List[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        RAG retrieval: search memories
        
        Args:
            query: Search query
            layers: List of layers to search (None = all)
            top_k: Number of results
        
        Returns:
            List of retrieved memory items
        """
        return self.memory_manager.retrieve(query, layers, top_k)
    
    @property
    def system_prompt(self) -> str:
        """System prompt for memory model"""
        return """You are a medical dialogue memory management agent. Your task is to manage patient memory information during medical consultations.

MEMORY LAYERS:
- working: Current dialogue key information (patient basic info, current symptoms, examination results, diagnosis conclusions)
- identity: Patient identity and basic information layer
- history: Historical diagnosis and treatment layer
- experience: Clinical experience and case layer

MEMORY ACTIONS:
1. memory_insert: Add new memory entry to specified layer
2. memory_update: Modify existing memory entry
3. memory_delete: Remove existing memory entry
4. memory_wait: No memory operation needed

At each step, you will receive:
- Current dialogue context
- Current global memory state

You should decide which memory operations (if any) are needed based on the dialogue context and current memory state."""


"""
Example implementation of chat and judge model functions for CareCall Memory RL training

This file shows how to implement chat_model_func and judge_model_func
that can be passed to the reward scoring system.

Note: The reward scoring system now uses agent_r1.utils.llm.query_llm by default,
so you don't need to provide these functions unless you want custom behavior.
"""

import os
from typing import Optional


def create_chat_model_func_using_query_llm(model_name: str = "gpt-4o-2024-11-20"):
    """
    Create a chat model function using query_llm from agent_r1.utils.llm
    
    Args:
        model_name: Model name (e.g., "gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18")
    
    Returns:
        Function that takes a prompt and returns an answer
    """
    from agent_r1.utils.llm import query_llm
    
    def chat_model_func(prompt: str) -> str:
        """
        Call chat model to answer question based on memory state
        
        Args:
            prompt: Prompt containing memory state and question
        
        Returns:
            Answer string
        """
        try:
            result = query_llm(
                model_name=model_name,
                messages=prompt,
                system="You are a medical assistant. Answer questions based on the provided patient memory information.",
                temperature=0.0,
                max_tokens=500
            )
            return result.get('response', '[No response from chat model]').strip()
        except Exception as e:
            print(f"Error calling chat model: {e}")
            return "[Error calling chat model]"
    
    return chat_model_func


def create_judge_model_func_using_query_llm(model_name: str = "DeepSeek-R1"):
    """
    Create a judge model function using query_llm from agent_r1.utils.llm
    
    Args:
        model_name: Model name (e.g., "DeepSeek-R1")
    
    Returns:
        Function that takes a prompt and returns a judgment
    """
    from agent_r1.utils.llm import query_llm
    
    def judge_model_func(prompt: str) -> str:
        """
        Call judge model to determine answer consistency
        
        Args:
            prompt: Prompt containing two answers to compare
        
        Returns:
            Judgment string ("Yes", "No", or score)
        """
        try:
            result = query_llm(
                model_name=model_name,
                messages=prompt,
                system="You are a judge that determines if two answers are consistent (meaning the same thing). Respond with only 'Yes' or 'No'.",
                temperature=0.0,
                max_tokens=50
            )
            return result.get('response', 'No').strip()
        except Exception as e:
            print(f"Error calling judge model: {e}")
            return "No"
    
    return judge_model_func


# Legacy functions (for backward compatibility)
def create_chat_model_func(api_key: Optional[str] = None, model_name: str = "gpt-4o"):
    """
    Create a chat model function for answering memory_query questions
    
    Args:
        api_key: API key for the chat model (if None, uses environment variable)
        model_name: Model name (e.g., "gpt-4o", "gpt-4-turbo")
    
    Returns:
        Function that takes a prompt and returns an answer
    """
    try:
        from openai import OpenAI
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        client = OpenAI(api_key=api_key)
        
        def chat_model_func(prompt: str) -> str:
            """
            Call chat model to answer question based on memory state
            
            Args:
                prompt: Prompt containing memory state and question
            
            Returns:
                Answer string
            """
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a medical assistant. Answer questions based on the provided patient memory information."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error calling chat model: {e}")
                return "[Error calling chat model]"
        
        return chat_model_func
    except ImportError:
        print("OpenAI library not installed. Install with: pip install openai")
        return None


def create_judge_model_func(api_key: Optional[str] = None, model_name: str = "deepseek-chat"):
    """
    Create a judge model function for comparing answers
    
    Args:
        api_key: API key for the judge model (if None, uses environment variable)
        model_name: Model name (e.g., "deepseek-chat" for DeepSeek R1)
    
    Returns:
        Function that takes a prompt and returns a judgment
    """
    try:
        from openai import OpenAI
        
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Judge model API key not provided")
        
        # For DeepSeek, you might need to use a different base URL
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.openai.com/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        def judge_model_func(prompt: str) -> str:
            """
            Call judge model to determine answer consistency
            
            Args:
                prompt: Prompt containing two answers to compare
            
            Returns:
                Judgment string ("Yes", "No", or score)
            """
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a judge that determines if two answers are consistent (meaning the same thing). Respond with only 'Yes' or 'No', or a score from 0.0 to 1.0."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=50
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error calling judge model: {e}")
                return "No"
        
        return judge_model_func
    except ImportError:
        print("OpenAI library not installed. Install with: pip install openai")
        return None


def create_simple_judge_model_func():
    """
    Create a simple judge model function using string comparison (fallback)
    
    Returns:
        Function that compares answers using simple heuristics
    """
    def judge_model_func(prompt: str) -> str:
        """
        Simple judge using string comparison
        
        Args:
            prompt: Prompt containing two answers
        
        Returns:
            "Yes" or "No"
        """
        # Extract answers from prompt (simple parsing)
        if "Answer 1:" in prompt and "Answer 2:" in prompt:
            parts = prompt.split("Answer 1:")
            if len(parts) > 1:
                answer1_part = parts[1].split("Answer 2:")[0].strip()
                answer2_part = parts[1].split("Answer 2:")[1].strip()
                
                answer1_lower = answer1_part.lower()
                answer2_lower = answer2_part.lower()
                
                # Simple comparison
                if answer1_lower == answer2_lower:
                    return "Yes"
                elif answer2_lower in answer1_lower or answer1_lower in answer2_lower:
                    return "Yes"
                else:
                    # Check word overlap
                    words1 = set(answer1_lower.split())
                    words2 = set(answer2_lower.split())
                    if len(words1) > 0 and len(words2) > 0:
                        overlap = len(words1 & words2) / len(words1)
                        if overlap > 0.7:
                            return "Yes"
        
        return "No"
    
    return judge_model_func


# Example usage:
if __name__ == "__main__":
    # Example 1: Using query_llm (recommended)
    print("=== Using query_llm (recommended) ===")
    chat_func = create_chat_model_func_using_query_llm(model_name="gpt-4o-2024-11-20")
    judge_func = create_judge_model_func_using_query_llm(model_name="DeepSeek-R1")
    
    answer = chat_func("Question: What is the patient's condition?")
    print(f"Chat answer: {answer}")
    
    judgment = judge_func("Answer 1: The patient has diabetes. Answer 2: The patient has diabetes.")
    print(f"Judge result: {judgment}")
    
    # Example 2: Using legacy OpenAI API (if needed)
    print("\n=== Using legacy OpenAI API ===")
    chat_func_legacy = create_chat_model_func(model_name="gpt-4o")
    judge_func_legacy = create_judge_model_func(model_name="deepseek-chat")
    
    if chat_func_legacy:
        answer = chat_func_legacy("Question: What is the patient's condition?")
        print(f"Chat answer: {answer}")
    
    if judge_func_legacy:
        judgment = judge_func_legacy("Answer 1: The patient has diabetes. Answer 2: The patient has diabetes.")
        print(f"Judge result: {judgment}")
    
    # Example 3: Using simple judge (no API needed)
    print("\n=== Using simple judge (fallback) ===")
    simple_judge = create_simple_judge_model_func()
    judgment = simple_judge("Answer 1: The patient has diabetes. Answer 2: The patient has diabetes.")
    print(f"Simple judge result: {judgment}")


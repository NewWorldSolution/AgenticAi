import os
import time
from typing import Any, List, Dict

class Agent:
    """Simple agent that can perform a specific task"""
    
    def __init__(self, name: str):
        self.name = name
    
    def run(self, input_data: Any) -> Any:
        """Execute the agent's task"""
        print(f"Agent '{self.name}' processing: {str(input_data)[:30]}...")
        # This would be implemented by specific agents
        return input_data


class ResearchAgent(Agent):
    """Agent that simulates researching a topic"""
    
    def run(self, query: str) -> str:
        print(f"🔍 {self.name} researching: '{query}'")
        time.sleep(0.5)  # Simulate API call
        return f"Research results for '{query}': Found 3 key points about this topic."


class SummarizerAgent(Agent):
    """Agent that summarizes information"""
    
    def run(self, text: str) -> str:
        print(f"📝 {self.name} summarizing text...")
        time.sleep(0.5)  # Simulate processing
        return f"Summary: {text.split(':', 1)[1][:30]}..."

class FactCheckerAgent(Agent):
    """Agent that verifies information and flags suspicious content"""
    
    suspicious_keywords = ["error", "uncertain", "debated"]
    
    def run(self, text: str) -> Dict:
        print(f"✓ {self.name} fact checking...")
        time.sleep(0.5)
        # TODO: Check if any suspicious keywords are in the text
        flags = []  # Replace this with real logic
        for keyword in self.suspicious_keywords:
            if keyword in text.lower():
                flags.append(f"Flagged keyword: {keyword}")
        return {
            "text": text,
            "accuracy": "high",
            "verified_claims": 3,
            "flags": flags
        }

print("=== AGENTIC WORKFLOW DEMO ===")
    
# Create agents
researcher = ResearchAgent("Research Assistant")
fact_checker = FactCheckerAgent("Fact Checker")
summarizer = SummarizerAgent("Summarizer")

print("\n🚀 Starting 'Information Processing' workflow\n")

# Initial input
query = "Agentic workflows in AI systems"

# Step 1: Research
research_results = researcher.run(query)
print(f"  → Output: {str(research_results)[:50]}...\n")

# Step 2: Fact check
fact_check_results = fact_checker.run(research_results)
print(f"  → Output: {str(fact_check_results)[:50]}...\n")

# Step 3: Summarize
summary = summarizer.run(fact_check_results["text"])
print(f"  → Output: {str(summary)[:50]}...\n")

print("✅ Workflow 'Information Processing' completed\n")

print("Final result:")
print(summary)

print("\nKey concepts demonstrated:")
print("1. Agents as components that perform specific tasks")
print("2. Workflow connecting agents in sequence")
print("3. Information flowing through the system")
print("4. Each agent transforming the data")

if __name__ == "__main__":
    # ... (existing main demo code from the demonstration) ...
    print("\n=== TESTING ENHANCED FactCheckerAgent ===")
    test_fact_checker = FactCheckerAgent("Test Enhanced FactChecker")
    text1 = "This report is clear and all facts are confirmed."
    result1 = test_fact_checker.run(text1)
    print(f"Input: '{text1}'\nOutput Flags: {result1['flags']}\n") # Expected: []
    text2 = "The findings suggest a positive trend, but the outcome is still debated and uncertain due to limited data."
    result2 = test_fact_checker.run(text2)
    print(f"Input: '{text2}'\nOutput Flags: {result2['flags']}\n") # Expected: ['debated', 'uncertain']
    text3 = "An error was found in the preliminary report, making some conclusions uncertain."
    result3 = test_fact_checker.run(text3)
    print(f"Input: '{text3}'\nOutput Flags: {result3['flags']}\n") # Expected: ['error', 'uncertain']
import os
from openai import OpenAI
from dotenv import load_dotenv
import threading

# Load environment variables and initialize OpenAI client
load_dotenv()
client = OpenAI(
    base_url="https://openai.vocareum.com/v1", api_key=os.getenv("OPENAI_API_KEY")
)

# Shared dict for thread-safe collection of agent outputs
agent_outputs = {}

# Example contract text (in a real application, this would be loaded from a file)
contract_text = """
CONSULTING AGREEMENT

This Consulting Agreement (the "Agreement") is made effective as of January 1, 2025 (the "Effective Date"), by and between ABC Corporation, a Delaware corporation ("Client"), and XYZ Consulting LLC, a California limited liability company ("Consultant").

1. SERVICES. Consultant shall provide Client with the following services: strategic business consulting, market analysis, and technology implementation advice (the "Services").

2. TERM. This Agreement shall commence on the Effective Date and shall continue for a period of 12 months, unless earlier terminated.

3. COMPENSATION. Client shall pay Consultant a fee of $10,000 per month for Services rendered. Payment shall be made within 30 days of receipt of Consultant's invoice.

4. CONFIDENTIALITY. Consultant acknowledges that during the engagement, Consultant may have access to confidential information. Consultant agrees to maintain the confidentiality of all such information.

5. INTELLECTUAL PROPERTY. All materials developed by Consultant shall be the property of Client. Consultant assigns all right, title, and interest in such materials to Client.

6. TERMINATION. Either party may terminate this Agreement with 30 days' written notice. Client shall pay Consultant for Services performed through the termination date.

7. GOVERNING LAW. This Agreement shall be governed by the laws of the State of Delaware.

8. LIMITATION OF LIABILITY. Consultant's liability shall be limited to the amount of fees paid by Client under this Agreement.

9. INDEMNIFICATION. Client shall indemnify Consultant against all claims arising from use of materials provided by Client.

10. ENTIRE AGREEMENT. This Agreement constitutes the entire understanding between the parties and supersedes all prior agreements.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first above written.
"""
user_prompt = contract_text


def call_openai(
    system_prompt,
    user_prompt,
    model="gpt-3.5-turbo",
    temperature=0.2,
):
    """Simple wrapper for OpenAI API calls."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


class LegalTermsChecker:
    """Agent that checks for problematic legal terms and clauses in contracts."""

    def run(self, contract_text):
        system_prompt = "You are a legal expert specializing in contract law. Review the provided contract text and identify any problematic clauses, ambiguous terms, or non-standard legal language. List your key findings."
        agent_outputs["legal"] = call_openai(system_prompt, contract_text)


class ComplianceValidator:
    """Agent that validates regulatory and industry compliance of contracts."""

    def run(self, contract_text):
        system_prompt = "You are a compliance expert. Analyze the provided contract text for adherence to relevant regulatory standards and industry best practices. Highlight any compliance issues or risks."
        agent_outputs["compliance"] = call_openai(system_prompt, contract_text)


class FinancialRiskAssessor:
    """Agent that assesses financial risks and liabilities in contracts."""

    def run(self, contract_text):
        system_prompt = "You are a financial risk assessor. Evaluate the provided contract text for potential financial risks and liabilities. Identify any clauses that may pose a financial threat to the client."
        agent_outputs["financial"] = call_openai(system_prompt, contract_text)


class SummaryAgent:
    """Agent that synthesizes findings from all specialized agents."""

    def run(self, contract_text, inputs):
        combined_prompt = (
            f"The user asked about to analyze: '{contract_text}'\n\n"
            f"Here are the expert responses:\n"
            f"- Legal Expert: {inputs['legal']}\n\n"
            f"- Compliance Expert: {inputs['compliance']}\n\n"
            f"- Financial Expert: {inputs['financial']}\n\n"
            "Please summarize the combined insights into a single clear and concise response."
        )
        print(f"Summary Agent resolving prompt: {combined_prompt}")
        system_prompt = "You are an expert summarizer. Combine the insights from various experts into a coherent summary."
        user_prompt_summary = combined_prompt
        return call_openai(system_prompt, user_prompt_summary)


# Main function to run all agents in parallel
def analyze_contract(contract_text):
    """Run all agents in parallel and summarize their findings."""
    # Initialize agents
    legal_agent = LegalTermsChecker()
    compliance_agent = ComplianceValidator()
    financial_agent = FinancialRiskAssessor()
    summary_agent = SummaryAgent()
    # Create threads for parallel execution
    threads = []
    threads.append(threading.Thread(target=legal_agent.run, args=(contract_text,)))
    threads.append(threading.Thread(target=compliance_agent.run, args=(contract_text,)))
    threads.append(threading.Thread(target=financial_agent.run, args=(contract_text,)))

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Run summary agent with collected outputs

    final_summary = summary_agent.run(contract_text, agent_outputs)

    return final_summary


def main():
    print("Enterprise Contract Analysis System")
    print("Analyzing contract...")

    # Call the analyze_contract function and print results
    final_analysis = analyze_contract(contract_text)
    print("\n=== FINAL CONTRACT ANALYSIS ===\n")
    print(final_analysis)


if __name__ == "__main__":
    main()

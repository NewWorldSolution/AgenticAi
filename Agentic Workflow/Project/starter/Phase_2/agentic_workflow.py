# agentic_workflow.py
import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
spec_path = BASE_DIR / "Product-Spec-Email-Router.txt"

# TODO: 1 - Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
from workflow_agents.base_agents import (  # noqa: E402
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
    RAGKnowledgePromptAgent,
)


## control the memorty usage of RAG agent
# def set_memory_limit_gb(gb: float):
#     try:
#         import resource  # Unix only

#         limit_bytes = int(gb * 1024**3)
#         resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
#         print(f"[SAFETY] Set hard memory limit to {gb} GB")
#     except Exception as e:
#         print(
#             f"[SAFETY] Could not set hard memory limit (platform may not support it): {e}"
#         )


# set_memory_limit_gb(18)

# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
with spec_path.open("r", encoding="utf-8") as f:
    product_spec = f.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(
    knowledge=knowledge_action_planning, openai_api_key=openai_api_key
)

# RAG Agent for Product Spec
persona_rag_agent = "You are a requirements analyst. Return only relevant product spec details, verbatim or tightly paraphrased."

rag_spec_agent = RAGKnowledgePromptAgent(
    openai_api_key=openai_api_key, persona=persona_rag_agent
)
rag_spec_agent.build_index(product_spec)


def get_relevant_product_spec(step_text: str) -> str:
    """Helper function to return the product specification relevant to the step_text."""
    query = (
        "From the Email Router product specification, extract the most relevant technical requirements "
        "and capabilities for this workflow step. Focus on: RAG, vector database, embeddings, LLM classifiers, "
        "SMTP/IMAP ingestion, confidence scoring, approval workflow, and monitoring metrics.\n\n"
        f"WORKFLOW STEP: {step_text}"
    )

    return rag_spec_agent.find_prompt_in_knowledge(query)


# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    "Each user story MUST reference at least one Email Router capability such as RAG responses, vector DB knowledge base / embeddings for RAG retrieval , LLM classification, SMTP/IMAP ingestion, confidence scoring, or approval workflow. "
    "At least one story MUST explicitly mention a vector database / embeddings for RAG retrieval.\n"
    "IMPORTANT: RAG refers to Retrieval-Augmented Generation (retrieving relevant documents from a vector database and using them as context for LLM-generated responses), NOT Red/Amber/Green classification.\n"
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
    "PRODUCT SPECIFICATION: \n"
    f"{product_spec}"
)


# TODO: 6 - Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    persona=persona_product_manager,
    knowledge=knowledge_product_manager,
    openai_api_key=openai_api_key,
)

# Product Manager - Evaluation Agent
# TODO: 7 - Define the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it as product_manager_evaluation_agent. This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona="You are an evaluation agent that checks the answers of other worker agents",
    evaluation_criteria=(
        "As a [type of user], I want [an action or feature] so that [benefit/value]. \n"
        "It should be at least 5 user stories. \n"
    ),
    worker_agent=product_manager_knowledge_agent,
    max_interactions=10,
)
# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = (
    "You MUST create product features by grouping the user stories from CONTEXT FROM PREVIOUS STEPS.\n\n"
    "Create AT LEAST ONE feature for EACH user story. Do not OMIT any user story.\n"
    "Return features using this EXACT structure for EACH feature (no markdown, no bullets):\n"
    "Feature Name: <clear, concise title>\n"
    "Description: <brief explanation of what the feature does and its purpose>\n"
    "Key Functionality: <specific capabilities/actions provided>\n"
    "User Benefit: <how this feature creates value for the user>\n\n"
    "When defining the RAG feature, Key Functionality MUST include embeddings + vector database retrieval.\n\n"
    "Rules:\n\n"
    "- Output ONLY feature blocks in the exact format above.\n"
    "- Do NOT output numbered lists.\n"
    "- Do NOT output bullet points.\n"
    "- Use the CONTEXT FROM PREVIOUS STEPS as the source of truth.\n"
    "Features must explicitly incorporate the technical mechanisms from the spec when relevant (RAG, vector DB, LLM classifiers, SMTP/IMAP, approval workflow, confidence scoring).\n"
    "IMPORTANT: RAG refers to Retrieval-Augmented Generation (retrieving relevant documents from a vector database and using them as context for LLM-generated responses), NOT Red/Amber/Green classification."
)
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
# (This is a necessary step before TODO 8. Students should add the instantiation code here.)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager,
)
# Program Manager - Evaluation Agent
persona_program_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)

# TODO: 8 - Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=(
        "The response must include at least as many Feature blocks as there are user stories in the context.\n"
        "The answer should be product features that follow the following structure:\n"
        "Feature Name: A clear, concise title that identifies the capability\n"
        "Description: A brief explanation of what the feature does and its purpose\n"
        "Key Functionality: The specific capabilities or actions the feature provides\n"
        "User Benefit: How this feature creates value for the user\n\n"
        "Each Feature must clearly map to exactly one user story.\n"
        "Scoring rules:\n\n"
        "- Score must be <= 5 if the response is not written as Feature blocks with those exact field labels.\n"
        "- If any feature is missing any field label, score must be <= 5.\n"
        "- Score must be <= 5 if any user story is not represented by a feature"
    ),
    worker_agent=program_manager_knowledge_agent,
    max_interactions=10,
    min_acceptable_score=7,
)
# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what needs to be built to implement each user story. "
    "Use the CONTEXT FROM PREVIOUS STEPS in the user prompt as the source of truth. "
    "IMPORTANT: The 'Related User Story' field MUST reference the actual user story text "
    "(starting with 'As a [type of user]...'), NOT the feature name. "
    "User stories are found in Step 1 of the context. Features are organizational groupings and should NOT be referenced in the Related User Story field.\n\n"
    "When applicable, include tasks like implementing RAG pipeline, vector DB setup, embedding ingestion, LLM classifier, confidence scoring, and approval workflow.\n\n"
    "Return tasks using this EXACT structure for EACH task (no markdown, no bullets, no numbered lists, no JSON):\n"
    "Task ID: <unique id>\n"
    "Task Title: <short title>\n"
    "Related User Story: <MUST be the full user story text starting with 'As a ...'>\n"
    "Description: <technical description>\n"
    "Acceptance Criteria: <measurable criteria>\n"
    "Estimated Effort: <time estimate>\n"
    "Dependencies: <task ids or None>\n\n"
    "Rules:\n\n"
    "- The 'Related User Story' must be the FULL user story sentence, not a feature name.\n"
    "- Tasks must include Email Router specifics when relevant: RAG pipeline, vector database, embeddings, "
    "LLM classifier, confidence scoring, SMTP/IMAP ingestion, approval workflow, monitoring metrics.\n"
    "IMPORTANT: RAG refers to Retrieval-Augmented Generation (retrieving relevant documents from a vector database and using them as context for LLM-generated responses), NOT Red/Amber/Green classification.\n"
    "When implementing RAG, tasks MUST involve:\n"
    "- document chunking\n"
    "- embedding generation\n"
    "- vector database retrieval\n"
    "- LLM-based response generation grounded in retrieved context "
    "You MUST create at least one task for EVERY user story in Step 1 (including SMTP/IMAP ingestion and dashboard/monitoring if present). Do not omit any story."
)
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
# (This is a necessary step before TODO 9. Students should add the instantiation code here.)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer,
)
# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)
# TODO: 9 - Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=(
        "The answer should be tasks following this exact structure: "
        "Each task block MUST start with 'Task ID:' and include all field labels exactly.\n"
        "Task ID: A unique identifier for tracking purposes\n"
        "Task Title: Brief description of the specific development work (must not be blank)\n"
        "Related User Story: Reference to the parent user story (must start with 'As a [type of user]...', not a feature name)\n"
        "Description: Detailed explanation of the technical work required (must not be blank)\n"
        "Acceptance Criteria: Specific requirements that must be met for completion (must not be blank)\n"
        "Estimated Effort: Time or complexity estimation\n"
        "Dependencies: Any tasks that must be completed first\n"
        "Validation rules (must be checkable from ANSWER only):\n"
        "- The answer must include at least 5 task blocks (count 'Task ID:' occurrences).\n"
        "- Every 'Related User Story:' line must start with 'As a '. If not, score must be <= 5.\n"
        "Scoring rules:\n"
        "- Score must be <= 5 if 'Related User Story' references a feature name instead of actual user story text.\n"
        "- The worker ANSWER must be plain text (NOT JSON). Only YOUR evaluation output is JSON.\n"
        '- If the answer starts with "{" or "[" OR contains a top-level JSON object/array, score must be 0.\n'
        "- Score must be <= 5 if any task is missing Task Title or Description fields.\n"
        "- Score must be <= 5 if any user story from Step 1 has zero tasks."
    ),
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=10,
    min_acceptable_score=7,
)
# Risk Manager - Knowledge Augmented Prompt Agent
persona_risk_manager = (
    "You are a Risk Manager. "
    "You identify delivery, technical, security, and operational risks based strictly on the provided product specification. "
    "You propose mitigations."
)

knowledge_risk_manager = (
    "Return risks using this exact structure for EACH risk (no markdown):\n"
    "Risk ID: <ID>\n"
    "Risk Title: <title>\n"
    "Description: <what could go wrong>\n"
    "Likelihood: Low/Medium/High\n"
    "Impact: Low/Medium/High\n"
    "Mitigation: <concrete action>\n"
    "Owner: <role>\n"
    "Trigger/Early Warning: <signal>\n"
    "Status: Open/Mitigating/Closed\n\n"
    "Base primarily on PRODUCT SPECIFICATION; context may provide implementation details\n\n"
    "The worker ANSWER must be plain text (NOT JSON). Only YOUR evaluation output is JSON.\n"
    "Output must be plain text in the required Risk ID / Risk Title / ... structure, not JSON.\n\n"
    'If the answer is JSON (starts with "{" or "[" ), score must be 0.\n'
    "Include at least one risk each for: RAG hallucinations/grounding, vector DB/privacy, SMTP/IMAP reliability, classifier confidence/routing errors, approval workflow bypass.\n "
    "IMPORTANT: RAG refers to Retrieval-Augmented Generation (retrieving relevant documents from a vector database and using them as context for LLM-generated responses), NOT Red/Amber/Green classification.\n"
    f"{product_spec}"
)
risk_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_risk_manager,
    knowledge=knowledge_risk_manager,
)
persona_risk_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents. "
    "Base risks primarily on the product specification. "
    "Ignore irrelevant context if present."
)
risk_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_risk_manager_eval,
    evaluation_criteria="""
         The answer must be a risk register with at least 3 risks.
            Each risk must include these fields EXACTLY (case-insensitive is acceptable):
            - Risk ID
            - Risk Title
            - Description
            - Likelihood
            - Impact
            - Mitigation
            - Owner
            - Trigger/Early Warning
            - Status

            Scoring rules:
            - If a field is present in the text, DO NOT claim it is missing.
            - Do NOT invent missing fields.
            - Minor formatting issues should not reduce the score below passing.
            - Passing requires: >=3 risks AND all required fields present for each risk.
            """,
    worker_agent=risk_manager_knowledge_agent,
    max_interactions=10,
    min_acceptable_score=4,
)
# Routing Agent
# TODO: 10 - Instantiate a routing_agent. You will need to define a list of agent dictionaries (routes) for Product Manager, Program Manager, and Development Engineer. Each dictionary should contain 'name', 'description', and 'func' (linking to a support function). Assign this list to the routing_agent's 'agents' attribute.
routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
    agents=[
        {
            "name": "product manager agent",
            "description": (
                "Responsible for defining product personas and user stories only. Does not define features or tasks. Does not group stories"
                "Triggered by prompting mentioning generating or creating user stories."
            ),
            "func": lambda x: product_manager_support_function(x),
        },
        {
            "name": "program manager agent",
            "description": (
                "Responsible for grouping approved user stories into high-level product features only. "
                "Defines feature names, descriptions, key functionality, and user benefits. "
                "Does not create user stories. Does not define development or engineering tasks."
            ),
            "func": lambda x: program_manager_support_function(x),
        },
        {
            "name": "development engineer agent",
            "description": (
                "Responsible for defining detailed engineering and development tasks only. "
                "Breaks approved user stories or features into implementable technical tasks. "
                "Includes task IDs, acceptance criteria, effort estimates, and dependencies. "
                "Triggered by prompts mentioning: tasks, engineering tasks, Task ID, Acceptance Criteria, Dependencies, Estimated Effort. "
                "Does not define user stories or product features."
            ),
            "func": lambda x: development_engineer_support_function(x),
        },
        {
            "name": "risk manager agent",
            "description": (
                "Responsible for identifying project risks and mitigations based on the product specification. "
                "Produces a structured risk assessment (Risk ID, Likelihood, Impact, Mitigation, Owner, Triggers). "
                "Triggered by prompts mentioning: risk assessment, risks, mitigations, risk register. "
                "Does not create user stories, features, or engineering tasks."
            ),
            "func": lambda x: risk_manager_support_function(x),
        },
    ],
)


# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.
def product_manager_support_function(query):
    """Support function for the Product Manager agent."""
    print("PM team working...")
    knowledge_response = product_manager_knowledge_agent.respond(query)

    result = product_manager_evaluation_agent.evaluate(
        {"prompt": query, "worker_response": knowledge_response}
    )
    role = "PM"
    if result["passed"]:
        print(
            f"\n[{role}] ✅ PASSED | iterations={result['iterations']} | score={result['score']}/10\n"
        )
    else:
        print(
            f"\n[{role}] ❌ FAILED | iterations={result['iterations']} | score={result['score']}/10\n"
        )
    return result["final_response"]


def program_manager_support_function(query):
    """Support function for the Program Manager agent."""
    print("PgM team working...")
    knowledge_response = program_manager_knowledge_agent.respond(query)

    result = program_manager_evaluation_agent.evaluate(
        {"prompt": query, "worker_response": knowledge_response}
    )
    role = "PgM"
    if result["passed"]:
        print(
            f"\n[{role}] ✅ PASSED | iterations={result['iterations']} | score={result['score']}/10\n"
        )
    else:
        print(
            f"\n[{role}] ❌ FAILED | iterations={result['iterations']} | score={result['score']}/10\n"
        )
    return result["final_response"]


def development_engineer_support_function(query):
    """Support function for the Development Engineer agent."""
    print("Dev team working...")
    knowledge_response = development_engineer_knowledge_agent.respond(query)

    result = development_engineer_evaluation_agent.evaluate(
        {"prompt": query, "worker_response": knowledge_response}
    )
    role = "Dev"
    if result["passed"]:
        print(
            f"\n[{role}] ✅ PASSED | iterations={result['iterations']} | score={result['score']}/10\n"
        )
    else:
        print(
            f"\n[{role}] ❌ FAILED | iterations={result['iterations']} | score={result['score']}/10\n"
        )
    return result["final_response"]


def risk_manager_support_function(query):
    """Support function for the Risk Manager agent."""
    print("Risk Management team working...")
    knowledge_response = risk_manager_knowledge_agent.respond(query)

    result = risk_manager_evaluation_agent.evaluate(
        {"prompt": query, "worker_response": knowledge_response}
    )
    role = "RiskMgr"
    if result["passed"]:
        print(
            f"\n[{role}] ✅ PASSED | iterations={result['iterations']} | score={result['score']}/10\n"
        )
    else:
        print(
            f"\n[{role}] ❌ FAILED | iterations={result['iterations']} | score={result['score']}/10\n"
        )
    return result["final_response"]


# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = (
    "Using the product specification, create a development plan by doing these steps:\n"
    "1) Generate user stories.\n"
    "2) Group them into product features.\n"
    "3) Create detailed engineering tasks for each user story (include Task ID, Task Title, Related User Story, Description, Acceptance Criteria, Estimated Effort, and Dependencies).\n"
    "4) Generate a risk assessment for the project with mitigations.\n"
    "Return only the plan steps as a numbered list."
)
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# TODO: 12 - Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).


workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []

for step in workflow_steps:
    print(f"\n--- Executing step: {step} ---")
    context = "\n\n".join(completed_steps)
    spec_details = get_relevant_product_spec(step)
    bad_markers = [
        "i don't have information",
        "i do not have information",
        "i don't know",
        "cannot find",
    ]
    if any(m in spec_details.lower() for m in bad_markers):
        spec_details = (
            "Use the Product-Spec-Email-Router.txt capabilities explicitly: "
            "SMTP/IMAP ingestion, LLM classification, confidence scoring, vector database + embeddings, "
            "RAG response generation, approval workflow, routing logic, dashboard metrics, AES-256 + TLS, RBAC + MFA."
        )

    step_prompt = (
        f"{step}\n\n"
        f"CONTEXT FROM PREVIOUS STEPS:\n{context}\n\n"
        f"RELEVANT PRODUCT SPEC DETAILS (use these explicitly; do not ignore):\n{spec_details}"
    )

    try:
        result = routing_agent.route(step_prompt)
        if not isinstance(result, str) or not result.strip():
            result = (
                f"[ERROR] Routing returned empty or non-string result for step: {step}"
            )

    except Exception as e:
        result = f"[ERROR] Step failed: {step}\nException: {type(e).__name__}: {e}"

    completed_steps.append(result)
    print(f"Step result:\n{result}")

sections = []
for i, step in enumerate(workflow_steps, 1):
    step_result = (
        completed_steps[i - 1] if i - 1 < len(completed_steps) else "[MISSING OUTPUT]"
    )
    sections.append(step_result.strip())
    if i < len(workflow_steps) - 1:
        sections.append("\n" + ("-" * 60) + "\n")
    # sections.append(f"=== STEP {i}: {step} ===\n{step_result}")

print("\n*** Workflow execution completed ***\n")
final_output = "=== CONSOLIDATED EMAIL ROUTER PROJECT PLAN ===\n\n" + "\n\n".join(
    sections
)
print(f"Final workflow output:\n{final_output}")

output_file = BASE_DIR / "agentic_workflow_output.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("AGENTIC WORKFLOW OUTPUT - EMAIL ROUTER PROJECT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Workflow Prompt: {workflow_prompt}\n\n")
    f.write("=" * 80 + "\n")
    f.write("FINAL OUTPUT:\n")
    f.write("=" * 80 + "\n\n")
    f.write(final_output + "\n\n")
    f.write("=" * 80 + "\n")
    f.write("FULL STEP OUTPUTS (FOR REFERENCE):\n")
    f.write("=" * 80 + "\n\n")
    for i, step_result in enumerate(completed_steps, 1):
        step_name = (
            workflow_steps[i - 1] if i - 1 < len(workflow_steps) else "[UNKNOWN STEP]"
        )
        f.write(f"\n--- Step {step_name} ---\n")
        f.write(step_result)
        f.write("\n\n")

print(f"\nSaved final workflow output to: {output_file}")
# output_file = BASE_DIR / "agentic_workflow_output.txt"
# with open(output_file, "w", encoding="utf-8") as f:
#     f.write("=" * 80 + "\n")
#     f.write("AGENTIC WORKFLOW OUTPUT - EMAIL ROUTER PROJECT\n")
#     f.write("=" * 80 + "\n\n")
#     f.write(f"Workflow Prompt: {workflow_prompt}\n\n")
#     f.write("=" * 80 + "\n")
#     f.write("FINAL OUTPUT:\n")
#     f.write("=" * 80 + "\n\n")
#     f.write(completed_steps[-1])
# print(f"\nOutput saved to: {output_file}")

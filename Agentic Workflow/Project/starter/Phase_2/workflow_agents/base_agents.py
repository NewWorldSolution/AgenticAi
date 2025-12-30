# TODO: 1 - import the OpenAI class from the openai library
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
from openai import OpenAI
import json

BASE_URL = "https://openai.vocareum.com/v1"


# DirectPromptAgent class definition
class DirectPromptAgent:

    def __init__(self, openai_api_key):
        # Initialize the agent
        # TODO: 2 - Define an attribute named openai_api_key to store the OpenAI API key provided to this class.
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        client = OpenAI(api_key=self.openai_api_key, base_url=BASE_URL)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # TODO: 4 - Provide the user's prompt here. Do not add a system prompt.
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        # TODO: 5 - Return only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content


# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        # TODO: 1 - Create an attribute for the agent's persona
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = OpenAI(api_key=self.openai_api_key, base_url=BASE_URL)

        # TODO: 2 - Declare a variable 'response' that calls OpenAI's API for a chat completion.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # TODO: 3 - Add a system prompt instructing the agent to assume the defined persona and explicitly forget previous context.
                {
                    "role": "system",
                    "content": f"{self.persona}\nFollow these persona instructions exactly.\nForget all previous context.",
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )

        return response.choices[0].message.content


# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        # TODO: 1 - Create an attribute to store the agent's knowledge.
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(api_key=self.openai_api_key, base_url=BASE_URL)
        system_message = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}\n"
            f"Answer the prompt based on this knowledge, not your own."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # TODO: 2 - Construct a system message including:
                #           - The persona with the following instruction:
                #             "You are _persona_ knowledge-based assistant. Forget all previous context."
                #           - The provided knowledge with this instruction:
                #             "Use only the following knowledge to answer, do not use your own knowledge: _knowledge_"
                #           - Final instruction:
                #             "Answer the prompt based on this knowledge, not your own."
                {"role": "system", "content": system_message},
                # TODO: 3 - Add the user's input prompt here as a user message.,
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(
        self,
        openai_api_key,
        persona,
        chunk_size=1200,
        chunk_overlap=100,
        max_chunks=80,
        top_k=3,
        max_context_chars=6000,
    ):
        """
        Memory-safe RAG agent:
        - Chunk knowledge once
        - Embed chunks once (bounded by max_chunks)
        - Store embeddings in-memory as float32 NumPy matrix
        """
        self.persona = persona
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.max_chunks = int(max_chunks)
        self.top_k = int(top_k)
        self.max_context_chars = int(max_context_chars)
        self.openai_api_key = openai_api_key

        # In-memory index
        self._chunks = []
        self._embeddings = None
        self.unique_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        )

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url=BASE_URL, api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r"[ \t]+", " ", text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunk = text[start:end].strip()

            if chunk:
                chunks.append({"chunk_id": chunk_id, "text": chunk})
                chunk_id += 1

            start = max(0, end - self.chunk_overlap)

            if chunk_id > self.max_chunks * 5:
                break
        return chunks

    def build_index(self, knowledge_text: str):
        """
        Build the in-memory vector index. Call ONCE (e.g., at startup).
        """
        chunks = self.chunk_text(knowledge_text)[: self.max_chunks]
        self._chunks = [c["text"] for c in chunks]

        if not self._chunks:
            self._embeddings = None
            return

        embs = []
        for t in self._chunks:
            e = np.array(self.get_embedding(t), dtype=np.float32)
            embs.append(e)

        self._embeddings = np.vstack(embs)  # shape: (n, d)

    def retrieve(self, prompt: str, top_k: int | None = None) -> str:
        """Return concatenated top-k most similar chunks."""
        if self._embeddings is None or not self._chunks:
            return ""

        k = self.top_k if top_k is None else int(top_k)

        q = np.array(self.get_embedding(prompt), dtype=np.float32)

        # cosine similarity
        denom = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q) + 1e-9
        sims = (self._embeddings @ q) / denom  # shape (n,)

        top_idx = np.argsort(-sims)[:k]
        retrieved = "\n\n".join(self._chunks[i] for i in top_idx)

        # hard-limit context size to keep prompts small and stable
        if len(retrieved) > self.max_context_chars:
            retrieved = retrieved[: self.max_context_chars] + "..."

        return retrieved

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["text"].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding="utf-8", index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        # prompt_embedding = self.get_embedding(prompt)
        # df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding="utf-8")
        # df["embeddings"] = df["embeddings"].apply(lambda x: np.array(eval(x)))
        # df["similarity"] = df["embeddings"].apply(
        #     lambda emb: self.calculate_similarity(prompt_embedding, emb)
        # )

        # best_chunk = df.loc[df["similarity"].idxmax(), "text"]
        retrieved = self.retrieve(prompt)

        client = OpenAI(base_url=BASE_URL, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona}. Forget previous context. "
                        "You MUST answer using ONLY the retrieved context. "
                        "If the retrieved context does not contain the answer, say you don't know."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Retrieved context:\n{retrieved}\n\nPrompt:\n{prompt}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content


class EvaluationAgent:

    def __init__(
        self,
        openai_api_key,
        persona,
        evaluation_criteria,
        worker_agent,
        max_interactions,
        min_acceptable_score=6,
    ):
        # Initialize the EvaluationAgent with given attributes.
        # TODO: 1 - Declare class attributes here
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions
        self.min_acceptable_score = min_acceptable_score

    def respond(self, initial_prompt):
        # This method is a placeholder to comply with the agent interface.
        return self.evaluate(initial_prompt)

    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        client = OpenAI(api_key=self.openai_api_key, base_url=BASE_URL)
        if isinstance(initial_prompt, dict):
            original_prompt = initial_prompt.get("prompt", "")
            prompt_to_evaluate = original_prompt
            provided_worker_response = initial_prompt.get("worker_response", None)
        else:
            original_prompt = initial_prompt
            prompt_to_evaluate = initial_prompt
            provided_worker_response = None
        final_response = ""
        final_evaluation = ""
        iterations_used = 0

        for i in range(
            self.max_interactions
        ):  # TODO: 2 - Set loop to iterate up to the maximum number of interactions:
            agent_name = self.persona.split(".")[0]
            print(f"\n[{agent_name}] Iteration {i+1}/{self.max_interactions}")
            iterations_used += 1
            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            if provided_worker_response is not None:
                response_from_worker = provided_worker_response
                provided_worker_response = None
            else:
                response_from_worker = self.worker_agent.respond(
                    prompt_to_evaluate
                )  # TODO: 3 - Obtain a response from the worker agent
            print(f"Worker Agent Response:\n{response_from_worker}")
            final_response = response_from_worker
            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = f"""
                Evaluate the ANSWER against the CRITERIA.

                CRITERIA:
                {self.evaluation_criteria}

                ANSWER:
                {response_from_worker}
                
                Return ONLY valid JSON (no markdown, no backticks, no extra text) that matches EXACTLY this schema:
                    {{
                    "score": 0-10 integer,
                    "issues": ["..."],
                    "fix_plan": {{
                        "must_add": ["..."],
                        "must_change": ["..."],
                        "must_remove": ["..."],
                        "rewrite_rules": ["..."]
                    }},
                    "revised_example": "one short corrected example",
                    "instructions": "1) ...\\n2) ...\\n3) ..."
                    }}

                    Rule:
                    - If score >= {self.min_acceptable_score}:
                    - issues must be []
                    - instructions must be ""
                    - fix_plan lists must all be []
                    """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[  # TODO: 5 - Define the message structure sent to the LLM for evaluation (use temperature=0)
                    {
                        "role": "system",
                        "content": self.persona,
                    },
                    {"role": "user", "content": eval_prompt},
                ],
                temperature=0,
            )
            data = None
            for eval_attempt in range(2):
                raw = response.choices[0].message.content.strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                try:
                    start = raw.find("{")
                    end = raw.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        raw = raw[start : end + 1]
                    data = json.loads(raw)
                    rev = data.get("revised_example", "")
                    if rev is None:
                        data["revised_example"] = ""
                    elif not isinstance(rev, str):
                        data["revised_example"] = json.dumps(rev, ensure_ascii=False)
                    break
                except json.JSONDecodeError:
                    if eval_attempt == 0:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {
                                    "role": "system",
                                    "content": self.persona,
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        "Your previous response was not valid JSON. "
                                        "Return ONLY valid JSON that matches the required schema.\n\n"
                                        f"{eval_prompt}\n\n"
                                        f"INVALID RESPONSE:\n{raw}"
                                    ),
                                },
                            ],
                            temperature=0,
                        )
                    else:
                        # fallback: treat as failure and request reformat next iteration
                        data = {
                            "score": 0,
                            "issues": ["Evaluator returned invalid JSON."],
                            "fix_plan": {
                                "must_add": [],
                                "must_change": [],
                                "must_remove": [],
                                "rewrite_rules": [],
                            },
                            "revised_example": "",
                            "instructions": (
                                "Keep the worker answer format unchanged. "
                                "Rewrite the answer to satisfy the CRITERIA (required fields/structure). "
                                "Do NOT output JSON unless the CRITERIA explicitly requires JSON."
                            ),
                        }

            print(f"Evaluator Agent Evaluation:\n{data}")
            final_evaluation = data
            print(" Step 3: Check if evaluation is positive")
            if data.get("score", 0) >= self.min_acceptable_score:
                print(f"✅ Accepted with score {data['score']}/10")
                break

            instructions = data.get("instructions", "").strip()
            if not instructions:
                instructions = "Rewrite the answer to match the CRITERIA exactly. Follow the required structure strictly."
            print(f"Instructions to fix:\n{instructions}")

            #               print(" Step 4: Generate instructions to correct the response")
            #              instruction_prompt = (
            #                    f"Evaluation criteria:\n{self.evaluation_criteria}\n\n"
            #                    f"The original prompt was: {initial_prompt}\n"
            #                    f"The response to that prompt was: {response_from_worker}\n"
            #                    f"It has been evaluated as insufficient (score {data.get('score', 0)}/10).\n"
            #                    f"Make only these corrections, do not alter content validity: {instructions}"
            #                    f"Return instructions only."
            #                )
            #                response = client.chat.completions.create(
            #                    model="gpt-3.5-turbo",
            #                    messages=[{"role": "user", "content": instruction_prompt}],
            #                    temperature=0,
            #                )

            print(" Step 5: Send feedback to worker agent for refinement")
            prompt_to_evaluate = (
                f"ORIGINAL PROMPT:\n{original_prompt}\n\n"
                f"PREVIOUS ANSWER:\n{response_from_worker}\n\n"
                f"REQUIRED FIXES (follow exactly):\n{instructions}\n\n"
                f"Now rewrite the answer so it fully satisfies the CRITERIA."
            )
        passed = (
            final_evaluation.get("score", 0) >= self.min_acceptable_score
            if isinstance(final_evaluation, dict)
            else False
        )
        score = (
            final_evaluation.get("score", 0)
            if isinstance(final_evaluation, dict)
            else 0
        )
        return {
            "final_response": final_response,
            "evaluation": final_evaluation,
            "iterations": iterations_used,
            "passed": passed,
            "score": score,
        }
        # TODO: 7 - Return a dictionary containing the final response, evaluation, and number of iterations


class RoutingAgent:

    def __init__(self, openai_api_key, agents):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        # TODO: 1 - Define an attribute to hold the agents, call it agents
        self.agents = agents
        # Precompute embeddings for each agent description once
        self.last_selected_agent_name = None
        self.last_selected_agent_score = None

        for agent in self.agents:
            agent["description_embedding"] = self.get_embedding(agent["description"])

    def respond(self, prompt):
        return self.route(prompt)

    def get_embedding(self, text):
        client = OpenAI(api_key=self.openai_api_key, base_url=BASE_URL)
        # TODO: 2 - Write code to calculate the embedding of the text using the text-embedding-3-large model
        # Extract and return the embedding vector from the response
        response = client.embeddings.create(
            model="text-embedding-3-large", input=text, encoding_format="float"
        )
        embedding = response.data[0].embedding
        return embedding

    # TODO: 3 - Define a method to route user prompts to the appropriate agent
    def route(self, user_input):
        """Route the user input to the most suitable agent based on similarity of embeddings."""
        # TODO: 4 - Compute the embedding of the user input prompt
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # TODO: 5 - Compute the embedding of the agent description
            agent_emb = agent["description_embedding"]
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (
                np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
            )
            print(similarity)

            # TODO: 6 - Add logic to select the best agent based on the similarity score between the user prompt and the agent descriptions
            if similarity > best_score:
                best_score = similarity
                best_agent = agent
        if best_agent is None:
            self.last_selected_agent_name = None
            self.last_selected_agent_score = None
            return "Sorry, no suitable agent could be selected."
        self.last_selected_agent_name = best_agent["name"]
        self.last_selected_agent_score = best_score
        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge):
        # TODO: 1 - Initialize the agent attributes here
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def respond(self, prompt):
        return self.extract_steps_from_prompt(prompt)

    def extract_steps_from_prompt(self, prompt):

        # TODO: 2 - Instantiate the OpenAI client using the provided API key
        client = OpenAI(api_key=self.openai_api_key, base_url=BASE_URL)
        # TODO: 3 - Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        # Provide the following system prompt along with the user's prompt:
        # "You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {pass the knowledge here}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an action planning agent. Using your knowledge, you extract from the user prompt the steps requested to complete the action the user is asking for. You return the steps as a list. Only return the steps in your knowledge. Forget any previous context. This is your knowledge: {self.knowledge}",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        response_text = response.choices[0].message.content
        # TODO: 4 - Extract the response text from the OpenAI API response

        # TODO: 5 - Clean and format the extracted steps by removing empty lines and unwanted text
        steps = [line.strip() for line in response_text.split("\n") if line.strip()]

        return steps

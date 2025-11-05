import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(
    base_url="https://openai.vocareum.com/v1", api_key=os.getenv("OPENAI_API_KEY")
)

MAX_RETRIES = 5

# Example user constraints
RECIPE_REQUEST = {
    "base_dish": "stew",
    "constraints": [
        "pork meat-free",
        "halal",
        "ketogenic",
        "high protein (>15g per serving)",
        "for 4 people",
        "taste must be rated 7/10 or higher",
        "similar to turkish cuisine",
        "easy to prepare (under 45 minutes total cook time)",
        "difficulty level must be rated easy",
    ],
}


class RecipeCreatorAgent:
    def create_recipe(self, recipe_request_dict, feedback=None):
        system_message = "You are an innovative and highly skilled chef, renowned for creating delicious recipes that also meet specific dietary and nutritional targets. You are good at interpreting user requests and also at refining your creations based on precise feedback"
        base_dish = recipe_request_dict["base_dish"]
        constraints_str = ", ".join(recipe_request_dict["constraints"])
        user_prompt_text = f"Create a detailed recipe for a {base_dish} that meets ALL of the following constraints: {constraints_str}."
        # Modify the prompt if feedback is provided

        if feedback:
            user_prompt_text += f"\n\nIMPORTANT: Your previous attempt had issues. Please revise the recipe based on this specific feedback: {feedback}\nEnsure all original constraints AND this feedback are addressed."
        else:
            user_prompt_text += "\nThis is the first attempt."
            user_prompt_text += "\n\nPlease provide: a creative name for the dish, a list of ingredients (with quantities), step-by-step instructions, an estimated calorie count per serving, an estimated protein content (grams) per serving, and a short description of its taste profile."

        print(f"\n🍳 Generating recipe with prompt:\n{user_prompt_text}\n")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt_text},
            ],
            temperature=1.0,
        )
        return response.choices[0].message.content


class NutritionEvaluatorAgent:
    def evaluate_recipe(self, recipe_text, original_request_dict):
        print("🔍 Evaluating recipe nutrition and constraints...")
        system_message = (
            "You are an extremely precise nutrition and dietary compliance evaluator. Your role is to meticulously assess a given recipe against a specific set of user-defined constraints. "
            "For each constraint, you must clearly state if it 'PASSED' or 'FAILED'. If a constraint FAILED, you must provide a concise reason and a actionable suggestion for improvement. "
            "You also need to provide an overall taste rating based on the recipe description."
        )

        eval_prompt = f"""Please evaluate the ""{recipe_text}"" above against EACH of the following constraints from the original REQUEST:
                    Original Request Constraints: ""{', '.join(original_request_dict['constraints'])}""
                    For each constraint, state the constraint verbatim, then write 'PASSED' or 'FAILED'. 
                    If 'FAILED', provide a brief reason and a specific suggestion for fixing it.
                    Example for one constraint:
                    'gluten-free: PASSED'
                    'under 500 calories per serving: FAILED - Estimated 650 calories. Suggest reducing oil by half.'
                    After evaluating all constraints, provide a line with 'Taste Rating: [N]/10' based on the recipe description (where N is a number).
                    Finally, on a new line, write 'Overall Status: PASSED' if ALL constraints are met (including taste rating of 7 or higher for the constraint 'taste must be rated 7/10 or higher')
                    OR 'Overall Status: FAILED' if ANY constraint is not met."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": eval_prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()


def optimize_recipe():
    recipe_agent = RecipeCreatorAgent()
    evaluator_agent = NutritionEvaluatorAgent()

    current_feedback = None

    for attempt in range(MAX_RETRIES):
        print(f"--- Attempt #{attempt + 1} ---")
        recipe_text = recipe_agent.create_recipe(RECIPE_REQUEST, current_feedback)
        evaluation = evaluator_agent.evaluate_recipe(recipe_text, RECIPE_REQUEST)

        print(f"\n🧾 Evaluation Result:\n{evaluation}\n")

        if "Overall Status: PASSED" in evaluation:
            print("\n✅ Final Approved Recipe:\n")
            print(recipe_text)
            break
        else:
            current_feedback = evaluation
    else:
        print("\n❌ Failed to meet all constraints after max retries.")
        print("Last version of the recipe:")
        print(recipe_text)


if __name__ == "__main__":
    print("🚀 Starting Recipe Optimization Process...")
    optimize_recipe()

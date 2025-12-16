import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    base_url="https://openai.vocareum.com/v1", api_key=os.getenv("OPENAI_API_KEY")
)


def get_llm_response(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM request failed: {str(e)}"


def feedstock_analyst_agent(feedstock_name):

    system_prompt = """
    You are a petrochemical expert analyzing hydrocarbon feedstocks. 
    Provide a concise analysis of the given feedstock, highlighting its key components and 
    general suitability for producing valuable refined products like gasoline, diesel, and kerosene
"""
    user_prompt = f"Analyze the following feedstock: {feedstock_name}"
    print("Feedstock Analyst Agent working...")
    result = get_llm_response(system_prompt, user_prompt)
    print("Feedstock Analyst Agent completed. Response received.")
    return result


def distillation_planner_agent(feedstock_analysis):
    system_prompt = """
    You are a refinery distillation tower operations planner. 
    Based on the provided feedstock analysis, estimate the potential percentage yields for major 
    products like gasoline, diesel, and kerosene. Be realistic.
    """
    user_prompt = "Based on this feedstock analysis, provide a distillation plan: {feedstock_analysis}. Example output: (Based on the analysis, potential yields are: Gasoline: 40%, Diesel: 30%, Kerosene: 20%, Other: 10%)"
    print("Distillation Planner Agent working...")
    result = get_llm_response(system_prompt, user_prompt)
    print("Distillation Planner Agent completed. Response received.")
    return result


def market_analyst_agent(product_list):
    system_prompt = """
    You are an energy market analyst. For the following list of refined products, provide a brief analysis of 
    current market demand (high, medium, low) and general profitability trends.
"""
    user_prompt = f"Analyze the market for the following products: {product_list}"
    print("Market Analyst Agent working...")
    result = get_llm_response(system_prompt, user_prompt)
    print("Market Analyst Agent completed. Response received.")
    return result


def production_optimizer_agent(distillation_plan, market_data):
    system_prompt = """
    You are a refinery production optimization expert. Your goal is to recommend a production strategy 
    based on potential yields and current market conditions.
"""
    user_prompt = f"Given the distillation plan: {distillation_plan} and market data: {market_data}, Please provide a concise recommendation on which products the refinery should prioritize or  focus on to maximize value, considering both the potential yield and market conditions."
    print("Production Optimizer Agent working...")
    result = get_llm_response(system_prompt, user_prompt)
    print("Production Optimizer Agent completed. Response received.")
    return result


if __name__ == "__main__":
    feedstock_name = "West Texas Intermediate Crude Oil"
    print(f"Processing feedstock: {feedstock_name}\n")
    feedstock_analysis = feedstock_analyst_agent(feedstock_name)
    print(f"\n--- Feedstock Analysis ---\n{feedstock_analysis}\n")
    distillation_plan = distillation_planner_agent(feedstock_analysis)
    print(f"\n--- Distillation Plan ---\n{distillation_plan}\n")
    market_data = market_analyst_agent(distillation_plan)
    print(f"\n--- Market Data ---\n{market_data}\n")
    optimization_recommendation = production_optimizer_agent(
        distillation_plan, market_data
    )
    print(
        f"\n--- Production Optimization Recommendation ---\n{optimization_recommendation}\n"
    )

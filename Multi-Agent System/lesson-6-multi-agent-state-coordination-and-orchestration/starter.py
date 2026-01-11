"""
Pasta Factory Exercise - Starter Code
====================================

In this exercise, you'll extend the Italian Pasta Factory multi-agent system
to handle more complex scenarios with shared state coordination and proper
multi-agent orchestration patterns.

You'll need to:
1. Implement the missing production and custom recipe tools
2. Create the CustomPastaDesignerAgent
3. Build the proper Orchestrator using ToolCallingAgent
4. Add coordination tools that route requests between specialized agents

This demonstrates extending multi-agent systems with new capabilities while
maintaining proper orchestration patterns.
"""

import re
import stat
from tabnanny import check
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import random
from dataclasses import dataclass, field, asdict

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)

# Load your OpenAI API key
import os
import dotenv

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)

# Pasta Factory State Management


@dataclass
class PastaOrder:
    order_id: str
    pasta_shape: str
    quantity: float  # in kg
    status: str = "pending"  # pending, queued, completed, cancelled
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: int = 1  # 1 = normal, 2 = rush, 3 = emergency
    customer_notes: str = ""
    estimated_delivery_date: str = ""


@dataclass
class FactoryState:
    inventory: Dict[str, float] = field(
        default_factory=lambda: {
            "flour": 10.0,  # kg
            "water": 5.0,  # liters
            "eggs": 24,  # count
            "semolina": 8.0,  # kg
        }
    )
    production_queue: List[PastaOrder] = field(default_factory=list)
    pasta_recipes: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "spaghetti": {"flour": 0.2, "water": 0.1},
            "fettuccine": {"flour": 0.25, "water": 0.1},
            "penne": {"flour": 0.2, "water": 0.1},
            "ravioli": {"flour": 0.3, "water": 0.1, "eggs": 2},
            "lasagna": {"flour": 0.3, "water": 0.15, "eggs": 3},
        }
    )
    custom_recipes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    order_counter: int = 0
    known_pasta_shapes: List[str] = field(
        default_factory=lambda: [
            "spaghetti",
            "fettuccine",
            "penne",
            "ravioli",
            "lasagna",
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inventory": self.inventory,
            "production_queue": [asdict(order) for order in self.production_queue],
            "pasta_recipes": self.pasta_recipes,
            "custom_recipes": self.custom_recipes,
        }

    def update_known_pasta_shapes(self):
        """Update the list of known pasta shapes based on recipes."""
        self.known_pasta_shapes = list(self.pasta_recipes.keys()) + list(
            self.custom_recipes.keys()
        )


# Initialize the shared factory state
factory_state = FactoryState()

# ======= Agent Tools =======


@tool
def check_pasta_recipe(pasta_shape: str) -> Dict[str, float]:
    """
    Check what ingredients are needed for a specific pasta shape.
    Returns a dictionary of ingredients and amounts needed per kg of pasta.
    """
    if pasta_shape in factory_state.pasta_recipes:
        return factory_state.pasta_recipes[pasta_shape]
    elif pasta_shape in factory_state.custom_recipes:
        return factory_state.custom_recipes[pasta_shape]
    return {}


@tool
def check_inventory() -> Dict[str, float]:
    """Check current inventory levels of all ingredients."""
    return factory_state.inventory


@tool
def generate_order_id() -> str:
    """Generate a unique order ID."""
    factory_state.order_counter += 1
    return f"ORD-{factory_state.order_counter:04d}"


@tool
def list_available_pasta_shapes() -> List[str]:
    """List all available pasta shapes that can be ordered."""
    return factory_state.known_pasta_shapes


@tool
def update_inventory(ingredient: str, amount: float) -> Dict[str, Any]:
    """
    Update the inventory amount for a specific ingredient.

    Args:
        ingredient: Name of the ingredient
        amount: New amount (will replace current amount)

    Returns:
        Status of the inventory update
    """
    if ingredient not in factory_state.inventory:
        return {
            "success": False,
            "message": f"Unknown ingredient: {ingredient}. Cannot update inventory.",
        }

    old_amount = factory_state.inventory[ingredient]
    factory_state.inventory[ingredient] = amount

    return {
        "success": True,
        "message": f"Inventory updated: {ingredient} from {old_amount} to {amount}.",
        "ingredient": ingredient,
        "old_amount": old_amount,
        "new_amount": amount,
    }


@tool
def check_production_capacity(days_ahead: int = 7) -> Dict[str, Any]:
    """
    Check the current production capacity and queue for the next X days.
    Returns information about queue size and estimated completion times.
    """
    queue_size = len(factory_state.production_queue)

    # Calculate the total production volume (in kg)
    total_volume = sum(order.quantity for order in factory_state.production_queue)

    # Simple capacity estimation: assume we can produce 10kg per day
    daily_capacity = 10.0  # kg per day
    days_to_complete = max(1, total_volume / daily_capacity)

    # Consider priority orders
    priority_orders = [o for o in factory_state.production_queue if o.priority > 1]
    priority_volume = sum(order.quantity for order in priority_orders)

    return {
        "queue_size": queue_size,
        "total_volume_kg": total_volume,
        "days_to_complete_current_queue": days_to_complete,
        "daily_capacity_kg": daily_capacity,
        "priority_orders": len(priority_orders),
        "priority_volume_kg": priority_volume,
    }


# TODO: Implement the following tools


@tool
def add_to_production_queue(
    order_id: str,
    pasta_shape: str,
    quantity: float,
    priority: int = 1,
    customer_notes: str = "",
) -> Dict[str, Any]:
    """
    Add an order to the production queue.

    Args:
        order_id: Unique order identifier
        pasta_shape: Type of pasta to produce
        quantity: Amount in kg
        priority: Order priority (1=normal, 2=rush, 3=emergency)
        customer_notes: Additional notes from customer

    Returns:
        Status of the queuing operation with estimated delivery date
    """
    # TODO: Implement this function
    # 1. Verify that pasta_shape is valid using check_pasta_recipe
    # 2. Calculate required ingredients and check inventory availability
    # 3. Create PastaOrder object and add to factory_state.production_queue
    # 4. Calculate estimated delivery date based on priority and production capacity
    # 5. Update inventory by subtracting required ingredients
    # 6. Return success status with delivery date
    if pasta_shape not in factory_state.known_pasta_shapes:
        return {
            "success": False,
            "message": f"Unknown pasta shape: {pasta_shape}. Cannot add to production queue.",
        }
    required_ingredients = check_pasta_recipe(pasta_shape)
    for ingredient, amount_per_kg in required_ingredients.items():
        total_required = amount_per_kg * quantity
        if factory_state.inventory.get(ingredient, 0) < total_required:
            return {
                "success": False,
                "message": f"Insufficient {ingredient} for order {order_id}. Required: {total_required}, Available: {factory_state.inventory.get(ingredient, 0)}",
            }

    # Simple estimated delivery date calculation
    capacity_info = check_production_capacity()

    days_to_complete = capacity_info["days_to_complete_current_queue"]

    if priority > 1:
        if priority == 3:
            days_to_complete = 1
        elif priority == 2:
            days_to_complete = max(1, days_to_complete // 2)
    total_days = max(1, int(days_to_complete) + 1)
    estimated_delivery_date = datetime.now() + timedelta(days=total_days).strftime(
        "%Y-%m-%d"
    )
    order = PastaOrder(
        order_id=order_id,
        pasta_shape=pasta_shape,
        quantity=quantity,
        priority=priority,
        customer_notes=customer_notes,
        estimated_delivery_date=estimated_delivery_date,
        status="queued",
    )
    factory_state.production_queue.append(order)
    # Update inventory
    for ingeredient, required in required_ingredients.items():
        factory_state.inventory[ingeredient] -= required
    return {
        "success": True,
        "message": f"Order {order_id} added to production queue.",
        "estimated_delivery_date": estimated_delivery_date,
        "priority": priority,
        "status": "queued",
    }


@tool
def create_custom_pasta_recipe(
    pasta_name: str, ingredients: Dict[str, float]
) -> Dict[str, Any]:
    """
    Create a custom pasta recipe with specific ingredient ratios.

    Args:
        pasta_name: Name of the custom pasta
        ingredients: Dictionary mapping ingredient names to amounts needed per kg

    Returns:
        Status of the recipe creation
    """
    # TODO: Implement this function
    # 1. Validate ingredients exist in factory_state.inventory
    # 2. Check if recipe name already exists
    # 3. Add the custom recipe to factory_state.custom_recipes
    # 4. Update factory_state.known_pasta_shapes using update_known_pasta_shapes()
    # 5. Return success status with recipe details
    for ingredient in ingredients:
        if ingredient not in factory_state.inventory:
            return {
                "success": False,
                "message": f"Unknown ingredient: {ingredient}. Cannot create custom recipe.",
            }
    if (
        pasta_name in factory_state.pasta_recipes
        or pasta_name in factory_state.custom_recipes
    ):
        return {
            "success": False,
            "message": f"Recipe name {pasta_name} already exists. Choose a different name.",
        }
    # Add custom recipe
    factory_state.custom_recipes[pasta_name] = ingredients
    # Update known pasta shapes
    factory_state.update_known_pasta_shapes()
    return {
        "success": True,
        "message": f"Custom pasta recipe {pasta_name} created successfully.",
        "recipe": ingredients,
    }


@tool
def prioritize_order(order_id: str, new_priority: int) -> Dict[str, Any]:
    """
    Change the priority of an existing order in the queue.

    Args:
        order_id: ID of the order to update
        new_priority: New priority level (1=normal, 2=rush, 3=emergency)

    Returns:
        Status of the priority change
    """
    # TODO: Implement this function
    # 1. Validate priority level (1, 2, or 3)
    # 2. Find the order in factory_state.production_queue
    # 3. Update the order's priority
    # 4. Recalculate estimated delivery date based on new priority
    # 5. Return success status with new delivery date
    if new_priority not in [1, 2, 3]:
        return {
            "success": False,
            "message": f"Invalid priority level: {new_priority}. Must be 1, 2, or 3.",
        }
    order_found = False
    for order in factory_state.production_queue:
        if order.order_id == order_id:
            order_found = True
            old_priority = order.priority
            order.priority = new_priority
            # Recalculate estimated delivery date
            capacity_info = check_production_capacity()
            days_to_complete = capacity_info["days_to_complete_current_queue"]
            if new_priority > 1:
                if new_priority == 3:
                    days_to_complete = 1
                elif new_priority == 2:
                    days_to_complete = max(1, days_to_complete // 2)
            total_days = max(1, int(days_to_complete) + 1)
            order.estimated_delivery_date = (
                datetime.now() + timedelta(days=total_days)
            ).strftime("%Y-%m-%d")
            return {
                "success": True,
                "message": f"Order {order_id} priority updated to {new_priority}.",
                "new_estimated_delivery_date": order.estimated_delivery_date,
            }
    if not order_found:
        return {
            "success": False,
            "message": f"Order ID {order_id} not found in production queue.",
        }


# ======= Agents =======


class OrderProcessorAgent(ToolCallingAgent):
    """Agent responsible for processing customer order requests."""

    def __init__(self, model):
        super().__init__(
            tools=[check_pasta_recipe, generate_order_id, list_available_pasta_shapes],
            model=model,
            name="order_processor",
            description="Agent responsible for processing customer orders. Parses requests, identifies pasta shapes and quantities.",
        )


class InventoryManagerAgent(ToolCallingAgent):
    """Agent responsible for managing ingredient inventory."""

    def __init__(self, model):
        super().__init__(
            tools=[check_inventory, check_pasta_recipe, update_inventory],
            model=model,
            name="inventory_manager",
            description="Agent responsible for tracking and managing ingredient inventory.",
        )


class ProductionManagerAgent(ToolCallingAgent):
    """Agent responsible for managing the production queue."""

    def __init__(self, model):
        super().__init__(
            tools=[
                check_production_capacity,
                add_to_production_queue,
                prioritize_order,
            ],  # TODO: Add add_to_production_queue, prioritize_order
            model=model,
            name="production_manager",
            description="Agent responsible for managing production scheduling and prioritization.",
        )


# TODO: Implement the CustomPastaDesignerAgent class
class CustomPastaDesignerAgent(ToolCallingAgent):
    """Agent responsible for designing custom pasta recipes."""

    def __init__(self, model):
        # TODO: Initialize with appropriate tools for custom pasta design
        super().__init__(
            tools=[
                check_inventory,
                create_custom_pasta_recipe,
            ],  # TODO: Add check_inventory, create_custom_pasta_recipe
            model=model,
            name="pasta_designer",
            description="Agent responsible for designing custom pasta recipes based on customer requirements.",
        )


# ======= Orchestrator =======


# TODO: Create proper Orchestrator using ToolCallingAgent pattern
class Orchestrator(ToolCallingAgent):
    """Orchestrator that coordinates workflow between specialized agents."""

    def __init__(self, model):
        self.model = model

        # TODO: Initialize specialized agents
        # self.order_processor = OrderProcessorAgent(model)
        # self.inventory_manager = InventoryManagerAgent(model)
        # self.production_manager = ProductionManagerAgent(model)
        # self.pasta_designer = CustomPastaDesignerAgent(model)
        self.order_processor = OrderProcessorAgent(model)
        self.inventory_manager = InventoryManagerAgent(model)
        self.production_manager = ProductionManagerAgent(model)
        self.pasta_designer = CustomPastaDesignerAgent(model)

        # TODO: Create coordination tools that route requests to different agents
        @tool
        def process_order_info(customer_request: str) -> str:
            """Process customer order information to extract details.

            Args:
                customer_request: The customer's order request

            Returns:
                Processed order information with pasta shape and quantity
            """
            return self.order_processor.run(
                f"""
            The customer request is: "{customer_request}"

            First check:
            1) What kind of pasta shape is being requested?
            2) What quantity (in kg) is being requested?

            Then check if we make the requested pasta shape. Use the check_pasta_recipe tool to verify.
            Generate a unique order ID using the generate_order_id tool.

            If the pasta shape is not one of we make, you can verify that we can offer using list_available_pasta_shapes tool.                                                         
         """
            )

        @tool
        def manage_inventory(order_details: str) -> str:
            """Check and manage inventory for an order.

            Args:
                order_details: Details about the order including pasta shape and quantity

            Returns:
                Inventory management result
            """
            return self.inventory_manager.run(
                f"""
            The order details are: "{order_details}"

            Check if we have sufficient ingredients in inventory to fulfill this order.
            1. Use the check_pasta_recipe tool to get required ingredients.
            2. Use the check_inventory tool to verify current inventory levels.
            3. Calculaate if we have enough ingredients to fulfill the order.
            """
            )

        @tool
        def schedule_production(order_info: str, priority: int = 1) -> str:
            """Schedule production for an order.

            Args:
                order_info: Information about the order to schedule
                priority: Order priority (1=normal, 2=rush, 3=emergency)

            Returns:
                Production scheduling result with delivery date
            """
            # TODO: Route this request to the ProductionManagerAgent
            return self.production_manager.run(
                f"""
            The order information is: "{order_info}"
            Priority level: {priority}

            Schedule production for this order:
            1. Add the order to the production queue using add_to_production_queue tool.
            2. USe specified priority level to determine scheduling.
            3. Provide customer production status and estimated delivery date.
            """
            )

        @tool
        def design_custom_pasta(customer_request: str) -> str:
            """Design a custom pasta recipe based on customer requirements.

            Args:
                customer_request: Customer's custom pasta request

            Returns:
                Custom pasta design result
            """
            return self.pasta_designer.run(
                f"""
            
            The customer request is: "{customer_request}"
            Customer wants a custom pasta design. Desing a custom pasta recipe based on their requirements.
            1. Check current inventory levels using check_inventory tool.
            2. Create a custom pasta recipe with appropriate ingredient ratios.
            3. Name the pasta appropriately based on pasta characteristics.
            4. Use the create_custom_pasta_recipe tool to save the recipe.
            """
            )

        super().__init__(
            tools=[
                process_order_info,
                manage_inventory,
                schedule_production,
                design_custom_pasta,
            ],  # TODO: Add the coordination tools
            model=model,
            name="orchestrator",
            description="""
            You are the orchestrator agent for pasta factory system.
            Your job is to coordinate order processing, inventory management, production scheduling,
            and custom pasta design through specialized agents.

            For each customer order, you will follow this workflow:
            1. Use process_order_info tool to understand the customer request and extract order details.
            If it is a custom pasta request, use design_custom_pasta tool to create the recipe first.
            2. Use manage_inventory tool to check if we have sufficient ingredients for the order.
            3. Use schedule_production tool to add the order to the production queue and get estimated delivery date.

            You need to determine priority based on customer language (e.g., "rush", "emergency", "urgent" = higher priority) and route requests accordingly.
            Always provide clear and comprehensive responses back to the customer about their order status.
            """,
        )

    def process_order(self, customer_request: str) -> str:
        """
        Process a customer order through coordinated agent workflow.
        """
        is_custom = any(
            word in customer_request.lower()
            for word in ["custom", "design", "unique", "special", "new", "different"]
        )

        priorty = 1
        if any(
            word in customer_request.lower()
            for word in ["rush", "emergency", "urgent", "asap", "immediate", "priority"]
        ):
            if "emergency" in customer_request.lower():
                priorty = 3
            else:
                priorty = 2
        context = f"""
        cutomer_request: {customer_request}
        is_custom: {is_custom}
        priority: {priorty}

        Process this order by coordinating the specialized agents.
        1. First process the order information to understand client requiest and requirements
        2. If it is a custom pasta request, design the custom pasta first.
        3. Check and manage inventory for the order to ensure we can fulfill it.
        4. Schedule production for the order with appropriate priority and provide estimated delivery date.

        In case we cannot fulfill the order, provide clear reasons to the customer why we cannot.
    """
        return self.run(context)


# ======= Main Demo =======


def run_demo():
    """Run a demonstration of the pasta factory system."""
    # TODO: Create orchestrator and test the multi-agent coordination
    orchestrator = Orchestrator(model)

    print("Welcome to the Pasta Factory Multi-Agent System!")
    print("Initial Factory State:", json.dumps(factory_state.to_dict(), indent=2))

    orders = [
        "I'd like to order 2kg of spaghetti please. When can I get it?",
        "I need a custom pasta with extra semolina and no eggs. Can you make that?",
        "Rush order! We need 5kg of fettuccine for a catering event tomorrow!",
    ]

    for i, order in enumerate(orders):
        print(f"\n--- Processing Order {i+1} ---")
        print(f"Customer: {order}")

        response = orchestrator.process_order(order)
        print(f"Factory: {response}")

    print("\n--- Final Factory State ---")
    print(json.dumps(factory_state.to_dict(), indent=2))
    print(
        "\nDemo complete! This demonstrates multi-agent coordination with shared state management."
    )


if __name__ == "__main__":
    run_demo()

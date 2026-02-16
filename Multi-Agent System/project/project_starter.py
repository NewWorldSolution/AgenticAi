from __future__ import annotations
from email import message
from multiprocessing import Value
from optparse import Option
from queue import Full
from re import I
import re
from tkinter import E
import unittest
from httpx import request
from openai import OpenAI
import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional
from sqlalchemy import create_engine, Engine
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)
from dataclasses import dataclass, field
from enum import Enum


# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = OpenAIServerModel(
    api_key=openai_api_key,
    api_base=base_url,
    model_id="gpt-4o-mini",
)


class Intent(str, Enum):
    QUOTE = "QUOTE"
    ORDER = "ORDER"

class TransactionType(str, Enum):
    STOCK_ORDERS = "stock_orders"
    SALES = "sales"


class FulfillmentStatus(str, Enum):
    FULFILLED = "FULFILLED"  # can meet requested_by or no requested_by constraint
    NOT_FULFILLED = (
        "NOT_FULFILLED"  # cannot meet requested_by => counter-offer returned
    )
    INVALID = "INVALID"  # missing fields, or unrecognized items


@dataclass(frozen=True)
class LineItem:
    raw_fragment: str  # the text chunk we extracted this from
    item_name: str  # normalized to a catalog name when possible
    quantity: int
    in_catalog: Optional[bool] = None
    alternatives: List[str] = field(
        default_factory=list
    )  # filled only if recognized is False


@dataclass(frozen=True)
class ParsedRequest:
    raw_text: str
    request_date: str  # ISO YYYY-MM-DD
    intent: Intent
    items: List[LineItem] = field(default_factory=list)
    job: Optional[str] = None
    need_size: Optional[str] = None
    event: Optional[str] = None
    requested_by: Optional[str] = None
    missing_fields: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class InventoryContext:
    item_name: str
    as_of_date: str  # ISO YYYY-MM-DD
    available_qty: int
    requested_qty: int
    shortage_qty: int
    in_catalog: bool


@dataclass(frozen=True)
class PricingResult:
    item_name: str
    requested_qty: int
    unit_price: float
    discount_rate: float  # 0.0 - 1.0
    subtotal: float
    discount_amount: float
    total_price: float
    rationale: str


@dataclass(frozen=True)
class LogisticsResult:
    item_name: str
    request_date: str  # ISO YYYY-MM-DD
    earliest_delivery_date: str
    status: FulfillmentStatus
    reason: Optional[str] = None  # Why not fulfilled


@dataclass(frozen=True)
class TransactionPlan:
    item_name: str

    sales_qty: int
    sales_total_price: float
    sales_date: str  # ISO YYYY-MM-DD

    create_reorder: bool
    reorder_qty: int = 0
    reorder_date: Optional[str] = None  # ISO YYYY-MM-DD
    reorder_total_cost: float = 0.00


@dataclass(frozen=True)
class LineResponse:
    item_name: str
    quantity: int
    status: FulfillmentStatus
    delivery_date: str  # ISO YYYY-MM-DD
    total_price: float
    reason: Optional[str] = None


@dataclass(frozen=True)
class SystemResponse:
    intent: Intent
    overall_status: FulfillmentStatus
    lines: List[LineResponse]
    message: str


@dataclass(frozen=True)
class TransactionRecord:
    item_name: str
    transaction_type: TransactionType  # 'stock_orders' or 'sales'
    units: int  # Quantity involved
    price: float  # Total price for the transaction
    transaction_date: str  # ISO 8601 string


FULFILLMENT_PRIORITY = {
    FulfillmentStatus.INVALID: 3,
    FulfillmentStatus.NOT_FULFILLED: 2,
    FulfillmentStatus.FULFILLED: 1,
}


def aggregate_overall_status(
    line_statuses: list[FulfillmentStatus],
) -> FulfillmentStatus:
    """
    INVALID > NOT_FULFILLED > FULFILLED
    """
    return max(line_statuses, key=lambda s: FULFILLMENT_PRIORITY[s])


# List containing the different kinds of papers
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Letter-sized paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Cardstock", "category": "paper", "unit_price": 0.15},
    {"item_name": "Colored paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Glossy paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Matte paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Recycled paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Eco-friendly paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Poster paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Banner paper", "category": "paper", "unit_price": 0.30},
    {"item_name": "Kraft paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Construction paper", "category": "paper", "unit_price": 0.07},
    {"item_name": "Wrapping paper", "category": "paper", "unit_price": 0.15},
    {"item_name": "Glitter paper", "category": "paper", "unit_price": 0.22},
    {"item_name": "Decorative paper", "category": "paper", "unit_price": 0.18},
    {"item_name": "Letterhead paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Legal-size paper", "category": "paper", "unit_price": 0.08},
    {"item_name": "Crepe paper", "category": "paper", "unit_price": 0.05},
    {"item_name": "Photo paper", "category": "paper", "unit_price": 0.25},
    {"item_name": "Uncoated paper", "category": "paper", "unit_price": 0.06},
    {"item_name": "Butcher paper", "category": "paper", "unit_price": 0.10},
    {"item_name": "Heavyweight paper", "category": "paper", "unit_price": 0.20},
    {"item_name": "Standard copy paper", "category": "paper", "unit_price": 0.04},
    {"item_name": "Bright-colored paper", "category": "paper", "unit_price": 0.12},
    {"item_name": "Patterned paper", "category": "paper", "unit_price": 0.15},
    # Product Types (priced per unit)
    {
        "item_name": "Paper plates",
        "category": "product",
        "unit_price": 0.10,
    },  # per plate
    {"item_name": "Paper cups", "category": "product", "unit_price": 0.08},  # per cup
    {
        "item_name": "Paper napkins",
        "category": "product",
        "unit_price": 0.02,
    },  # per napkin
    {
        "item_name": "Disposable cups",
        "category": "product",
        "unit_price": 0.10,
    },  # per cup
    {
        "item_name": "Table covers",
        "category": "product",
        "unit_price": 1.50,
    },  # per cover
    {
        "item_name": "Envelopes",
        "category": "product",
        "unit_price": 0.05,
    },  # per envelope
    {
        "item_name": "Sticky notes",
        "category": "product",
        "unit_price": 0.03,
    },  # per sheet
    {"item_name": "Notepads", "category": "product", "unit_price": 2.00},  # per pad
    {
        "item_name": "Invitation cards",
        "category": "product",
        "unit_price": 0.50,
    },  # per card
    {"item_name": "Flyers", "category": "product", "unit_price": 0.15},  # per flyer
    {
        "item_name": "Party streamers",
        "category": "product",
        "unit_price": 0.05,
    },  # per roll
    {
        "item_name": "Decorative adhesive tape (washi tape)",
        "category": "product",
        "unit_price": 0.20,
    },  # per roll
    {
        "item_name": "Paper party bags",
        "category": "product",
        "unit_price": 0.25,
    },  # per bag
    {
        "item_name": "Name tags with lanyards",
        "category": "product",
        "unit_price": 0.75,
    },  # per tag
    {
        "item_name": "Presentation folders",
        "category": "product",
        "unit_price": 0.50,
    },  # per folder
    # Large-format items (priced per unit)
    {
        "item_name": "Large poster paper (24x36 inches)",
        "category": "large_format",
        "unit_price": 1.00,
    },
    {
        "item_name": "Rolls of banner paper (36-inch width)",
        "category": "large_format",
        "unit_price": 2.50,
    },
    # Specialty papers
    {"item_name": "100 lb cover stock", "category": "specialty", "unit_price": 0.50},
    {"item_name": "80 lb text paper", "category": "specialty", "unit_price": 0.40},
    {"item_name": "250 gsm cardstock", "category": "specialty", "unit_price": 0.30},
    {"item_name": "220 gsm poster paper", "category": "specialty", "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system


def generate_sample_inventory(
    paper_supplies: list, coverage: float = 0.4, seed: int = 137
) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)), size=num_items, replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append(
            {
                "item_name": item["item_name"],
                "category": item["category"],
                "unit_price": item["unit_price"],
                "current_stock": np.random.randint(200, 800),  # Realistic stock range
                "min_stock_level": np.random.randint(
                    50, 150
                ),  # Reasonable threshold for reordering
            }
        )

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)


def init_database(db_engine: Engine = db_engine, seed: int = 137) -> Engine:
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame(
            {
                "id": [],
                "item_name": [],
                "transaction_type": [],  # 'stock_orders' or 'sales'
                "units": [],  # Quantity involved
                "price": [],  # Total price for the transaction
                "transaction_date": [],  # ISO-formatted date
            }
        )
        transactions_schema.to_sql(
            "transactions", db_engine, if_exists="replace", index=False
        )

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql(
            "quote_requests", db_engine, if_exists="replace", index=False
        )

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("job_type", "")
            )
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("order_size", "")
            )
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(
                lambda x: x.get("event_type", "")
            )

        # Retain only relevant columns
        quotes_df = quotes_df[
            [
                "request_id",
                "total_amount",
                "quote_explanation",
                "order_date",
                "job_type",
                "order_size",
                "event_type",
            ]
        ]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append(
            {
                "item_name": None,
                "transaction_type": "sales",
                "units": None,
                "price": 50000.0,
                "transaction_date": initial_date,
            }
        )

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append(
                {
                    "item_name": item["item_name"],
                    "transaction_type": "stock_orders",
                    "units": item["current_stock"],
                    "price": item["current_stock"] * item["unit_price"],
                    "transaction_date": initial_date,
                }
            )

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql(
            "transactions", db_engine, if_exists="append", index=False
        )

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame(
            [
                {
                    "item_name": item_name,
                    "transaction_type": transaction_type,
                    "units": quantity,
                    "price": price,
                    "transaction_date": date_str,
                }
            ]
        )

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise


def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))


def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(
        f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'"
    )

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(
            f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base."
        )
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")


def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[
                transactions["transaction_type"] == "sales", "price"
            ].sum()
            total_purchases = transactions.loc[
                transactions["transaction_type"] == "stock_orders", "price"
            ].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append(
            {
                "item_name": item["item_name"],
                "stock": stock,
                "unit_price": item["unit_price"],
                "value": item_value,
            }
        )

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]


########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


def _normalize_date(input_date_str: str) -> str:
    """
    Accepts 'YYYY-MM-DD' or full ISO 'YYYY-MM-DDTHH:MM:SS...'
    Returns 'YYYY-MM-DD'
    """
    if not input_date_str:
        raise ValueError("input_date_str is required")
    return input_date_str.split("T")[0]


@tool
def ia_get_stock_level(item_name: str, as_of_date: str) -> Dict:
    """
    IA tool (read-only): get stock level for one item as of date.
    Uses: get_stock_level
    """
    as_of_date = _normalize_date(as_of_date)
    stock_info = get_stock_level(item_name, as_of_date)
    if stock_info.empty:
        return {"item_name": item_name, "current_stock": 0, "as_of_date": as_of_date}
    return {
        "item_name": str(stock_info["item_name"].iloc[0]),
        "current_stock": int(stock_info["current_stock"].iloc[0]),
        "as_of_date": as_of_date,
    }


# Tools for inventory agent
@tool
def ia_get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    IA tool (read-only): get all inventory as of date.
    Uses: get_all_inventory
    """
    as_of_date = _normalize_date(as_of_date)
    return {k: int(v) for k, v in get_all_inventory(as_of_date=as_of_date).items()}


# Tools for quoting agent


@tool
def pa_search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    PA tool (read-only): search quote history for similar requests.
    Uses: search_quote_history
    """
    return search_quote_history(search_terms=search_terms, limit=int(limit))


# Tools for ordering agent
@tool
def tla_get_supplier_delivery_date(request_date: str, quantity: int) -> str:
    """
    TLA tool (read-only): get supplier delivery date based on requested date and quantity.
    Uses: get_supplier_delivery_date
    """
    return get_supplier_delivery_date(
        input_date_str=_normalize_date(request_date), quantity=quantity
    )


@tool
def tla_create_transaction(
    item_name: str, transaction_type: str, quantity: int, price: float, date: str
) -> int:
    """TLA tool (DB write): create a sales or stock_orders transaction. Uses create_transaction."""
    return int(
        create_transaction(
            item_name=item_name,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
            date=date,
        )
    )


@tool
def tla_get_cash_balance(as_of_date: str) -> float:
    """TLA tool (read-only): get current cash balance as of date. Uses get_cash_balance."""
    return float(get_cash_balance(as_of_date=_normalize_date(as_of_date)))


@tool
def tla_generate_financial_report(as_of_date: str) -> Dict:
    """TLA tool (read-only): generate financial report as of date. Uses generate_financial_report."""
    return generate_financial_report(as_of_date=_normalize_date(as_of_date))


# Set up your agents and create an orchestration agent that will manage them.


# Agent propmts

IA_SYSTEM_PROMPT = """
You are the Inventory Agent (IA). You are read-only.
Use only the inventory tools to answer questions about current stock.
Return factual, structured output. Do not guess. Do not write to the DB.
"""
PA_SYSTEM_PROMPT = """
You are the Pricing Agent (PA) for Beaver’s Choice Paper Company.
You are strictly READ-ONLY and must never write to the database.

TASK:
Given a line item, quantity, and a base unit price (from catalog),
use quote history (via pa_search_quote_history) as retrieval context to recommend a discount.

RETURN FORMAT:
Return ONLY a Python dictionary with exactly these keys:
{
  "discount_rate": float,   # between 0.0 and 0.30
  "rationale": str          # short explanation referencing history patterns
}

RULES:
- Do NOT invent or output unit_price. The orchestrator provides base_unit_price.
- Do NOT compute totals.
- Prefer discount_rate based on: quantity, job/event/size context, and similarity to past quotes.
- If history is weak/empty, use a conservative discount based on quantity:
  <100 -> 0.00
  100-499 -> 0.05
  500-999 -> 0.10
  1000+ -> 0.15
- Keep discount_rate <= 0.30.

Return ONLY the dictionary (no extra text).
"""

TLA_SYSTEM_PROMPT = """
You are the Transactions & Logistics Agent (TLA).

ROLE:
You compute the earliest supplier delivery date for a requested item.
You do NOT decide fulfillment status.
You do NOT write to the database unless explicitly instructed.
You do NOT apply business rules about requested_by dates.

RULES:
- Return only a Python dictionary.
- The dictionary must contain:
    - earliest_delivery_date (string YYYY-MM-DD)
    - reason (string, may be empty)

You must NOT include fulfillment status.
You must NOT include pricing.
You must NOT include any explanation text outside the dictionary.

Example output:
{
    "earliest_delivery_date": "2025-01-04",
    "reason": ""
}

Return ONLY the dictionary.
"""

# Instantiate agents

inventory_agent = ToolCallingAgent(
    name="Inventory Agent",
    description="Read-only agent for inventory checks like stock level and inventory snapshot.",
    model=model,
    tools=[ia_get_stock_level, ia_get_all_inventory],
    system_prompt=IA_SYSTEM_PROMPT,
)

pricing_agent = ToolCallingAgent(
    name="Pricing Agent",
    description="Read-only agent for pricing and quote history reference, may consult quote history.",
    model=model,
    tools=[pa_search_quote_history],
    system_prompt=PA_SYSTEM_PROMPT,
)
transactions_logistics_agent = ToolCallingAgent(
    name="TransactionsLogisticsAgent",
    description="Agent for delivery feasibility and the only agent permitted to write transactions.",
    model=model,
    tools=[
        tla_get_supplier_delivery_date,
        tla_create_transaction,  # DB write (must be gated by orchestrator)
        tla_get_cash_balance,
        tla_generate_financial_report,
    ],
    system_prompt=TLA_SYSTEM_PROMPT,
)
def get_catalog_item(item_name: str) -> Optional[Dict]:
    for p in paper_supplies:
        if p["item_name"].lower() == item_name.lower():
            return p
    return None

class FrontDeskOrchestratorAgent:
    """
    Orchestrator agent that manages the flow of information and decision-making between the Inventory Agent, Pricing Agent, and Transactions & Logistics Agent.

    Responsibilities:
    - Receive incoming customer requests and determine which agents to consult.
    - Coordinate the sequence of tool calls across agents based on the request context and intermediate results.
    - Make final decisions on order fulfillment, pricing, and transaction recording based on aggregated information from all agents.
    - Applies gating rules for DB writes to ensure that only the Transactions & Logistics Agent writes to the database, and only when explicitly authorized.
    - Never write to DB
    - Calls Inventory Agent -> Pricing Agent -> Transactions & Logistics Agent in sequence, passing necessary information and context.
    """

    def __init__(self, inventory_agent, pricing_agent, transactions_logistics_agent):
        self.inventory_agent = inventory_agent
        self.pricing_agent = pricing_agent
        self.transactions_logistics_agent = transactions_logistics_agent

    def handle_request(self, parsed: ParsedRequest) -> SystemResponse:
        # TODO: implement to your locked flow:
        # 1) validate parsed (missing fields / invalid items)
        # 2) IA: inventory context per line
        # 3) PA: pricing per line
        # 4) TLA: delivery + fulfillment per line
        # 5) aggregate overall_status
        # 6) gate DB writes: ORDER + overall FULFILLED only
        # 7) produce customer message + line responses

        """
        Deterministic orchestration entry point. Follows locked flowchart exactly.
        """
        if parsed.missing_fields:
            return SystemResponse(
                intent=parsed.intent,
                overall_status=FulfillmentStatus.INVALID,
                lines=[],
                message=f"Request is missing required fields: {', '.join(parsed.missing_fields)}",
            )
        if not parsed.items:
            return SystemResponse(
                intent=parsed.intent,
                overall_status=FulfillmentStatus.INVALID,
                lines=[],
                message="Request must include at least one item.",
            )
        line_responses: List[LineResponse] = []
        line_statuses: List[FulfillmentStatus] = []

        # Step 1: Process each line item in the request
        for item in parsed.items:
            # 1) Validate item (e.g., check if item_name is valid)
            if item.quantity <= 0 or not item.item_name:         
                line_responses.append(
                    LineResponse(
                        item_name=item.item_name or "UNKNOWN",
                        quantity=item.quantity,
                        status=FulfillmentStatus.INVALID,
                        delivery_date=parsed.request_date,  # Echo back request date for reference
                        total_price=0.0,
                        reason="Invalid quantity or missing item name.",
                    )
                )
                line_statuses.append(FulfillmentStatus.INVALID)
                continue
            catalog_item = get_catalog_item(item.item_name)
            if not catalog_item:
                line_responses.append(
                    LineResponse(
                        item_name=item.item_name,
                        quantity=item.quantity,
                        status=FulfillmentStatus.INVALID,
                        delivery_date=parsed.request_date,
                        total_price=0.0,
                        reason="Unknown item (not in catalog).",
                    )
                )
                line_statuses.append(FulfillmentStatus.INVALID)
                continue

            base_unit_price = float(catalog_item["unit_price"])
            # 1B. Inventory Agent (IA)
            ia_reply = self.inventory_agent.run(
                f"Get stock for item '{item.item_name}' as of {parsed.request_date}. "
                f"Return a Python dict with keys: item_name, current_stock, as_of_date."
            )
            try:
                ia_data = ast.literal_eval(ia_reply)  # Convert string dict to actual dict

                inventory_ctx = InventoryContext(
                item_name=ia_data["item_name"],
                as_of_date=ia_data["as_of_date"],
                available_qty=ia_data["current_stock"],
                requested_qty=item.quantity,
                shortage_qty=max(0, item.quantity - ia_data["current_stock"]),
                in_catalog=True,
            )
            except Exception:
                line_responses.append(
                    LineResponse(
                        item_name=item.item_name,
                        quantity=item.quantity,
                        status=FulfillmentStatus.INVALID,
                        delivery_date=parsed.request_date,
                        total_price=0.0,
                        reason="Inventory Agent returned invalid response format.",
                    )
                )
                line_statuses.append(FulfillmentStatus.INVALID)
                continue
            

            # 1C. Pricing Agent (PA)
            # (read-only)
            search_terms = [
                item.item_name.lower(),
                (parsed.job or "").lower(),
                (parsed.event or "").lower(),
                (parsed.need_size or "").lower(),
            ]
            history = pa_search_quote_history(
            search_terms=search_terms,
            limit=5
            )

            pa_reply = self.pricing_agent.run(
                "Use the provided historical quote data to recommend a discount rate.\n\n"
                f"item_name={item.item_name}\n"
                f"quantity={item.quantity}\n"
                f"base_unit_price={base_unit_price}\n"
                f"quote_history={history}\n\n"
                "Rules:\n"
                "- Analyze similarities in quantity, job_type, order_size, and event_type.\n"
                "- If history is weak or empty, fall back to conservative quantity-based discount.\n"
                "- Return ONLY a Python dictionary:\n"
                "{'discount_rate': float, 'rationale': str}"
            )

            try:
                pa_data = ast.literal_eval(pa_reply)  # Convert string dict to actual dict
                discount_rate = float(pa_data["discount_rate"])
                discount_rate = max(0.0, min(discount_rate, 0.30))  # safety clamp


                subtotal = round(base_unit_price * item.quantity, 2)
                discount_amount = round(subtotal * discount_rate, 2)
                total_price = round(subtotal - discount_amount, 2)

                pricing = PricingResult(
                    item_name=item.item_name,
                    requested_qty=item.quantity,
                    unit_price=base_unit_price,
                    discount_rate=discount_rate,
                    subtotal=subtotal,
                    discount_amount=discount_amount,
                    total_price=total_price,
                    rationale=pa_data["rationale"],
                )
            except Exception:
                line_responses.append(
                    LineResponse(
                        item_name=item.item_name,
                        quantity=item.quantity,
                        status=FulfillmentStatus.INVALID,
                        delivery_date=parsed.request_date,
                        total_price=0.0,
                        reason="Pricing Agent returned invalid response format.",
                    )
                )
                line_statuses.append(FulfillmentStatus.INVALID)
                continue
            
            # 1D. Transactions & Logistics Agent (TLA)
            # Computes delivery date + fulfillment status
            tla_reply = self.transactions_logistics_agent.run(
                f"For item '{item.item_name}', quantity {item.quantity}, "
                f"request_date {parsed.request_date}, requested_by {parsed.requested_by}. "
                f"Return a Python dict with earliest_delivery_date and reason."
            )
            try:
                tla_data = ast.literal_eval(tla_reply)  # Convert string dict to actual dict
                delivery_date = tla_data["earliest_delivery_date"]
                tla_reason = tla_data.get("reason")
            except Exception:
                line_responses.append(
                    LineResponse(
                        item_name=item.item_name,
                        quantity=item.quantity,
                        status=FulfillmentStatus.INVALID,
                        delivery_date=parsed.request_date,
                        total_price=0.0,
                        reason="Transactions & Logistics Agent returned invalid response format.",
                    )
                )
                line_statuses.append(FulfillmentStatus.INVALID)
                continue
            
            if parsed.requested_by:
                if delivery_date > parsed.requested_by:
                    status = FulfillmentStatus.NOT_FULFILLED
                    reason = f"Requested delivery date cannot be met. Earliest possible: {delivery_date}."
                else:
                    status = FulfillmentStatus.FULFILLED
                    reason = tla_reason
            else:
                status = FulfillmentStatus.FULFILLED
                reason = tla_reason

            logistics = LogisticsResult(
                item_name=item.item_name,
                request_date=parsed.request_date,
                earliest_delivery_date=delivery_date,
                status=status,
                reason=reason,
            )
            # 1E. Build LineResponse
            line_responses.append(
                LineResponse(
                    item_name=item.item_name,
                    quantity=item.quantity,
                    status=logistics.status,
                    delivery_date=logistics.earliest_delivery_date,
                    total_price=pricing.total_price,
                    reason=logistics.reason,
                )
            )
            line_statuses.append(logistics.status)
        # Step 2: Aggregate overall status
        if not line_statuses:
            overall_status = FulfillmentStatus.INVALID
        else:
            overall_status = aggregate_overall_status(line_statuses)

        # Step 3: Intent + DB write gating
        if (
            parsed.intent == Intent.ORDER
            and overall_status == FulfillmentStatus.FULFILLED
        ):
            # Only the TLA can write to the DB, and only when the intent is ORDER and overall status is FULFILLED
            for line in line_responses:
                if line.status != FulfillmentStatus.FULFILLED:
                    continue  # Skip lines that are not fulfilled 
                tla_create_transaction(
                    item_name=line.item_name,
                    transaction_type="sales",
                    quantity=line.quantity,
                    price=line.total_price,
                    date=parsed.request_date,
                )
                # After sales transaction, check if reorder needed
                # Get current stock AFTER the sale date
                stock_info = get_stock_level(line.item_name, parsed.request_date)
                current_stock = int(stock_info["current_stock"].iloc[0])

                # Get min_stock_level from inventory table
                inventory_df = pd.read_sql(
                    "SELECT min_stock_level FROM inventory WHERE item_name = :item_name",
                    db_engine,
                    params={"item_name": line.item_name},
                )

                if inventory_df.empty:
                        continue

                min_stock_level = int(inventory_df.iloc[0]["min_stock_level"])
                base_unit_price = float(inventory_df.iloc[0]["unit_price"])

                if current_stock < min_stock_level:
                    reorder_qty = int(max(min_stock_level * 2 - current_stock, line.quantity))

                    # Future-dated reorder using supplier lead time
                    reorder_date = tla_get_supplier_delivery_date(parsed.request_date, reorder_qty)
                    tla_create_transaction(
                        item_name=line.item_name,
                        transaction_type="stock_orders",
                        quantity=reorder_qty,
                        price=round(reorder_qty * base_unit_price, 2),
                        date=reorder_date,
                    )

        # Step 4: Build customer message

        message = self._build_customer_message(
            parsed.intent, overall_status, line_responses
        )

        return SystemResponse(
            overall_status=overall_status,
            lines=line_responses,
            message=message,
            intent=parsed.intent,
        )

    def _build_customer_message(
        self,
        intent: Intent,
        overall_status: FulfillmentStatus,
        lines: List[LineResponse],
    ) -> str:
        """
        Customer-facing summary only.
        No internal reasoning.
        """
        if overall_status == FulfillmentStatus.INVALID:
            return (
                "Your request could not be processed due to invalid item information."
            )

        if intent == Intent.QUOTE:
            return "Here is your quote with pricing and estimated delivery dates."

        if intent == Intent.ORDER and overall_status == FulfillmentStatus.FULFILLED:
            return "Your order has been successfully placed."

        if intent == Intent.ORDER and overall_status == FulfillmentStatus.NOT_FULFILLED:
            return (
                "We cannot meet your requested delivery date. Here is a counter-offer."
            )

        return "Request processed."


fdo = FrontDeskOrchestratorAgent(
    inventory_agent, pricing_agent, transactions_logistics_agent
)
# Run your test scenarios by writing them here. Make sure to keep track of them.


def run_test_scenarios():

    print("Initializing Database...")
    init_database()
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]
    # overall_status = aggregate_overall_status([line.status for line in line_responses])
    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        # response = call_your_multi_agent_system(request_with_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()

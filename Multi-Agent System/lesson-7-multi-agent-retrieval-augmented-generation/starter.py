# Insurance Claims RAG System Exercise - Starter

from typing import Dict, List, Any, Optional, Union, Set
import random
import json
import os
from arrow import get
from huggingface_hub import notebook_login
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)
import os
import dotenv

# Note: Make sure to set up your .env file with your API key before running
dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)

# Import core components from the demo file
# Note: In a real application, you would organize this better with proper imports
from demo import (
    PrivacyLevel,
    AccessControl,
    Claim,
    PatientRecord,
    ComplaintRecord,
    Database,
    VectorKnowledgeBase,
    VectorClaimSearch,
    DataGenerator,
    database,
    vector_kb,
    vector_claim_search,
    search_knowledge_base,
    retrieve_claim_history,
    get_claim_details,
    get_patient_info,
    find_similar_claims,
    submit_complaint,
    respond_to_complaint,
    get_complaint_history,
    process_new_claim,
    ClaimProcessingAgent,
    CustomerServiceAgent,
    MedicalReviewAgent,
    ComplaintResolutionOrchestrator,
)

"""
EXERCISE: CLAIM FRAUD DETECTION WITH RAG

In this exercise, you'll enhance the insurance claims processing system by adding 
fraud detection capabilities powered by RAG. Fraud detection is a critical component 
of insurance claims processing, saving the industry billions of dollars annually.

Your task is to:

1. Implement a FraudDetectionAgent class that leverages RAG to identify potentially 
   fraudulent claims by comparing them with known fraud patterns
   
2. Create a fraud knowledge base with common fraud indicators and patterns
   
3. Implement vector search functionality to identify similar fraud patterns
   
4. Integrate the agent into the existing workflow, adding a fraud review step to the
   claim processing pipeline

HINTS:
- You can use the existing VectorKnowledgeBase and VectorClaimSearch as references
- Your fraud detection component should consider multiple factors like claim frequency,
  unusual patterns, and similarity to known fraud cases
- Make sure to respect the privacy levels and access controls already in place
"""

# STEP 1: Create a knowledge base of fraud patterns
# TODO: Implement a fraud knowledge base with common fraud patterns
fraud_patterns = [
    {
        "pattern_id": "FP-001",
        "pattern_name": "Repeated claims without documentation",
        "description": (
            "Multiple claims submitted over a short period with little or no supporting "
            "documentation. This pattern is commonly associated with attempts to exploit "
            "automated claim approval processes."
        ),
        "indicator": (
            "No supporting documentation provided for multiple claims; "
            "repeated submissions rely on similar narratives."
        ),
        "severity": "high",
        "privacy_level": PrivacyLevel.AGENT,
    },
    {
        "pattern_id": "FP-002",
        "pattern_name": "Patient denies receiving service",
        "description": (
            "A claim is submitted for a medical service that the patient later denies "
            "receiving. This can indicate provider fraud or identity misuse."
        ),
        "indicator": (
            "Patient states they did not receive the claimed service or treatment."
        ),
        "severity": "critical",
        "privacy_level": PrivacyLevel.AGENT,
    },
    {
        "pattern_id": "FP-003",
        "pattern_name": "Unusual claim amount compared to history",
        "description": (
            "A claim amount that significantly deviates from the patient’s historical "
            "claim amounts, without clear justification in medical records."
        ),
        "indicator": (
            "Claim amount appears unusually high relative to the patient’s past claims."
        ),
        "severity": "medium",
        "privacy_level": PrivacyLevel.AGENT,
    },
    {
        "pattern_id": "FP-004",
        "pattern_name": "Highly similar claim narratives",
        "description": (
            "Claim descriptions across multiple submissions are highly similar or nearly "
            "identical, suggesting reused or templated language rather than independent events."
        ),
        "indicator": (
            "Repeated use of nearly identical wording across different claims."
        ),
        "severity": "medium",
        "privacy_level": PrivacyLevel.AGENT,
    },
    {
        "pattern_id": "FP-005",
        "pattern_name": "Inconsistent or implausible service timelines",
        "description": (
            "Claims contain service dates or timelines that conflict with other known claims "
            "or appear implausible given medical or logistical constraints."
        ),
        "indicator": (
            "Overlapping service dates or timelines that do not align with patient history."
        ),
        "severity": "high",
        "privacy_level": PrivacyLevel.AGENT,
    },
    {
        "pattern_id": "FP-006",
        "pattern_name": "Missing or vague service details",
        "description": (
            "Claims lack specific information about the service provided, such as procedure "
            "details, provider information, or treatment justification."
        ),
        "indicator": (
            "Claim description is vague, incomplete, or avoids specifying concrete services."
        ),
        "severity": "low",
        "privacy_level": PrivacyLevel.AGENT,
    },
]


# STEP 2: Implement a vector-based fraud pattern detector
class FraudPatternDetector:
    def __init__(self):
        # TODO: Initialize the fraud detector with vector embeddings
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.fraud_patterns = []
        self.pattern_vectors = None
        self.is_initialized = False

    def update_patterns(self, fraud_patterns):
        # TODO: Update the patterns database
        """Update the fraud patterns database and compute their vector embeddings."""
        self.fraud_patterns = fraud_patterns
        pattern_texts = []
        for pattern in fraud_patterns:
            pattern_texts.append(
                f"{pattern['pattern_name']}: {pattern['description']} Indicator: {pattern['indicator']}"
            )
        if pattern_texts:
            self.pattern_vectors = self.vectorizer.fit_transform(pattern_texts)
            self.is_initialized = True

    def detect_fraud_indicators(
        self, claim, patient_history, access_level=PrivacyLevel.AGENT
    ):
        # TODO: Implement fraud detection logic
        # Use vector similarity and rule-based methods to identify potential fraud
        """Detect potential fraud indicators in a claim using vector similarity and rule_based methods.
        Args:
            claim: The claim to evaluate
            patient_history: The patient's claim history. Optional list.
            access_level: The access level of the requester
        Returns:
            Dictionary with fraud indicators and risk assessment
        """
        if not self.is_initialized:
            return {
                "claim_id": claim.claim_id if hasattr(claim, "claim_id") else "unknown",
                "error": "Fraud pattern detector not initialized.",
                "fraud_risk": "unknown",
            }
        # Prepare claim text for vectorization
        claim_text = f"Procedure: {claim.procedure_description}. Amount: {claim.amount} Patient {claim.patient_id} Status: {claim.status}"

        # Add patient history to claim text if available
        if patient_history:
            recent_procedure = []
            for past_claim in patient_history:
                if past_claim["id"] != claim.claim_id:
                    recent_procedure.append(past_claim["procedure_code"])
                    claim_text += f"Recent {past_claim['procedure_code']}"

        claim_vector = self.vectorizer.transform([claim_text])

        similarities = cosine_similarity(claim_vector, self.pattern_vectors).flatten()

        # Identify top matching patterns
        matching_patterns = []
        for i, score in enumerate(similarities):
            if score > 0.1:  # Threshold for similarity
                pattern = self.fraud_patterns[i]
                if AccessControl.has_access(access_level, pattern["privacy_level"]):
                    matching_patterns.append(
                        {
                            "pattern_id": pattern["pattern_id"],
                            "pattern_name": pattern["pattern_name"],
                            "similarity_score": float(score),
                            "severity": pattern["severity"],
                        }
                    )
        # Add rule-based fraud indicators
        rule_indicators = self._apply_fraud_rules(claim, patient_history)

        # Calculate overall fraud risk

        fraud_risk = self._calculate_fraud_risk(matching_patterns, rule_indicators)

        return {
            "claim_id": claim.id,
            "matching_patterns": sorted(
                matching_patterns, key=lambda x: x["similarity_score"], reverse=True
            ),
            "rule_indicators": rule_indicators,
            "fraud_risk": fraud_risk,
        }

    def _apply_fraud_rules(self, claim, patient_history):
        """Apply rule-based fraud detection"""
        indicators = []

        if claim.amount > 400:
            indicators.append(
                {
                    "rule": "high_amount",
                    "description": "claim amount is significantly higher than average",
                    "confidence": 0.6,
                }
            )
        if patient_history:
            similar_claim = [
                c
                for c in patient_history
                if c["procedure_code"] == claim.procedure_code
            ] and c["id"] != claim.claim_id
            if len(similar_claim) >= 2:
                indicators.append(
                    {
                        "rule": "repeat_procedure",
                        "description": f"Procedure {claim.procedure_code} claimed multiple times recently",
                        "confidence": 0.7,
                    }
                )

        return indicators

    def _calculate_fraud_risk(self, matching_patterns, rule_indicators):
        "Calculate overall fraud risk level based on matches and rules"
        if not matching_patterns and not rule_indicators:
            return "low"

        pattern_score = 0
        for match in matching_patterns:
            severity_multiplier = {
                "low": 0.3,
                "medium": 0.6,
                "high": 0.8,
                "critical": 1,
            }.get(match["severity"], 0.5)
            pattern_score += match["similarity_score"] * severity_multiplier

        rule_score = sum(indicator["confidence"] for indicator in rule_indicators)

        total_score = pattern_score + rule_score

        if total_score < 0.3:
            return "low"
        elif total_score < 0.7:
            return "medium"
        elif total_score < 1.2:
            return "high"
        else:
            return "critical"


fraud_detector = FraudPatternDetector()
fraud_detector.update_patterns(fraud_patterns)


# STEP 3: Implement a tool for fraud detection
@tool
def check_claim_for_fraud(
    claim_id: str, access_level: str = PrivacyLevel.AGENT
) -> Dict:
    """
    Check a claim for potential fraud indicators.

    Args:
        claim_id: The claim ID to check
        access_level: The access level of the requester

    Returns:
        Dictionary containing fraud assessment results
    """
    # Get claim details
    claim_data = database.get_claim(claim_id, access_level)
    if not claim_data:
        return {"error": f"Claim {claim_id} not found.", "success": False}

    # Get the actuual claim object
    claim = database.claims[claim_id]

    # Get patient history and context
    patient_id = claim.patient_id
    patient_claim = database.get_patient_claims(patient_id, access_level)

    # Run fraud detection
    fraud_analysis = fraud_detector.detect_fraud_indicators(
        claim, patient_claim, access_level
    )
    return {
        "success": True,
        "fraud_analysis": fraud_analysis,
        "claim_id": claim_id,
        "recommendation": _get_fraud_recommendation(fraud_analysis),
    }


def _get_fraud_recommendation(risk_level):
    """Generate recommendation based on fraud risk level"""
    if risk_level == "low":
        return "Approved. Proceed normal processing."
    elif risk_level == "medium":
        return "Flag for manual review before approval."
    elif risk_level == "high":
        return "Hold claim and initiate fraud investigation."
    elif risk_level == "critical":
        return "Deny claim and escalate to fraud department."
    else:
        return "Unknown risk level. Further analysis required."


# STEP 4: Create a FraudDetectionAgent
class FraudDetectionAgent(ToolCallingAgent):
    """Agent for detecting potential fraud in insurance claims."""

    def __init__(self, model: OpenAIServerModel):
        # TODO: Implement the fraud detection agent
        super().__init__(
            model=model,
            tools=[
                check_claim_for_fraud,
                get_claim_details,
                retrieve_claim_history,
                search_knowledge_base,
                get_patient_info,
            ],
            instructions=(
                "You are a Fraud Detection Agent specialized in identifying potentially "
                "fraudulent insurance claims. You have AGENT level access to the database. Use the provided tools to analyze claims "
                "and provide recommendations based on fraud risk."
                "Your main job is to assess claims for potential fraud indicators and recommend appropriate actions,"
                " such as approval, manual review, or flag suspicious and escalation to the fraud department."
            ),
        )
        self.access_level = PrivacyLevel.AGENT


# STEP 5: Update the orchestrator to include fraud detection
# TODO: Modify ComplaintResolutionOrchestrator to include fraud detection in the workflow
class EnhancedOrchestrator(ComplaintResolutionOrchestrator):
    """Enhanced orchestrator that includes fraud detection in the workflow."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(model)
        self.fraud_detector = FraudDetectionAgent(model)

        @tool
        def handle_claim_with_fraud_check(claim_data: Dict) -> Dict:
            """
            Process a new claim with integrated fraud detection.

            Args:
                claim_data: The claim data to process

            Returns:
                Dictionary containing the claim processing result with fraud assessment
            """
            # Step 1: Process the new claim
            process_result = self.claim_processor.run(
                f"""
                Process this new claim:
                Patient ID: {claim_data['patient_id']}
                Service Date: {claim_data['service_date']}
                Procedure Code: {claim_data['procedure_code']}
                Amount: ${claim_data['amount']}
                
                Use the process_new_claim tool.
                """
            )

            # Extract claim_id from the result
            claim_id = None
            if hasattr(process_result, "tool_calls") and process_result.tool_calls:
                for call in process_result.tool_calls:
                    if (
                        call.name == "process_new_claim"
                        and "claim_id" in call.arguments
                    ):
                        claim_id = call.arguments["claim_id"]

            if not claim_id:
                return {"success": False, "error": "Failed to process claim"}

            # Step 2: Run fraud detection on the new claim
            fraud_result = self.fraud_detector.run(
                f"""
                Analyze claim {claim_id} for potential fraud.
                
                First, get details about the claim using get_claim_details tool.
                Then check for fraud indicators using check_claim_for_fraud tool.
                Provide a detailed assessment of any potential fraud risks.
                """
            )

            # Extract fraud analysis
            fraud_analysis = {"fraud_risk": "unknown"}
            if hasattr(fraud_result, "tool_calls") and fraud_result.tool_calls:
                for call in fraud_result.tool_calls:
                    if (
                        call.name == "check_claim_for_fraud"
                        and "fraud_analysis" in call.arguments
                    ):
                        fraud_analysis = call.arguments["fraud_analysis"]

            # Get the claim details
            claim = database.get_claim(claim_id, PrivacyLevel.ADMIN)

            return {
                "success": True,
                "claim_id": claim_id,
                "claim_status": claim["status"],
                "decision_reason": claim["decision_reason"],
                "fraud_risk": fraud_analysis.get("fraud_risk", "unknown"),
                "recommendation": _get_fraud_recommendation(
                    fraud_analysis.get("fraud_risk", "unknown")
                ),
            }

        # Add the new tool to the orchestrator
        self.tools["handle_claim_with_fraud_check"] = handle_claim_with_fraud_check


# STEP 6: Function to demonstrate the fraud detection capabilities
def demonstrate_fraud_detection():
    """
    Run a demonstration of the fraud detection capabilities.
    """
    # TODO: Implement a demonstration of the fraud detection feature
    # Create orchestrator with fraud detection
    orchestrator = EnhancedOrchestrator(model)

    print("Generating a legitimate claim for processing...")
    # Generate a legitimate claim
    legitimate_claim = {
        "patient_id": random.choice(list(database.patients.keys())),
        "service_date": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        "procedure_code": random.choice(
            ["71020", "81003", "85025"]
        ),  # Choose common procedures
        "amount": random.uniform(50, 200),  # Reasonable amount
    }

    print(f"Processing legitimate claim: {json.dumps(legitimate_claim, indent=2)}")
    legitimate_result = orchestrator.run(
        f"""
        Process this claim and check for fraud:
        Patient ID: {legitimate_claim['patient_id']}
        Service Date: {legitimate_claim['service_date']}
        Procedure Code: {legitimate_claim['procedure_code']}
        Amount: ${legitimate_claim['amount']:.2f}
        
        Use the handle_claim_with_fraud_check tool.
        """
    )

    print("\n" + "=" * 50 + "\n")

    print("Now generating a suspicious claim with fraud indicators...")
    # Generate a suspicious claim (high amount, unusual procedure)
    # Find a patient with existing claims for this example
    patients_with_claims = {
        patient_id: len(patient.claim_ids)
        for patient_id, patient in database.patients.items()
        if patient.claim_ids
    }

    if patients_with_claims:
        # Get patient with most claims
        patient_id = max(patients_with_claims.items(), key=lambda x: x[1])[0]
    else:
        # If no patients with claims, just pick a random one
        patient_id = random.choice(list(database.patients.keys()))

    suspicious_claim = {
        "patient_id": patient_id,
        "service_date": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        "procedure_code": "43239",  # Expensive procedure
        "amount": random.uniform(800, 1200),  # Unusually high amount
    }

    print(f"Processing suspicious claim: {json.dumps(suspicious_claim, indent=2)}")
    suspicious_result = orchestrator.run(
        f"""
        Process this claim and check for fraud:
        Patient ID: {suspicious_claim['patient_id']}
        Service Date: {suspicious_claim['service_date']}
        Procedure Code: {suspicious_claim['procedure_code']}
        Amount: ${suspicious_claim['amount']:.2f}
        
        Use the handle_claim_with_fraud_check tool.
        """
    )

    print("\nFraud detection demonstration completed.")
    return True


if __name__ == "__main__":
    # Initialize and populate database
    print("Initializing and populating database...")
    DataGenerator.populate_database(num_patients=20, num_claims=50, num_complaints=10)
    print(
        f"Database contains {len(database.patients)} patients, {len(database.claims)} claims, and {len(database.complaints)} complaints"
    )

    # Run the fraud detection demo
    print("\n=== Insurance Claim Fraud Detection Demo ===\n")
    demonstrate_fraud_detection()

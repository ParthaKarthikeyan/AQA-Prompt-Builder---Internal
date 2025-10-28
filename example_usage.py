"""
Example of how to use the AQA Prompt Builder Streamlit app
"""

# This is a reference file showing example inputs for the Streamlit app

EXAMPLE_QUESTION = """Did the agent demonstrate empathy and understanding throughout the call?"""

EXAMPLE_RATING_OPTIONS = """Excellent - Agent consistently demonstrated empathy and understanding
Good - Agent showed empathy most of the time
Fair - Agent showed limited empathy
Poor - Agent did not demonstrate empathy
N/A - Not applicable for this call type"""

EXAMPLE_GUIDELINE = """Evaluate based on:
- Use of empathetic language (e.g., "I understand", "I can see how that would be frustrating")
- Acknowledging customer's feelings and concerns
- Validating customer's emotions
- Offering appropriate support and reassurance
- Tone of voice indicators (if available)

Evidence should be directly quoted from the transcript to support the rating."""

EXAMPLE_TRANSCRIPT = """Agent: Thank you for calling [Company], my name is Sarah. How can I help you today?
Customer: Hi, I'm really frustrated. My package hasn't arrived and it's been a week!
Agent: I completely understand your frustration. I can see how that would be upsetting, especially if you were expecting it. Let me check the tracking information for you right away. Can you please provide me with your order number or tracking number?
Customer: Sure, it's 12345.
Agent: Thank you. Let me look that up for you... I can see here that your package was delayed due to weather conditions in the shipping area. I know this is inconvenient, but your package is now in transit and should arrive within the next 2 business days.
Customer: Okay, that's better than I thought.
Agent: I'm glad I could help clarify that for you. I completely understand waiting for packages can be stressful. I've also sent you an email with updated tracking information so you can monitor its progress. Is there anything else I can help you with today?
Customer: No, that's everything. Thanks!
Agent: You're very welcome! Have a wonderful day."""

# Expected output format:
EXPECTED_OUTPUT = {
    "Question": "Did the agent demonstrate empathy and understanding throughout the call?",
    "Answer": "Excellent - Agent consistently demonstrated empathy and understanding",
    "Justification": "The agent demonstrated empathy through multiple instances: 1) Immediate acknowledgment of the customer's frustration ('I completely understand your frustration'), 2) Validation of customer's emotion ('I can see how that would be upsetting'), 3) Appropriate reassurance ('I'm glad I could help clarify that for you'), 4) Ongoing support ('I completely understand waiting for packages can be stressful'). The agent consistently used empathetic language and validated the customer's concerns throughout the interaction."
}


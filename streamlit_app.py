import streamlit as st
import requests
import json
import time
import re
import ast
import pandas as pd
import tiktoken
from typing import List, Dict, Any
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="AQA Prompt Builder",
    page_icon="🔨",
    layout="wide"
)

# RunPod configuration
RUNPOD_ENDPOINT_ID = "cj0k04fo6vknjh"
RUNPOD_API_KEY = "rpa_ARG4EDO1OIMKM70C4J04YBVR1685WN3VB46AFUSU1c54vp"
RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run"
HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

# Token counting function
def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoder = tiktoken.encoding_for_model("gpt-4")
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception as e:
        st.error(f"Error counting tokens: {e}")
        return 0

# Template content from Template - Prompt Dev Request.docx
PROMPT_DEV_TEMPLATE = """
PROMPT DEVELOPMENT TEMPLATE

Task: Generate a comprehensive evaluation prompt for call center quality assurance.

Inputs Required:
1. Question - The specific question to be evaluated
2. Rating Options - The available rating/answer options
3. Guideline - Detailed guidelines and criteria for evaluation

Output Format:
Generate a structured prompt that includes:
- Clear question statement
- Rating options
- Evaluation guidelines
- Output format specification (JSON)
- Instructions for analysis

The generated prompt should be ready for use with the RunPod inference API.
"""

# Submit job to RunPod to generate prompt
def submit_prompt_generation_job(question: str, rating_options: str, guideline: str, max_tokens: int = 2048, temperature: float = 0.4) -> str:
    """Submit a job to RunPod to generate a prompt based on the template and inputs"""
    
    system_prompt = f"""{PROMPT_DEV_TEMPLATE}

Based on the following inputs, generate a comprehensive evaluation prompt:

QUESTION TO BE EVALUATED:
{question}

RATING OPTIONS:
{rating_options}

EVALUATION GUIDELINE:
{guideline}

Please generate a complete, well-structured prompt that:
1. Clearly presents the question
2. Lists all rating options
3. Provides detailed guidelines for evaluation
4. Specifies the exact JSON output format
5. Includes instructions for analyzing the interaction
6. Is ready for use with call center transcript evaluation

Generate only the prompt text, nothing else."""

    payload = {
        "input": {
            "prompt": f"< | User | >{system_prompt}< | Assistant | >",
            "sampling_params": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }
    }

    try:
        response = requests.post(RUNPOD_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            response_json = response.json()
            return response_json["id"]
        else:
            st.error(f"Failed to submit prompt generation job: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error submitting prompt generation job: {e}")
        return None

# Get the generated prompt from job result
def extract_generated_prompt_from_response(response_text: str) -> str:
    """Extract the generated prompt from the RunPod response"""
    try:
        # Remove any JSON formatting or extra text
        # The prompt should be in the response
        cleaned_text = response_text.strip()
        
        # Try to extract content between any JSON tags if present
        if '<think>' in cleaned_text:
            # Extract content after the redacted_reasoning tag
            parts = cleaned_text.split('</think>')
            if len(parts) > 1:
                cleaned_text = parts[-1].strip()
        
        # Remove any leading/trailing JSON artifacts
        cleaned_text = re.sub(r'^[^A-Za-z]*', '', cleaned_text)
        
        return cleaned_text
    except Exception as e:
        st.error(f"Error extracting generated prompt: {e}")
        return response_text

# Submit job to RunPod
def submit_job(transcript: str, user_prompt: str, max_tokens: int = 32768, temperature: float = 0.4) -> str:
    """Submit a job to RunPod for inference"""
    system_prompt = f"""You are an evaluation system for call center interactions. Analyze the transcript below and answer the question using ONLY ONE rating option from the available options provided in the prompt.

IMPORTANT: Select ONLY ONE rating option based on your analysis. Return ONLY valid JSON output, nothing else.

Transcript:
{transcript}
"""

    payload = {
        "input": {
            "prompt": f"{system_prompt}< | User | >{user_prompt}< | Assistant | >",
            "sampling_params": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }
    }

    try:
        response = requests.post(RUNPOD_URL, headers=HEADERS, json=payload)
        if response.status_code == 200:
            response_json = response.json()
            return response_json["id"]
        else:
            st.error(f"Failed to submit job: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error submitting job: {e}")
        return None

# RunPod job status checking
def check_job_status(job_id: str) -> Dict[str, Any]:
    """Check the status of a RunPod job"""
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{job_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    try:
        response = requests.get(url, headers=headers)
        return response.json()
    except Exception as e:
        st.error(f"Error checking job status: {e}")
        return {"status": "ERROR", "error": str(e)}

# JSON response parsing and validation
def extract_jsons_from_response(raw_response: str) -> List[Dict]:
    """Extract JSON content from the response text"""
    try:
        # Function to extract content after reasoning tags
        def extract_think_content(response_text):
            """Extracts content after </think> tags."""
            # Handle </think> tag
            if '</think>' in response_text:
                remaining_text = response_text.split('</think>')[-1].strip()
            else:
                remaining_text = response_text.strip()
            return remaining_text

        # Clean content and convert to dictionary
        def clean_and_dict(text):
            # Remove all non-JSON content before the JSON object starts and clean up
            # Find the first { and keep everything from there
            idx = text.find('{')
            if idx == -1:
                raise ValueError("No JSON object found")
            cleaned_text = text[idx:]
            return ast.literal_eval(cleaned_text)

        # Extract the final answer after any reasoning tags
        final_ans = extract_think_content(raw_response)

        # Try to find and parse JSON
        # Look for the first valid JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'  # Better JSON pattern
        json_matches = re.findall(json_pattern, final_ans, re.DOTALL)

        # If no matches, try simpler approach
        if not json_matches:
            # Just try to parse the whole thing as JSON
            try:
                dict_result = clean_and_dict(final_ans)
                return [dict_result]
            except:
                st.error(f"Could not parse response as JSON: {final_ans[:200]}")
                return []

        # Clean and convert each JSON string to a dictionary
        json_dicts = []
        for json_str in json_matches:
            try:
                # Clean up the JSON string
                cleaned_str = json_str.strip()
                json_dict = ast.literal_eval(cleaned_str)
                json_dicts.append(json_dict)
            except Exception as e:
                st.warning(f"Could not parse JSON block: {e}")
                continue

        return json_dicts if json_dicts else []
        
    except Exception as e:
        st.error(f"Error parsing JSON response: {e}")
        st.error(f"Raw response: {raw_response[:500]}")
        return []

# Main Streamlit app
def main():
    st.title("🔨 AQA Prompt Builder")
    st.markdown("Build, test, and refine prompts for AQA evaluation")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        max_tokens = st.number_input("Max Tokens", min_value=1000, max_value=32768, value=32768)
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.4, step=0.1)
        
        st.header("RunPod Status")
        try:
            response_health = requests.get(f'https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/health', headers=HEADERS)
            if response_health.status_code == 200:
                health_data = response_health.json()
                st.success("✅ RunPod Endpoint Online")
                st.metric("Jobs Completed", health_data.get("jobs", {}).get("completed", "N/A"))
            else:
                st.error("❌ RunPod Endpoint Offline")
        except Exception as e:
            st.error(f"❌ Error checking endpoint: {e}")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Build Prompt", "🚀 Test Prompt", "📊 Results", "📋 Batch Evaluation"])
    
    # Tab 1: Build Prompt
    with tab1:
        st.header("Build Your Evaluation Prompt")
        st.markdown("Enter your inputs below and RunPod will generate a comprehensive prompt based on the template")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Components")
            
            question = st.text_area(
                "Question:",
                height=150,
                placeholder="e.g., Did the agent greet the customer professionally?"
            )
            
            rating_options = st.text_area(
                "Rating Options:",
                height=100,
                placeholder="e.g., Yes / No / N/A"
            )
            
            guideline = st.text_area(
                "Guideline:",
                height=200,
                placeholder="e.g., A proper greeting should include professional language, warm tone, identification of the agent, and offer of assistance."
            )
            
            # Generate button
            if st.button("🔨 Generate Prompt via RunPod", disabled=not (question and rating_options and guideline)):
                with st.spinner("Submitting prompt generation job to RunPod..."):
                    job_id = submit_prompt_generation_job(question, rating_options, guideline, max_tokens=2048, temperature=temperature)
                    
                if job_id:
                    st.success(f"✅ Prompt generation job submitted!")
                    st.info(f"Job ID: `{job_id}`")
                    
                    # Store job ID in session state
                    st.session_state.prompt_gen_job_id = job_id
                    st.session_state.prompt_gen_question = question
                    st.session_state.prompt_gen_rating_options = rating_options
                    st.session_state.prompt_gen_guideline = guideline
                else:
                    st.error("Failed to submit prompt generation job")
        
        with col2:
            st.subheader("Generated Prompt")
            
            # Check for prompt generation job
            if 'prompt_gen_job_id' in st.session_state:
                job_id = st.session_state.prompt_gen_job_id
                
                if st.button("🔄 Check Generation Status"):
                    with st.spinner("Checking job status..."):
                        status = check_job_status(job_id)
                    
                    if status.get('status') == 'COMPLETED':
                        st.success("✅ Prompt generated successfully!")
                        
                        # Extract and display the generated prompt
                        try:
                            response_text = status.get('output')[0].get('choices')[0].get('tokens')[0]
                            generated_prompt = extract_generated_prompt_from_response(response_text)
                            
                            st.text_area(
                                "Generated Prompt:",
                                value=generated_prompt,
                                height=500,
                                disabled=True,
                                key="display_generated_prompt"
                            )
                            
                            # Save to session state
                            st.session_state.generated_prompt = generated_prompt
                            st.session_state.question = st.session_state.prompt_gen_question
                            st.session_state.rating_options = st.session_state.prompt_gen_rating_options
                            st.session_state.guideline = st.session_state.prompt_gen_guideline
                            
                            # Token count
                            token_count = count_tokens(generated_prompt)
                            st.metric("Prompt Token Count", token_count)
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Prompt",
                                data=generated_prompt,
                                file_name="generated_prompt.txt",
                                mime="text/plain"
                            )
                            
                            # Clear the job ID
                            del st.session_state.prompt_gen_job_id
                            
                        except Exception as e:
                            st.error(f"Error extracting generated prompt: {e}")
                            
                    elif status.get('status') == 'IN_PROGRESS':
                        st.info("⏳ Prompt generation in progress...")
                    elif status.get('status') == 'FAILED':
                        st.error("❌ Prompt generation failed")
                        if 'error' in status:
                            st.error(f"Error: {status['error']}")
                    else:
                        st.warning(f"Job status: {status.get('status', 'Unknown')}")
                else:
                    st.info(f"📝 Job ID: `{job_id}` - Click 'Check Generation Status' to view the generated prompt")
            
            # Display previously generated prompt
            elif 'generated_prompt' in st.session_state:
                st.text_area(
                    "Generated Prompt:",
                    value=st.session_state.generated_prompt,
                    height=500,
                    disabled=True
                )
                
                # Token count
                token_count = count_tokens(st.session_state.generated_prompt)
                st.metric("Prompt Token Count", token_count)
                
                # Download button
                st.download_button(
                    label="📥 Download Prompt",
                    data=st.session_state.generated_prompt,
                    file_name="generated_prompt.txt",
                    mime="text/plain"
                )
            else:
                st.info("👆 Fill in the inputs on the left and click 'Generate Prompt via RunPod' to create a prompt")
    
    # Tab 2: Test Prompt
    with tab2:
        st.header("Test Your Prompt on Transcripts")
        
        # Check if prompt exists
        if 'generated_prompt' not in st.session_state or not st.session_state.generated_prompt:
            st.warning("⚠️ Please build a prompt first in the 'Build Prompt' tab")
            
            # Allow manual prompt entry for testing
            manual_prompt = st.text_area(
                "Or enter a custom prompt to test:",
                height=200,
                placeholder="Enter your prompt here..."
            )
            if manual_prompt:
                st.session_state.current_test_prompt = manual_prompt
        else:
            st.success("✅ Using prompt from 'Build Prompt' tab")
            st.session_state.current_test_prompt = st.session_state.generated_prompt
            st.text_area(
                "Prompt to be tested:",
                value=st.session_state.current_test_prompt,
                height=200,
                disabled=True
            )
        
        # Transcript input
        st.subheader("Transcript Input")
        transcript = st.text_area(
            "Enter or paste transcript:",
            height=300,
            placeholder="Agent: Hello, thank you for calling [Company]. My name is Sarah. How can I assist you today?\nCustomer: Hi, I need help with my recent order...\n\n[Continue transcript here]"
        )
        
        # Submit button for testing
        col1, col2 = st.columns([1, 3])
        with col1:
            test_clicked = st.button("🚀 Test Prompt", disabled=not (transcript and ('current_test_prompt' in st.session_state)))
        
        if test_clicked:
            if not transcript.strip():
                st.error("Please enter a transcript")
            elif 'current_test_prompt' not in st.session_state:
                st.error("Please build or enter a prompt")
            else:
                with st.spinner("Submitting job to RunPod..."):
                    job_id = submit_job(transcript, st.session_state.current_test_prompt, max_tokens, temperature)
                    
                if job_id:
                    st.success(f"✅ Job submitted successfully!")
                    st.info(f"Job ID: `{job_id}`")
                    
                    # Store job ID in session state
                    if 'test_job_ids' not in st.session_state:
                        st.session_state.test_job_ids = []
                    st.session_state.test_job_ids.append(job_id)
                    st.session_state.current_test_job = job_id
                    st.session_state.test_job_status = "SUBMITTED"
                else:
                    st.error("Failed to submit job")
    
    # Tab 3: Results
    with tab3:
        st.header("View Results")
        
        # Check for pending job
        if 'current_test_job' in st.session_state:
            job_id = st.session_state.current_test_job
            
            if st.button("🔄 Check Status"):
                with st.spinner("Checking job status..."):
                    status = check_job_status(job_id)
                    
                if status.get('status') == 'COMPLETED':
                    st.success("✅ Job completed!")
                    
                    # Extract and display results
                    try:
                        tokens = status.get('output')[0].get('choices')[0].get('tokens')[0]
                        jsons = extract_jsons_from_response(tokens)
                        
                        if jsons:
                            st.subheader("📋 Results")
                            
                            # Store results
                            if 'test_results' not in st.session_state:
                                st.session_state.test_results = {}
                            st.session_state.test_results[job_id] = jsons
                            
                            # Display results
                            for i, result in enumerate(jsons):
                                st.write(f"**Result {i+1}:**")
                                st.json(result)
                            
                            # Download results
                            results_json = json.dumps(jsons, indent=2)
                            st.download_button(
                                label="📥 Download Results as JSON",
                                data=results_json,
                                file_name=f"prompt_test_results_{job_id}.json",
                                mime="application/json"
                            )
                            
                            # Clear current job
                            if 'current_test_job' in st.session_state:
                                del st.session_state.current_test_job
                        else:
                            st.error("No valid JSON results found in response")
                            
                    except Exception as e:
                        st.error(f"Error processing results: {e}")
                        
                elif status.get('status') == 'IN_PROGRESS':
                    st.info("⏳ Job is still in progress...")
                elif status.get('status') == 'FAILED':
                    st.error("❌ Job failed")
                    if 'error' in status:
                        st.error(f"Error: {status['error']}")
                else:
                    st.warning(f"Job status: {status.get('status', 'Unknown')}")
            else:
                st.info(f"📝 Job ID: `{job_id}` - Click 'Check Status' to view results")
        
        # Show stored results
        if 'test_results' in st.session_state and st.session_state.test_results:
            st.subheader("📊 Stored Results")
            
            for job_id, results in st.session_state.test_results.items():
                with st.expander(f"📄 Results for Job {job_id[:8]}... ({len(results)} results)"):
                    for i, result in enumerate(results):
                        st.write(f"**Result {i+1}:**")
                        st.json(result)
                    
                    # Download individual results
                    results_json = json.dumps(results, indent=2)
                    st.download_button(
                        label=f"📥 Download These Results",
                        data=results_json,
                        file_name=f"results_{job_id}.json",
                        mime="application/json",
                        key=f"download_{job_id}"
                    )
        
        # Job history
        if 'test_job_ids' in st.session_state and st.session_state.test_job_ids:
            st.subheader("📚 Job History")
            for i, job_id in enumerate(reversed(st.session_state.test_job_ids)):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.code(job_id)
                with col2:
                    if st.button(f"Check", key=f"check_{i}"):
                        st.session_state.current_test_job = job_id
                        st.rerun()
        
        if 'current_test_job' not in st.session_state and 'test_job_ids' not in st.session_state:
            st.info("👆 No tests yet. Build a prompt and test it in the 'Test Prompt' tab")
    
    # Tab 4: Batch Evaluation
    with tab4:
        st.header("📋 Batch Evaluation - Multiple Questions")
        st.markdown("Evaluate multiple questions on the same transcript")
        
        # Input for number of questions
        num_questions = st.number_input(
            "Number of Questions", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="How many questions do you want to evaluate?"
        )
        
        # Container for questions
        questions_data = []
        
        with st.expander(f"📝 Define {num_questions} Questions", expanded=True):
            for i in range(num_questions):
                st.subheader(f"Question {i+1}")
                
                question_text = st.text_area(
                    f"Question {i+1}:",
                    height=100,
                    key=f"batch_q_{i}",
                    placeholder=f"e.g., Did the agent greet the customer properly?"
                )
                
                rating_options = st.text_area(
                    f"Rating Options:",
                    height=60,
                    key=f"batch_r_{i}",
                    placeholder="e.g., Yes / No / N/A"
                )
                
                guideline = st.text_area(
                    f"Guideline:",
                    height=100,
                    key=f"batch_g_{i}",
                    placeholder="e.g., A proper greeting includes professional language and warm tone."
                )
                
                if question_text and rating_options and guideline:
                    questions_data.append({
                        'index': i + 1,
                        'question': question_text,
                        'rating_options': rating_options,
                        'guideline': guideline
                    })
                
                st.divider()
        
        # Transcript input
        st.subheader("📄 Transcript")
        batch_transcript = st.text_area(
            "Enter or paste transcript:",
            height=300,
            key="batch_transcript",
            placeholder="Agent: Hello, thank you for calling [Company]. My name is Sarah. How can I assist you today?\nCustomer: Hi, I need help with my recent order..."
        )
        
        # Submit button
        col1, col2 = st.columns([1, 4])
        with col1:
            batch_submit = st.button(
                "🚀 Generate & Test All", 
                disabled=not (questions_data and batch_transcript and len(questions_data) == num_questions)
            )
        
        if batch_submit:
            if not questions_data or len(questions_data) != num_questions:
                st.error(f"Please fill in all {num_questions} questions with their rating options and guidelines")
            elif not batch_transcript.strip():
                st.error("Please enter a transcript")
            else:
                with st.spinner("Generating prompts and submitting jobs..."):
                    # Generate prompts for each question
                    generated_prompts = []
                    job_ids = []
                    
                    for q_data in questions_data:
                        # Generate prompt
                        prompt = f"""Answer questions based on the interaction between a call-center agent and a customer.

QUESTION:
{q_data['question']}

RATING OPTIONS:
{q_data['rating_options']}

GUIDELINE:
{q_data['guideline']}

Please analyze the interaction and provide your answer in the following JSON format:
{{
    "Question": "{q_data['question']}",
    "Answer": "[Select appropriate option from: {q_data['rating_options']}]",
    "Justification": "[Provide detailed justification based on the interaction and guidelines, including relevant evidence from the interaction]"
}}

IMPORTANT: Return ONLY valid JSON, nothing else."""

                        generated_prompts.append(prompt)
                        
                        # Submit job
                        job_id = submit_job(batch_transcript, prompt, max_tokens, temperature)
                        if job_id:
                            job_ids.append({
                                'question_num': q_data['index'],
                                'question': q_data['question'],
                                'job_id': job_id
                            })
                    
                    if job_ids:
                        st.success(f"✅ Submitted {len(job_ids)} evaluation jobs!")
                        
                        # Store in session state
                        if 'batch_jobs' not in st.session_state:
                            st.session_state.batch_jobs = []
                        
                        for job_info in job_ids:
                            st.session_state.batch_jobs.append(job_info)
        
        # Display batch results section
        st.header("📊 Batch Results")
        
        if 'batch_jobs' in st.session_state and st.session_state.batch_jobs:
            for i, job_info in enumerate(st.session_state.batch_jobs):
                with st.expander(f"📋 Question {job_info['question_num']}: {job_info['question'][:50]}...", expanded=(i == 0)):
                    st.write(f"**Job ID:** `{job_info['job_id']}`")
                    
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        if st.button(f"🔄 Check Status", key=f"batch_check_{i}"):
                            with st.spinner("Checking status..."):
                                status = check_job_status(job_info['job_id'])
                            
                            if status.get('status') == 'COMPLETED':
                                st.success("✅ Completed!")
                                
                                try:
                                    response_text = status.get('output')[0].get('choices')[0].get('tokens')[0]
                                    jsons = extract_jsons_from_response(response_text)
                                    
                                    if jsons:
                                        for result in jsons:
                                            st.json(result)
                                        
                                        # Store result
                                        if 'batch_results' not in st.session_state:
                                            st.session_state.batch_results = {}
                                        st.session_state.batch_results[job_info['job_id']] = jsons
                                        
                                    else:
                                        st.error("No valid JSON found in response")
                                        
                                except Exception as e:
                                    st.error(f"Error: {e}")
                                    st.text(response_text[:500] if 'response_text' in locals() else "No response")
                            
                            elif status.get('status') == 'IN_PROGRESS':
                                st.info("⏳ Still processing...")
                            elif status.get('status') == 'FAILED':
                                st.error(f"❌ Failed: {status.get('error', 'Unknown error')}")
                    
                    # Show stored results if available
                    if 'batch_results' in st.session_state and job_info['job_id'] in st.session_state.batch_results:
                        st.write("**Result:**")
                        for result in st.session_state.batch_results[job_info['job_id']]:
                            st.json(result)
                    
                    with col2:
                        if st.button(f"🗑️ Remove", key=f"batch_remove_{i}"):
                            st.session_state.batch_jobs.pop(i)
                            st.rerun()
            
            # Download all results
            if 'batch_results' in st.session_state and st.session_state.batch_results:
                all_results_json = json.dumps(st.session_state.batch_results, indent=2)
                st.download_button(
                    label="📥 Download All Batch Results",
                    data=all_results_json,
                    file_name=f"batch_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("👆 Fill in the questions and transcript above, then click 'Generate & Test All'")

if __name__ == "__main__":
    main()


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
    system_prompt = f"""Evaluate this transcript between an agent and a customer. Provide output in the following format:

**Output Format:**  
```json  
{{  
  "question": "The question being evaluated",  
  "rating": "Select ONE option from what is specified in the prompt",  
  "explanation": "Provide a detailed explanation of the evaluation based on the guidelines, including relevant evidence from the transcript."  
}}  
```

IMPORTANT: Return ONLY valid JSON output in the exact format above. Nothing else.

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
                
                # Show worker information if available
                if "workers" in health_data:
                    workers_data = health_data["workers"]
                    if isinstance(workers_data, dict):
                        total_workers = len(workers_data)
                        busy_workers = sum(1 for w in workers_data.values() if w.get("status") == "BUSY")
                        st.metric("Workers", f"{busy_workers}/{total_workers} busy", delta=f"{total_workers} total")
        except Exception as e:
            st.error(f"❌ Error checking endpoint: {e}")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Build Prompt", "🚀 Test Prompt", "📊 Results", "📋 Batch Evaluation", "📦 Bulk Testing"])
    
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
                    height=68,
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
                "🚀 Generate Prompts & Test All", 
                disabled=not (questions_data and batch_transcript and len(questions_data) == num_questions)
            )
        
        if batch_submit:
            if not questions_data or len(questions_data) != num_questions:
                st.error(f"Please fill in all {num_questions} questions with their rating options and guidelines")
            elif not batch_transcript.strip():
                st.error("Please enter a transcript")
            else:
                with st.spinner("Step 1/2: Generating prompts via RunPod..."):
                    # First, generate prompts for each question via RunPod
                    prompt_generation_jobs = []
                    
                    for q_data in questions_data:
                        prompt_gen_job_id = submit_prompt_generation_job(
                            q_data['question'], 
                            q_data['rating_options'], 
                            q_data['guideline'], 
                            max_tokens=2048, 
                            temperature=temperature
                        )
                        if prompt_gen_job_id:
                            prompt_generation_jobs.append({
                                'question_num': q_data['index'],
                                'question': q_data['question'],
                                'prompt_gen_job_id': prompt_gen_job_id,
                                'rating_options': q_data['rating_options'],
                                'guideline': q_data['guideline']
                            })
                    
                    st.success(f"✅ Submitted {len(prompt_generation_jobs)} prompt generation jobs!")
                    
                    # Store prompt generation job IDs
                    st.session_state.batch_prompt_gen_jobs = prompt_generation_jobs
                    st.session_state.batch_transcript = batch_transcript
                    st.session_state.batch_waiting_for_prompts = True
        
        # Show prompt generation status
        if 'batch_prompt_gen_jobs' in st.session_state and st.session_state.batch_prompt_gen_jobs:
            st.subheader("📝 Prompt Generation Status")
            
            all_prompts_generated = True
            generated_prompts = []
            
            for i, job_info in enumerate(st.session_state.batch_prompt_gen_jobs):
                with st.expander(f"Question {job_info['question_num']}: {job_info['question'][:50]}...", expanded=True):
                    st.write(f"**Prompt Generation Job ID:** `{job_info['prompt_gen_job_id']}`")
                    
                    # Check status
                    if st.button(f"🔄 Check Prompt Status", key=f"check_prompt_gen_{i}"):
                        with st.spinner("Checking status..."):
                            status = check_job_status(job_info['prompt_gen_job_id'])
                        
                        if status.get('status') == 'COMPLETED':
                            st.success("✅ Prompt generated!")
                            try:
                                response_text = status.get('output')[0].get('choices')[0].get('tokens')[0]
                                generated_prompt = extract_generated_prompt_from_response(response_text)
                                
                                job_info['generated_prompt'] = generated_prompt
                                job_info['prompt_completed'] = True
                                
                            except Exception as e:
                                st.error(f"Error extracting prompt: {e}")
                                
                        elif status.get('status') == 'IN_PROGRESS':
                            st.info("⏳ Still generating...")
                        elif status.get('status') == 'FAILED':
                            st.error(f"❌ Failed: {status.get('error', 'Unknown error')}")
            
            # Check if all prompts are generated
            all_complete = all(job.get('prompt_completed', False) for job in st.session_state.batch_prompt_gen_jobs)
            
            if all_complete:
                st.success("✅ All prompts generated! Ready to test.")
                
                # Now test all prompts on transcript
                if st.button("🧪 Test All Prompts on Transcript"):
                    with st.spinner("Step 2/2: Testing prompts on transcript..."):
                        job_ids = []
                        
                        for job_info in st.session_state.batch_prompt_gen_jobs:
                            if 'generated_prompt' in job_info:
                                # Submit test job
                                test_job_id = submit_job(
                                    st.session_state.batch_transcript, 
                                    job_info['generated_prompt'], 
                                    max_tokens, 
                                    temperature
                                )
                                
                                if test_job_id:
                                    job_ids.append({
                                        'question_num': job_info['question_num'],
                                        'question': job_info['question'],
                                        'job_id': test_job_id
                                    })
                        
                        if job_ids:
                            st.success(f"✅ Submitted {len(job_ids)} evaluation jobs!")
                            
                            # Store evaluation job IDs
                            if 'batch_jobs' not in st.session_state:
                                st.session_state.batch_jobs = []
                            
                            for job_info in job_ids:
                                st.session_state.batch_jobs.append(job_info)
                            
                            # Clear prompt generation tracking
                            if 'batch_prompt_gen_jobs' in st.session_state:
                                del st.session_state.batch_prompt_gen_jobs
                            if 'batch_waiting_for_prompts' in st.session_state:
                                del st.session_state.batch_waiting_for_prompts
        
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
    
    # Tab 5: Bulk Testing
    with tab5:
        st.header("📦 Bulk Testing - Multiple Transcripts")
        st.markdown("Upload multiple transcripts and test them against your generated prompts")
        
        # Prompt input section with three methods
        st.subheader("📝 Prompts to Test")
        
        # Method selection
        prompt_input_method = st.radio(
            "Choose prompt input method:",
            ["Use Generated Prompt", "Upload Prompts File (CSV/Excel)", "Manual Entry"],
            key="bulk_prompt_method"
        )
        
        if prompt_input_method == "Use Generated Prompt":
            if 'generated_prompt' not in st.session_state or not st.session_state.generated_prompt:
                st.warning("⚠️ Please build a prompt in the 'Build Prompt' tab first")
            else:
                st.success("✅ Using prompt from 'Build Prompt' tab")
                st.session_state.bulk_test_prompts = {
                    'single': True,
                    'prompt': st.session_state.generated_prompt
                }
                st.text_area(
                    "Prompt to be used:",
                    value=st.session_state.generated_prompt,
                    height=150,
                    disabled=True,
                    key="bulk_prompt_display"
                )
        
        elif prompt_input_method == "Upload Prompts File (CSV/Excel)":
            st.markdown("Upload a CSV or Excel file with columns: `question_number`, `question`, `prompt`")
            
            prompts_file = st.file_uploader(
                "Choose a prompts file",
                type=['csv', 'xlsx'],
                key="bulk_prompts_file"
            )
            
            if prompts_file is not None:
                try:
                    if prompts_file.name.endswith('.csv'):
                        prompts_df = pd.read_csv(prompts_file)
                    else:
                        prompts_df = pd.read_excel(prompts_file)
                    
                    st.success(f"✅ Prompts file loaded: {len(prompts_df)} prompts")
                    
                    # Check required columns
                    if 'question' not in prompts_df.columns or 'prompt' not in prompts_df.columns:
                        st.error("❌ File must have 'question' and 'prompt' columns")
                        st.write(f"Found columns: {', '.join(prompts_df.columns)}")
                    else:
                        # Add question_number if it doesn't exist
                        if 'question_number' not in prompts_df.columns:
                            prompts_df['question_number'] = range(1, len(prompts_df) + 1)
                            st.info("ℹ️ Added 'question_number' column (1, 2, 3, ...)")
                        
                        st.write(f"**Preview ({len(prompts_df)} rows):**")
                        st.dataframe(prompts_df.head(), use_container_width=True)
                        
                        # Store in session state with question numbers
                        prompts_dict = {}
                        prompts_with_numbers = {}
                        for _, row in prompts_df.iterrows():
                            prompts_dict[row['question']] = row['prompt']
                            prompts_with_numbers[row['question']] = {
                                'prompt': row['prompt'],
                                'question_number': row['question_number']
                            }
                        
                        st.session_state.bulk_test_prompts = {
                            'single': False,
                            'prompts': prompts_dict,
                            'prompts_with_numbers': prompts_with_numbers,
                            'df': prompts_df
                        }
                
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        else:  # Manual Entry
            manual_prompt = st.text_area(
                "Enter a custom prompt to test:",
                height=200,
                key="bulk_manual_prompt",
                placeholder="Enter your prompt here..."
            )
            if manual_prompt:
                st.session_state.bulk_test_prompts = {
                    'single': True,
                    'prompt': manual_prompt
                }
        
        # File upload section
        st.subheader("📄 Upload Transcripts File")
        st.markdown("Upload a CSV or Excel file with columns: `interactionid`, `transcript`")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx'],
            key="bulk_file_upload"
        )
        
        if uploaded_file is not None:
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File loaded: {len(df)} transcripts")
                
                # Check required columns
                if 'interactionid' not in df.columns or 'transcript' not in df.columns:
                    st.error("❌ File must have 'interactionid' and 'transcript' columns")
                    st.write(f"Found columns: {', '.join(df.columns)}")
                else:
                    st.write(f"**Preview ({len(df)} rows):**")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Store in session state
                    st.session_state.bulk_transcripts_df = df
                    
                    # Submit button
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        is_ready = 'bulk_test_prompts' in st.session_state and st.session_state.bulk_test_prompts is not None
                        if st.button("🚀 Start Bulk Testing", disabled=not is_ready):
                            if 'bulk_test_prompts' not in st.session_state or st.session_state.bulk_test_prompts is None:
                                st.error("Please enter or generate prompts first")
                            else:
                                prompts_info = st.session_state.bulk_test_prompts
                                
                                if prompts_info['single']:
                                    # Single prompt - test against all transcripts
                                    total_jobs = len(df)
                                    with st.spinner(f"Submitting {total_jobs} jobs to RunPod..."):
                                        bulk_job_ids = []
                                        
                                        for idx, row in df.iterrows():
                                            interaction_id = row['interactionid']
                                            transcript = row['transcript']
                                            
                                            job_id = submit_job(
                                                transcript,
                                                prompts_info['prompt'],
                                                max_tokens,
                                                temperature
                                            )
                                            
                                            if job_id:
                                                bulk_job_ids.append({
                                                    'interactionid': interaction_id,
                                                    'transcript': transcript,
                                                    'job_id': job_id,
                                                    'index': idx,
                                                    'prompt': 'Single Prompt',
                                                    'question': ''
                                                })
                                        
                                        if bulk_job_ids:
                                            st.success(f"✅ Submitted {len(bulk_job_ids)} jobs!")
                                            
                                            # Store in session state
                                            if 'bulk_jobs' not in st.session_state:
                                                st.session_state.bulk_jobs = []
                                            
                                            st.session_state.bulk_jobs.extend(bulk_job_ids)
                                
                                else:
                                    # Multiple prompts - send ALL prompts in ONE request per transcript
                                    prompts_dict = prompts_info['prompts']
                                    prompts_with_numbers = prompts_info.get('prompts_with_numbers', {})
                                    
                                    # Build combined prompt with all questions
                                    questions_list = []
                                    for q_idx, (question, prompt_text) in enumerate(prompts_dict.items(), 1):
                                        question_num = prompts_with_numbers.get(question, {}).get('question_number', q_idx)
                                        questions_list.append(f"Question {question_num}: {question}")
                                    
                                    combined_prompt = f"""Please evaluate the following {len(prompts_dict)} questions based on the transcript provided.

QUESTIONS TO EVALUATE:
{chr(10).join(questions_list)}

For each question, please provide your answer in the JSON format below. Return an array of JSON objects, one for each question.
""" + "\n\n".join([prompt_text for prompt_text in prompts_dict.values()])
                                    
                                    total_jobs = len(df)
                                    with st.spinner(f"Submitting {total_jobs} jobs to RunPod (1 per transcript with all prompts)..."):
                                        bulk_job_ids = []
                                        
                                        for idx, row in df.iterrows():
                                            interaction_id = row['interactionid']
                                            transcript = row['transcript']
                                            
                                            # Single job per transcript with all prompts
                                            job_id = submit_job(
                                                transcript,
                                                combined_prompt,
                                                max_tokens,
                                                temperature
                                            )
                                            
                                            if job_id:
                                                bulk_job_ids.append({
                                                    'interactionid': interaction_id,
                                                    'transcript': transcript,
                                                    'job_id': job_id,
                                                    'index': idx,
                                                    'prompts': list(prompts_dict.keys()),  # Store all questions
                                                    'prompts_with_numbers': prompts_with_numbers
                                                })
                                        
                                        if bulk_job_ids:
                                            st.success(f"✅ Submitted {len(bulk_job_ids)} jobs!")
                                            
                                            # Store in session state
                                            if 'bulk_jobs' not in st.session_state:
                                                st.session_state.bulk_jobs = []
                                            
                                            st.session_state.bulk_jobs.extend(bulk_job_ids)
            
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        # Bulk results section
        st.subheader("📊 Bulk Results")
        
        if 'bulk_jobs' in st.session_state and st.session_state.bulk_jobs:
            st.write(f"**Total jobs submitted:** {len(st.session_state.bulk_jobs)}")
            
            # Store results
            if 'bulk_results' not in st.session_state:
                st.session_state.bulk_results = {}
            
            # Progress tracking
            completed = len(st.session_state.bulk_results)
            total = len(st.session_state.bulk_jobs)
            
            st.metric("Progress", f"{completed}/{total} completed", f"{int(completed/total*100) if total > 0 else 0}%")
            
            # Check status for all jobs
            if st.button("🔄 Check All Statuses"):
                with st.spinner("Checking job statuses..."):
                    for job_info in st.session_state.bulk_jobs:
                        # Check if this job is already processed
                        if 'prompts' in job_info and len(job_info['prompts']) > 0:
                            # Multi-prompt mode: check if any result exists for this job
                            processed = any(key.startswith(f"{job_info['job_id']}_") for key in st.session_state.bulk_results.keys())
                            should_process = not processed
                        else:
                            # Single prompt mode: check if job_id exists
                            should_process = job_info['job_id'] not in st.session_state.bulk_results
                        
                        if should_process:
                            status = check_job_status(job_info['job_id'])
                            
                            if status.get('status') == 'COMPLETED':
                                try:
                                    response_text = status.get('output')[0].get('choices')[0].get('tokens')[0]
                                    jsons = extract_jsons_from_response(response_text)
                                    
                                    if jsons:
                                        # Check if this job has multiple prompts (bulk multi-prompt mode)
                                        if 'prompts' in job_info and len(job_info['prompts']) > 0:
                                            # Process multiple responses for multiple prompts
                                            prompts_list = job_info['prompts']
                                            prompts_with_numbers = job_info.get('prompts_with_numbers', {})
                                            
                                            # Store each result separately
                                            for i, json_result in enumerate(jsons):
                                                if i < len(prompts_list):
                                                    question = prompts_list[i]
                                                    question_number = prompts_with_numbers.get(question, {}).get('question_number', '')
                                                    
                                                    st.session_state.bulk_results[f"{job_info['job_id']}_{i}"] = {
                                                        'interactionid': job_info['interactionid'],
                                                        'transcript': job_info['transcript'],
                                                        'result': json_result,
                                                        'index': job_info['index'],
                                                        'question': question,
                                                        'question_number': question_number,
                                                        'prompt': 'Multi-Prompt'
                                                    }
                                        else:
                                            # Single prompt mode
                                            st.session_state.bulk_results[job_info['job_id']] = {
                                                'interactionid': job_info['interactionid'],
                                                'transcript': job_info['transcript'],
                                                'result': jsons[0] if jsons else None,
                                                'index': job_info['index'],
                                                'question': job_info.get('question', ''),
                                                'question_number': job_info.get('question_number', ''),
                                                'prompt': job_info.get('prompt', '')
                                            }
                                except Exception as e:
                                    st.error(f"Error processing {job_info['interactionid']}: {e}")
                            elif status.get('status') == 'IN_PROGRESS':
                                pass  # Don't show message for every IN_PROGRESS
                            elif status.get('status') == 'FAILED':
                                st.error(f"❌ {job_info['interactionid']} - Failed")
                    
                st.rerun()  # Rerun after checking all jobs
            
            # Display results
            if st.session_state.bulk_results:
                # Convert to DataFrame for download (pivoted format)
                results_list = []
                for job_id, result_data in st.session_state.bulk_results.items():
                    result = result_data['result']
                    if result:
                        # Store both rating and explanation with question number
                        results_list.append({
                            'question_number': result_data.get('question_number', ''),
                            'interactionid': result_data['interactionid'],
                            'question': result_data.get('question', result.get('question', result.get('Question', ''))),
                            'rating': result.get('rating', result.get('Rating', result.get('Answer', ''))),
                            'explanation': result.get('explanation', result.get('Explanation', result.get('Justification', '')))
                        })
                
                if results_list:
                    # Create original DataFrame
                    results_df = pd.DataFrame(results_list)
                    
                    st.write("**Results (Original Format):**")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Pivot the data: Questions as rows, InteractionIDs as columns
                    try:
                        # Add combined question label if question_number exists
                        if 'question_number' in results_df.columns:
                            results_df['question_label'] = results_df.apply(
                                lambda row: f"Q{row['question_number']}: {row['question']}" if row['question_number'] else row['question'], 
                                axis=1
                            )
                        else:
                            results_df['question_label'] = results_df['question']
                        
                        # Create pivoted table with ratings
                        rating_df = results_df.pivot_table(
                            index='question_label',
                            columns='interactionid',
                            values='rating',
                            aggfunc='first'
                        )
                        rating_df = rating_df.fillna('').reset_index()
                        
                        st.write("**Pivoted Results (Ratings by Question):**")
                        st.dataframe(rating_df, use_container_width=True)
                        
                        # Download as CSV
                        csv = rating_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Ratings as CSV",
                            data=csv,
                            file_name=f"bulk_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Download as Excel with multiple sheets
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            # Pivoted ratings
                            rating_df.to_excel(writer, sheet_name='Ratings (Pivoted)', index=False)
                            
                            # Create pivoted explanations
                            explanation_df = results_df.pivot_table(
                                index='question_label',
                                columns='interactionid',
                                values='explanation',
                                aggfunc='first'
                            )
                            explanation_df = explanation_df.fillna('').reset_index()
                            explanation_df.to_excel(writer, sheet_name='Explanations (Pivoted)', index=False)
                            
                            # Original format
                            results_df.to_excel(writer, sheet_name='Original Format', index=False)
                            
                            # Transcripts
                            if 'bulk_transcripts_df' in st.session_state:
                                st.session_state.bulk_transcripts_df.to_excel(writer, sheet_name='Transcripts', index=False)
                        
                        st.download_button(
                            label="📥 Download Results as Excel",
                            data=buffer.getvalue(),
                            file_name=f"bulk_results_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.warning(f"Could not pivot results: {e}")
                        # Fall back to original format if pivot fails
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"bulk_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            # Display job list
            with st.expander("📋 Job Details"):
                for job_info in st.session_state.bulk_jobs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Check status for multi-prompt or single-prompt jobs
                        if 'prompts' in job_info and len(job_info['prompts']) > 0:
                            # Multi-prompt mode: check if results exist
                            exists = any(key.startswith(f"{job_info['job_id']}_") for key in st.session_state.bulk_results.keys())
                            status = "✅ Completed" if exists else "⏳ Pending"
                            prompt_count = f" ({len(job_info['prompts'])} prompts)" if 'prompts' in job_info else ""
                        else:
                            status = "✅ Completed" if job_info['job_id'] in st.session_state.bulk_results else "⏳ Pending"
                            prompt_count = ""
                        
                        st.write(f"**{job_info['interactionid']}** - Job ID: `{job_info['job_id']}` - {status}{prompt_count}")
                    with col2:
                        if st.button("🗑️", key=f"bulk_remove_{job_info['job_id']}"):
                            st.session_state.bulk_jobs.remove(job_info)
                            
                            # Remove all related results
                            if 'prompts' in job_info and len(job_info['prompts']) > 0:
                                # Remove multi-prompt results
                                keys_to_remove = [key for key in st.session_state.bulk_results.keys() if key.startswith(f"{job_info['job_id']}_")]
                                for key in keys_to_remove:
                                    del st.session_state.bulk_results[key]
                            else:
                                # Single prompt result
                                if job_info['job_id'] in st.session_state.bulk_results:
                                    del st.session_state.bulk_results[job_info['job_id']]
                            st.rerun()
        else:
            st.info("👆 Upload a CSV/Excel file and click 'Start Bulk Testing'")

if __name__ == "__main__":
    main()


# AQA Prompt Builder

A Streamlit application for building, testing, and refining prompts for Automated Quality Assurance (AQA) evaluation.

## Features

### 1. Build Prompt Tab
- **Question**: Define the evaluation question
- **Rating Options**: Specify the possible answers (e.g., Yes/No/NA, Excellent/Good/Fair/Poor)
- **Guideline**: Provide detailed evaluation criteria and guidelines
- Generates a structured prompt based on the template from "Template - Prompt Dev Request.docx"
- Shows token count for the generated prompt
- Downloads the generated prompt as a text file

### 2. Test Prompt Tab
- Uses the generated prompt from the Build Prompt tab
- Allows manual transcript entry in a text box
- Submits the prompt to RunPod inference API
- Provides real-time job status tracking

### 3. Results Tab
- Displays inference results from RunPod
- Shows JSON-formatted answers
- Downloads results as JSON files
- Maintains a history of all test jobs

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. **Build a Prompt**:
   - Enter a question you want to evaluate
   - Define rating options (e.g., "Yes / No / NA")
   - Provide detailed guidelines
   - Review the generated prompt

3. **Test Your Prompt**:
   - Go to the "Test Prompt" tab
   - Enter or paste a transcript in the text box
   - Click "Test Prompt" to submit to RunPod
   - Monitor job status

4. **View Results**:
   - Go to the "Results" tab
   - Click "Check Status" to retrieve results
   - Review the JSON output
   - Download results if needed

## Configuration

The app uses RunPod for inference. Update these configurations in the code:
- `RUNPOD_ENDPOINT_ID`: Your RunPod endpoint ID
- `RUNPOD_API_KEY`: Your RunPod API key

Adjustable parameters (in sidebar):
- **Max Tokens**: Maximum tokens for inference (default: 32768)
- **Temperature**: Sampling temperature (default: 0.4)

## Workflow

1. **Build**: Create prompts using Question, Rating Options, and Guidelines
2. **Test**: Test prompts on sample transcripts
3. **Refine**: Adjust prompts based on results
4. **Deploy**: Use successful prompts for production evaluation

## Prompt Template Structure

The generated prompt follows this structure:
1. **Question**: The evaluation question
2. **Rating Options**: Available answer options
3. **Guideline**: Evaluation criteria and guidelines
4. **Output Format**: JSON format specification

## Sample Transcripts

You can manually enter transcripts in the Test Prompt tab, or paste from Excel/CSV files. The format should be:
```
Agent: [Agent's dialogue]
Customer: [Customer's dialogue]
...
```

## Troubleshooting

- **RunPod Offline**: Check your API key and endpoint ID
- **Job Failed**: Review the error message and check your prompt formatting
- **No Results**: Ensure your prompt is well-formed and the transcript is valid
- **Token Limit Exceeded**: Reduce max_tokens or simplify your prompt

## File Structure

```
AQA Prompt Builder - Internal/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── Template - Prompt Dev Request.docx  # Prompt template reference
```


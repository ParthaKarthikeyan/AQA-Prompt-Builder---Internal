@echo off
echo Initializing git repository...
git init
echo.
echo Adding files to git...
git add .
echo.
echo Creating initial commit...
git commit -m "Initial commit: AQA Prompt Builder Streamlit app

- Created streamlit app with 3 tabs: Build Prompt, Test Prompt, Results
- Prompt generation via RunPod API using Question, Rating Options, and Guideline
- Testing capabilities on transcripts
- Results viewing and download functionality
- Based on Template - Prompt Dev Request.docx structure"

echo.
echo Git repository initialized and files committed!
pause


@echo off
if "%1"=="" (
    echo Usage: quick-commit.bat "Your commit message"
    echo Example: quick-commit.bat "Added new feature"
    exit /b 1
)

echo.
echo Staging all changes...
git add .

echo.
echo Committing changes...
git commit -m "%*"

echo.
echo Pushing to GitHub...
git push origin main

echo.
echo Done! Changes pushed to GitHub.
echo.


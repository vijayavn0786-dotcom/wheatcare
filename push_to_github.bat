@echo off
title Push WheatCare AI to GitHub
color 0B
echo =======================================================
echo          PUSHING WHEATCARE AI TO GITHUB
echo =======================================================
echo.
echo Target Repository: https://github.com/vijayavn0786-dotcom/wheatcare.git
echo Branch: main
echo.
cd /d "%~dp0"
git push -u origin main
echo.
pause

@echo off
title Simulateur de Securite VANET - Dashboard
echo =====================================================================
echo    🛡️ INITIALISATION DU SIMULATEUR DE SECURITE VANET (INF4258)
echo =====================================================================
echo.

:: Se deplacer dans le repertoire du script
cd /d "%~dp0"

:: Verification de la variable SUMO_HOME
if "%SUMO_HOME%"=="" (
    echo [ERROR] La variable d'environnement SUMO_HOME n'est pas configuree.
    echo Veuillez installer Eclipse SUMO et ajouter SUMO_HOME a vos variables d'environnement.
    pause
    exit /b 1
)

:: Detection automatique de l'environnement contenant les dependances (gymnasium, streamlit, stable-baselines3)
set RUN_CMD=

:: Test 1: Environnement virtuel (.venv)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import gymnasium, streamlit, stable_baselines3" >nul 2>&1
    if errorlevel 0 (
        echo [*] Environnement virtuel valide detecte (.venv).
        set RUN_CMD=".venv\Scripts\python.exe" -m streamlit run app_dashboard.py
    )
)

:: Test 2: Python global
if "%RUN_CMD%"=="" (
    python -c "import gymnasium, streamlit, stable_baselines3" >nul 2>&1
    if %errorlevel% equ 0 (
        echo [*] Dependances trouvees dans l'environnement Python global.
        set RUN_CMD=python -m streamlit run app_dashboard.py
    )
)

:: Si aucun environnement n'est valide
if "%RUN_CMD%"=="" (
    echo [ERROR] Les dependances requises (gymnasium, stable-baselines3, streamlit) ne sont pas installees.
    echo.
    echo Veuillez les installer en executant la commande suivante dans votre terminal :
    echo     pip install gymnasium stable-baselines3 streamlit sumo-rl
    echo.
    pause
    exit /b 1
)

echo.
echo [*] Lancement du Dashboard de Supervision...
echo [INFO] Le navigateur web va s'ouvrir automatiquement sur http://localhost:8501
echo.
%RUN_CMD%


pause

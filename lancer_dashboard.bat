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

:: Activation de l'environnement virtuel
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activation de l'environnement virtuel (.venv)...
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] Environnement virtuel .venv non detecte. Tentative avec le python systeme...
)

echo.
echo [*] Lancement du Dashboard de Supervision Streamlit...
echo [INFO] Le navigateur web va s'ouvrir automatiquement sur http://localhost:8501
echo.
streamlit run app_dashboard.py

pause

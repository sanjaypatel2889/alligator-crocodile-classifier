@echo off
title Tilesview AI — Streamlit Dashboard
set KMP_DUPLICATE_LIB_OK=TRUE
echo ============================================================
echo  Alligator vs Crocodile — Interactive Dashboard
echo  Opening at: http://localhost:8501
echo ============================================================
echo.
streamlit run app.py --server.port 8501
pause

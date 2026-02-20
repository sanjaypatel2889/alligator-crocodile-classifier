@echo off
title Tilesview AI — Step 3: Prediction
echo ============================================================
echo  Alligator vs Crocodile — Prediction
echo ============================================================
echo.
echo Usage options:
echo   Default (40 val images):  just press Enter below
echo   Custom folder:            python predict.py --input path\to\folder
echo   Single image:             python predict.py --input path\to\image.jpg
echo.
python predict.py
echo.
echo [Done] Annotated images saved to predictions\
echo [Done] Summary saved to predictions\prediction_summary.txt
pause

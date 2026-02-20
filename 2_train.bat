@echo off
title Tilesview AI — Step 1: Training
echo ============================================================
echo  Alligator vs Crocodile — Training Pipeline
echo  Models: MobileNetV2  +  ResNet50
echo ============================================================
echo.
python train.py
echo.
echo [Done] Best model saved to models\best_model.pth
echo [Done] Outputs: outputs\confusion_matrix\  outputs\training_plots\
echo [Done] Reports: outputs\model_comparison.txt  conclusion\conclusion.txt
pause

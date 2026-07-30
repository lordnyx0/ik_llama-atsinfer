@echo off
title Qwen 3.6 - ATSInfer CUDA Web Server
echo ========================================================
echo   Iniciando Servidor Web/API llama-server (ATSInfer CUDA)
echo   Acesse em: http://localhost:8080
echo ========================================================

cd /d "C:\Users\Nyx\Desktop\Qwen-AI-Server\ik_llama.cpp\build\bin\Release"

.\llama-server.exe ^
  -m "C:\Users\Nyx\.lmstudio\models\HauhauCS\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf" ^
  --alias qwen36 ^
  --fit ^
  --fit-margin 256 ^
  -rtr ^
  -fa on ^
  -ctk q8_0 ^
  -ctv q8_0 ^
  -c 32000 ^
  --temp 0.7 ^
  --top-k 40 ^
  -t 8 ^
  --no-mmap ^
  --port 8080 ^
  --chat-template-file "C:\Users\Nyx\Desktop\Qwen-AI-Server\qwen_fixed.jinja" ^
  --jinja

pause

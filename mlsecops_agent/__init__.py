"""
MLSecOps Agent — модуль DevSecOps для сканирования агентных сценариев Langflow.

Pipeline:
  1. Static Workflow Analysis — загрузка и парсинг флоу
  2. Attack Surface Modeling — MAESTRO threat modeling
  3. BORAT Scenario Generation — генерация целей атаки
  4. Runtime Attack Execution — атакующий мультиагент BORAT
  5. LLM Oracle Evaluation — оценка судьёй + JSON-отчёт
"""

__version__ = "0.1.0"

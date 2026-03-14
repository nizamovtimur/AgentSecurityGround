# Agentic Radar
[GitHub - splx-ai/agentic-radar: A security scanner for your LLM agentic workflows](https://github.com/splx-ai/agentic-radar)

* анализирует **source code workflow**
* строит **граф взаимодействия агентов и инструментов**
* обнаруживает **структурные уязвимости**
* затем может запускать тесты.

## Workflow graph extraction

Radar строит **граф agent workflow**:

* agents
* tools
* handoffs
* MCP servers
* execution path

Это можно использовать в работе как:

### Graph-based attack surface analysis

Идея:

```
Agent Graph
  nodes = agents/tools
  edges = calls / handoffs
```

На этом графе можно автоматически искать:

* attack entrypoints
* privilege escalation paths
* tool abuse chains

Пример:

```
User → PlannerAgent → WebSearchTool → BrowserAgent → FileSystem
```

такая цепочка сразу показывает:

```
prompt injection → tool invocation → data exfiltration
```

## Tool vulnerability mapping

Radar автоматически:

* находит **используемые tools**
* анализирует по фреймворку моделирования угроз MAESTRO

Пример логики:

```
Tool: WebSearch
Risk: Prompt Injection
Tool: FileSystem
Risk: Data Exfiltration
Tool: Browser
Risk: Remote Code Execution
```

## Что можно написать в related work об Agentic Radar

```
While most existing approaches rely on black-box adversarial prompting,
Agentic Radar introduces architecture-aware analysis of agent workflows.
However, its runtime attack generation capabilities remain limited.

Our framework extends this idea by introducing structured adversarial
scenario generation (BORAT) and systematic evaluation of agent responses.
```

# Модуль AppSec-платформы для сканирования агентных сценариев

```
        ┌────────────────────┐
        │ Static Workflow    │
        │ Analysis           │
        └─────────┬──────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ Attack Surface     │
        │ Modeling           │
        └─────────┬──────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ BORAT Scenario     │
        │ Generation         │
        └─────────┬──────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ Runtime Attack     │
        │ Execution          │
        └─────────┬──────────┘
                  │
                  ▼
        ┌────────────────────┐
        │ LLM Oracle         │
        │ Evaluation         │
        └────────────────────┘
```

Это превращает систему в **complete agent red-teaming pipeline**.

## Модуль 1 — Agent Workflow Extraction

Это самая сильная идея Radar.

Он:

* сканирует source code
* находит agents
* находит tools
* строит **workflow graph**

Пример графа:

```
User
  ↓
PlannerAgent
  ↓
SearchTool
  ↓
BrowserAgent
  ↓
FileSystemTool
```

### Что это дает стенду

* data flow
* tool dependencies
* trust boundaries

Graph model:

```
nodes:
   agent
   tool
   external_api

edges:
   invoke
   delegate
   read
   write
```

## Модуль 2 — Attack Surface Modeling

После построения графа можно вычислять **attack surface**. Обозначать активы по фреймфорку MAESTRO и описывать актуальные угрозы в зависимости от реализации, подсвечивая риски

## Модуль 3 — Automatic Attack Generation

Сейчас BORAT = predefined attack.

Но можно сделать:

```
tool type
   ↓
attack template
```

Например:

### WebSearchTool

```
prompt injection page
malicious HTML
indirect prompt injection
```

### FileSystemTool

```
exfiltration attack
path traversal
sensitive file access
```

### CodeExecutionTool

```
RCE payload
tool abuse
```

Это превращает BORAT в:

**context-aware attack generator**.

## CI/CD Security Testing

**Continuous Agent Red Teaming**

Pipeline:

```
commit
 ↓
workflow scan
 ↓
attack generation
 ↓
runtime tests
 ↓
security report
```

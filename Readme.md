# Investment Analysis Agent

> A FinTech decision-support application that uses a modular multi-agent architecture to analyze equities, assess investor risk, and generate personalized investment insights.

## 1. Project Overview

The **Investment Analysis Agent** is an AI-oriented FinTech application designed to help retail investors interpret multiple dimensions of equity investment analysis in one workflow.

Instead of relying on a single monolithic analysis, the application separates the task into specialized agents for:

- User profiling and risk assessment
- Market analysis
- Fundamental analysis
- News and sentiment analysis
- Strategy/recommendation generation
- Portfolio analysis
- Report generation

The application is intended as a **decision-support and educational system**, not as a replacement for a registered financial advisor.

## 2. Problem Statement

Retail investors often have access to large amounts of financial information but may find it difficult to combine:

- Market trends and technical indicators
- Company fundamentals
- News and sentiment
- Personal risk tolerance
- Investment horizon
- Portfolio constraints

The project addresses this problem by bringing these analytical dimensions into a single modular workflow and producing a consolidated result.

## 3. Key Features

### Investor Profiling
Processes user information and establishes an investor/risk profile that can be used to personalize the analysis.

### Market Analysis
Analyzes market-related information such as price behaviour, trend, momentum and technical indicators supported by the application's data layer.

### Fundamental Analysis
Evaluates company-level financial indicators and produces a structured view of financial health.

### News & Sentiment Analysis
Processes company-related news/context and converts it into a structured sentiment signal where the configured data/AI pipeline is available.

### Risk Analysis
Combines relevant market and fundamental signals to produce a risk-oriented assessment.

### Strategy Generation
Uses the outputs of the analytical components to generate an investment strategy/recommendation rather than relying on one isolated indicator.

### Portfolio Analysis
Uses portfolio-related information to calculate position/allocation-oriented insights.

### Report Generation
Consolidates the analysis into a user-facing report, including the relevant recommendation, risk information and supporting analysis.

## 4. AI / Agentic AI Component

The central AI contribution of the project is the **multi-agent design**.

Each agent has a focused responsibility and produces structured information that can be consumed by other components. This separation makes it possible to improve one analytical capability without redesigning the complete application.

The AI-oriented components include:

1. **User Agent** — user/investor profiling.
2. **Market Agent** — market-related analysis and interpretation.
3. **Fundamentals Agent** — company fundamental analysis.
4. **News Sentiment Agent** — news/context and sentiment processing.
5. **Risk Agent** — risk evaluation from analytical signals.
6. **Strategy Agent** — strategy/recommendation generation.
7. **Portfolio Agent** — portfolio-oriented advice.
8. **Portfolio Analyzer** — portfolio calculations/analysis.
9. **Report Agent** — consolidation and report generation.

The project also contains an LLM service, allowing the application to use an LLM-based component where configured. The news/sentiment and report-generation layers are the natural locations for language-model-assisted reasoning and summarization.

## 5. Architecture

```text
                         ┌─────────────────────┐
                         │       User/UI       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Flask / app.py  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Orchestrator     │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
       ┌────────────┐        ┌────────────┐        ┌────────────┐
       │ User Agent │        │Market Agent│        │Fundamentals│
       └─────┬──────┘        └─────┬──────┘        │   Agent    │
             │                     │               └─────┬──────┘
             └─────────────────────┼─────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
       ┌──────────────┐      ┌────────────┐      ┌──────────────┐
       │News/Sentiment│      │ Risk Agent │      │Strategy Agent│
       │    Agent     │      └─────┬──────┘      └──────┬───────┘
       └──────┬───────┘            │                    │
              └────────────────────┼────────────────────┘
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                  ┌────────────┐      ┌───────────────┐
                  │Portfolio   │      │Portfolio      │
                  │Agent       │      │Analyzer       │
                  └─────┬──────┘      └───────┬───────┘
                        └──────────┬───────────┘
                                   ▼
                         ┌─────────────────────┐
                         │    Report Agent     │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Final Analysis      │
                         │ / Report            │
                         └─────────────────────┘
```

### Supporting Services

The `services/` layer separates external/data/model integrations from the agent logic:

- `llm_service.py` — LLM integration/service layer
- `marketstack_service.py` — MarketStack data integration
- `price_service.py` — Price/data retrieval and processing

The separation between **agents**, **services**, and the **orchestrator** keeps the application modular and easier to maintain.

## 6. Repository Structure

```text
Investment-Analysis-Agent/
│
├── agents/
│   ├── fundamentals_agent.py
│   ├── market_agent.py
│   ├── news_sentiment_agent.py
│   ├── portfolio_agent.py
│   ├── portfolio_analyzer.py
│   ├── report_agent.py
│   ├── risk_agent.py
│   ├── strategy_agent.py
│   └── user_agent.py
│
├── services/
│   ├── llm_service.py
│   ├── marketstack_service.py
│   └── price_service.py
│
├── app.py
├── config.py
├── orchestrator.py
├── requirements.txt
├── test_accuracy.py
├── test_report_*.html
└── users.db
```

### Important GitHub cleanup

Do **not** commit the local Python environment or generated cache files. Add these to `.gitignore`:

```gitignore
venv/
__pycache__/
*.pyc
.env
```

If `users.db` contains real/personal user information, it should also be excluded from the public repository and replaced with a safe sample database.

## 7. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Backend | Flask |
| Cross-Origin Support | Flask-CORS |
| AI/LLM | OpenAI-compatible LLM service layer |
| Market Data | MarketStack / price service |
| Financial Data | Market/price service layer |
| Database | SQLite (`users.db`) |
| Testing | Python test script + generated HTML reports |
| Development | VS Code / Git / GitHub |

## 8. Installation

### Prerequisites

- Python 3.8+
- Git
- Internet connection when external APIs are enabled

### Clone the repository

```bash
git clone https://github.com/srishti2005/Quantam_Advisor
cd <REPOSITORY_FOLDER>
```

### Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 9. Configuration

Keep API keys and other secrets outside the source code.

Create a `.env` file if the implementation/configuration expects environment variables:

```env
OPENAI_API_KEY=your_key_here
MARKETSTACK_API_KEY=your_key_here
```

Use the variable names defined in `config.py` if they differ from the examples above.

**Never commit API keys to GitHub.**

## 10. Running the Application

From the project root:

```bash
python app.py
```

The Flask application should then expose the routes configured in `app.py`.

If the application prints a local URL in the terminal, open that URL in a browser.

## 11. How the Analysis Works

A typical analysis follows this conceptual flow:

```text
User Inputs
     ↓
Investor Profiling
     ↓
Market / Price Analysis
     ↓
Fundamental Analysis
     ↓
News & Sentiment
     ↓
Risk Assessment
     ↓
Strategy Generation
     ↓
Portfolio Analysis
     ↓
Report Generation
     ↓
Final Investment Analysis
```

The orchestrator coordinates the processing rather than placing the complete investment-analysis logic inside a single function.

## 12. Testing

The repository contains:

- `test_accuracy.py` for accuracy/logic validation
- HTML test reports generated during testing

Run the available test script with:

```bash
python test_accuracy.py
```

For additional tests, follow the test cases and expected outputs implemented in the repository.

## 13. Data and Model Limitations

The application should be interpreted according to the data sources and API configuration used during execution.

Important limitations include:

- Market data may depend on external API availability.
- API responses can vary with provider limits and market-data freshness.
- LLM-generated text can contain errors and should not be treated as authoritative financial advice.
- Scoring/rule-based recommendations depend on the formulas implemented in the project.
- The prototype is intended for educational and decision-support purposes.

## 14. Responsible AI / Financial Safety

The application is designed as a **financial decision-support prototype**.

It should not be used as a substitute for:

- A SEBI-registered investment adviser
- Professional financial planning
- Independent investment research
- Verification of current market information

Users should independently verify market data and consider their own financial circumstances before making investment decisions.

## 15. Future Scope

Possible extensions include:

- More reliable real-time market-data integration
- Stronger RAG-based news analysis
- Improved LLM reasoning and report generation
- Full portfolio-level optimization
- Backtesting and recommendation evaluation
- Multilingual investment reports
- More advanced risk models
- Cloud deployment and monitoring
- Improved authentication and data privacy controls

## 16. Academic Context

**CA-2: Coding-Based Assignment — FinTech + AI**

**Project:** Investment Analysis Agent

**Team:**
- Parnika Jain — 23070126087
- Srishti Tripathi — 23070126131

---



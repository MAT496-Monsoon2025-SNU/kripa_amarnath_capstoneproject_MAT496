# kripa_amarnath_capstoneproject_MAT496

# Portfolio Analysis

# Overview

This project is an advanced conversational AI agent designed to act as an "Autonomous Hedge Fund Team." A user interacts with the system, providing a stock ticker (e.g., RELIANCE.NS, TCS.NS). The system orchestrates a team of specialist agents—a Quantitative Analyst (for technical signals) and a Fundamental Researcher (for PDF analysis)—to research the asset in parallel.

Using yfinance for market data and ChromaDB for internal knowledge (PDFs), the agents fetch real-time price action, calculate Entry/Exit zones, and cross-reference financial health. The system then consolidates these findings into a comprehensive "Buy/Sell/Wait" report. The entire application state is managed using LangGraph, featuring persistent memory to track the analysis context.

# Reason for picking up this project

I'm interested in mathematical finance and this project ended up being a good blend of both finance and LLM. This project was selected to synthesise and demonstrate all the major advanced topics covered in the course:

LangGraph (State, Nodes, Graph):
The core architecture is built as a stateful graph using StateGraph, managing complex data flow between the math-heavy Quant agent and the text-heavy Research agent using a custom AgentState.

Tool Calling & RAG:
The agents use custom tools (get_trade_signals) to perform sophisticated Feature Engineering (Bollinger Bands, Support/Resistance) and RAG (Retrieval Augmented Generation) to query unstructured PDF data (Annual Reports).

Persistent Memory:
The graph utilises a MemorySaver checkpointer. This allows the agent to recall previous analysis if the user asks follow-up questions (e.g., "What was the Stop Loss again?").

Multi-Agent Parallelization:
This is the core of the capstone.

Sub-Graphs/Nodes: Specialised nodes (Quant & Research) operate independently.

Parallelisation: The main graph orchestrator executes the Technical Analysis and Fundamental Analysis simultaneously to reduce latency.

Map-Reduce: The system "maps" the ticker to both specialists and "reduces" their independent findings into a single strategic verdict.

# The Plan

[DONE] Step 1: Define State & Graph with Persistent Memory.
Defined AgentState to hold ticker, technical signals, and fundamental summaries. Initialized memory checkpointers.

[DONE] Step 2: Implement Core Tools.
Created placeholder tools returning hardcoded "Buy at 100" signals. This allowed verification of the tool-calling logic without hitting Yahoo Finance API limits during initial testing.

[DONE] Step 3: Create "Quant Agent" Node.
Built the dedicated technical node to handle price logic using mock data.

[DONE] Step 4: Create "Research Agent" Node.
Built the dedicated RAG node for handling PDF queries using mock text.

[DONE] Step 5: Implement Map-Reduce Orchestrator.
Built the Main Graph to coordinate the workflow. Verified that the "Aggregator" node correctly combined the mock results from both branches.

[DONE] Step 6: Integrate Real APIs (yfinance & ChromaDB).
Upgraded Tools: Replaced mock tools with live logic.

Logic: Implemented Pandas calculations for Entry (Support/Lower Band), Exit (Resistance), and Stop Loss (3% risk).

RAG: Implemented PDF ingestion for Reliance/TCS annual reports.

[DONE] Step 7: Deploy & Build Web Interface.
Create a streamlit web app (app.py) that acts as the front end for the LangGraph agent.

[DONE] Step 8: Final Test.
Test the full flow with live tickers to ensure the math and text analysis align in the final report.

# Conclusion
I had planned to achieve the development of "HedgeGraph India," an autonomous multi-agent system capable of performing institutional-grade stock analysis. The primary objective was to solve the problem of information overload by orchestrating two specialized agents—a Quantitative Agent for technical analysis and a Research Agent for fundamental retrieval—using a Map-Reduce architecture.
I think I have achieved the conclusion satisfactorily.



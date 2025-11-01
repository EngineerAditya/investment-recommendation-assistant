"""
FinBuddy - AI-Powered Financial Assistant
A modern LangGraph agent for portfolio recommendations using RAG and live market data.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import yfinance as yf
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Application configuration."""
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_db_path: Path = Path("data/vector_db/chroma_db")
    ollama_model: str = "llama3:8b"  # Changed to match original code
    retriever_k: int = 5
    
    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from JSON file with fallback to defaults."""
        try:
            with open(config_path) as f:
                data = json.load(f)
            
            chroma_path = Path(data.get("output_directory", cls.chroma_db_path))
            if not chroma_path.is_absolute():
                chroma_path = config_path.parent / chroma_path
                
            return cls(
                embedding_model=data.get("embedding_model", cls.embedding_model),
                chroma_db_path=chroma_path,
            )
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {config_path}. Using defaults.")
            return cls()


# ============================================================================
# Tool Definitions
# ============================================================================

def create_financial_tools(retriever, config: Config):
    """Create tool functions with proper decorators."""
    
    @tool
    def get_market_news(query: str) -> str:
        """
        Search the vector database for relevant market news.
        Use this to find news articles about companies, sectors, or market events.
        
        Args:
            query: Search query for market news (e.g., 'tech sector', 'AAPL', 'positive news')
            
        Returns:
            Formatted news articles with headlines, summaries, and metadata
        """
        print(f"🔍 Searching news: '{query}'")
        docs = retriever.invoke(query)
        
        if not docs:
            return "No relevant market news found in the database."
        
        results = []
        for doc in docs:
            meta = doc.metadata
            results.append(
                f"📰 {meta.get('headline', 'N/A')}\n"
                f"Summary: {doc.page_content}\n"
                f"Source: {meta.get('source_name', 'N/A')} | "
                f"Sentiment: {meta.get('sentiment', 'N/A')} | "
                f"Published: {meta.get('published_at', 'N/A')}\n"
                f"URL: {meta.get('url', 'N/A')}"
            )
        
        return "\n\n" + "\n\n".join(results)
    
    @tool
    def get_stock_data(ticker: str) -> str:
        """
        Retrieve live financial data for a stock ticker.
        For Indian stocks, include the .NS suffix (e.g., 'RELIANCE.NS').
        
        Args:
            ticker: Stock ticker symbol (e.g., 'NVDA', 'MSFT', 'RELIANCE.NS')
            
        Returns:
            JSON string with company financials and key metrics
        """
        print(f"📊 Fetching data: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or info.get("regularMarketPrice") is None:
                return f"❌ No data found for '{ticker}'. Verify the ticker symbol."
            
            data = {
                "ticker": ticker,
                "company_name": info.get("longName"),
                "current_price": info.get("regularMarketPrice"),
                "currency": info.get("currency", "USD"),
                "market_cap": info.get("marketCap"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"❌ Error fetching {ticker}: {str(e)}"
    
    @tool
    def create_portfolio(
        investment_amount: float,
        risk_profile: str,
        time_horizon_years: int,
        suggested_tickers: list[str],
    ) -> str:
        """
        Generate a portfolio allocation based on user preferences and selected stocks.
        
        Args:
            investment_amount: Total amount to invest (must be positive)
            risk_profile: Risk tolerance - must be 'low', 'medium', or 'high'
            time_horizon_years: Investment time horizon (positive integer)
            suggested_tickers: List of stock tickers to include (at least one required)
            
        Returns:
            Formatted portfolio allocation with percentages and amounts
        """
        print(f"💼 Creating {risk_profile} risk portfolio")
        
        if not suggested_tickers:
            return "❌ No tickers provided for portfolio construction."
        
        num_stocks = len(suggested_tickers)
        allocation = {}
        
        # Risk-based allocation strategy
        if risk_profile == "low":
            cash_pct = 0.4
            stock_pct = 0.6
            allocation["Cash/Bonds"] = cash_pct
            for ticker in suggested_tickers:
                allocation[ticker] = stock_pct / num_stocks
                
        elif risk_profile == "medium":
            cash_pct = 0.15
            stock_pct = 0.85
            allocation["Cash/Bonds"] = cash_pct
            for ticker in suggested_tickers:
                allocation[ticker] = stock_pct / num_stocks
        else:  # high risk
            for ticker in suggested_tickers:
                allocation[ticker] = 1.0 / num_stocks
        
        # Format output
        lines = [
            f"📊 Portfolio Summary",
            f"Risk Profile: {risk_profile.upper()}",
            f"Total Investment: ${investment_amount:,.2f}",
            f"Time Horizon: {time_horizon_years} years",
            f"\n{'Asset':<20} {'Allocation':<12} {'Amount':<15}",
            "-" * 50,
        ]
        
        for asset, pct in allocation.items():
            amount = investment_amount * pct
            lines.append(f"{asset:<20} {pct*100:>6.1f}%      ${amount:>12,.2f}")
        
        return "\n".join(lines)
    
    return [get_market_news, get_stock_data, create_portfolio]


# ============================================================================
# Agent State
# ============================================================================

class AgentState(TypedDict):
    """State for the FinBuddy agent graph."""
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================================
# Agent Graph
# ============================================================================

def create_agent_graph(llm, tools: list, checkpointer):
    """Create the LangGraph agent workflow."""
    
    # Bind tools to LLM using format_tool_to_openai_function for compatibility
    from langchain_core.utils.function_calling import convert_to_openai_function
    
    functions = [convert_to_openai_function(t) for t in tools]
    llm_with_tools = llm.bind(functions=functions)
    
    def call_model(state: AgentState) -> dict:
        """Call the LLM with current state."""
        print("🤖 LLM processing...")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        """Determine if we should call tools or end."""
        last_message = state["messages"][-1]
        
        # Check for function calls in the response
        if hasattr(last_message, "additional_kwargs"):
            if "function_call" in last_message.additional_kwargs:
                return "tools"
        
        return "__end__"
    
    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", "__end__"])
    workflow.add_edge("tools", "agent")
    
    return workflow.compile(checkpointer=checkpointer)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Run the FinBuddy assistant."""
    
    # Load configuration
    script_dir = Path(__file__).parent
    config_path = script_dir / "embeddings" / "config.json"
    config = Config.from_file(config_path)
    
    # Initialize components
    print("🚀 Initializing FinBuddy...")
    
    try:
        llm = ChatOllama(model=config.ollama_model, temperature=0)
        llm.invoke("test")  # Verify connection
        print(f"✓ Connected to Ollama ({config.ollama_model})")
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print(f"\n💡 Available models: Run 'ollama list' to see installed models")
        print(f"💡 Pull the model: ollama pull {config.ollama_model}")
        print(f"💡 Or edit the config to use an available model")
        return
    
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    print(f"✓ Loaded embeddings ({config.embedding_model})")
    
    try:
        vector_store = Chroma(
            persist_directory=str(config.chroma_db_path),
            embedding_function=embeddings,
        )
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.retriever_k},
        )
        print(f"✓ Loaded vector store ({config.chroma_db_path})")
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        print("Run 'python embeddings/create_index_chroma.py' first.")
        return
    
    # Create tools
    tools = create_financial_tools(retriever, config)
    
    # Create agent
    checkpointer = MemorySaver()
    graph = create_agent_graph(llm, tools, checkpointer)
    
    # Run conversation loop
    print("\n" + "="*60)
    print("💰 FinBuddy - Your AI Investment Assistant")
    print("="*60)
    print("\nHello! I'm FinBuddy. I'll help you create a personalized portfolio.")
    print("To begin, please tell me:")
    print("  • How much you want to invest")
    print("  • Your risk tolerance (low, medium, or high)")
    print("  • Your investment time horizon in years")
    print("\nType 'quit' to exit.\n")
    
    config_dict = {"configurable": {"thread_id": "session_1"}}
    
    system_prompt = """You are FinBuddy, an expert financial assistant helping users build investment portfolios.

**Knowledge Base Scope:**
Your vector database contains ONLY recent market news articles. It does NOT contain educational content about investment basics.

**When to Use Tools:**
- Use `get_market_news` for: current market sentiment, company news, sector trends, recent events
- DO NOT use `get_market_news` for: general investment advice, educational content, investment principles
- For general investment questions, use your own knowledge and explain that your news database focuses on current events

**Portfolio Building Workflow:**
1. **Gather Requirements**: Ask for investment_amount, risk_profile (low/medium/high), and time_horizon_years
2. **General Advice**: Provide investment principles from your knowledge (no tool needed)
3. **Research**: Use get_market_news to understand current sentiment and trends for specific sectors/companies
4. **Analyze**: Use get_stock_data for 3-5 promising tickers based on research and risk profile
5. **Build**: Call create_portfolio with selected tickers
6. **Explain**: Present the portfolio with clear reasoning from both news and financial data

**Important Notes:**
- For Indian stocks, use .NS suffix (e.g., 'RELIANCE.NS')
- This is educational only - not real financial advice
- Be conversational and helpful
- If asked general investment questions, answer from your knowledge - don't search the news database
- Always justify portfolio recommendations with specific data points from tools"""
    
    messages = [HumanMessage(content=system_prompt)]
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thanks for using FinBuddy! Goodbye!")
                break
            
            if not user_input:
                continue
            
            messages.append(HumanMessage(content=user_input))
            
            # Stream agent execution
            for event in graph.stream(
                {"messages": messages},
                config_dict,
                stream_mode="values",
            ):
                last_msg = event["messages"][-1]
                
                if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                    print(f"\nFinBuddy: {last_msg.content}\n")
                    messages = event["messages"]
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
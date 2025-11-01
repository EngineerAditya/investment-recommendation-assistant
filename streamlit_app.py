"""
FinBuddy - AI-Powered Financial Assistant (Streamlit Frontend)
A modern web interface for portfolio recommendations using RAG and live market data.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import streamlit as st
import yfinance as yf
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="FinBuddy - AI Investment Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container background */
    .main {
        background-color: #ffffff;
    }
    
    /* Headers */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #555555;
        margin-bottom: 2rem;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        color: #1a1a1a;
    }
    .user-message b {
        color: #0d47a1;
    }
    .assistant-message {
        background-color: #f1f8f4;
        border-left: 5px solid #4caf50;
        color: #1a1a1a;
    }
    .assistant-message b {
        color: #2e7d32;
    }
    
    /* Markdown inside messages */
    .assistant-message p {
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .assistant-message h1, .assistant-message h2, .assistant-message h3 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .assistant-message ul, .assistant-message ol {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
    }
    .assistant-message li {
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    .assistant-message strong {
        font-weight: 600;
        color: #1a1a1a;
    }
    .assistant-message em {
        font-style: italic;
    }
    .assistant-message blockquote {
        border-left: 3px solid #4caf50;
        padding-left: 1rem;
        margin: 1rem 0;
        color: #555;
    }
    
    /* Horizontal rules */
    .assistant-message hr {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 1.5rem 0;
    }
    
    /* Better spacing for structured content */
    .assistant-message .stMarkdown {
        color: #1a1a1a;
    }
    
    /* Warning/Info boxes styling */
    .assistant-message [data-testid="stAlert"] {
        margin: 1rem 0;
    }
    .tool-message {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        font-size: 0.9rem;
        color: #1a1a1a;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0.4rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #155a8a;
        border: none;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #1a1a1a;
    }
    
    /* Chat input */
    .stChatInput {
        border-color: #1f77b4;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        color: #1a1a1a;
        font-weight: 600;
    }
    
    /* Ensure all text is readable */
    p, span, div, li, label, .stMarkdown {
        color: #1a1a1a !important;
    }
    
    /* Success/Info/Warning messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        color: #1a1a1a !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Headers in content */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: #1a1a1a !important;
    }
    
    /* Code blocks */
    .assistant-message code {
        background-color: #f5f5f5;
        color: #d32f2f;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        font-size: 0.9em;
    }
    
    .assistant-message pre {
        background-color: #f5f5f5;
        color: #1a1a1a;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        overflow-x: auto;
        margin: 1rem 0;
    }
    
    .assistant-message pre code {
        background-color: transparent;
        color: #1a1a1a;
        padding: 0;
    }
    
    /* Tables inside messages */
    .assistant-message table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        background-color: white;
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .assistant-message th {
        background-color: #e8f5e9;
        color: #1a1a1a;
        font-weight: 600;
        padding: 0.75rem;
        text-align: left;
        border-bottom: 2px solid #4caf50;
    }
    
    .assistant-message td {
        color: #1a1a1a;
        padding: 0.75rem;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .assistant-message tr:last-child td {
        border-bottom: none;
    }
    
    .assistant-message tr:hover {
        background-color: #f5f5f5;
    }
    
    /* Links */
    a {
        color: #1f77b4 !important;
        text-decoration: none !important;
    }
    
    a:hover {
        color: #155a8a !important;
        text-decoration: underline !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Application configuration."""
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_db_path: Path = Path("data/vector_db/chroma_db")
    ollama_model: str = "llama3:8b"
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
        st.session_state.tool_calls.append(f"🔍 Searching news: '{query}'")
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
        st.session_state.tool_calls.append(f"📊 Fetching data: {ticker}")
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
        st.session_state.tool_calls.append(f"💼 Creating {risk_profile} risk portfolio")
        
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
    
    from langchain_core.utils.function_calling import convert_to_openai_function
    
    functions = [convert_to_openai_function(t) for t in tools]
    llm_with_tools = llm.bind(functions=functions)
    
    def call_model(state: AgentState) -> dict:
        """Call the LLM with current state."""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        """Determine if we should call tools or end."""
        last_message = state["messages"][-1]
        
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
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are FinBuddy, an expert financial advisor specializing in Indian markets and investment portfolios.

**SCOPE RESTRICTION - CRITICAL:**
You are STRICTLY a financial assistant. You can ONLY help with:
- Investment planning and portfolio construction
- Stock market analysis and recommendations
- Mutual funds, ETFs, bonds, and other financial instruments
- Risk profiling and asset allocation
- Tax-saving investments (80C, ELSS, etc.)
- Market analysis and financial news
- Retirement planning and wealth management

For ANY non-financial query (weather, recipes, general knowledge, coding, etc.), respond EXACTLY with:
"I apologize, but I'm a specialized financial assistant. I can only help with investment planning, portfolio management, and financial advisory. Please ask me about your investment goals, portfolio allocation, or market analysis."

**Knowledge Base Scope:**
- Your vector database contains recent market news articles for current sentiment analysis
- Use your financial expertise for investment principles, strategies, and calculations
- DO NOT search news database for educational content - use your own knowledge

**ENHANCED ADVISORY CAPABILITIES:**

1. **Comprehensive Risk Profiling:**
   - Age-based risk assessment (younger = higher equity tolerance)
   - Income stability analysis (salaried vs business)
   - Financial goals (short-term liquidity vs long-term wealth)
   - Existing liabilities and emergency fund status
   - Family responsibilities and insurance coverage
   - Map risk profiles: Low (conservative), Medium (balanced), High (aggressive)

2. **Indian Investment Landscape:**
   - Asset Classes: Equity (stocks, mutual funds, ELSS), Debt (FDs, bonds, PPF), Gold (ETFs, Sovereign Gold Bonds), Real Estate, Alternative investments
   - Tax-saving instruments: ELSS, PPF, NSC, SSY, NPS, life insurance (80C limit ₹1.5L)
   - Index funds: Nifty 50, Sensex, Nifty Next 50, sectoral indices
   - SIP benefits for rupee cost averaging
   - Always use Indian Rupees (₹) for calculations

3. **Portfolio Construction Framework:**
   
   **Step 1 - Gather Complete Information:**
   - Investment amount (one-time or SIP)
   - Current age and income level
   - Risk tolerance (derived from age, goals, stability)
   - Time horizon (short: <3yr, medium: 3-7yr, long: >7yr)
   - Financial goals (retirement, house, education, wealth creation)
   - Existing portfolio (if any)
   - Tax optimization needs
   
   **Step 2 - Asset Allocation Strategy:**
   
   For **Low Risk** (Conservative):
   - Equity: 30-40% (large-cap funds, Nifty 50 index)
   - Debt: 50-60% (FDs, PPF, debt funds, bonds)
   - Gold: 5-10% (Gold ETFs, SGBs)
   - Emergency fund: 6 months expenses in liquid funds
   
   For **Medium Risk** (Balanced):
   - Equity: 60-70% (mix of large-cap 40%, mid-cap 20%, index funds 10%)
   - Debt: 20-25% (mix of FDs, corporate bonds, debt funds)
   - Gold: 5-10%
   - Emergency fund: 6 months expenses
   
   For **High Risk** (Aggressive):
   - Equity: 80-90% (large-cap 30%, mid-cap 30%, small-cap 20%, sectoral 10%)
   - Debt: 5-10% (minimal safety net)
   - Gold: 5%
   - Higher volatility tolerance
   
   **Step 3 - Use Tools for Research:**
   - Use `get_market_news` to analyze current sentiment for sectors/companies
   - Use `get_stock_data` to evaluate 5-7 specific stocks based on:
     * PE ratio (compare with sector average)
     * Market cap (stability indicator)
     * 52-week performance
     * Dividend yield (for income seekers)
     * Sector fundamentals
   - For Indian stocks, ALWAYS use .NS suffix (e.g., 'RELIANCE.NS', 'TCS.NS', 'INFY.NS')
   
   **Step 4 - Build Portfolio:**
   - Select 3-5 stocks/funds based on research
   - Call `create_portfolio` with appropriate allocation
   - Ensure diversification across sectors
   
   **Step 5 - Detailed Explanation:**
   Provide reasoning covering:
   - Why this allocation suits their risk profile and age
   - How each stock/fund was selected (fundamentals + news sentiment)
   - Expected returns range (conservative estimates)
   - Tax implications (LTCG, STCG rates)
   - Rebalancing frequency (quarterly/annually)
   - Exit strategy and review triggers

4. **Advanced Scenarios Handling:**

   **Rebalancing Existing Portfolios:**
   - Analyze current allocation vs target allocation
   - Identify overweight/underweight positions
   - Suggest sell/buy actions to restore balance
   - Consider tax implications of selling (LTCG after 1 year: 10% above ₹1L)
   - Recommend gradual rebalancing vs immediate changes
   
   **Market Volatility Response:**
   - Market correction (10-20% fall): Opportunity for SIPs, avoid lump sum panic
   - Bear market (>20% fall): Dollar-cost averaging, focus on quality large-caps
   - Bull market peaks: Book partial profits, increase debt allocation
   - Rising interest rates: Favor FDs and debt funds, reduce equity slightly
   - Always emphasize long-term perspective over market timing
   
   **Tax Optimization:**
   - Calculate 80C utilization (ELSS, PPF, insurance up to ₹1.5L)
   - Suggest NPS for additional ₹50K deduction under 80CCD(1B)
   - Explain LTCG vs STCG implications (equity: 10% vs 15%)
   - Consider tax-efficient debt funds vs FDs based on income slab
   - Health insurance deductions under 80D
   
   **Guaranteed Returns + Growth Balance:**
   - Allocate portion to guaranteed instruments (PPF 7.1%, FDs 6-7%, NSC)
   - Remaining to equity for growth potential
   - Example: For "medium risk with guaranteed returns" → 40% PPF/FD, 60% equity
   - Clearly separate guaranteed vs market-linked components

5. **Response Format Standards:**

   Always structure your responses clearly with proper formatting.
   
   **Expected Returns:**
   - Conservative: X-Y% annually
   - Optimistic: Y-Z% annually
   - (Based on historical averages and current market conditions)
   
   **Tax Implications:**
   - 80C savings: ₹[amount]
   - LTCG considerations: [explanation]
   
   **Review & Rebalancing:**
   - Review frequency: [Quarterly/Annually]
   - Rebalance if allocation drifts >10% from target
   
   **Risk Disclaimer:**
   Past performance doesn't guarantee future returns. This is educational guidance, not SEBI-registered financial advice.

6. **Important Calculation Guidelines:**
   - Use realistic return expectations (equity: 10-12%, debt: 6-7%, PPF: 7.1%)
   - Always mention that returns are not guaranteed for market-linked instruments
   - Calculate post-tax returns when relevant
   - For SIPs, emphasize rupee-cost averaging benefits
   - Consider inflation impact for long-term goals

**Tool Usage Guidelines:**
- `get_market_news`: For current events, sector sentiment, company-specific news
- `get_stock_data`: For fundamental analysis (PE, market cap, financials)
- `create_portfolio`: After gathering requirements and completing research
- Never search news database for general investment education

**Critical Reminders:**
- Maintain professional yet conversational tone
- Ask clarifying questions when information is incomplete
- For Indian stocks: ALWAYS append .NS to tickers
- This is educational guidance, not SEBI-registered advice
- Encourage users to consult certified financial planners for personalized advice
- Stay within financial domain - reject all non-financial queries politely
"""


# ============================================================================
# Initialize Session State
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [HumanMessage(content=SYSTEM_PROMPT)]
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "tool_calls" not in st.session_state:
        st.session_state.tool_calls = []
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.graph = None
        st.session_state.config_dict = {"configurable": {"thread_id": "session_1"}}


# ============================================================================
# Initialize Components
# ============================================================================

@st.cache_resource
def initialize_components():
    """Initialize LLM, embeddings, and vector store (cached)."""
    script_dir = Path(__file__).parent
    config_path = script_dir / "embeddings" / "config.json"
    config = Config.from_file(config_path)
    
    try:
        llm = ChatOllama(model=config.ollama_model, temperature=0)
        llm.invoke("test")
    except Exception as e:
        st.error(f"❌ Ollama connection failed: {e}")
        st.info(f"💡 Make sure Ollama is running and model '{config.ollama_model}' is installed")
        st.stop()
    
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    
    try:
        vector_store = Chroma(
            persist_directory=str(config.chroma_db_path),
            embedding_function=embeddings,
        )
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.retriever_k},
        )
    except Exception as e:
        st.error(f"❌ Vector store error: {e}")
        st.info("Run 'python embeddings/create_index_chroma.py' first.")
        st.stop()
    
    tools = create_financial_tools(retriever, config)
    checkpointer = MemorySaver()
    graph = create_agent_graph(llm, tools, checkpointer)
    
    return graph, config


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Run the Streamlit app."""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">💰 FinBuddy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your AI-Powered Investment Assistant</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Quick Start Guide")
        st.markdown("""
        **To get personalized investment advice, tell me:**
        
        1. 💵 **Investment Amount**  
           How much you want to invest
        
        2. 🎯 **Risk Tolerance**  
           Low, Medium, or High
        
        3. ⏰ **Time Horizon**  
           Your investment period in years
        
        4. 🎓 **Financial Goals**  
           Retirement, house, education, etc.
        
        5. 👤 **Personal Details** (optional)  
           Age, income, existing portfolio
        """)
        
        st.divider()
        
        st.header("🛠️ Features")
        st.markdown("""
        - 📰 Live market news analysis
        - 📊 Real-time stock data
        - 💼 Personalized portfolio creation
        - 🇮🇳 Indian market expertise
        - 💡 Tax optimization guidance
        """)
        
        st.divider()
        
        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = [HumanMessage(content=SYSTEM_PROMPT)]
            st.session_state.chat_history = []
            st.session_state.tool_calls = []
            st.rerun()
        
        st.divider()
        st.caption("⚠️ **Disclaimer:** This is educational guidance only, not SEBI-registered financial advice.")
    
    # Initialize components
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing FinBuddy..."):
            st.session_state.graph, config = initialize_components()
            st.session_state.initialized = True
        st.success(f"✓ Connected to Ollama ({config.ollama_model})")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.container():
                st.markdown(f"""
                <div class="chat-message user-message">
                    <b style="color: #0d47a1;">You:</b><br>
                    <span style="color: #1a1a1a;">{msg["content"]}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Render assistant response with proper markdown
            with st.container():
                st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
                st.markdown('<b style="color: #2e7d32;">FinBuddy:</b>', unsafe_allow_html=True)
                # Use native Streamlit markdown for proper rendering
                st.markdown(msg["content"])
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Display tool calls in an expander if any
    if st.session_state.tool_calls:
        with st.expander("🔧 Tool Activity", expanded=False):
            for tool_call in st.session_state.tool_calls:
                st.markdown(f'<div class="tool-message" style="color: #1a1a1a !important;">{tool_call}</div>', 
                           unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask me about investment planning, portfolio recommendations, or market analysis...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        # Clear previous tool calls
        st.session_state.tool_calls = []
        
        # Display user message immediately
        with st.container():
            st.markdown(f"""
            <div class="chat-message user-message">
                <b style="color: #0d47a1;">You:</b><br>
                <span style="color: #1a1a1a;">{user_input}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Process with agent
        with st.spinner("🤔 Thinking..."):
            try:
                for event in st.session_state.graph.stream(
                    {"messages": st.session_state.messages},
                    st.session_state.config_dict,
                    stream_mode="values",
                ):
                    last_msg = event["messages"][-1]
                    
                    if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                        # Add assistant response to history
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": last_msg.content
                        })
                        st.session_state.messages = event["messages"]
                        
                        # Display assistant message with proper markdown rendering
                        with st.container():
                            st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
                            st.markdown('<b style="color: #2e7d32;">FinBuddy:</b>', unsafe_allow_html=True)
                            # Use native Streamlit markdown for proper rendering
                            st.markdown(last_msg.content)
                            st.markdown('</div>', unsafe_allow_html=True)
                        break
            except Exception as e:
                st.error(f"❌ Error: {e}")
        
        # Display tool calls if any
        if st.session_state.tool_calls:
            with st.expander("🔧 Tool Activity", expanded=True):
                for tool_call in st.session_state.tool_calls:
                    st.markdown(f'<div class="tool-message" style="color: #1a1a1a !important;">{tool_call}</div>', 
                               unsafe_allow_html=True)
        
        st.rerun()


if __name__ == "__main__":
    main()

"""
Main Streamlit dashboard for the Swing Trade Stock Screener
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.config import config
from src.analysis.eligibility_scanner import EligibilityScanner
from src.data.stock_loader import StockLoader
from src.data.monitor import ContinuousMonitor, AlertManager

# Page configuration
st.set_page_config(
    page_title="Swing Trade Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .eligible-row {
        background-color: #d4edda !important;
        border-left: 4px solid #28a745 !important;
    }
    .error-row {
        background-color: #f8d7da !important;
        border-left: 4px solid #dc3545 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">Swing Trade Stock Screener</h1>', unsafe_allow_html=True)
    st.markdown("**Fibonacci Analysis for Nifty 50 Stocks**")
    
    # Initialize session state
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = []
    if 'scan_timestamp' not in st.session_state:
        st.session_state.scan_timestamp = None
    if 'monitor' not in st.session_state:
        st.session_state.monitor = None
    if 'alert_manager' not in st.session_state:
        st.session_state.alert_manager = AlertManager()
    if 'monitoring_active' not in st.session_state:
        st.session_state.monitoring_active = False
    if 'swing_high_index' not in st.session_state:
        st.session_state.swing_high_index = -2
    if 'swing_low_index' not in st.session_state:
        st.session_state.swing_low_index = -2
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        
        # Timeframe selector
        timeframe_options = {
            "5m": "5m",
            "15m": "15m", 
            "30m": "30m",
            "1h": "1h",
            "1d": "1d"
        }
        selected_timeframe = st.selectbox("Timeframe", list(timeframe_options.keys()), index=1)
        
        # Update config with selected timeframe
        config.data.data_interval = timeframe_options[selected_timeframe]
        
        # Swing point selection
        st.subheader("Swing Point Selection")
        
        # Time interval selection (15 to 375 minutes in 15-minute steps)
        time_interval_minutes = st.slider(
            "Time Interval (minutes)",
            min_value=15,
            max_value=375,
            value=30,
            step=15,
            help="Look back this many minutes from effective analysis time to find swing points"
        )
        
        # Store in session state
        st.session_state.time_interval_minutes = time_interval_minutes
        
        # Market hours information
        st.subheader("Market Hours")
        from src.utils.market_hours import create_market_hours_manager
        market_manager = create_market_hours_manager(config)
        market_info = market_manager.get_market_hours_info()
        
        if market_info['is_market_open']:
            st.success(f"🟢 Market is OPEN (9:15 AM - 3:30 PM)")
        else:
            st.warning(f"🔴 Market is CLOSED (9:15 AM - 3:30 PM)")
            st.info(f"Using last market close: {market_info['last_market_close'].strftime('%Y-%m-%d %H:%M')}")
        
        st.caption(f"Market Days: {', '.join(market_info['market_days'])}")
        
        # Eligibility parameters
        st.subheader("Eligibility Criteria")
        min_level = st.slider("Min Fibonacci Level", 0.0, 1.0, config.analysis.eligibility_min_level, 0.01)
        max_level = st.slider("Max Fibonacci Level", 0.0, 1.0, config.analysis.eligibility_max_level, 0.01)
        
        # Data source
        st.subheader("Data Source")
        use_default_stocks = st.checkbox("Use Default Nifty 50", value=True)
        
        if not use_default_stocks:
            uploaded_file = st.file_uploader("Upload Stock List", type=['txt', 'csv'])
        else:
            uploaded_file = None
        
        # Continuous monitoring
        st.subheader("Continuous Monitoring")
        enable_monitoring = st.checkbox("Enable Real-time Monitoring", value=False)
        
        if enable_monitoring:
            monitor_interval = st.slider("Update Interval (seconds)", 10, 300, 30)
            
            # Kite API Configuration
            st.subheader("Kite API Configuration")
            kite_api_key = st.text_input("Kite API Key", type="password", help="Enter your Kite Connect API key")
            kite_access_token = st.text_input("Kite Access Token", type="password", help="Enter your Kite Connect access token")
            
            if kite_api_key and kite_access_token:
                # Update config with API credentials
                config.api.kite_api_key = kite_api_key
                config.api.kite_access_token = kite_access_token
                config.api.kite_connect_enabled = True
                config.api.yahoo_finance_enabled = False
    
    # Main content
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.header("Stock Analysis")
    
    with col2:
        if st.button("Scan Stocks", type="primary"):
            scan_stocks()
    
    with col3:
        if st.button("Export Results"):
            export_results()
    
    # Monitoring controls
    if enable_monitoring and kite_api_key and kite_access_token:
        col4, col5 = st.columns(2)
        
        with col4:
            if not st.session_state.monitoring_active:
                if st.button("Start Monitoring", type="primary"):
                    start_monitoring()
            else:
                if st.button("Stop Monitoring"):
                    stop_monitoring()
        
        with col5:
            if st.session_state.monitoring_active:
                stats = st.session_state.monitor.get_monitoring_stats()
                st.metric("Scans", stats['scan_count'])
                st.metric("Alerts", stats['total_alerts'])
    
    # Display results
    if st.session_state.monitoring_active:
        display_monitoring_results()
    elif st.session_state.scan_results:
        display_results()
    else:
        st.info("Click 'Scan Stocks' to analyze stocks for swing trade opportunities")
    
    # Footer
    st.markdown("---")
    st.markdown("**Swing Trade Stock Screener** - Powered by Fibonacci Analysis")

def scan_stocks():
    """Scan stocks for swing trade eligibility"""
    
    with st.spinner("Scanning stocks..."):
        try:
            # Update configuration
            config.analysis.lookback_period = st.session_state.get('lookback_period', config.analysis.lookback_period)
            config.analysis.eligibility_min_level = st.session_state.get('min_level', config.analysis.eligibility_min_level)
            config.analysis.eligibility_max_level = st.session_state.get('max_level', config.analysis.eligibility_max_level)
            
            # Get time interval selection
            time_interval_minutes = st.session_state.get('time_interval_minutes', 30)
            
            # Initialize scanner
            scanner = EligibilityScanner(config)
            
            # Determine stock list
            if st.session_state.get('use_default_stocks', True):
                # Use default Nifty 50
                file_path = config.data.default_stock_list
                results = scanner.scan_stocks_from_file_with_time_interval(file_path, time_interval_minutes)
            else:
                # Use uploaded file
                uploaded_file = st.session_state.get('uploaded_file')
                if uploaded_file:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    results = scanner.scan_stocks_from_file_with_time_interval(temp_path, time_interval_minutes)
                    
                    # Clean up temp file
                    os.remove(temp_path)
                else:
                    st.error("Please upload a stock list file")
                    return
            
            # Store results
            st.session_state.scan_results = results
            st.session_state.scan_timestamp = datetime.now()
            
            st.success(f"Scanned {len(results)} stocks")
            
        except Exception as e:
            st.error(f"Error scanning stocks: {e}")

def display_results():
    """Display scan results"""
    
    results = st.session_state.scan_results
    
    if not results:
        st.warning("No results to display")
        return
    
    # Summary metrics
    display_summary_metrics(results)
    
    # Swing point information
    display_swing_point_info(results)
    
    # Trend analysis information
    display_trend_analysis_info(results)
    
    # Results table
    st.subheader("Analysis Results")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        st.warning("No data to display")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_eligible_only = st.checkbox("Show Eligible Only", value=False)
    
    with col2:
        show_errors = st.checkbox("Show Errors", value=True)
    
    with col3:
        trend_filter = st.selectbox("Filter by Trend", ["All", "uptrend", "downtrend", "sideways"])
    
    # Apply filters
    filtered_df = df.copy()
    
    if show_eligible_only:
        filtered_df = filtered_df[filtered_df['eligible'] == True]
    
    if not show_errors:
        filtered_df = filtered_df[filtered_df['error'].isna()]
    
    if trend_filter != "All":
        filtered_df = filtered_df[filtered_df['trend'] == trend_filter]
    
    # Display table
    if not filtered_df.empty:
        # Select display columns
        display_columns = ['symbol', 'current_price', 'swing_high', 'swing_low', 
                          'swing_high_datetime', 'swing_low_datetime', 'trend', 'trend_confidence', 'fib_1618', 'fib_0786', 'fib_0500', 'eligible']
        
        # Filter to available columns
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        display_df = filtered_df[available_columns].copy()
        
        # Rename columns
        column_names = {
            'symbol': 'Symbol',
            'current_price': 'Current Price',
            'swing_high': 'Swing High',
            'swing_low': 'Swing Low',
            'swing_high_datetime': 'Swing High Date/Time',
            'swing_low_datetime': 'Swing Low Date/Time',
            'trend': 'Trend',
            'trend_confidence': 'Trend Confidence',
            'fib_1618': '1.618 Level',
            'fib_0786': '0.786 Level',
            'fib_0500': '0.5 Level',
            'eligible': 'Eligible'
        }
        
        display_df = display_df.rename(columns=column_names)
        
        # Format numbers
        numeric_columns = ['Current Price', 'Swing High', 'Swing Low', '1.618 Level', '0.786 Level', '0.5 Level']
        for col in numeric_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        # Format trend confidence as percentage
        if 'Trend Confidence' in display_df.columns:
            display_df['Trend Confidence'] = (display_df['Trend Confidence'] * 100).round(1).astype(str) + '%'
        
        # Display table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Display count
        st.info(f"Showing {len(display_df)} stocks")
    else:
        st.warning("No stocks match the current filters")

def display_summary_metrics(results):
    """Display summary metrics"""
    
    if not results:
        return
    
    # Calculate metrics
    total_stocks = len(results)
    eligible_stocks = len([r for r in results if r.get('eligible', False)])
    error_stocks = len([r for r in results if r.get('error') is not None])
    eligibility_rate = (eligible_stocks / total_stocks) * 100 if total_stocks > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Stocks", total_stocks)
    
    with col2:
        st.metric("Eligible Stocks", eligible_stocks, f"{eligibility_rate:.1f}%")
    
    with col3:
        st.metric("Error Stocks", error_stocks)
    
    with col4:
        if st.session_state.scan_timestamp:
            st.metric("Last Scan", st.session_state.scan_timestamp.strftime("%H:%M:%S"))

def display_swing_point_info(results):
    """Display swing point selection information"""
    
    if not results:
        return
    
    st.subheader("Swing Point Selection")
    
    # Get current swing point selection
    swing_high_index = st.session_state.get('swing_high_index', -2)
    swing_low_index = st.session_state.get('swing_low_index', -2)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Swing High Index:** {swing_high_index}")
        if swing_high_index == -1:
            st.caption("Using: Last swing high")
        elif swing_high_index == -2:
            st.caption("Using: Second last swing high")
        elif swing_high_index == -3:
            st.caption("Using: Third last swing high")
        else:
            st.caption(f"Using: {abs(swing_high_index)}th last swing high")
    
    with col2:
        st.info(f"**Swing Low Index:** {swing_low_index}")
        if swing_low_index == -1:
            st.caption("Using: Last swing low")
        elif swing_low_index == -2:
            st.caption("Using: Second last swing low")
        elif swing_low_index == -3:
            st.caption("Using: Third last swing low")
        else:
            st.caption(f"Using: {abs(swing_low_index)}th last swing low")
    
    with col3:
        # Show statistics about available swing points
        valid_results = [r for r in results if not r.get('error')]
        if valid_results:
            avg_highs = sum(r.get('available_swing_highs', 0) for r in valid_results) / len(valid_results)
            avg_lows = sum(r.get('available_swing_lows', 0) for r in valid_results) / len(valid_results)
            
            st.metric("Avg Swing Highs", f"{avg_highs:.1f}")
            st.metric("Avg Swing Lows", f"{avg_lows:.1f}")
    
    # Show examples of swing points for a few stocks
    st.subheader("Swing Point Examples")
    
    # Get a few examples
    examples = [r for r in results if not r.get('error') and r.get('available_swing_highs', 0) > 0][:3]
    
    if examples:
        for result in examples:
            symbol = result.get('symbol', 'Unknown')
            swing_high = result.get('swing_high', 0)
            swing_low = result.get('swing_low', 0)
            available_highs = result.get('available_swing_highs', 0)
            available_lows = result.get('available_swing_lows', 0)
            
            st.write(f"**{symbol}**: Swing High: {swing_high:.2f}, Swing Low: {swing_low:.2f} (Available: {available_highs} highs, {available_lows} lows)")
    else:
        st.warning("No valid swing point data available")
    
    # Trend distribution
    if results:
        trends = [r.get('trend', 'unknown') for r in results if r.get('trend')]
        if trends:
            st.subheader("Trend Distribution")
            
            trend_counts = pd.Series(trends).value_counts()
            
            # Create pie chart
            fig = px.pie(
                values=trend_counts.values,
                names=trend_counts.index,
                title="Trend Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

def display_trend_analysis_info(results):
    """Display trend analysis information"""
    
    if not results:
        return
    
    st.subheader("Trend Analysis")
    
    # Get trend distribution with confidence
    trend_data = []
    for result in results:
        if not result.get('error'):
            trend_data.append({
                'trend': result.get('trend', 'unknown'),
                'confidence': result.get('trend_confidence', 0),
                'explanation': result.get('trend_explanation', ''),
                'symbol': result.get('symbol', 'Unknown')
            })
    
    if not trend_data:
        st.warning("No valid trend data available")
        return
    
    # Display trend summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Trend distribution
        trend_counts = pd.Series([d['trend'] for d in trend_data]).value_counts()
        st.write("**Trend Distribution:**")
        for trend, count in trend_counts.items():
            st.write(f"- {trend.title()}: {count} stocks")
    
    with col2:
        # Average confidence by trend
        trend_confidences = {}
        for data in trend_data:
            trend = data['trend']
            if trend not in trend_confidences:
                trend_confidences[trend] = []
            trend_confidences[trend].append(data['confidence'])
        
        st.write("**Average Confidence:**")
        for trend, confidences in trend_confidences.items():
            avg_conf = sum(confidences) / len(confidences)
            st.write(f"- {trend.title()}: {avg_conf:.1%}")
    
    with col3:
        # High confidence trends
        high_conf_stocks = [d for d in trend_data if d['confidence'] > 0.7]
        st.write("**High Confidence Trends:**")
        st.write(f"{len(high_conf_stocks)} stocks with >70% confidence")
    
    # Show detailed trend explanations for a few stocks
    st.subheader("Detailed Trend Analysis")
    
    # Get a few examples with explanations
    examples = [d for d in trend_data if d['explanation']][:3]
    
    if examples:
        for i, example in enumerate(examples, 1):
            with st.expander(f"{i}. {example['symbol']} - {example['trend'].title()} (Confidence: {example['confidence']:.1%})"):
                st.write(example['explanation'])
    else:
        st.info("No detailed trend explanations available")

def start_monitoring():
    """Start continuous monitoring"""
    try:
        # Get symbols to monitor
        if st.session_state.get('use_default_stocks', True):
            symbols = config.get_stock_symbols()
        else:
            uploaded_file = st.session_state.get('uploaded_file')
            if uploaded_file:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                loader = StockLoader()
                symbols = loader.load_stocks(temp_path)
                symbols = loader.validate_symbols(symbols)
                
                # Clean up temp file
                os.remove(temp_path)
            else:
                st.error("Please upload a stock list file")
                return
        
        # Create monitor
        monitor_interval = st.session_state.get('monitor_interval', 30)
        monitor = ContinuousMonitor(config, symbols, monitor_interval)
        
        # Add alert callback
        from src.data.monitor import create_alert_callback
        alert_callback = create_alert_callback(st.session_state.alert_manager)
        monitor.add_callback(alert_callback)
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Update session state
        st.session_state.monitor = monitor
        st.session_state.monitoring_active = True
        
        st.success(f"Started monitoring {len(symbols)} symbols")
        
    except Exception as e:
        st.error(f"Error starting monitoring: {e}")

def stop_monitoring():
    """Stop continuous monitoring"""
    try:
        if st.session_state.monitor:
            st.session_state.monitor.stop_monitoring()
            st.session_state.monitoring_active = False
            st.success("Monitoring stopped")
        else:
            st.warning("No active monitoring to stop")
    except Exception as e:
        st.error(f"Error stopping monitoring: {e}")

def display_monitoring_results():
    """Display real-time monitoring results"""
    if not st.session_state.monitoring_active or not st.session_state.monitor:
        return
    
    # Get latest results
    latest_results = st.session_state.monitor.get_latest_results()
    eligible_stocks = st.session_state.monitor.get_eligible_stocks()
    
    if not latest_results:
        st.info("Monitoring in progress...")
        return
    
    # Display monitoring status
    st.subheader("Real-time Monitoring")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Scanned", len(latest_results))
    
    with col2:
        st.metric("Eligible Now", len(eligible_stocks))
    
    with col3:
        stats = st.session_state.monitor.get_monitoring_stats()
        st.metric("Total Scans", stats['scan_count'])
    
    with col4:
        st.metric("Total Alerts", stats['total_alerts'])
    
    # Display eligible stocks
    if eligible_stocks:
        st.subheader("Current Opportunities")
        
        df = pd.DataFrame(eligible_stocks)
        display_columns = ['symbol', 'current_price', 'swing_high', 'swing_low', 'swing_high_index', 'swing_low_index', 
                          'swing_high_datetime', 'swing_low_datetime', 'trend', 'fib_1618', 'fib_0786', 'fib_0500']
        
        available_columns = [col for col in display_columns if col in df.columns]
        display_df = df[available_columns].copy()
        
        # Rename columns
        column_names = {
            'symbol': 'Symbol',
            'current_price': 'Current Price',
            'swing_high': 'Swing High',
            'swing_low': 'Swing Low',
            'swing_high_index': 'Swing High Index',
            'swing_low_index': 'Swing Low Index',
            'swing_high_datetime': 'Swing High Date/Time',
            'swing_low_datetime': 'Swing Low Date/Time',
            'trend': 'Trend',
            'fib_1618': '1.618 Level',
            'fib_0786': '0.786 Level',
            'fib_0500': '0.5 Level'
        }
        
        display_df = display_df.rename(columns=column_names)
        
        # Format numbers
        numeric_columns = ['Current Price', 'Swing High', 'Swing Low', '1.618 Level', '0.786 Level', '0.5 Level']
        for col in numeric_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Display recent alerts
    recent_alerts = st.session_state.alert_manager.get_recent_alerts(hours=1)
    if recent_alerts:
        st.subheader("Recent Alerts (Last Hour)")
        
        alert_df = pd.DataFrame(recent_alerts)
        alert_columns = ['symbol', 'timestamp', 'current_price', 'trend']
        available_alert_columns = [col for col in alert_columns if col in alert_df.columns]
        
        if available_alert_columns:
            alert_display_df = alert_df[available_alert_columns].copy()
            alert_display_df = alert_display_df.rename(columns={
                'symbol': 'Symbol',
                'timestamp': 'Time',
                'current_price': 'Price',
                'trend': 'Trend'
            })
            
            st.dataframe(alert_display_df, use_container_width=True, hide_index=True)

def export_results():
    """Export results to CSV"""
    
    if not st.session_state.scan_results:
        st.warning("No results to export")
        return
    
    try:
        # Create export directory
        export_dir = "exports"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"swing_trade_results_{timestamp}.csv"
        filepath = os.path.join(export_dir, filename)
        
        # Create DataFrame
        df = pd.DataFrame(st.session_state.scan_results)
        
        # Select columns for export
        export_columns = ['symbol', 'current_price', 'swing_high', 'swing_low', 'trend', 
                         'fib_1618', 'fib_0786', 'fib_0500', 'eligible', 'error']
        
        # Filter to available columns
        available_columns = [col for col in export_columns if col in df.columns]
        export_df = df[available_columns].copy()
        
        # Export to CSV
        export_df.to_csv(filepath, index=False)
        
        st.success(f"Results exported to {filepath}")
        
        # Provide download link
        with open(filepath, "rb") as f:
            st.download_button(
                label="Download CSV",
                data=f.read(),
                file_name=filename,
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error exporting results: {e}")

if __name__ == "__main__":
    main()

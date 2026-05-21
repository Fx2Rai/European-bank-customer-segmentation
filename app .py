import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE & THEME CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Eurozone Retail Banking Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size:30px !important; font-weight: bold; color: #1e3a8a; margin-bottom: 2px; }
    .sub-header { font-size:15px !important; color: #475569; margin-bottom: 20px; }
    .insight-box { background-color: #f8fafc; border: 1px solid #e2e8f0; border-left: 5px solid #1e3a8a; padding: 18px; border-radius: 6px; margin-top: 10px; }
    .critical-box { background-color: #fff5f5; border: 1px solid #fed7d7; border-left: 5px solid #e53e3e; padding: 18px; border-radius: 6px; margin-top: 10px; }
    .gov-box { background-color: #f0fdfa; border: 1px solid #ccfbf1; border-left: 5px solid #0d9488; padding: 20px; border-radius: 6px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA ENGINE WITH EXPLICIT MANDATED SEGMENTATION (STEPS 1-3)
# -----------------------------------------------------------------------------
@st.cache_data
def generate_aligned_portfolio_data():
    np.random.seed(42)
    n_samples = 10000
    
    customer_ids = 15600000 + np.random.permutation(n_samples)
    surnames = np.random.choice(['Smith', 'Martin', 'Garcia', 'Müller', 'Dupont', 'Laval', 'Schmidt', 'Romero'], n_samples)
    credit_scores = np.clip(np.random.normal(655, 90, n_samples).astype(int), 350, 850)
    
    geography = np.random.choice(['France', 'Spain', 'Germany'], n_samples, p=[0.5014, 0.2477, 0.2509])
    gender = np.random.choice(['Female', 'Male'], n_samples, p=[0.45, 0.55])
    age = np.clip(np.random.normal(38.9, 10.5, n_samples).astype(int), 18, 92)
    tenure = np.random.randint(0, 11, n_samples)
    balance = np.where(np.random.rand(n_samples) > 0.36, np.random.uniform(15000, 210000, n_samples), 0.0)
    num_products = np.random.choice([1, 2, 3, 4], n_samples, p=[0.5084, 0.4590, 0.0266, 0.0060])
    has_cr_card = np.random.choice([1, 0], n_samples, p=[0.7055, 0.2945])
    is_active_member = np.random.choice([1, 0], n_samples, p=[0.5151, 0.4849])
    estimated_salary = np.random.uniform(11000, 199000, n_samples)
    
    df = pd.DataFrame({
        'CustomerId': customer_ids, 'Surname': surnames, 'CreditScore': credit_scores,
        'Geography': geography, 'Gender': gender, 'Age': age, 'Tenure': tenure,
        'Balance': balance, 'NumOfProducts': num_products, 'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member, 'EstimatedSalary': estimated_salary
    })
    
    # Mathematical churn probability simulator matching empirical papers
    churn_prob = np.zeros(n_samples)
    for i in range(n_samples):
        p = 0.10
        if df.iloc[i]['Geography'] == 'Germany': p += 0.18
        if 46 <= df.iloc[i]['Age'] <= 60: p += 0.28
        if df.iloc[i]['NumOfProducts'] >= 3: p += 0.70
        if df.iloc[i]['IsActiveMember'] == 0: p += 0.14
        churn_prob[i] = np.clip(p, 0.0, 1.0)
        
    df['Exited'] = (np.random.rand(n_samples) < churn_prob).astype(int)
    
    # --- MANDATED METHODOLOGY STEP 3: EXPLICIT CATEGORICAL BINNING ---
    df['Age_Segment'] = pd.cut(df['Age'], bins=[0, 29, 45, 60, 120], labels=['<30', '30–45', '46–60', '60+'])
    df['Credit_Band'] = pd.cut(df['CreditScore'], bins=[349, 579, 669, 851], labels=['Low Risk', 'Medium Risk', 'High Risk'])
    df['Tenure_Group'] = pd.cut(df['Tenure'], bins=[-1, 2, 7, 11], labels=['New', 'Mid-term', 'Long-term'])
    df['Balance_Segment'] = df['Balance'].apply(lambda x: 'Zero-balance' if x == 0 else ('Low-balance' if x <= 100000 else 'High-balance'))
    
    return df

df_clean = generate_aligned_portfolio_data()

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROL WORKSPACE
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🏛️ **ECB Framework**")
st.sidebar.markdown("<h3 style='color:#1e3a8a;'>Portfolio Control Center</h3>", unsafe_allow_html=True)
st.sidebar.markdown("---")

geo_filter = st.sidebar.multiselect("Sovereign Territory", options=['All', 'France', 'Germany', 'Spain'], default=['All'])
df_geo = df_clean if 'All' in geo_filter or len(geo_filter) == 0 else df_clean[df_clean['Geography'].isin(geo_filter)]

gender_filter = st.sidebar.radio("Gender Segment", options=['All', 'Female', 'Male'])
df_gender = df_geo if gender_filter == 'All' else df_geo[df_geo['Gender'] == gender_filter]

activity_filter = st.sidebar.selectbox("Engagement Status", options=['All Active/Inactive', 'Active Members Only', 'Inactive Members Only'])
if activity_filter == 'Active Members Only':
    df_final = df_gender[df_gender['IsActiveMember'] == 1]
elif activity_filter == 'Inactive Members Only':
    df_final = df_gender[df_gender['IsActiveMember'] == 0]
else:
    df_final = df_gender

st.sidebar.markdown("---")
# Toggle to switch into the required Sovereign/Government Stakeholder Briefing View
view_mode = st.sidebar.radio("Dashboard View Mode", options=["Operational Analytics Deck", "Government Stakeholder Summary"])

# -----------------------------------------------------------------------------
# 4. APP BRANDING HEADERS
# -----------------------------------------------------------------------------
st.markdown("<div class='main-header'>Eurozone Retail Portfolio Supervision Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predictive Churn Engine & High-Value Asset Capital Preservation Dashboard</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. DYNAMIC NATIVE KEY PERFORMANCE INDICATORS (KPIs)
# -----------------------------------------------------------------------------
total_customers = len(df_final)
overall_churn_rate = (df_final['Exited'].sum() / total_customers * 100) if total_customers > 0 else 0

high_value_df = df_final[df_final['Balance_Segment'] == 'High-balance']
high_value_churn = (high_value_df['Exited'].sum() / len(high_value_df) * 100) if len(high_value_df) > 0 else 0

if total_customers > 0 and df_final['Geography'].nunique() > 0:
    geo_churn_series = df_final.groupby('Geography')['Exited'].mean()
    highest_risk_country = geo_churn_series.idxmax()
    highest_risk_churn_rate = geo_churn_series.max() * 100
else:
    highest_risk_country, highest_risk_churn_rate = "None", 0.0

inactive_churners = len(df_final[(df_final['IsActiveMember'] == 0) & (df_final['Exited'] == 1)])
active_churners = len(df_final[(df_final['IsActiveMember'] == 1) & (df_final['Exited'] == 1)])
engagement_drop_idx = (inactive_churners / (active_churners + inactive_churners) * 100) if (active_churners + inactive_churners) > 0 else 0

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Portfolio Base", f"{total_customers:,}", "Active Rows")
kpi2.metric("Overall Churn Rate", f"{overall_churn_rate:.2f}%", "Portfolio Avg")
kpi3.metric("High-Value Churn", f"{high_value_churn:.2f}%", "Balances > €100K")
kpi4.metric("Geographic Risk Index", f"{highest_risk_churn_rate:.2f}%", f"{highest_risk_country} Hotspot")
kpi5.metric("Engagement Drop Indicator", f"{engagement_drop_idx:.1f}%", "Inactive Share")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. CONDITIONAL VIEW LOGIC (OPERATIONAL VS GOVERNMENT BRIEFING)
# -----------------------------------------------------------------------------
if view_mode == "Government Stakeholder Summary":
    st.markdown("<div class='gov-box'>", unsafe_allow_html=True)
    st.markdown("## 🏛️ Executive Summary for Government & Central Bank Stakeholders")
    st.markdown(f"""
    **Macro Supervision Briefing Framework:** Under Basel III risk governance, structural liquidity outflows within systemic commercial entities pose sovereign risk. This summary provides policy directives derived from macro portfolio analytics.
    
    ### 📊 Key Macro Findings:
    1. **Systemic Attrition Corridors:** Portfolio attrition is aggressively concentrated within the **46–60 age demographic**, exposing institutional wealth transfers out of traditional banking infrastructures.
    2. **Geographic Divergence Index:** The territory of **{highest_risk_country}** exhibits acute financial stress profiles with peak out-of-system asset movement hitting **{highest_risk_churn_rate:.2f}%**.
    3. **The Cross-Product Trap:** Retail banking distribution structures that enforce product bundling beyond two accounts hit a systemic break point, accelerating risk vectors exponentially rather than securing loyalty.
    
    ### 📜 Recommended Legislative & Regulatory Directives:
    * **Targeted Capital Conservation Buffers:** Mandate increased localized liquidity coverage allocations for branches operating within high-attrition regions.
    * **Product Governance Auditing:** Require retail institutions to deploy mandatory customer health index tracking protocols prior to executing secondary and tertiary automated cross-sell events.
    * **Demographic Retention Incentives:** Introduce state-aligned premium yield structures targeting long-term capital protection for middle-to-senior age cohorts to protect domestic deposit bases.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    tab_summary, tab_geo, tab_demographic, tab_premium = st.tabs([
        "📊 Overall Churn Summary", "🌍 Geography-Wise Churn", 
        "📈 Mandatory Segment Dimensions", "💎 High-Value Customer Explorer"
    ])

    # --- MODULE 1: OVERALL CHURN SUMMARY ---
    with tab_summary:
        m1, m2, m3 = st.columns([1.5, 1.2, 1.3])
        with m1:
            prod_churn = df_final.groupby('NumOfProducts')['Exited'].mean().reset_index()
            prod_churn['Exited'] *= 100
            fig_prod = px.bar(
                prod_churn, x='NumOfProducts', y='Exited',
                labels={'NumOfProducts': 'Products Held', 'Exited': 'Churn Rate (%)'},
                title='<b>The Cross-Selling Paradox Matrix</b>',
                color='Exited', color_continuous_scale='RdBu_r'
            )
            fig_prod.update_layout(xaxis=dict(tickmode='array', tickvals=[1,2,3,4]), coloraxis_showscale=False)
            st.plotly_chart(fig_prod, use_container_width=True)
        with m2:
            # MANDATED STEP 5: GENDER-BASED COMPARATIVE VISUALIZATION
            gender_churn = df_final.groupby('Gender')['Exited'].mean().reset_index()
            gender_churn['Exited'] *= 100
            fig_gender = px.bar(
                gender_churn, x='Gender', y='Exited', color='Gender',
                title='<b>Gender Attrition Variance</b>',
                labels={'Exited': 'Churn Rate (%)'},
                color_discrete_map={'Female': '#e53e3e', 'Male': '#1e3a8a'}
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        with m3:
            if len(df_final) > 0 and df_final['NumOfProducts'].nunique() > 0:
                prod_lookup = df_final.groupby('NumOfProducts')['Exited'].mean() * 100
                best_prod_node = prod_lookup.idxmin()
                best_prod_rate = prod_lookup.min()
            else:
                best_prod_node, best_prod_rate = 0, 0
                
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 📋 Macro Distribution Insights")
            st.markdown(f"""
            * **The Cross-Sell Cap:** Standard cross-selling assumptions state that bundling increases loyalty. However, the portfolio data exposes a critical threshold at 3 and 4 products, where churn jumps exponentially.
            * **Optimal Node:** Holding exactly **{best_prod_node} products** results in the lowest structural attrition rate (**{best_prod_rate:.2f}%**). 
            * **Policy Action:** Enforce automated flags on retail account managers to audit user satisfaction before cross-selling a 3rd asset line.
            """)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- MODULE 2: GEOGRAPHY-WISE CHURN ---
    with tab_geo:
        g1, g2 = st.columns([2, 1])
        with g1:
            geo_stats = df_final.groupby('Geography').agg(
                Total_Accounts=('CustomerId', 'count'),
                Churn_Rate=('Exited', lambda x: x.mean() * 100)
            ).reset_index()
            fig_geo_bar = go.Figure()
            fig_geo_bar.add_trace(go.Bar(x=geo_stats['Geography'], y=geo_stats['Total_Accounts'], name='Total Accounts', yaxis='y1', marker_color='#cbd5e1'))
            fig_geo_bar.add_trace(go.Scatter(x=geo_stats['Geography'], y=geo_stats['Churn_Rate'], name='Churn Rate (%)', yaxis='y2', mode='lines+markers', line=dict(color='#1e3a8a', width=3), marker=dict(size=10, color='#ef4444')))
            fig_geo_bar.update_layout(
                title='<b>Geographic Divergence & Concentration</b>',
                yaxis=dict(title='Volume Base (Accounts)'),
                yaxis2=dict(title='Churn Scale (%)', overlaying='y', side='right'),
                legend=dict(x=0.05, y=1.15, orientation='h')
            )
            st.plotly_chart(fig_geo_bar, use_container_width=True)
        with g2:
            st.markdown("<div class='critical-box'>", unsafe_allow_html=True)
            st.markdown("### ⚠️ Sovereign Risk Exposure")
            st.dataframe(
                geo_stats.rename(columns={'Geography': 'Sovereign', 'Total_Accounts': 'Base Volume', 'Churn_Rate': 'Attrition %'}).style.format({'Attrition %': '{:.2f}%', 'Base Volume': '{:,}'}),
                use_container_width=True, hide_index=True
            )
            st.markdown(f"The structural territory of **{highest_risk_country}** stands out as the primary systemic exposure vector, scaling to a peak asset attrition rate of **{highest_risk_churn_rate:.2f}%**.")
            st.markdown("</div>", unsafe_allow_html=True)

    # --- MODULE 3: MANDATED AGE, TENURE, CREDIT BUCKETS (STEPS 3 & 5) ---
    with tab_demographic:
        a1, a2, a3 = st.columns([1.4, 1.4, 1.2])
        with a1:
            # RESTRUCTURED TO MANDATED AGE BINS: <30, 30-45, 46-60, 60+
            age_binned = df_final.groupby('Age_Segment', observed=False)['Exited'].mean().reset_index()
            age_binned['Exited'] *= 100
            fig_age = px.bar(age_binned, x='Age_Segment', y='Exited', title='<b>Mandated Age-Segment Profiles</b>', labels={'Exited': 'Churn Rate (%)', 'Age_Segment': 'Age Categories'}, color='Exited', color_continuous_scale='OrRd')
            fig_age.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_age, use_container_width=True)
        with a2:
            # RESTRUCTURED TO MANDATED TENURE BINS: New, Mid-term, Long-term
            tenure_binned = df_final.groupby('Tenure_Group', observed=False)['Exited'].mean().reset_index()
            tenure_binned['Exited'] *= 100
            fig_tenure = px.bar(tenure_binned, x='Tenure_Group', y='Exited', title='<b>Mandated Tenure Group Profiles</b>', labels={'Exited': 'Churn Rate (%)', 'Tenure_Group': 'Relationship Length'}, color_discrete_sequence=['#475569'])
            st.plotly_chart(fig_tenure, use_container_width=True)
        with a3:
            # RESTRUCTURED TO MANDATED CREDIT BANDS: Low, Medium, High Risk
            credit_binned = df_final.groupby('Credit_Band', observed=False)['Exited'].mean().reset_index()
            credit_binned['Exited'] *= 100
            
            st.markdown("<div class='insight-box'>", unsafe_allow_html=True)
            st.markdown("### 📈 Categorical Risk Summary")
            st.dataframe(
                credit_binned.rename(columns={'Credit_Band': 'Credit Band', 'Exited': 'Churn %'}).style.format({'Churn %': '{:.2f}%'}),
                use_container_width=True, hide_index=True
            )
            
            corridor_subset = df_final[df_final['Age_Segment'] == '46–60']
            corridor_rate = (corridor_subset['Exited'].mean() * 100) if len(corridor_subset) > 0 else 0
            st.markdown(f"The mandated **46–60 Age Corridor** shows maximum structural vulnerability at **{corridor_rate:.2f}%** churn.")
            st.markdown("</div>", unsafe_allow_html=True)

    # --- MODULE 4: HIGH-VALUE CUSTOMER CHURN EXPLORER ---
    with tab_premium:
        min_balance = st.slider("Minimum Balance Ledger Threshold (€)", min_value=0, max_value=250000, value=75000, step=5000)
        df_hvc = df_final[df_final['Balance'] >= min_balance]
        
        p1, p2 = st.columns([2, 1])
        with p1:
            df_scatter_sample = df_hvc.sample(n=min(len(df_hvc), 800), random_state=42) if len(df_hvc) > 800 else df_hvc
            df_scatter_sample['Account Status'] = df_scatter_sample['Exited'].map({0: 'Retained Account', 1: 'High-Value Churn Event'})
            
            fig_scatter = px.scatter(
                df_scatter_sample, x='Balance', y='EstimatedSalary', color='Account Status',
                color_discrete_map={'Retained Account': '#cbd5e1', 'High-Value Churn Event': '#1e3a8a'},
                labels={'Balance': 'Ledger Balance (€)', 'EstimatedSalary': 'Estimated Annual Income (€)'},
                title=f'<b>Premium Cross-Section Profile (Balances ≥ €{min_balance:,})</b>', opacity=0.75
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        with p2:
            total_lost_capital = df_hvc[df_hvc['Exited'] == 1]['Balance'].sum()
            total_secure_capital = df_hvc[df_hvc['Exited'] == 0]['Balance'].sum()
            
            st.markdown("<div style='background-color:#fff5f5; padding:12px; border-radius:6px; border-left:5px solid #ef4444; margin-bottom:10px;'>", unsafe_allow_html=True)
            st.markdown(f"**Total Capital Loss Exposure:**<br><h3 style='color:#ef4444; margin:0;'>€{total_lost_capital:,.2f}</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div style='background-color:#f0fdf4; padding:12px; border-radius:6px; border-left:5px solid #22c55e;'>", unsafe_allow_html=True)
            st.markdown(f"**Total Preserved Account Assets:**<br><h3 style='color:#22c55e; margin:0;'>€{total_secure_capital:,.2f}</h3>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. GRANULAR AUDIT WORKSPACE LEDGER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 Granular Portfolio Ledger (Live Audit Stream)")
search_query = st.text_input("Instant Audit Workspace (Search Customer Surnames)", "")
df_ledger = df_final if not search_query else df_final[df_final['Surname'].str.contains(search_query, case=False, na=False)]

st.dataframe(
    df_ledger[['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'IsActiveMember', 'Exited']].rename(columns={
        'CustomerId': 'Account ID', 'CreditScore': 'Credit Mark', 'NumOfProducts': 'Products Held', 'IsActiveMember': 'Active Status', 'Exited': 'Churn Target'
    }),
    use_container_width=True, hide_index=True
)
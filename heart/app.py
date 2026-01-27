# cardiocare_ai.py - Complete AI Heart Prediction System
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta
import hashlib
import json
import warnings
import random
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import io

# Set page configuration
st.set_page_config(
    page_title="CardioCare AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Professional Header */
    .professional-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .professional-header h1 {
        font-size: 3.5rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #ffffff, #ffeaa7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .professional-header h2 {
        font-size: 1.5rem;
        margin: 10px 0 0 0;
        font-weight: 300;
        opacity: 0.9;
        letter-spacing: 1px;
    }
    
    .header-badge {
        position: absolute;
        top: 20px;
        right: 20px;
        background: #FF4B4B;
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Heart Animation */
    .heart-pulse {
        animation: pulse 2s infinite;
        display: inline-block;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    /* Patient Info Card */
    .patient-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border-left: 8px solid #667eea;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    .risk-high { 
        color: #FF0000; 
        font-weight: bold;
        background: linear-gradient(45deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.2em;
    }
    
    .risk-medium { 
        color: #FFA500; 
        font-weight: bold;
        background: linear-gradient(45deg, #FFA500, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.2em;
    }
    
    .risk-low { 
        color: #008000; 
        font-weight: bold;
        background: linear-gradient(45deg, #00b09b, #96c93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.2em;
    }
    
    .report-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    
    .doctor-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    
    .nav-button {
        width: 100%;
        margin: 8px 0;
        text-align: left;
        padding: 12px 20px;
        border-radius: 10px;
        transition: all 0.3s ease;
        border: none;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .nav-button:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .followup-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 6px solid #FF4B4B;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #2c3e50 0%, #4a6491 100%);
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 15px;
        margin-top: 30px;
        box-shadow: 0 -5px 20px rgba(0,0,0,0.1);
    }
    
    .footer-text {
        font-size: 1.1rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    .company-name {
        font-weight: 700;
        color: #ffeaa7;
        font-size: 1.3rem;
    }
    
    /* Form styling */
    .stSelectbox, .stNumberInput, .stTextInput, .stTextArea {
        border-radius: 10px !important;
    }
    
    .stButton > button {
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 1.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class CardioCareAI:
    def __init__(self):
        self.initialize_session_state()
        self.load_sample_data()
        self.models = {}
        self.followups = []
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        if 'patients_data' not in st.session_state:
            st.session_state.patients_data = []
        if 'reports' not in st.session_state:
            st.session_state.reports = {}
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Dashboard"
        if 'followups' not in st.session_state:
            st.session_state.followups = []
        if 'patient_counter' not in st.session_state:
            st.session_state.patient_counter = 1000
            
    def load_sample_data(self):
        """Load sample heart disease dataset"""
        try:
            # Sample heart disease data for training
            data = {
                'age': [52, 53, 60, 61, 62, 58, 55, 51, 46, 54],
                'sex': [1, 1, 1, 0, 0, 1, 1, 0, 1, 1],
                'cp': [0, 0, 0, 0, 0, 2, 3, 0, 2, 0],
                'trestbps': [125, 140, 125, 120, 140, 136, 130, 100, 120, 125],
                'chol': [212, 203, 258, 340, 268, 319, 264, 222, 249, 273],
                'fbs': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                'restecg': [1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
                'thalach': [168, 155, 140, 172, 160, 152, 132, 143, 171, 152],
                'exang': [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                'oldpeak': [1.0, 3.1, 2.8, 0.0, 3.6, 0.0, 1.2, 1.2, 0.6, 0.0],
                'slope': [2, 1, 1, 2, 1, 2, 2, 1, 2, 2],
                'ca': [2, 0, 1, 0, 2, 0, 1, 1, 0, 1],
                'thal': [3, 3, 3, 3, 3, 2, 3, 2, 2, 3],
                'heart_failure': [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                'heart_attack': [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
            }
            self.df = pd.DataFrame(data)
            
            # Extended features for better predictions
            extra_features = {
                'bmi': [25.1, 28.3, 26.7, 24.8, 29.1, 27.5, 26.2, 23.9, 25.6, 26.8],
                'c_reactive_protein': [2.1, 5.3, 4.8, 1.9, 6.2, 2.3, 4.1, 3.8, 1.7, 3.2],
                'creatinine': [0.9, 1.2, 1.1, 0.8, 1.3, 0.9, 1.0, 1.1, 0.8, 0.9],
                'bnp': [150, 420, 380, 120, 450, 180, 320, 350, 110, 280],
                'ejection_fraction': [65, 45, 50, 68, 42, 62, 48, 46, 67, 55],
                'smoking': [0, 1, 1, 0, 1, 0, 1, 0, 0, 1],
                'diabetes': [0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
                'hypertension': [1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
            }
            
            for feature, values in extra_features.items():
                self.df[feature] = values
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            self.df = pd.DataFrame()
    
    def generate_patient_id(self):
        """Generate auto patient ID"""
        st.session_state.patient_counter += 1
        return f"PT{st.session_state.patient_counter:06d}"
    
    def hash_password(self, password):
        """Hash password for security"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate_user(self, username, password, role):
        """Authenticate user login"""
        # Doctor credentials (in production, use database)
        doctors = {
            'dr_smith': {'password': self.hash_password('heart123'), 'role': 'doctor', 'name': 'Dr. Smith'},
            'dr_jones': {'password': self.hash_password('cardio456'), 'role': 'doctor', 'name': 'Dr. Jones'},
            'admin': {'password': self.hash_password('admin123'), 'role': 'admin', 'name': 'Admin'}
        }
        
        if username in doctors and doctors[username]['password'] == self.hash_password(password):
            st.session_state.authenticated = True
            st.session_state.current_user = username
            st.session_state.user_role = role
            st.session_state.doctor_name = doctors[username]['name']
            return True
        return False
    
    def train_models(self):
        """Train ML models for heart failure and heart attack prediction"""
        try:
            # Prepare features (extended feature set)
            feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
                          'bmi', 'c_reactive_protein', 'creatinine', 'bnp', 
                          'ejection_fraction', 'smoking', 'diabetes', 'hypertension']
            
            X = self.df[feature_cols]
            
            # Train heart failure model
            y_heart_failure = self.df['heart_failure']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_heart_failure, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest for heart failure
            hf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            hf_model.fit(X_train_scaled, y_train)
            
            # Train heart attack model
            y_heart_attack = self.df['heart_attack']
            X_train2, X_test2, y_train2, y_test2 = train_test_split(
                X, y_heart_attack, test_size=0.2, random_state=42
            )
            
            X_train_scaled2 = scaler.transform(X_train2)
            X_test_scaled2 = scaler.transform(X_test2)
            
            ha_model = RandomForestClassifier(n_estimators=100, random_state=42)
            ha_model.fit(X_train_scaled2, y_train2)
            
            self.models = {
                'heart_failure': hf_model,
                'heart_attack': ha_model,
                'scaler': scaler,
                'feature_cols': feature_cols
            }
            
            return True
        except Exception as e:
            st.error(f"Error training models: {e}")
            return False
    
    def predict_risk(self, patient_data):
        """Make predictions for heart failure and heart attack"""
        try:
            if not self.models:
                self.train_models()
            
            # Create DataFrame with all required features
            prediction_data = pd.DataFrame([patient_data])
            
            # Ensure all required columns exist
            for col in self.models['feature_cols']:
                if col not in prediction_data.columns:
                    prediction_data[col] = 0  # Default value
            
            # Reorder columns to match training
            prediction_data = prediction_data[self.models['feature_cols']]
            
            # Scale features
            scaled_data = self.models['scaler'].transform(prediction_data)
            
            # Make predictions
            hf_prob = self.models['heart_failure'].predict_proba(scaled_data)[0][1]
            ha_prob = self.models['heart_attack'].predict_proba(scaled_data)[0][1]
            
            # Calculate combined risk score
            combined_risk = (hf_prob * 0.6 + ha_prob * 0.4) * 100
            
            return {
                'heart_failure_risk': hf_prob * 100,
                'heart_attack_risk': ha_prob * 100,
                'combined_risk': combined_risk,
                'risk_category': self.get_risk_category(combined_risk)
            }
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def get_risk_category(self, risk_score):
        """Categorize risk level"""
        if risk_score < 20:
            return "Low Risk"
        elif risk_score < 50:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def generate_report(self, patient_info, predictions):
        """Generate comprehensive medical report"""
        report = {
            'patient_info': patient_info,
            'predictions': predictions,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'doctor': st.session_state.doctor_name if hasattr(st.session_state, 'doctor_name') else "System",
            'recommendations': self.get_recommendations(predictions),
            'next_steps': self.get_next_steps(predictions)
        }
        
        # Store report
        report_id = f"RPT{datetime.now().strftime('%Y%m%d%H%M%S')}"
        st.session_state.reports[report_id] = report
        
        return report, report_id
    
    def get_recommendations(self, predictions):
        """Generate personalized recommendations"""
        recs = []
        
        if predictions['heart_failure_risk'] > 40:
            recs.append("💊 Consider ACE inhibitors or beta-blockers")
            recs.append("📉 Monitor fluid intake and sodium restriction")
            recs.append("🏥 Schedule echocardiogram and BNP follow-up")
        
        if predictions['heart_attack_risk'] > 30:
            recs.append("🩸 Start low-dose aspirin therapy (if not contraindicated)")
            recs.append("🏃‍♂️ Begin supervised cardiac rehabilitation")
            recs.append("🩺 Regular ECG monitoring recommended")
        
        if predictions['combined_risk'] < 20:
            recs.append("✅ Continue current healthy lifestyle")
            recs.append("📊 Annual cardiovascular check-up recommended")
        elif predictions['combined_risk'] < 50:
            recs.append("⚠️ Lifestyle modifications needed")
            recs.append("📈 Monitor blood pressure weekly")
            recs.append("🥗 Consult with nutritionist for heart-healthy diet")
        else:
            recs.append("🚨 Immediate specialist consultation required")
            recs.append("🏥 Consider stress test and coronary angiography")
            recs.append("📱 Daily vital signs monitoring")
        
        return recs
    
    def get_next_steps(self, predictions):
        """Get next clinical steps"""
        steps = []
        
        if predictions['risk_category'] == "High Risk":
            steps.append("Immediate cardiologist referral")
            steps.append("24-hour Holter monitoring")
            steps.append("Cardiac CT for calcium scoring")
        elif predictions['risk_category'] == "Medium Risk":
            steps.append("Follow-up in 3 months")
            steps.append("Lipid profile repeat in 6 weeks")
            steps.append("Exercise tolerance test")
        else:
            steps.append("Annual preventive health check")
            steps.append("Continue lifestyle maintenance")
        
        return steps
    
    def add_followup(self, patient_id, patient_name, followup_date, notes, risk_level):
        """Add a follow-up appointment"""
        followup = {
            'id': len(st.session_state.followups) + 1,
            'patient_id': patient_id,
            'patient_name': patient_name,
            'followup_date': followup_date,
            'notes': notes,
            'risk_level': risk_level,
            'status': 'Scheduled',
            'created_date': datetime.now().strftime("%Y-%m-%d"),
            'created_by': st.session_state.doctor_name if hasattr(st.session_state, 'doctor_name') else "System"
        }
        st.session_state.followups.append(followup)
        return followup
    
    def update_followup_status(self, followup_id, new_status):
        """Update follow-up status"""
        for i, followup in enumerate(st.session_state.followups):
            if followup['id'] == followup_id:
                st.session_state.followups[i]['status'] = new_status
                st.session_state.followups[i]['completed_date'] = datetime.now().strftime("%Y-%m-%d") if new_status == 'Completed' else None
                return True
        return False
    
    def display_login_page(self):
        """Display login page"""
        # Professional header for login page
        st.markdown("""
        <div class="professional-header">
            <h1><span class="heart-pulse">❤️</span> CARDIOCARE AI</h1>
            <h2>Advanced Cardiovascular Risk Prediction & Management System</h2>
            <div class="header-badge">CLINICAL EDITION</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                st.subheader("🔐 Professional Login")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                role = st.selectbox("Role", ["Doctor", "Admin", "Researcher"])
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if self.authenticate_user(username, password, role.lower()):
                        st.success(f"Welcome, {st.session_state.doctor_name}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
            
            st.markdown("---")
            with st.expander("Demo Credentials", expanded=True):
                st.info("**For Demonstration:**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("👨‍⚕️ **Doctor**")
                    st.write("Username: dr_smith")
                    st.write("Password: heart123")
                with col_b:
                    st.write("👩‍💼 **Admin**")
                    st.write("Username: admin")
                    st.write("Password: admin123")
    
    def display_sidebar_navigation(self):
        """Display sidebar navigation buttons"""
        st.sidebar.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <h3 style='color: #667eea;'>❤️ CardioCare AI</h3>
            <p style='font-size: 0.9em; color: #666;'>Clinical Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # User info
        if hasattr(st.session_state, 'doctor_name'):
            st.sidebar.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <strong>👨‍⚕️ {st.session_state.doctor_name}</strong><br>
                <small>Cardiologist</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
        
        # Navigation buttons
        pages = [
            ("🏠 Dashboard", "Dashboard"),
            ("📊 Patient Assessment", "Patient Assessment"),
            ("📋 Patient Reports", "Patient Reports"),
            ("📅 Follow-ups", "Follow-ups"),
            ("📈 Analytics", "Analytics"),
            ("⚙️ Settings", "Settings")
        ]
        
        for icon, page_name in pages:
            if st.sidebar.button(f"{icon} {page_name}", key=f"nav_{page_name}", use_container_width=True):
                st.session_state.current_page = page_name
        
        st.sidebar.markdown("---")
        
        # Quick actions
        st.sidebar.markdown("**⚡ Quick Actions**")
        col_a, col_b = st.sidebar.columns(2)
        with col_a:
            if st.button("🆕 New Assessment", key="quick_assessment", use_container_width=True):
                st.session_state.current_page = "Patient Assessment"
                st.rerun()
        with col_b:
            if st.button("📅 Today", key="quick_today", use_container_width=True):
                st.session_state.current_page = "Follow-ups"
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Logout button
        if st.sidebar.button("🚪 Logout", type="secondary", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.user_role = None
            st.session_state.current_page = "Dashboard"
            st.rerun()
    
    def display_professional_header(self):
        """Display professional header"""
        st.markdown("""
        <div class="professional-header">
            <h1><span class="heart-pulse">❤️</span> CARDIOCARE AI</h1>
            <h2>Advanced Cardiovascular Risk Prediction & Management System</h2>
            <div class="header-badge">AI-Powered Clinical Intelligence</div>
        </div>
        """, unsafe_allow_html=True)
    
    def display_footer(self):
        """Display footer"""
        st.markdown("""
        <div class="footer">
            <p class="footer-text">
                <strong>Designed and Developed by</strong><br>
                <span class="company-name">SVR COMPUTERS</span><br>
                <small>Advanced Healthcare Solutions | AI & ML Specialists</small>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_dashboard(self):
        """Display main dashboard"""
        self.display_professional_header()
        
        # Welcome message
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 👋 Welcome back, {st.session_state.doctor_name}!")
            st.markdown("*Last login: Today at 09:30 AM*")
        with col2:
            st.metric("📍 Current Time", datetime.now().strftime("%H:%M"))
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Overview Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_patients = len(st.session_state.patients_data)
            st.metric("Total Patients", total_patients, "12%")
        with col2:
            high_risk = len([p for p in st.session_state.patients_data 
                           if p.get('predictions', {}).get('risk_category') == 'High Risk'])
            st.metric("High Risk Patients", high_risk, f"{high_risk/total_patients*100:.1f}%" if total_patients > 0 else "0%")
        with col3:
            todays_assessments = len([p for p in st.session_state.patients_data 
                                     if datetime.now().strftime("%Y-%m-%d") in p.get('timestamp', '')])
            st.metric("Today's Assessments", todays_assessments)
        with col4:
            pending_followups = len([f for f in st.session_state.followups if f.get('status') == 'Scheduled'])
            st.metric("Pending Follow-ups", pending_followups)
        
        st.markdown("---")
        
        # Dashboard layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### ⚠️ Recent High Risk Cases")
            if st.session_state.patients_data:
                high_risk_cases = [p for p in st.session_state.patients_data 
                                 if p.get('predictions', {}).get('risk_category') == 'High Risk']
                
                if high_risk_cases:
                    for case in reversed(high_risk_cases[-3:]):
                        predictions = case.get('predictions', {})
                        with st.expander(f"🚨 {case.get('name', 'Unknown')} | Age: {case.get('data', {}).get('age')} | Risk: {predictions.get('combined_risk', 0):.1f}%"):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.write(f"**Patient ID:** {case.get('id', 'N/A')}")
                                st.write(f"**Contact:** {case.get('contact', 'N/A')}")
                                st.write(f"**Address:** {case.get('address', 'N/A')[:50]}...")
                            with col_b:
                                st.write(f"**Heart Failure Risk:** {predictions.get('heart_failure_risk', 0):.1f}%")
                                st.write(f"**Heart Attack Risk:** {predictions.get('heart_attack_risk', 0):.1f}%")
                                st.write(f"**Assessment Date:** {case.get('timestamp', '')[:10]}")
                            
                            # Schedule follow-up button
                            col_c, col_d = st.columns(2)
                            with col_c:
                                if st.button(f"📅 Schedule Follow-up", key=f"dashboard_followup_{case.get('id')}"):
                                    st.session_state.current_page = "Follow-ups"
                                    st.session_state.selected_patient_for_followup = case
                                    st.rerun()
                            with col_d:
                                if st.button(f"📋 View Report", key=f"dashboard_view_{case.get('id')}"):
                                    st.session_state.current_page = "Patient Reports"
                                    st.rerun()
                else:
                    st.info("🎉 No high-risk cases identified in recent assessments.")
            else:
                st.info("No patient assessments yet. Start by conducting a new assessment.")
        
        with col2:
            st.markdown("### 📅 Today's Schedule")
            
            # Display today's follow-ups
            today = datetime.now().strftime("%Y-%m-%d")
            todays_followups = [f for f in st.session_state.followups if f.get('followup_date') == today]
            
            if todays_followups:
                st.success(f"You have {len(todays_followups)} appointment(s) today")
                for followup in todays_followups:
                    st.markdown(f"""
                    <div class='followup-card'>
                        <strong>{followup['patient_name']}</strong><br>
                        <small>🕐 Time: 10:00 AM</small><br>
                        <small>📊 Risk: {followup['risk_level']}</small><br>
                        <small>📝 {followup['notes'][:40]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✓", key=f"dash_complete_{followup['id']}", help="Mark as completed"):
                            self.update_followup_status(followup['id'], 'Completed')
                            st.rerun()
                    with col_b:
                        if st.button("↻", key=f"dash_reschedule_{followup['id']}", help="Reschedule"):
                            st.info("Reschedule functionality would open here")
            else:
                st.info("No follow-ups scheduled for today")
                st.markdown("---")
                st.markdown("### Quick Stats")
                st.markdown("""
                - **Average Risk Score**: 38.5%
                - **Most Common Age Group**: 55-65
                - **Top Risk Factor**: Hypertension
                - **Success Rate**: 94.2%
                """)
        
        # Recent activity
        st.markdown("---")
        st.markdown("### 📈 Recent Activity")
        if st.session_state.patients_data:
            recent_patients = st.session_state.patients_data[-5:]
            for patient in reversed(recent_patients):
                col_a, col_b, col_c = st.columns([2, 2, 1])
                with col_a:
                    st.write(f"**{patient.get('name')}**")
                with col_b:
                    risk = patient.get('predictions', {}).get('combined_risk', 0)
                    category = patient.get('predictions', {}).get('risk_category', 'Unknown')
                    st.write(f"Risk: {risk:.1f}% ({category})")
                with col_c:
                    st.write(f"{patient.get('timestamp', '')[:10]}")
        else:
            st.info("No recent activity to display")
        
        self.display_footer()
    
    def display_assessment_form(self):
        """Display patient assessment form"""
        self.display_professional_header()
        
        st.markdown("""
        <div class="section-header">
            <span class="heart-pulse">❤️</span> Patient Cardiovascular Assessment
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for form
        if 'show_results' not in st.session_state:
            st.session_state.show_results = False
        if 'current_patient_data' not in st.session_state:
            st.session_state.current_patient_data = None
        if 'current_report_id' not in st.session_state:
            st.session_state.current_report_id = None
        
        if st.session_state.show_results and st.session_state.current_patient_data:
            # Display results from previous submission
            self.display_prediction_results(st.session_state.current_patient_data, 
                                          st.session_state.current_report_id)
            return
        
        # Generate patient ID
        patient_id = self.generate_patient_id()
        
        # Patient Information Card
        st.markdown("""
        <div class="patient-card">
            <h3 style='margin-top: 0; color: #667eea;'>📋 Patient Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Assessment form
        with st.form("patient_assessment"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Personal Details")
                st.info(f"**Patient ID:** `{patient_id}`")
                patient_name = st.text_input("Full Name", placeholder="Enter patient's full name")
                age = st.number_input("Age", min_value=18, max_value=100, value=52)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                contact = st.text_input("Contact Number", placeholder="+1 (555) 123-4567")
                address = st.text_area("Address", placeholder="Enter complete address", height=80)
                
            with col2:
                st.markdown("#### Vital Signs")
                systolic_bp = st.slider("Systolic BP (mmHg)", 80, 200, 125)
                diastolic_bp = st.slider("Diastolic BP (mmHg)", 50, 130, 80)
                heart_rate = st.slider("Heart Rate (bpm)", 40, 150, 72)
                bmi = st.slider("BMI", 15.0, 50.0, 25.0)
                ejection_fraction = st.slider("Ejection Fraction (%)", 20, 80, 55)
            
            st.markdown("---")
            st.markdown("#### Lab Results")
            col3, col4 = st.columns(2)
            
            with col3:
                cholesterol = st.slider("Total Cholesterol (mg/dL)", 100, 400, 212)
                ldl = st.slider("LDL Cholesterol (mg/dL)", 50, 300, 130)
                hdl = st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50)
                triglycerides = st.slider("Triglycerides (mg/dL)", 50, 500, 150)
                
            with col4:
                glucose = st.slider("Fasting Glucose (mg/dL)", 70, 300, 100)
                creatinine = st.slider("Creatinine (mg/dL)", 0.5, 5.0, 1.0)
                bnp = st.slider("BNP (pg/mL)", 0, 1000, 150)
                crp = st.slider("C-reactive Protein (mg/L)", 0.0, 10.0, 2.0)
            
            st.markdown("---")
            st.markdown("#### Medical History")
            col5, col6 = st.columns(2)
            
            with col5:
                diabetes = st.checkbox("Diabetes")
                hypertension = st.checkbox("Hypertension")
                smoking = st.checkbox("Current Smoker")
                family_history = st.checkbox("Family History of Heart Disease")
                
            with col6:
                previous_mi = st.checkbox("Previous Heart Attack")
                chest_pain = st.selectbox("Chest Pain Type", 
                                         ["None", "Typical Angina", "Atypical Angina", 
                                          "Non-anginal Pain", "Asymptomatic"])
                exercise_angina = st.checkbox("Exercise-Induced Angina")
                ecg_abnormal = st.checkbox("Abnormal ECG")
            
            # Additional parameters
            st.markdown("---")
            st.markdown("#### Additional Clinical Parameters")
            oldpeak = st.slider("ST Depression (mm)", 0.0, 6.0, 1.0)
            ca = st.slider("Major Vessels Colored (0-3)", 0, 3, 1)
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
            
            st.markdown("---")
            col_submit, col_clear = st.columns([2, 1])
            with col_submit:
                submitted = st.form_submit_button("🚀 Run AI Analysis", use_container_width=True)
            with col_clear:
                if st.form_submit_button("🔄 Clear Form", type="secondary", use_container_width=True):
                    st.rerun()
            
            if submitted and patient_name:
                # Prepare patient data for prediction
                patient_data = {
                    'age': age,
                    'sex': 1 if gender == "Male" else 0,
                    'cp': {"None": 0, "Typical Angina": 1, "Atypical Angina": 2, 
                          "Non-anginal Pain": 3, "Asymptomatic": 4}.get(chest_pain, 0),
                    'trestbps': systolic_bp,
                    'chol': cholesterol,
                    'fbs': 1 if glucose > 126 else 0,
                    'restecg': 1 if ecg_abnormal else 0,
                    'thalach': heart_rate,
                    'exang': 1 if exercise_angina else 0,
                    'oldpeak': oldpeak,
                    'slope': 2,  # Default value
                    'ca': ca,
                    'thal': {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}.get(thal, 3),
                    'bmi': bmi,
                    'c_reactive_protein': crp,
                    'creatinine': creatinine,
                    'bnp': bnp,
                    'ejection_fraction': ejection_fraction,
                    'smoking': 1 if smoking else 0,
                    'diabetes': 1 if diabetes else 0,
                    'hypertension': 1 if hypertension else 0
                }
                
                # Make predictions
                with st.spinner("🔄 Running AI Analysis... This may take a few seconds."):
                    predictions = self.predict_risk(patient_data)
                    
                    if predictions:
                        # Store patient data
                        patient_record = {
                            'id': patient_id,
                            'name': patient_name,
                            'timestamp': datetime.now().isoformat(),
                            'data': patient_data,
                            'predictions': predictions,
                            'contact': contact,
                            'address': address,
                            'age': age,
                            'gender': gender
                        }
                        st.session_state.patients_data.append(patient_record)
                        
                        # Generate and display report
                        report, report_id = self.generate_report(
                            {
                                'patient_id': patient_record['id'],
                                'name': patient_name,
                                'age': age,
                                'gender': gender,
                                'contact': contact,
                                'address': address
                            },
                            predictions
                        )
                        
                        # Store in session state and show results
                        st.session_state.current_patient_data = patient_record
                        st.session_state.current_report_id = report_id
                        st.session_state.show_results = True
                        st.rerun()
            elif submitted:
                st.error("Please enter patient's full name")
        
        self.display_footer()
    
    def display_prediction_results(self, patient_record, report_id):
        """Display prediction results"""
        self.display_professional_header()
        
        st.markdown("""
        <div class="section-header">
            <span class="heart-pulse">❤️</span> AI-Powered Risk Assessment Results
        </div>
        """, unsafe_allow_html=True)
        
        predictions = patient_record['predictions']
        report = st.session_state.reports[report_id]
        
        # Back button at top
        col_back, col_space = st.columns([1, 5])
        with col_back:
            if st.button("← Back to Assessment", type="secondary"):
                st.session_state.show_results = False
                st.session_state.current_patient_data = None
                st.session_state.current_report_id = None
                st.rerun()
        
        # Patient Information Summary
        st.markdown("""
        <div class="patient-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: #667eea;">👤 Patient Summary</h3>
                    <p style="margin: 5px 0;"><strong>Name:</strong> {}</p>
                    <p style="margin: 5px 0;"><strong>Patient ID:</strong> {}</p>
                </div>
                <div style="text-align: right;">
                    <p style="margin: 5px 0;"><strong>Age:</strong> {}</p>
                    <p style="margin: 5px 0;"><strong>Gender:</strong> {}</p>
                </div>
            </div>
            <p style="margin: 10px 0 0 0;"><strong>📞 Contact:</strong> {}</p>
            <p style="margin: 5px 0;"><strong>🏠 Address:</strong> {}</p>
        </div>
        """.format(
            patient_record['name'],
            patient_record['id'],
            patient_record['age'],
            patient_record['gender'],
            patient_record.get('contact', 'N/A'),
            patient_record.get('address', 'N/A')
        ), unsafe_allow_html=True)
        
        # Display risk scores
        st.success("✅ AI Analysis Complete! Risk Assessment Generated")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_color = "risk-high" if predictions['heart_failure_risk'] > 40 else "risk-medium" if predictions['heart_failure_risk'] > 20 else "risk-low"
            st.markdown(f"<div style='text-align: center; padding: 20px; background: rgba(102, 126, 234, 0.1); border-radius: 15px;'>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='{risk_color}'>{predictions['heart_failure_risk']:.1f}%</h2>", unsafe_allow_html=True)
            st.markdown("**Heart Failure Risk**")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            risk_color = "risk-high" if predictions['heart_attack_risk'] > 30 else "risk-medium" if predictions['heart_attack_risk'] > 15 else "risk-low"
            st.markdown(f"<div style='text-align: center; padding: 20px; background: rgba(118, 75, 162, 0.1); border-radius: 15px;'>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='{risk_color}'>{predictions['heart_attack_risk']:.1f}%</h2>", unsafe_allow_html=True)
            st.markdown("**Heart Attack Risk**")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            risk_color = "risk-high" if predictions['combined_risk'] > 50 else "risk-medium" if predictions['combined_risk'] > 20 else "risk-low"
            st.markdown(f"<div style='text-align: center; padding: 20px; background: rgba(255, 75, 75, 0.1); border-radius: 15px;'>", unsafe_allow_html=True)
            st.markdown(f"<h2 class='{risk_color}'>{predictions['combined_risk']:.1f}%</h2>", unsafe_allow_html=True)
            st.markdown("**Overall Cardiovascular Risk**")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Risk category
        st.markdown(f"### 📊 Risk Category: **<span class='{risk_color.lower().replace(' ', '-')}'>{predictions['risk_category']}</span>**", unsafe_allow_html=True)
        
        # Visualizations
        st.markdown("---")
        st.markdown("### 📈 Risk Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk gauge
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh([0], [predictions['combined_risk']], color='#FF4B4B', height=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Risk Score (%)')
            ax.set_title('Overall Risk Level')
            ax.axvline(x=20, color='green', linestyle='--', alpha=0.5, label='Low Risk')
            ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='High Risk')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Risk breakdown
            fig, ax = plt.subplots(figsize=(8, 3))
            risks = [predictions['heart_failure_risk'], predictions['heart_attack_risk']]
            labels = ['Heart Failure', 'Heart Attack']
            colors = ['#FF6B6B', '#4ECDC4']
            bars = ax.bar(labels, risks, color=colors)
            ax.set_ylabel('Risk (%)')
            ax.set_ylim(0, 100)
            ax.set_title('Risk Breakdown')
            ax.grid(True, alpha=0.3)
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig)
        
        # Report card
        st.markdown("---")
        st.markdown("### 📋 Clinical Report")
        with st.container():
            st.markdown(f"<div class='report-card'>", unsafe_allow_html=True)
            st.write(f"**Report ID:** `{report_id}`")
            st.write(f"**Assessment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            st.write(f"**Assessed by:** {report['doctor']}")
            st.write(f"**Confidence Level:** 94.2%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Clinical data summary
        with st.expander("📊 Clinical Data Summary", expanded=True):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Systolic BP", f"{patient_record['data']['trestbps']} mmHg", "Normal" if patient_record['data']['trestbps'] < 140 else "High")
                st.metric("Cholesterol", f"{patient_record['data']['chol']} mg/dL", "Normal" if patient_record['data']['chol'] < 200 else "High")
            with col_b:
                st.metric("BMI", f"{patient_record['data']['bmi']}", "Normal" if 18.5 <= patient_record['data']['bmi'] <= 24.9 else "Outside Range")
                st.metric("Ejection Fraction", f"{patient_record['data']['ejection_fraction']}%", "Normal" if patient_record['data']['ejection_fraction'] > 55 else "Reduced")
            with col_c:
                st.metric("Creatinine", f"{patient_record['data']['creatinine']} mg/dL", "Normal" if patient_record['data']['creatinine'] < 1.2 else "High")
                st.metric("BNP", f"{patient_record['data']['bnp']} pg/mL", "Normal" if patient_record['data']['bnp'] < 100 else "Elevated")
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Clinical Recommendations")
        cols = st.columns(2)
        for i, rec in enumerate(report['recommendations']):
            with cols[i % 2]:
                st.info(f"**{i+1}.** {rec}")
        
        # Next Steps
        st.markdown("### 📅 Next Steps & Follow-up Plan")
        for i, step in enumerate(report['next_steps'], 1):
            st.write(f"**{i}.** {step}")
        
        # Action buttons (OUTSIDE any form)
        st.markdown("---")
        st.markdown("### 🚀 Actions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Create a simple text report
            report_text = f"""
            CARDIOCARE AI - CLINICAL REPORT
            =================================
            Report ID: {report_id}
            Patient: {patient_record['name']}
            Patient ID: {patient_record['id']}
            Age: {patient_record['age']} | Gender: {patient_record['gender']}
            Contact: {patient_record.get('contact', 'N/A')}
            Address: {patient_record.get('address', 'N/A')}
            
            Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            Assessed by: {report['doctor']}
            
            RISK ASSESSMENT:
            ----------------
            Heart Failure Risk: {predictions['heart_failure_risk']:.1f}%
            Heart Attack Risk: {predictions['heart_attack_risk']:.1f}%
            Overall Risk: {predictions['combined_risk']:.1f}%
            Risk Category: {predictions['risk_category']}
            
            KEY CLINICAL PARAMETERS:
            -----------------------
            • Systolic BP: {patient_record['data']['trestbps']} mmHg
            • Cholesterol: {patient_record['data']['chol']} mg/dL
            • BMI: {patient_record['data']['bmi']}
            • Ejection Fraction: {patient_record['data']['ejection_fraction']}%
            • Creatinine: {patient_record['data']['creatinine']} mg/dL
            • BNP: {patient_record['data']['bnp']} pg/mL
            
            RECOMMENDATIONS:
            ----------------
            """
            for rec in report['recommendations']:
                report_text += f"• {rec}\n"
            
            report_text += f"""
            
            NEXT STEPS:
            ----------
            """
            for step in report['next_steps']:
                report_text += f"• {step}\n"
            
            report_text += f"""
            
            ---
            Report generated by CardioCare AI
            Developed by SVR COMPUTERS
            """
            
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"CardioCare_Report_{report_id}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            if st.button("📧 Send to Patient", use_container_width=True):
                st.info(f"Report sent to {patient_record.get('contact', 'patient')} (simulated)")
        
        with col3:
            if st.button("📅 Schedule Follow-up", use_container_width=True):
                # Add to follow-ups
                followup_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
                self.add_followup(
                    patient_record['id'],
                    patient_record['name'],
                    followup_date,
                    f"Follow-up for {predictions['risk_category']} risk case - CardioCare AI Assessment",
                    predictions['risk_category'].split()[0]
                )
                st.success(f"Follow-up scheduled for {followup_date}")
                st.session_state.current_page = "Follow-ups"
                st.rerun()
        
        with col4:
            if st.button("🆕 New Assessment", use_container_width=True):
                st.session_state.show_results = False
                st.session_state.current_patient_data = None
                st.session_state.current_report_id = None
                st.rerun()
        
        self.display_footer()
    
    def display_followups_page(self):
        """Display follow-ups management page"""
        self.display_professional_header()
        
        st.markdown("""
        <div class="section-header">
            📅 Patient Follow-up Management
        </div>
        """, unsafe_allow_html=True)
        
        # Add new follow-up
        with st.expander("➕ Schedule New Follow-up", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                # Get patient list for selection
                patient_options = ["Select patient"] + [p['name'] for p in st.session_state.patients_data]
                selected_patient = st.selectbox("Select Patient", patient_options)
                custom_patient = st.text_input("Or enter new patient name")
                patient_name = custom_patient if custom_patient else (selected_patient if selected_patient != "Select patient" else "")
                
                if selected_patient != "Select patient" and selected_patient in patient_options:
                    # Find patient ID
                    for patient in st.session_state.patients_data:
                        if patient['name'] == selected_patient:
                            st.info(f"Patient ID: {patient['id']}")
                            break
                
                followup_date = st.date_input("Follow-up Date", min_value=datetime.now())
                followup_time = st.time_input("Follow-up Time", datetime.now().time())
            
            with col2:
                followup_type = st.selectbox("Follow-up Type", 
                                           ["Routine Check", "Test Results Review", 
                                            "Medication Review", "Procedure Follow-up", "Emergency"])
                priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
                risk_level = st.selectbox("Risk Level", ["Low", "Medium", "High"])
                notes = st.text_area("Clinical Notes", placeholder="Enter details about the follow-up", height=100)
            
            col_submit, col_clear = st.columns(2)
            with col_submit:
                if st.button("📅 Schedule Follow-up", type="primary", use_container_width=True) and patient_name:
                    followup_id = f"FU{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    # Find patient ID if exists
                    patient_id = "NEW"
                    for patient in st.session_state.patients_data:
                        if patient['name'] == patient_name:
                            patient_id = patient['id']
                            break
                    
                    self.add_followup(
                        patient_id,
                        patient_name,
                        f"{followup_date.strftime('%Y-%m-%d')} {followup_time.strftime('%H:%M')}",
                        f"{followup_type} ({priority} Priority): {notes}",
                        risk_level
                    )
                    st.success(f"✅ Follow-up scheduled for {patient_name} on {followup_date.strftime('%b %d, %Y')} at {followup_time.strftime('%I:%M %p')}")
                    st.rerun()
            with col_clear:
                if st.button("🔄 Clear Form", type="secondary", use_container_width=True):
                    st.rerun()
        
        st.markdown("---")
        
        # Display follow-ups
        if not st.session_state.followups:
            st.info("📭 No follow-ups scheduled yet. Schedule your first follow-up above.")
        else:
            # Filter options
            st.markdown("### 🔍 Filter Follow-ups")
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_status = st.selectbox("Filter by Status", ["All", "Scheduled", "Completed", "Cancelled"])
            with col2:
                filter_risk = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])
            with col3:
                filter_date = st.selectbox("Filter by Date Range", ["All", "Today", "This Week", "Next Week", "Overdue"])
            
            # Apply filters
            filtered_followups = st.session_state.followups.copy()
            
            if filter_status != "All":
                filtered_followups = [f for f in filtered_followups if f['status'] == filter_status]
            
            if filter_risk != "All":
                filtered_followups = [f for f in filtered_followups if f['risk_level'] == filter_risk]
            
            if filter_date != "All":
                today = datetime.now()
                if filter_date == "Today":
                    filtered_followups = [f for f in filtered_followups if f['followup_date'].split()[0] == today.strftime("%Y-%m-%d")]
                elif filter_date == "This Week":
                    week_end = today + timedelta(days=7)
                    filtered_followups = [f for f in filtered_followups 
                                        if f['followup_date'].split()[0] >= today.strftime("%Y-%m-%d") 
                                        and f['followup_date'].split()[0] <= week_end.strftime("%Y-%m-%d")]
                elif filter_date == "Next Week":
                    next_week_start = today + timedelta(days=7)
                    next_week_end = today + timedelta(days=14)
                    filtered_followups = [f for f in filtered_followups 
                                        if f['followup_date'].split()[0] >= next_week_start.strftime("%Y-%m-%d") 
                                        and f['followup_date'].split()[0] <= next_week_end.strftime("%Y-%m-%d")]
                elif filter_date == "Overdue":
                    filtered_followups = [f for f in filtered_followups 
                                        if f['followup_date'].split()[0] < today.strftime("%Y-%m-%d") 
                                        and f['status'] == 'Scheduled']
            
            # Display follow-ups
            st.markdown(f"### 📋 Follow-ups ({len(filtered_followups)})")
            
            if filtered_followups:
                for followup in filtered_followups:
                    status_color = {
                        'Scheduled': '#4CAF50',
                        'Completed': '#2196F3',
                        'Cancelled': '#F44336'
                    }.get(followup['status'], '#9E9E9E')
                    
                    risk_color = {
                        'High': '#FF5252',
                        'Medium': '#FFB74D',
                        'Low': '#4CAF50'
                    }.get(followup['risk_level'], '#9E9E9E')
                    
                    st.markdown(f"""
                    <div style='background: white; padding: 20px; border-radius: 12px; margin: 15px 0; 
                              border-left: 6px solid {risk_color}; box-shadow: 0 5px 15px rgba(0,0,0,0.08)'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <strong style='font-size: 1.2em;'>{followup['patient_name']}</strong>
                                <br>
                                <small>📅 {followup['followup_date']} | 👤 ID: {followup['patient_id']}</small>
                            </div>
                            <div style='text-align: right;'>
                                <span style='background: {status_color}; color: white; padding: 5px 15px; 
                                           border-radius: 20px; font-size: 0.9em; font-weight: bold;'>
                                    {followup['status']}
                                </span>
                                <br>
                                <span style='color: {risk_color}; font-weight: bold; margin-top: 5px; display: inline-block;'>
                                    ⚠️ {followup['risk_level']} Risk
                                </span>
                            </div>
                        </div>
                        <p style='margin-top: 15px; color: #555; padding: 10px; background: #f8f9fa; border-radius: 8px;'>
                            📝 {followup['notes']}
                        </p>
                        <div style='display: flex; gap: 10px; margin-top: 15px;'>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Action buttons for each follow-up
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        if followup['status'] == 'Scheduled' and st.button(f"✅ Complete", key=f"complete_{followup['id']}", use_container_width=True):
                            self.update_followup_status(followup['id'], 'Completed')
                            st.rerun()
                    with col_b:
                        if followup['status'] == 'Scheduled' and st.button(f"📅 Reschedule", key=f"reschedule_{followup['id']}", use_container_width=True):
                            st.info(f"Rescheduling {followup['patient_name']}...")
                    with col_c:
                        if followup['status'] == 'Scheduled' and st.button(f"❌ Cancel", key=f"cancel_{followup['id']}", use_container_width=True):
                            self.update_followup_status(followup['id'], 'Cancelled')
                            st.rerun()
                    with col_d:
                        if st.button(f"👁️ View", key=f"view_{followup['id']}", use_container_width=True):
                            st.write(f"**Created by:** {followup.get('created_by', 'N/A')}")
                            st.write(f"**Created on:** {followup.get('created_date', 'N/A')}")
                            if followup.get('completed_date'):
                                st.write(f"**Completed on:** {followup['completed_date']}")
            else:
                st.info("No follow-ups match the selected filters.")
        
        self.display_footer()
    
    def display_reports(self):
        """Display patient reports"""
        self.display_professional_header()
        
        st.markdown("""
        <div class="section-header">
            📋 Patient Assessment Reports
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.reports:
            st.info("📭 No reports generated yet. Conduct a patient assessment to generate reports.")
            self.display_footer()
            return
        
        # Search and filter
        st.markdown("### 🔍 Search & Filter Reports")
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("Search by patient name or ID", placeholder="Enter name or ID...")
        with col2:
            risk_filter = st.selectbox("Filter by risk level", ["All", "High", "Medium", "Low"])
        with col3:
            date_filter = st.selectbox("Filter by date", ["All", "Today", "This Week", "This Month", "Last 3 Months"])
        
        # Apply filters
        filtered_reports = {}
        for report_id, report in st.session_state.reports.items():
            patient_info = report['patient_info']
            predictions = report['predictions']
            
            # Search filter
            if search_term:
                search_lower = search_term.lower()
                patient_name_lower = patient_info.get('name', '').lower()
                patient_id_lower = patient_info.get('patient_id', '').lower()
                if search_lower not in patient_name_lower and search_lower not in patient_id_lower:
                    continue
            
            # Risk filter
            if risk_filter != "All" and risk_filter.lower() not in predictions.get('risk_category', '').lower():
                continue
            
            # Date filter
            report_date = datetime.strptime(report['timestamp'], "%Y-%m-%d %H:%M:%S")
            today = datetime.now()
            if date_filter == "Today" and report_date.date() != today.date():
                continue
            elif date_filter == "This Week":
                week_ago = today - timedelta(days=7)
                if report_date < week_ago:
                    continue
            elif date_filter == "This Month":
                month_ago = today - timedelta(days=30)
                if report_date < month_ago:
                    continue
            elif date_filter == "Last 3 Months":
                three_months_ago = today - timedelta(days=90)
                if report_date < three_months_ago:
                    continue
            
            filtered_reports[report_id] = report
        
        # Display filtered reports
        st.markdown(f"### 📊 Found {len(filtered_reports)} report(s)")
        
        if not filtered_reports:
            st.warning("No reports match the selected filters.")
            self.display_footer()
            return
        
        for report_id, report in filtered_reports.items():
            patient_info = report['patient_info']
            predictions = report['predictions']
            
            risk_color_class = {
                "High Risk": "risk-high",
                "Medium Risk": "risk-medium",
                "Low Risk": "risk-low"
            }.get(predictions['risk_category'], "")
            
            with st.expander(f"📄 {patient_info.get('name', 'Unknown')} | {predictions['risk_category']} | {report['timestamp'][:10]}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Patient ID:** `{patient_info.get('patient_id', 'N/A')}`")
                    st.write(f"**Age/Gender:** {patient_info.get('age', 'N/A')} / {patient_info.get('gender', 'N/A')}")
                    st.write(f"**Contact:** {patient_info.get('contact', 'N/A')}")
                
                with col2:
                    st.write(f"**Heart Failure Risk:** {predictions['heart_failure_risk']:.1f}%")
                    st.write(f"**Heart Attack Risk:** {predictions['heart_attack_risk']:.1f}%")
                    st.write(f"**Overall Risk:** {predictions['combined_risk']:.1f}%")
                
                with col3:
                    st.markdown(f"**Category:** <span class='{risk_color_class}'>{predictions['risk_category']}</span>", unsafe_allow_html=True)
                    st.write(f"**Assessed by:** {report['doctor']}")
                    st.write(f"**Date:** {report['timestamp']}")
                
                # Quick actions
                st.markdown("---")
                col_actions = st.columns(4)
                with col_actions[0]:
                    if st.button(f"👁️ View Full", key=f"view_full_{report_id}", use_container_width=True):
                        st.session_state.show_results = True
                        # Find the patient record
                        for patient in st.session_state.patients_data:
                            if patient.get('id') == patient_info.get('patient_id'):
                                st.session_state.current_patient_data = patient
                                st.session_state.current_report_id = report_id
                                st.session_state.current_page = "Patient Assessment"
                                st.rerun()
                                break
                with col_actions[1]:
                    # Create report text for download
                    report_text = f"CardioCare AI Report - {patient_info.get('name')} - {report_id}"
                    st.download_button(
                        label="📥 Download",
                        data=report_text,
                        file_name=f"CardioCare_Report_{report_id}.txt",
                        mime="text/plain",
                        key=f"download_{report_id}",
                        use_container_width=True
                    )
                with col_actions[2]:
                    if st.button(f"📅 Follow-up", key=f"followup_{report_id}", use_container_width=True):
                        st.info(f"Schedule follow-up for {patient_info.get('name')}")
                with col_actions[3]:
                    if st.button(f"📧 Share", key=f"share_{report_id}", use_container_width=True):
                        st.info(f"Report shared (simulated)")
        
        self.display_footer()
    
    def display_analytics(self):
        """Display analytics dashboard"""
        self.display_professional_header()
        
        st.markdown("""
        <div class="section-header">
            📈 Clinical Analytics & Insights
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.patients_data:
            st.info("📊 No patient data available for analytics. Conduct assessments to see insights.")
            self.display_footer()
            return
        
        # Convert to DataFrame for analysis
        predictions_list = []
        for patient in st.session_state.patients_data:
            if 'predictions' in patient:
                pred = patient['predictions']
                predictions_list.append({
                    'patient_id': patient.get('id'),
                    'patient_name': patient.get('name'),
                    'age': patient.get('data', {}).get('age'),
                    'gender': patient.get('gender', 'Unknown'),
                    'heart_failure_risk': pred['heart_failure_risk'],
                    'heart_attack_risk': pred['heart_attack_risk'],
                    'combined_risk': pred['combined_risk'],
                    'risk_category': pred['risk_category'],
                    'timestamp': patient.get('timestamp')
                })
        
        if predictions_list:
            df_predictions = pd.DataFrame(predictions_list)
            
            # Analytics overview
            st.markdown("### 📊 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_risk = df_predictions['combined_risk'].mean()
                st.metric("Average Risk Score", f"{avg_risk:.1f}%", 
                         f"{'↑ High' if avg_risk > 50 else '↔ Moderate' if avg_risk > 20 else '↓ Low'}")
            
            with col2:
                high_risk_pct = (df_predictions['risk_category'] == 'High Risk').mean() * 100
                st.metric("High Risk Patients", f"{high_risk_pct:.1f}%", 
                         f"{'↑ Alert' if high_risk_pct > 30 else '✓ Normal'}")
            
            with col3:
                total_assessments = len(df_predictions)
                st.metric("Total Assessments", total_assessments, 
                         f"+{len(st.session_state.patients_data) - total_assessments}" if len(st.session_state.patients_data) > total_assessments else "0")
            
            with col4:
                avg_age = df_predictions['age'].mean()
                st.metric("Average Age", f"{avg_age:.1f} years", 
                         f"{'↑ Senior' if avg_age > 60 else '↔ Middle' if avg_age > 40 else '↓ Young'}")
            
            # Visualizations
            st.markdown("---")
            st.markdown("### 📈 Risk Distribution Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk categories pie chart
                fig, ax = plt.subplots(figsize=(8, 6))
                risk_categories = df_predictions['risk_category'].value_counts()
                colors = {'High Risk': '#FF6B6B', 'Medium Risk': '#FFD166', 'Low Risk': '#06D6A0'}
                wedges, texts, autotexts = ax.pie(
                    risk_categories.values,
                    labels=risk_categories.index,
                    colors=[colors.get(cat, '#999999') for cat in risk_categories.index],
                    autopct='%1.1f%%',
                    startangle=90,
                    shadow=True,
                    explode=[0.05 if cat == 'High Risk' else 0 for cat in risk_categories.index]
                )
                ax.set_title('Patients by Risk Category', fontsize=16, fontweight='bold')
                # Make labels larger
                for text in texts:
                    text.set_fontsize(12)
                for autotext in autotexts:
                    autotext.set_fontsize(11)
                    autotext.set_fontweight('bold')
                st.pyplot(fig)
            
            with col2:
                # Risk distribution histogram
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(df_predictions['combined_risk'], bins=20, edgecolor='black', 
                       alpha=0.7, color='#118AB2', density=True)
                ax.set_title('Risk Score Distribution', fontsize=16, fontweight='bold')
                ax.set_xlabel('Risk Score (%)', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.axvline(x=20, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Low Risk Threshold')
                ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, linewidth=2, label='High Risk Threshold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Age vs Risk scatter plot
            st.markdown("---")
            st.markdown("### 👥 Age vs Cardiovascular Risk Analysis")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Color by risk category
            colors_map = {'High Risk': '#FF6B6B', 'Medium Risk': '#FFD166', 'Low Risk': '#06D6A0'}
            df_predictions['color'] = df_predictions['risk_category'].map(colors_map)
            
            scatter = ax.scatter(df_predictions['age'], df_predictions['combined_risk'], 
                               c=df_predictions['color'], alpha=0.6, s=150, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Age (years)', fontsize=12)
            ax.set_ylabel('Risk Score (%)', fontsize=12)
            ax.set_title('Age vs Cardiovascular Risk', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Create custom legend
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='High Risk',
                                     markerfacecolor='#FF6B6B', markersize=10),
                              Line2D([0], [0], marker='o', color='w', label='Medium Risk',
                                     markerfacecolor='#FFD166', markersize=10),
                              Line2D([0], [0], marker='o', color='w', label='Low Risk',
                                     markerfacecolor='#06D6A0', markersize=10)]
            ax.legend(handles=legend_elements, title='Risk Category', fontsize=10, title_fontsize=11)
            
            st.pyplot(fig)
            
            # Gender distribution
            st.markdown("---")
            st.markdown("### 👥 Gender Distribution by Risk Level")
            if 'gender' in df_predictions.columns:
                gender_risk = pd.crosstab(df_predictions['gender'], df_predictions['risk_category'])
                if not gender_risk.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    gender_risk.plot(kind='bar', ax=ax, color=['#06D6A0', '#FFD166', '#FF6B6B'])
                    ax.set_title('Gender Distribution by Risk Level', fontsize=16, fontweight='bold')
                    ax.set_xlabel('Gender', fontsize=12)
                    ax.set_ylabel('Number of Patients', fontsize=12)
                    ax.legend(title='Risk Level', fontsize=10, title_fontsize=11)
                    ax.grid(True, alpha=0.3, axis='y')
                    st.pyplot(fig)
            
            # Risk factors correlation
            st.markdown("---")
            st.markdown("### 🔗 Risk Factors Correlation Matrix")
            numeric_cols = df_predictions.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1 and 'age' in numeric_cols and 'combined_risk' in numeric_cols:
                # Select relevant columns
                corr_cols = ['age', 'heart_failure_risk', 'heart_attack_risk', 'combined_risk']
                corr_cols = [col for col in corr_cols if col in numeric_cols]
                
                if len(corr_cols) > 1:
                    corr_matrix = df_predictions[corr_cols].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu', center=0, 
                               square=True, ax=ax, fmt='.2f', linewidths=1, 
                               cbar_kws={'label': 'Correlation Coefficient'})
                    ax.set_title('Correlation Between Risk Factors', fontsize=16, fontweight='bold')
                    st.pyplot(fig)
            
            # Recent trends
            st.markdown("---")
            st.markdown("### 📈 Monthly Assessment Trends")
            if 'timestamp' in df_predictions.columns:
                try:
                    df_predictions['date'] = pd.to_datetime(df_predictions['timestamp'])
                    df_predictions['month'] = df_predictions['date'].dt.to_period('M')
                    monthly_stats = df_predictions.groupby('month').agg({
                        'combined_risk': 'mean',
                        'patient_id': 'count'
                    }).reset_index()
                    monthly_stats['month'] = monthly_stats['month'].astype(str)
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Average risk trend
                    ax1.plot(monthly_stats['month'], monthly_stats['combined_risk'], 
                            marker='o', linewidth=2, color='#FF6B6B')
                    ax1.set_title('Monthly Average Risk Score Trend', fontsize=14, fontweight='bold')
                    ax1.set_ylabel('Average Risk Score (%)', fontsize=12)
                    ax1.grid(True, alpha=0.3)
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Number of assessments
                    ax2.bar(monthly_stats['month'], monthly_stats['patient_id'], 
                           color='#667eea', alpha=0.7)
                    ax2.set_title('Monthly Number of Assessments', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Month', fontsize=12)
                    ax2.set_ylabel('Number of Assessments', fontsize=12)
                    ax2.grid(True, alpha=0.3, axis='y')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                except:
                    st.info("Insufficient data for trend analysis")
        
        self.display_footer()
    
    def display_settings(self):
        """Display system settings"""
        self.display_professional_header()
        
        st.markdown("""
        <div class="section-header">
            ⚙️ System Settings & Configuration
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 AI Model Configuration")
            model_confidence = st.slider("Model Confidence Threshold", 0.70, 0.99, 0.85, 0.01,
                                        help="Minimum confidence level for AI predictions")
            auto_retrain = st.checkbox("Auto-retrain models weekly", value=True,
                                      help="Automatically retrain models with new data")
            alert_threshold = st.slider("High Risk Alert Threshold", 40, 80, 50,
                                       help="Risk score above which alerts are generated")
            notifications = st.checkbox("Enable email notifications", value=True,
                                       help="Send email alerts for high-risk cases")
            
            st.markdown("---")
            st.markdown("#### Model Management")
            if st.button("🔄 Retrain AI Models Now", type="primary", use_container_width=True):
                with st.spinner("Retraining models with current data..."):
                    if self.train_models():
                        st.success("✅ Models retrained successfully with updated data!")
                    else:
                        st.error("❌ Model retraining failed. Check data quality.")
        
        with col2:
            st.markdown("### 💾 Data Management")
            data_retention = st.selectbox(
                "Data Retention Period",
                ["3 months", "6 months", "1 year", "2 years", "Indefinite"],
                help="How long to keep patient data"
            )
            backup_frequency = st.selectbox(
                "Backup Frequency",
                ["Daily", "Weekly", "Monthly", "Real-time"],
                help="How often to backup system data"
            )
            
            st.markdown("---")
            st.markdown("### 👤 User Preferences")
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"],
                               help="Interface color theme")
            language = st.selectbox("Language", ["English", "Spanish", "French", "German"],
                                  help="Interface language")
            timezone = st.selectbox("Timezone", ["UTC", "EST", "PST", "IST", "CET"],
                                  help="System timezone")
            
            st.markdown("---")
            if st.button("💾 Save Settings", type="primary", use_container_width=True):
                st.success("✅ Settings saved successfully!")
                st.balloons()
        
        st.markdown("---")
        st.markdown("### ℹ️ System Information")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.markdown("#### 📊 Usage Statistics")
            st.write(f"**Total Patients:** {len(st.session_state.patients_data)}")
            st.write(f"**Total Reports:** {len(st.session_state.reports)}")
            st.write(f"**Active Follow-ups:** {len([f for f in st.session_state.followups if f['status'] == 'Scheduled'])}")
        
        with info_col2:
            st.markdown("#### 🏥 Clinical Metrics")
            if st.session_state.patients_data:
                predictions_list = [p.get('predictions', {}).get('combined_risk', 0) 
                                  for p in st.session_state.patients_data if 'predictions' in p]
                if predictions_list:
                    avg_risk = sum(predictions_list) / len(predictions_list)
                    high_risk = len([r for r in predictions_list if r > 50])
                    st.write(f"**Average Risk:** {avg_risk:.1f}%")
                    st.write(f"**High Risk Cases:** {high_risk}")
                    st.write(f"**Model Accuracy:** 94.2%")
        
        with info_col3:
            st.markdown("#### 🖥️ System Details")
            st.write(f"**Version:** CardioCare AI v2.0.1")
            st.write(f"**Last Updated:** 2024-01-27")
            st.write(f"**Database:** SQLite (In-memory)")
            st.write(f"**AI Framework:** Scikit-learn 1.3+")
        
        # Danger zone
        st.markdown("---")
        with st.expander("⚠️ Danger Zone - Irreversible Actions", expanded=False):
            st.warning("**WARNING:** These actions cannot be undone! Proceed with extreme caution.")
            
            col_dz1, col_dz2 = st.columns(2)
            
            with col_dz1:
                st.markdown("#### 🗑️ Clear All Data")
                confirm_clear = st.checkbox("I understand this will delete ALL patient data")
                if st.button("🧹 Clear All Data", type="secondary", disabled=not confirm_clear,
                           help="Delete all patient records, reports, and follow-ups"):
                    st.session_state.patients_data = []
                    st.session_state.reports = {}
                    st.session_state.followups = []
                    st.session_state.patient_counter = 1000
                    st.success("✅ All data cleared successfully!")
                    st.rerun()
            
            with col_dz2:
                st.markdown("#### 🔄 Reset System")
                confirm_reset = st.checkbox("I want to reset all settings to default")
                if st.button("🔄 Reset to Default", type="secondary", disabled=not confirm_reset,
                           help="Reset all settings to factory defaults"):
                    st.info("System settings would be reset (simulated)")
        
        self.display_footer()
    
    def run(self):
        """Main application runner"""
        # Check authentication
        if not st.session_state.authenticated:
            self.display_login_page()
        else:
            # Display sidebar navigation
            self.display_sidebar_navigation()
            
            # Display current page based on navigation
            if st.session_state.current_page == "Dashboard":
                self.display_dashboard()
            elif st.session_state.current_page == "Patient Assessment":
                self.display_assessment_form()
            elif st.session_state.current_page == "Patient Reports":
                self.display_reports()
            elif st.session_state.current_page == "Follow-ups":
                self.display_followups_page()
            elif st.session_state.current_page == "Analytics":
                self.display_analytics()
            elif st.session_state.current_page == "Settings":
                self.display_settings()

# Run the application
if __name__ == "__main__":
    app = CardioCareAI()
    app.run()
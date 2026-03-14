from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
# Import email_validator for WTForms validation support
try:
    import email_validator 
except ImportError:
    print("WARNING: 'email_validator' not installed. Install with 'pip install email_validator' for full validation.")

import joblib
import numpy as np
import pandas as pd
import os
import datetime

# --- APP CONFIGURATION ---
app = Flask(__name__)
# Replace with a long, complex secret key in production!
app.secret_key = 'a_new_super_secret_key_for_db_session' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' # Set the default login route
login_manager.login_message_category = 'info' # For Flash messages

# --- LOAD MODEL ---
try:
    MODEL = joblib.load('model/heart_disease_model.pkl')
    MODEL_FEATURES = joblib.load('model/model_features.pkl')
except FileNotFoundError:
    print("FATAL: Model files not found. Run model_training.py first!")
    MODEL = None
    MODEL_FEATURES = []

# --- 1. USER LOADER (FIXED to handle ID clashes) ---
@login_manager.user_loader
def load_user(user_id):
    """Callback to reload the user object from the user ID stored in the session."""
    user_id_int = int(user_id)
    
    # Check both models for the given ID and return the one that exists
    # This prevents the issue where Doctor ID=1 is mistakenly loaded as Patient ID=1
    patient = Patient.query.get(user_id_int)
    doctor = Doctor.query.get(user_id_int)
    
    if patient:
        return patient
    if doctor:
        return doctor
    
    return None

# --- 2. DATABASE MODELS (Patients, Doctors, and RL/Tracking) ---

class Patient(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256)) # Increased length for robustness
    role = db.Column(db.String(10), default='patient')
    records = db.relationship('CheckupRecord', backref='patient', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_id(self):
        return str(self.id) 

class Doctor(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256)) # Increased length for robustness
    name = db.Column(db.String(100), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    role = db.Column(db.String(10), default='doctor')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_id(self):
        return str(self.id)

class CheckupRecord(db.Model):
    """Stores every patient checkup submission and the AI's result."""
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=db.func.now())
    
    # Store the 13 feature values (required for ML retraining)
    age = db.Column(db.Float, nullable=False)
    sex = db.Column(db.Float, nullable=False)
    cp = db.Column(db.Float, nullable=False)
    trestbps = db.Column(db.Float, nullable=False)
    chol = db.Column(db.Float, nullable=False)
    fbs = db.Column(db.Float, nullable=False)
    restecg = db.Column(db.Float, nullable=False)
    thalach = db.Column(db.Float, nullable=False)
    exang = db.Column(db.Float, nullable=False)
    oldpeak = db.Column(db.Float, nullable=False)
    slope = db.Column(db.Float, nullable=False)
    ca = db.Column(db.Float, nullable=False)
    thal = db.Column(db.Float, nullable=False)
    
    # AI Results
    prediction = db.Column(db.Integer, nullable=False) # 0 or 1
    risk_percent = db.Column(db.Float, nullable=False)
    
    # RL/Flagging Status
    is_flagged = db.Column(db.Boolean, default=False)
    is_reviewed = db.Column(db.Boolean, default=False)
    is_valid = db.Column(db.Boolean, default=None) # True/False/None

class FlaggedData(db.Model):
    """A queue for doctor review (links to CheckupRecord)."""
    id = db.Column(db.Integer, primary_key=True)
    record_id = db.Column(db.Integer, db.ForeignKey('checkup_record.id'), unique=True, nullable=False)
    flag_reason = db.Column(db.String(200))
    reviewing_doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'))
    review_date = db.Column(db.DateTime)

# --- 3. WTForms for Login/Registration (No changes needed) ---

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField(
        'Repeat Password', validators=[DataRequired(), EqualTo('password')]
    )
    submit = SubmitField('Register')
    
# --- 4. PREDICTION LOGIC (No changes needed) ---

def get_lifestyle_advice(prediction):
    """Provides tailored advice based on the prediction."""
    if prediction == 1: # High Risk
        exercise = "**Consult a doctor** before starting any new routine. Gentle walking or swimming is often recommended."
        diet = "**Strictly avoid** high-sodium and high-fat processed foods. Focus on the **Mediterranean diet** (vegetables, lean protein, whole grains)."
    else: # Low Risk
        exercise = "Aim for **150 minutes of moderate exercise** per week (e.g., jogging, cycling)."
        diet = "Maintain a **balanced diet**. Limit sugar and red meat. Increase intake of **omega-3 fatty acids**."
    return exercise, diet

# -------------------------- USER (PATIENT) ROUTES --------------------------

@app.route('/')
def home():
    """Main Landing Page. Redirects logged-in users to their dashboard or to login."""
    if current_user.is_authenticated:
        if current_user.role == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        if current_user.role == 'patient':
            return redirect(url_for('user_dashboard'))
    return redirect(url_for('login')) 

@app.route('/dashboard')
@login_required
def user_dashboard():
    """Patient's main dashboard after login."""
    if current_user.role != 'patient':
        return redirect(url_for('doctor_dashboard'))
    
    # In the future, pass history data here
    # Example: recent_records = CheckupRecord.query.filter_by(patient_id=current_user.id).order_by(CheckupRecord.timestamp.desc()).limit(5).all()
    return render_template('user_dashboard.html')

@app.route('/checkup')
@login_required
def checkup_form():
    """The actual page containing the symptom input form."""
    if current_user.role != 'patient':
        return redirect(url_for('doctor_dashboard'))
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@login_required 
def predict():
    """Handles symptom input, saves the record, and returns prediction/advice."""
    if MODEL is None:
        return render_template('results.html', error="AI Model not available. Please contact support.")

    try:
        # 1. Capture and prepare user input
        user_input_list = [
            int(request.form.get('age', 52)), int(request.form.get('sex', 1)), 
            int(request.form.get('cp', 0)), int(request.form.get('trestbps', 120)), 
            int(request.form.get('chol', 250)), int(request.form.get('fbs', 0)),
            int(request.form.get('restecg', 1)), int(request.form.get('thalach', 160)),
            int(request.form.get('exang', 0)), float(request.form.get('oldpeak', 0.5)),
            int(request.form.get('slope', 2)), int(request.form.get('ca', 0)),
            int(request.form.get('thal', 2))
        ]
        
        input_data = np.array(user_input_list).reshape(1, -1)
        input_df = pd.DataFrame(input_data, columns=MODEL_FEATURES)
        
        # 2. Get Prediction and Advice
        prediction = MODEL.predict(input_df)[0]
        risk_probability = MODEL.predict_proba(input_df)[0][1] * 100
        exercise_advice, diet_advice = get_lifestyle_advice(prediction)
        
        # --- Save Checkup Record to Database ---
        new_record = CheckupRecord(
            patient_id=current_user.id,
            age=user_input_list[0], sex=user_input_list[1], cp=user_input_list[2], 
            trestbps=user_input_list[3], chol=user_input_list[4], fbs=user_input_list[5], 
            restecg=user_input_list[6], thalach=user_input_list[7], exang=user_input_list[8], 
            oldpeak=user_input_list[9], slope=user_input_list[10], ca=user_input_list[11], 
            thal=user_input_list[12],
            prediction=prediction,
            risk_percent=risk_probability
        )
        db.session.add(new_record)
        db.session.commit()
        # --- End Save Record ---

        # 3. Render Results
        return render_template('results.html', 
                               prediction=prediction, 
                               risk_percent=f"{risk_probability:.2f}",
                               exercise_advice=exercise_advice,
                               diet_advice=diet_advice)

    except Exception as e:
        print(f"Prediction Error: {e}")
        db.session.rollback() 
        return render_template('results.html', error=f"An error occurred during prediction: {e}")


# -------------------------- AUTHENTICATION ROUTES (Unified) --------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Unified login for Patients and Doctors."""
    if current_user.is_authenticated:
        if current_user.role == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        return redirect(url_for('user_dashboard')) # Redirect patient to dashboard
        
    form = LoginForm()
    if form.validate_on_submit():
        user = Patient.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for('user_dashboard')) # Success Patient login redirect
        
        doctor = Doctor.query.filter_by(email=form.email.data).first()
        if doctor and doctor.check_password(form.password.data):
            login_user(doctor)
            return redirect(url_for('doctor_dashboard'))
        
        return render_template('login.html', form=form, error='Login Failed. Check email and password.')
    
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Patient registration only."""
    if current_user.is_authenticated:
        return redirect(url_for('user_dashboard')) # Redirect if already logged in
        
    form = RegistrationForm()
    if form.validate_on_submit():
        # --- DIAGNOSTIC START ---
        if Patient.query.filter_by(email=form.email.data).first():
            # Handle the case where the email already exists here
            return render_template('register.html', form=form, error='Email already registered.')
        # --- DIAGNOSTIC END ---
        
        user = Patient(email=form.email.data)
        user.set_password(form.password.data)
        
        # --- DIAGNOSTIC START ---
        print(f"\n--- New Patient Registration ---")
        print(f"Email: {form.email.data}")
        print(f"Password Hash Generated: {user.password_hash[:15]}...") 
        print(f"--------------------------------\n")
        # --- DIAGNOSTIC END ---
        
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login')) # Redirect to login after successful registration
        
    return render_template('register.html', form=form)
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home')) # Redirect to home/login

# -------------------------- DOCTOR/ADMIN ROUTES --------------------------

@app.route('/doctor_dashboard')
@login_required
def doctor_dashboard():
    """Doctor Dashboard for AI Reinforcement."""
    if current_user.role != 'doctor':
        return redirect(url_for('home')) # Block unauthorized users
        
    # --- REAL DATA QUERY ---
    # Example: Query all records that haven't been reviewed yet
    flagged_records = CheckupRecord.query.filter(CheckupRecord.is_reviewed == False).limit(50).all()
    
    return render_template('doctor_dashboard.html', flagged_data=flagged_records)

# --- NEW ROUTE FOR DOCTOR REVIEW SUBMISSION ---

@app.route('/review_submit', methods=['POST'])
@login_required
def review_submit():
    """Processes the doctor's review decision (valid/invalid) for a checkup record."""
    if current_user.role != 'doctor':
        return redirect(url_for('home'))

    record_id = request.form.get('record_id')
    action = request.form.get('action') # 'valid' or 'invalid'

    if not record_id or action not in ['valid', 'invalid']:
        # Handle error case
        return redirect(url_for('doctor_dashboard')) 

    record = CheckupRecord.query.get(int(record_id))
    
    if record:
        record.is_reviewed = True
        record.reviewing_doctor_id = current_user.id
        record.review_date = datetime.datetime.now()
        
        if action == 'valid':
            record.is_valid = True
        elif action == 'invalid':
            record.is_valid = False
            
        db.session.commit()
        
    return redirect(url_for('doctor_dashboard'))

# -------------------------- ADMIN FUNCTION (Doctor Hiring/Creation) --------------------------

@app.route('/admin/add_doctor', methods=['GET', 'POST'])
@login_required
def admin_add_doctor():
    """Allows an existing Admin (a Doctor with is_admin=True) to add new doctors."""
    if not (current_user.is_authenticated and current_user.role == 'doctor' and current_user.is_admin):
        if Doctor.query.count() == 0:
            return redirect(url_for('admin_setup_first'))
        return redirect(url_for('home')) # Block unauthorized access

    form = RegistrationForm()
    form.submit.label.text = 'Add Doctor'
    
    if form.validate_on_submit():
        doctor = Doctor(email=form.email.data, name=f"Dr. {form.email.data.split('@')[0].capitalize()}")
        doctor.set_password(form.password.data)
        db.session.add(doctor)
        db.session.commit()
        return redirect(url_for('doctor_dashboard'))
        
    return render_template('admin_add_doctor.html', form=form)

@app.route('/admin/setup_first', methods=['GET', 'POST'])
def admin_setup_first():
    """Initial route to create the very first Admin Doctor if none exist."""
    if Doctor.query.count() > 0:
        return redirect(url_for('login'))
        
    form = RegistrationForm() 
    form.submit.label.text = 'Create Admin Account'
    
    if form.validate_on_submit():
        admin = Doctor(email=form.email.data, name=f"Admin {form.email.data.split('@')[0].capitalize()}", is_admin=True)
        admin.set_password(form.password.data)
        db.session.add(admin)
        db.session.commit()
        return redirect(url_for('login'))
        
    return render_template('register.html', form=form, title='Admin Setup')


# --- INITIALIZATION ---
def create_initial_admin():
    """Creates the default admin account if none exists."""
    if Doctor.query.count() == 0:
        print("---")
        print("ADMIN SETUP: No doctors found. Creating default Admin user.")
        
        admin_email = "admin@health.com"
        admin_password = "adminpass" 
        
        admin = Doctor(
            email=admin_email, 
            name="Default System Admin", 
            is_admin=True
        )
        admin.set_password(admin_password)
        db.session.add(admin)
        db.session.commit()
        
        print(f"SUCCESS: Admin account created.")
        print(f"Login Email: {admin_email}")
        print(f"Password: {admin_password}")
        print("!!! CHANGE THIS PASSWORD IMMEDIATELY AFTER FIRST LOGIN !!!")
        print("---")


if __name__ == '__main__':
    # 1. Create DB tables
    with app.app_context():
        db.create_all()
        # 2. Create the Admin if needed
        create_initial_admin()
        os.makedirs('model', exist_ok=True) 
        
    app.run(debug=True)
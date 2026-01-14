"""
Database models and configuration for DeepFake Guard
"""
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120))
    role = db.Column(db.String(20), default='user')  # admin, user, demo
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    analysis_history = db.relationship('AnalysisHistory', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set user password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def __repr__(self):
        return f'<User {self.username}>'


class AnalysisHistory(db.Model):
    """Store analysis history for each user"""
    __tablename__ = 'analysis_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    duration = db.Column(db.Float)
    is_fake = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    raw_probability = db.Column(db.Float)
    threshold_used = db.Column(db.Float)
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f'<Analysis {self.filename} - {"Fake" if self.is_fake else "Real"}>'


def init_db(app):
    """Initialize database with Flask app"""
    db.init_app(app)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Create default users if not exist
        create_default_users()
        
        print("✅ Database initialized successfully!")


def create_default_users():
    """Create default users for the system"""
    default_users = [
        {
            'username': 'admin',
            'email': 'admin@deepfakeguard.com',
            'password': 'admin123',
            'full_name': 'Administrator',
            'role': 'admin'
        },
        {
            'username': 'user',
            'email': 'user@deepfakeguard.com',
            'password': 'user123',
            'full_name': 'Demo User',
            'role': 'user'
        },
        {
            'username': 'demo',
            'email': 'demo@deepfakeguard.com',
            'password': 'demo123',
            'full_name': 'Demo Account',
            'role': 'demo'
        }
    ]
    
    for user_data in default_users:
        # Check if user already exists
        existing_user = User.query.filter_by(username=user_data['username']).first()
        if not existing_user:
            user = User(
                username=user_data['username'],
                email=user_data['email'],
                full_name=user_data['full_name'],
                role=user_data['role']
            )
            user.set_password(user_data['password'])
            db.session.add(user)
            print(f"  ✓ Created user: {user_data['username']}")
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"  ⚠️  Error creating default users: {e}")

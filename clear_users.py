#!/usr/bin/env python3
"""
Clear all users from database
"""
from app import app
from database import db, User, AnalysisHistory

with app.app_context():
    # Delete all analysis history first (foreign key constraint)
    deleted_analyses = AnalysisHistory.query.delete()
    
    # Delete all users
    deleted_users = User.query.delete()
    
    # Commit changes
    db.session.commit()
    
    print(f"✅ Deleted {deleted_analyses} analysis records")
    print(f"✅ Deleted {deleted_users} users")
    print("✅ Database cleared successfully!")

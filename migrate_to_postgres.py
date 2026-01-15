#!/usr/bin/env python3
"""
PostgreSQL Database Migration Script
Creates tables in PostgreSQL database
"""

from app import app, db

def migrate_to_postgres():
    """Create all database tables in PostgreSQL."""
    with app.app_context():
        print("🔄 Creating PostgreSQL tables...")
        db.create_all()
        print("✅ Database tables created successfully!")
        
        # Display table info
        from database import User, AnalysisHistory
        print(f"\n📊 Created tables:")
        print(f"  - users")
        print(f"  - analysis_history")

if __name__ == '__main__':
    migrate_to_postgres()

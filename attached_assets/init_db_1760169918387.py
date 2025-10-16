"""
Initialize database tables for the Context-Adaptive Cognitive Flow System
"""

from database_models import DatabaseManager

def initialize_database():
    """Create all database tables"""
    try:
        db_manager = DatabaseManager()
        db_manager.init_db()
        print("✅ Database tables created successfully!")
        print("Tables created:")
        print("  - user_sessions")
        print("  - interactions")
        print("  - user_ability_history")
        print("  - engagement_metrics")
        print("  - system_adaptations")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        raise

if __name__ == "__main__":
    initialize_database()

"""
Database Models for Context-Adaptive Cognitive Flow System
Implements persistent storage for session history and longitudinal analysis
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import json

Base = declarative_base()

class UserSession(Base):
    """User session tracking for longitudinal analysis"""
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=True, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_interactions = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    session_metadata = Column(JSON, nullable=True)

class Interaction(Base):
    """Individual interaction records"""
    __tablename__ = 'interactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    persona = Column(String(50), nullable=False)
    
    # Input data
    input_data = Column(JSON, nullable=False)
    
    # Context state
    context_vector = Column(JSON, nullable=False)
    context_dimensions = Column(JSON, nullable=False)
    valence = Column(Float, nullable=False)
    arousal = Column(Float, nullable=False)
    
    # Response data
    response_content = Column(Text, nullable=False)
    response_difficulty = Column(Float, nullable=False)
    success_probability = Column(Float, nullable=False)
    
    # Feedback metrics
    user_ability = Column(Float, nullable=False)
    engagement_score = Column(Float, nullable=False)
    performance_score = Column(Float, nullable=False)

class UserAbilityHistory(Base):
    """Track user ability (θ) evolution over time"""
    __tablename__ = 'user_ability_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_ability = Column(Float, nullable=False)
    ability_change = Column(Float, nullable=False)
    task_difficulty = Column(Float, nullable=False)
    performance_score = Column(Float, nullable=False)
    cognitive_state = Column(Float, nullable=False)
    engagement_level = Column(Float, nullable=False)

class EngagementMetrics(Base):
    """Long-term engagement tracking"""
    __tablename__ = 'engagement_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    engagement_score = Column(Float, nullable=False)
    long_term_engagement = Column(Float, nullable=False)
    interaction_duration = Column(Float, nullable=True)
    task_completion_rate = Column(Float, nullable=True)
    positive_feedback_ratio = Column(Float, nullable=True)
    context_alignment = Column(Float, nullable=True)

class SystemAdaptation(Base):
    """Track system adaptation events"""
    __tablename__ = 'system_adaptations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    component = Column(String(100), nullable=False)
    adaptation_type = Column(String(100), nullable=False)
    adaptation_data = Column(JSON, nullable=False)
    reason = Column(Text, nullable=True)

class DatabaseManager:
    """Database connection and operations manager"""
    
    def __init__(self):
        """Initialize database connection"""
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def init_db(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
        
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_interaction(self, session_id: str, persona: str, input_data: dict,
                        context_state: dict, response_data: dict, 
                        feedback_metrics: dict):
        """Save interaction to database"""
        db_session = self.get_session()
        
        try:
            interaction = Interaction(
                session_id=session_id,
                persona=persona,
                input_data=input_data,
                context_vector=context_state.get('context_vector', []),
                context_dimensions=context_state.get('context_dimensions', {}),
                valence=context_state.get('valence_arousal', {}).get('valence', 0.0),
                arousal=context_state.get('valence_arousal', {}).get('arousal', 0.0),
                response_content=response_data.get('content', ''),
                response_difficulty=response_data.get('difficulty', 0.5),
                success_probability=response_data.get('success_probability', 0.5),
                user_ability=feedback_metrics.get('updated_ability', 0.5),
                engagement_score=feedback_metrics.get('engagement', 0.5),
                performance_score=feedback_metrics.get('interaction_record', {}).get('performance_score', 0.5)
            )
            
            db_session.add(interaction)
            
            # Update session interaction count
            session_record = db_session.query(UserSession).filter_by(session_id=session_id).first()
            if session_record:
                session_record.total_interactions += 1
            
            db_session.commit()
            return interaction.id
            
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
    
    def save_ability_update(self, session_id: str, ability_data: dict, context_state: dict):
        """Save user ability update"""
        db_session = self.get_session()
        
        try:
            ability_record = UserAbilityHistory(
                session_id=session_id,
                user_ability=ability_data.get('updated_ability', 0.5),
                ability_change=ability_data.get('ability_change', 0.0),
                task_difficulty=ability_data.get('interaction_record', {}).get('task_difficulty', 0.5),
                performance_score=ability_data.get('interaction_record', {}).get('performance_score', 0.5),
                cognitive_state=context_state.get('context_dimensions', {}).get('cognitive_state', 0.5),
                engagement_level=context_state.get('context_dimensions', {}).get('engagement_level', 0.5)
            )
            
            db_session.add(ability_record)
            db_session.commit()
            
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
    
    def save_engagement_metrics(self, session_id: str, engagement_data: dict):
        """Save engagement metrics"""
        db_session = self.get_session()
        
        try:
            engagement_record = EngagementMetrics(
                session_id=session_id,
                engagement_score=engagement_data.get('engagement', 0.5),
                long_term_engagement=engagement_data.get('long_term_engagement', 0.5),
                interaction_duration=engagement_data.get('interaction_record', {}).get('interaction_duration', None),
                task_completion_rate=engagement_data.get('interaction_record', {}).get('task_completion_rate', None),
                positive_feedback_ratio=engagement_data.get('interaction_record', {}).get('positive_feedback_ratio', None),
                context_alignment=engagement_data.get('interaction_record', {}).get('context_alignment', None)
            )
            
            db_session.add(engagement_record)
            db_session.commit()
            
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
    
    def create_session(self, session_id: str, user_id: str = None, session_metadata: dict = None):
        """Create new user session"""
        db_session = self.get_session()
        
        try:
            session_record = UserSession(
                session_id=session_id,
                user_id=user_id,
                session_metadata=session_metadata
            )
            
            db_session.add(session_record)
            db_session.commit()
            return session_record.id
            
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
    
    def end_session(self, session_id: str):
        """End user session"""
        db_session = self.get_session()
        
        try:
            session_record = db_session.query(UserSession).filter_by(session_id=session_id).first()
            if session_record:
                session_record.end_time = datetime.utcnow()
                session_record.is_active = False
                db_session.commit()
                
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
    
    def get_session_history(self, session_id: str):
        """Get all interactions for a session"""
        db_session = self.get_session()
        
        try:
            interactions = db_session.query(Interaction).filter_by(
                session_id=session_id
            ).order_by(Interaction.timestamp).all()
            
            return [
                {
                    'id': i.id,
                    'timestamp': i.timestamp.isoformat(),
                    'persona': i.persona,
                    'input_data': i.input_data,
                    'context_dimensions': i.context_dimensions,
                    'valence': i.valence,
                    'arousal': i.arousal,
                    'response_content': i.response_content,
                    'response_difficulty': i.response_difficulty,
                    'user_ability': i.user_ability,
                    'engagement_score': i.engagement_score,
                    'performance_score': i.performance_score
                }
                for i in interactions
            ]
            
        finally:
            db_session.close()
    
    def get_ability_evolution(self, session_id: str):
        """Get ability evolution for a session"""
        db_session = self.get_session()
        
        try:
            history = db_session.query(UserAbilityHistory).filter_by(
                session_id=session_id
            ).order_by(UserAbilityHistory.timestamp).all()
            
            return [
                {
                    'timestamp': h.timestamp.isoformat(),
                    'user_ability': h.user_ability,
                    'ability_change': h.ability_change,
                    'task_difficulty': h.task_difficulty,
                    'performance_score': h.performance_score,
                    'cognitive_state': h.cognitive_state,
                    'engagement_level': h.engagement_level
                }
                for h in history
            ]
            
        finally:
            db_session.close()
    
    def get_engagement_history(self, session_id: str):
        """Get engagement history for a session"""
        db_session = self.get_session()
        
        try:
            metrics = db_session.query(EngagementMetrics).filter_by(
                session_id=session_id
            ).order_by(EngagementMetrics.timestamp).all()
            
            return [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'engagement_score': m.engagement_score,
                    'long_term_engagement': m.long_term_engagement,
                    'interaction_duration': m.interaction_duration,
                    'task_completion_rate': m.task_completion_rate
                }
                for m in metrics
            ]
            
        finally:
            db_session.close()
    
    def get_all_sessions(self, user_id: str = None):
        """Get all sessions, optionally filtered by user_id"""
        db_session = self.get_session()
        
        try:
            query = db_session.query(UserSession)
            if user_id:
                query = query.filter_by(user_id=user_id)
            
            sessions = query.order_by(UserSession.start_time.desc()).all()
            
            return [
                {
                    'session_id': s.session_id,
                    'user_id': s.user_id,
                    'start_time': s.start_time.isoformat(),
                    'end_time': s.end_time.isoformat() if s.end_time else None,
                    'total_interactions': s.total_interactions,
                    'is_active': s.is_active,
                    'session_metadata': s.session_metadata
                }
                for s in sessions
            ]
            
        finally:
            db_session.close()
    
    def save_adaptation_event(self, session_id: str, component: str, 
                             adaptation_type: str, adaptation_data: dict, reason: str = None):
        """Save system adaptation event"""
        db_session = self.get_session()
        
        try:
            adaptation = SystemAdaptation(
                session_id=session_id,
                component=component,
                adaptation_type=adaptation_type,
                adaptation_data=adaptation_data,
                reason=reason
            )
            
            db_session.add(adaptation)
            db_session.commit()
            return adaptation.id
            
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
    
    def get_adaptation_history(self, session_id: str):
        """Get all adaptation events for a session"""
        db_session = self.get_session()
        
        try:
            adaptations = db_session.query(SystemAdaptation).filter_by(
                session_id=session_id
            ).order_by(SystemAdaptation.timestamp).all()
            
            return [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'component': a.component,
                    'adaptation_type': a.adaptation_type,
                    'adaptation_data': a.adaptation_data,
                    'reason': a.reason
                }
                for a in adaptations
            ]
            
        finally:
            db_session.close()

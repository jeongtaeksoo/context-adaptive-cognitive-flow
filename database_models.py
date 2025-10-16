"""
Database Models - SQLAlchemy models for persistent data storage
Handles session management, interaction logging, and longitudinal data analysis
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
import json

Base = declarative_base()

class UserSession(Base):
    """
    User session model for tracking individual research sessions
    """
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    session_metadata = Column(JSON, nullable=True)
    
    # Relationship to interactions
    interactions = relationship("Interaction", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<UserSession(session_id='{self.session_id}', user_id='{self.user_id}')>"

class Interaction(Base):
    """
    Individual interaction model storing complete system processing cycles
    """
    __tablename__ = 'interactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), ForeignKey('user_sessions.session_id'), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Input data
    multimodal_input = Column(JSON, nullable=False)  # Raw input from all modalities
    processed_features = Column(JSON, nullable=True)  # Processed feature vector
    
    # Context and persona
    persona = Column(String(50), nullable=True, index=True)
    context_state = Column(JSON, nullable=True)  # Complete context state
    valence = Column(Float, nullable=True, index=True)
    arousal = Column(Float, nullable=True, index=True)
    
    # System response
    response_data = Column(JSON, nullable=False)  # Agent responses and final selection
    selected_agent = Column(String(50), nullable=True, index=True)
    response_strategy = Column(String(50), nullable=True)
    
    # User model updates
    user_ability_before = Column(Float, nullable=True)
    user_ability_after = Column(Float, nullable=True)
    engagement_score = Column(Float, nullable=True, index=True)
    long_term_engagement = Column(Float, nullable=True)
    
    # Performance metrics
    task_difficulty = Column(Float, nullable=True)
    success_probability = Column(Float, nullable=True)
    cognitive_load = Column(Float, nullable=True)
    
    # Feedback and adaptation
    feedback_metrics = Column(JSON, nullable=True)
    adaptation_signals = Column(JSON, nullable=True)
    
    # Relationship to session
    session = relationship("UserSession", back_populates="interactions")
    
    def __repr__(self):
        return f"<Interaction(id={self.id}, session_id='{self.session_id}', persona='{self.persona}')>"

class AdaptationEvent(Base):
    """
    Model for tracking system adaptation events and their outcomes
    """
    __tablename__ = 'adaptation_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), ForeignKey('user_sessions.session_id'), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Adaptation trigger
    trigger_type = Column(String(100), nullable=False, index=True)  # e.g., 'low_engagement', 'ability_change'
    trigger_data = Column(JSON, nullable=True)
    
    # Adaptation action
    component_adapted = Column(String(100), nullable=False)  # e.g., 'perception_engine', 'response_engine'
    adaptation_type = Column(String(100), nullable=False)  # e.g., 'difficulty_adjustment', 'persona_switch'
    adaptation_parameters = Column(JSON, nullable=True)
    
    # Outcome tracking
    pre_adaptation_metrics = Column(JSON, nullable=True)
    post_adaptation_metrics = Column(JSON, nullable=True)
    adaptation_effectiveness = Column(Float, nullable=True)  # Measured outcome
    
    def __repr__(self):
        return f"<AdaptationEvent(id={self.id}, trigger_type='{self.trigger_type}', component='{self.component_adapted}')>"

class SystemMetrics(Base):
    """
    Model for tracking overall system performance metrics over time
    """
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Session aggregates
    active_sessions_count = Column(Integer, default=0)
    total_interactions_today = Column(Integer, default=0)
    
    # Performance aggregates
    avg_engagement_score = Column(Float, nullable=True)
    avg_user_ability = Column(Float, nullable=True)
    avg_response_time_ms = Column(Float, nullable=True)
    
    # Agent usage statistics
    teacher_agent_usage_pct = Column(Float, nullable=True)
    companion_agent_usage_pct = Column(Float, nullable=True)
    coach_agent_usage_pct = Column(Float, nullable=True)
    
    # System health indicators
    error_rate_pct = Column(Float, nullable=True)
    adaptation_frequency = Column(Float, nullable=True)
    context_confidence_avg = Column(Float, nullable=True)
    
    # Additional metrics
    metrics_data = Column(JSON, nullable=True)  # Flexible storage for additional metrics
    
    def __repr__(self):
        return f"<SystemMetrics(timestamp='{self.timestamp}', avg_engagement={self.avg_engagement_score})>"

class DatabaseManager:
    """
    Database manager for handling all database operations
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            database_url: Optional database URL (defaults to environment variable)
        """
        self.database_url = database_url or os.environ.get('DATABASE_URL')
        
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=300
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def init_db(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
        
    def create_session(self, session_id: str, user_id: str, session_metadata: Dict[str, Any] = None) -> UserSession:
        """
        Create new user session
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            session_metadata: Optional session metadata
            
        Returns:
            Created UserSession object
        """
        db_session = self.get_session()
        try:
            user_session = UserSession(
                session_id=session_id,
                user_id=user_id,
                session_metadata=session_metadata
            )
            db_session.add(user_session)
            db_session.commit()
            db_session.refresh(user_session)
            return user_session
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
    
    def end_session(self, session_id: str):
        """
        Mark session as ended
        
        Args:
            session_id: Session to end
        """
        db_session = self.get_session()
        try:
            user_session = db_session.query(UserSession).filter(
                UserSession.session_id == session_id
            ).first()
            
            if user_session:
                user_session.ended_at = datetime.utcnow()
                db_session.commit()
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
    
    def save_interaction(self, session_id: str, persona: str, input_data: Dict[str, Any],
                        context_state: Dict[str, Any], response_data: Dict[str, Any],
                        feedback_metrics: Dict[str, Any] = None, **kwargs) -> Interaction:
        """
        Save complete interaction to database
        
        Args:
            session_id: Session identifier
            persona: Active persona/agent
            input_data: Multimodal input data
            context_state: Context state from context hub
            response_data: Response data from agents
            feedback_metrics: Feedback loop metrics
            **kwargs: Additional interaction data
            
        Returns:
            Created Interaction object
        """
        db_session = self.get_session()
        try:
            # Extract key metrics for indexing
            valence_arousal = context_state.get('valence_arousal', {})
            
            interaction = Interaction(
                session_id=session_id,
                multimodal_input=input_data,
                processed_features=kwargs.get('processed_features'),
                persona=persona,
                context_state=context_state,
                valence=valence_arousal.get('valence'),
                arousal=valence_arousal.get('arousal'),
                response_data=response_data,
                selected_agent=kwargs.get('selected_agent'),
                response_strategy=kwargs.get('response_strategy'),
                user_ability_before=kwargs.get('user_ability_before'),
                user_ability_after=kwargs.get('user_ability_after'),
                engagement_score=feedback_metrics.get('engagement') if feedback_metrics else None,
                long_term_engagement=feedback_metrics.get('long_term_engagement') if feedback_metrics else None,
                task_difficulty=kwargs.get('task_difficulty'),
                success_probability=kwargs.get('success_probability'),
                cognitive_load=kwargs.get('cognitive_load'),
                feedback_metrics=feedback_metrics,
                adaptation_signals=kwargs.get('adaptation_signals')
            )
            
            db_session.add(interaction)
            db_session.commit()
            db_session.refresh(interaction)
            return interaction
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
    
    def log_adaptation_event(self, session_id: str, trigger_type: str, component_adapted: str,
                            adaptation_type: str, trigger_data: Dict[str, Any] = None,
                            adaptation_parameters: Dict[str, Any] = None) -> AdaptationEvent:
        """
        Log system adaptation event
        
        Args:
            session_id: Session identifier
            trigger_type: What triggered the adaptation
            component_adapted: Which component was adapted
            adaptation_type: Type of adaptation performed
            trigger_data: Data about adaptation trigger
            adaptation_parameters: Parameters of the adaptation
            
        Returns:
            Created AdaptationEvent object
        """
        db_session = self.get_session()
        try:
            adaptation_event = AdaptationEvent(
                session_id=session_id,
                trigger_type=trigger_type,
                trigger_data=trigger_data,
                component_adapted=component_adapted,
                adaptation_type=adaptation_type,
                adaptation_parameters=adaptation_parameters
            )
            
            db_session.add(adaptation_event)
            db_session.commit()
            db_session.refresh(adaptation_event)
            return adaptation_event
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
    
    def get_session_interactions(self, session_id: str, limit: int = None) -> List[Interaction]:
        """
        Get all interactions for a session
        
        Args:
            session_id: Session identifier
            limit: Optional limit on number of interactions
            
        Returns:
            List of Interaction objects
        """
        db_session = self.get_session()
        try:
            query = db_session.query(Interaction).filter(
                Interaction.session_id == session_id
            ).order_by(Interaction.timestamp.desc())
            
            if limit:
                query = query.limit(limit)
                
            return query.all()
        finally:
            db_session.close()
    
    def get_user_engagement_history(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get user engagement history over specified period
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            
        Returns:
            List of engagement data points
        """
        db_session = self.get_session()
        try:
            # Query interactions with engagement scores
            cutoff_date = datetime.utcnow() - datetime.timedelta(days=days)
            
            interactions = db_session.query(Interaction).join(UserSession).filter(
                UserSession.user_id == user_id,
                Interaction.timestamp >= cutoff_date,
                Interaction.engagement_score.isnot(None)
            ).order_by(Interaction.timestamp.asc()).all()
            
            return [
                {
                    'timestamp': interaction.timestamp,
                    'engagement_score': interaction.engagement_score,
                    'long_term_engagement': interaction.long_term_engagement,
                    'persona': interaction.persona,
                    'user_ability': interaction.user_ability_after
                }
                for interaction in interactions
            ]
        finally:
            db_session.close()
    
    def compute_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of session statistics
        """
        db_session = self.get_session()
        try:
            interactions = self.get_session_interactions(session_id)
            
            if not interactions:
                return {'status': 'no_data'}
            
            # Basic stats
            stats = {
                'total_interactions': len(interactions),
                'session_duration_minutes': (interactions[0].timestamp - interactions[-1].timestamp).total_seconds() / 60,
                'avg_engagement': np.mean([i.engagement_score for i in interactions if i.engagement_score]),
                'final_user_ability': interactions[0].user_ability_after,
                'initial_user_ability': interactions[-1].user_ability_before or 0.5,
            }
            
            # Agent usage
            agent_counts = {}
            for interaction in interactions:
                agent = interaction.selected_agent or 'unknown'
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            stats['agent_usage'] = agent_counts
            
            # Emotional state analysis
            valence_scores = [i.valence for i in interactions if i.valence is not None]
            arousal_scores = [i.arousal for i in interactions if i.arousal is not None]
            
            if valence_scores:
                stats['avg_valence'] = np.mean(valence_scores)
                stats['valence_range'] = [min(valence_scores), max(valence_scores)]
            
            if arousal_scores:
                stats['avg_arousal'] = np.mean(arousal_scores)
                stats['arousal_range'] = [min(arousal_scores), max(arousal_scores)]
            
            # Performance trends
            abilities = [i.user_ability_after for i in interactions if i.user_ability_after]
            if len(abilities) >= 2:
                stats['ability_change'] = abilities[0] - abilities[-1]  # Latest - earliest
                stats['ability_trend'] = 'improving' if stats['ability_change'] > 0.05 else 'declining' if stats['ability_change'] < -0.05 else 'stable'
            
            return stats
            
        finally:
            db_session.close()
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """
        Clean up old data beyond retention period
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        db_session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - datetime.timedelta(days=days_to_keep)
            
            # Delete old sessions and their interactions (cascaded)
            old_sessions = db_session.query(UserSession).filter(
                UserSession.created_at < cutoff_date
            )
            
            deleted_count = old_sessions.count()
            old_sessions.delete()
            
            # Delete old adaptation events
            db_session.query(AdaptationEvent).filter(
                AdaptationEvent.timestamp < cutoff_date
            ).delete()
            
            # Delete old system metrics
            db_session.query(SystemMetrics).filter(
                SystemMetrics.timestamp < cutoff_date
            ).delete()
            
            db_session.commit()
            return deleted_count
            
        except Exception as e:
            db_session.rollback()
            raise e
        finally:
            db_session.close()
    
    def get_system_health_metrics(self) -> Dict[str, Any]:
        """
        Get current system health metrics
        
        Returns:
            System health metrics dictionary
        """
        db_session = self.get_session()
        try:
            # Get recent metrics (last 24 hours)
            cutoff_date = datetime.utcnow() - datetime.timedelta(hours=24)
            
            recent_interactions = db_session.query(Interaction).filter(
                Interaction.timestamp >= cutoff_date
            ).all()
            
            if not recent_interactions:
                return {'status': 'no_recent_data'}
            
            # Compute health metrics
            engagement_scores = [i.engagement_score for i in recent_interactions if i.engagement_score]
            user_abilities = [i.user_ability_after for i in recent_interactions if i.user_ability_after]
            
            # Agent distribution
            agent_counts = {}
            for interaction in recent_interactions:
                agent = interaction.selected_agent or 'unknown'
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            total_interactions = len(recent_interactions)
            
            return {
                'total_recent_interactions': total_interactions,
                'avg_engagement': np.mean(engagement_scores) if engagement_scores else 0,
                'avg_user_ability': np.mean(user_abilities) if user_abilities else 0.5,
                'agent_distribution': {
                    agent: count / total_interactions 
                    for agent, count in agent_counts.items()
                },
                'engagement_std': np.std(engagement_scores) if len(engagement_scores) > 1 else 0,
                'ability_std': np.std(user_abilities) if len(user_abilities) > 1 else 0,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        finally:
            db_session.close()

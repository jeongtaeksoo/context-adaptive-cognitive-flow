"""
Streamlit Web Demo for Context-Adaptive Cognitive Flow System
Interactive visualization of the cognitive rehabilitation framework
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from context_adaptive_cognitive_flow.main import CognitiveFlowSimulator

st.set_page_config(
    page_title="Context-Adaptive Cognitive Flow",
    layout="wide"
)

st.title("Context-Adaptive Cognitive Flow System")
st.markdown("**Cognitive Rehabilitation Framework for Older Adults**")
st.markdown("---")

with st.sidebar:
    st.header("Simulation Parameters")
    
    st.subheader("Performance Target")
    P_star = st.slider(
        "P* (Target Performance Probability)", 
        min_value=0.5, 
        max_value=0.95, 
        value=0.85, 
        step=0.05,
        help="Optimal learning zone target (Eq.3)"
    )
    
    st.subheader("Learning Rate")
    eta = st.slider(
        "η (Difficulty Adjustment Rate)", 
        min_value=0.05, 
        max_value=0.30, 
        value=0.15, 
        step=0.05,
        help="Rate of item difficulty adaptation (Eq.3)"
    )
    
    st.subheader("Initial Conditions")
    theta_0 = st.slider(
        "θ₀ (Initial User Ability)", 
        min_value=0.5, 
        max_value=2.5, 
        value=1.5, 
        step=0.1,
        help="Starting user ability parameter (Eq.2, 4)"
    )
    
    b_0 = st.slider(
        "b₀ (Initial Item Difficulty)", 
        min_value=0.5, 
        max_value=2.5, 
        value=1.2, 
        step=0.1,
        help="Starting item difficulty parameter (Eq.2, 3)"
    )
    
    st.subheader("Simulation Steps")
    num_steps = st.slider(
        "Number of Time Steps", 
        min_value=5, 
        max_value=20, 
        value=10, 
        step=1
    )
    
    st.markdown("---")
    run_button = st.button("Run Simulation", type="primary", use_container_width=True)

if run_button:
    with st.spinner("Running cognitive flow simulation..."):
        simulator = CognitiveFlowSimulator(
            P_star=P_star,
            eta=eta,
            theta_0=theta_0,
            b_0=b_0
        )
        
        results = []
        for t in range(num_steps):
            result = simulator.simulate_step(t)
            results.append(result)
        
        st.success(f"Simulation completed: {num_steps} time steps")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Step-by-Step Results")
            
            for i, result in enumerate(results):
                state = result['state']
                with st.expander(f"**Step {i}**: L_cog={state['L_cog']:.2f}, P_t={state['P_t']:.2f}"):
                    st.markdown(f"**State:**")
                    st.markdown(f"- Cognitive Load: `{state['L_cog']:.3f}`")
                    st.markdown(f"- Performance Prob: `{state['P_t']:.3f}`")
                    st.markdown(f"- Item Difficulty: `{state['b_t']:.3f}`")
                    st.markdown(f"- User Ability: `{state['theta_t']:.3f}`")
                    st.markdown(f"- Response Delay: `{state['delay']:.2f}s`")
                    st.markdown(f"- Emotion: ({state['valence']:+.1f}, {state['arousal']:+.1f})")
                    
                    st.markdown(f"**Agent Messages:**")
                    st.info(f"Teacher: {result['teacher']}")
                    st.success(f"Companion: {result['companion']}")
                    st.warning(f"Coach: {result['coach']}")
        
        with col2:
            st.subheader("Time Evolution Visualization")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Context-Adaptive Cognitive Flow: Time Evolution', 
                         fontsize=14, fontweight='bold')
            
            time_steps = np.arange(len(simulator.history['L_cog']))
            
            ax1 = axes[0, 0]
            ax1.plot(time_steps, simulator.history['L_cog'], 'o-', linewidth=2, 
                     markersize=6, color='#e74c3c', label='L_cog')
            ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Low threshold')
            ax1.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5, label='High threshold')
            ax1.fill_between(time_steps, 0.5, 1.5, alpha=0.2, color='green', label='Optimal zone')
            ax1.set_xlabel('Time Step', fontsize=10)
            ax1.set_ylabel('Cognitive Load', fontsize=10)
            ax1.set_title('Stage II: Cognitive Load (Eq.1)', fontsize=11, fontweight='bold')
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            ax2 = axes[0, 1]
            ax2.plot(time_steps, simulator.history['P_t'], 's-', linewidth=2, 
                     markersize=6, color='#3498db', label='P_t')
            ax2.axhline(y=P_star, color='green', linestyle='--', alpha=0.7, 
                       label=f'Target P*={P_star:.2f}')
            ax2.set_xlabel('Time Step', fontsize=10)
            ax2.set_ylabel('Performance Probability', fontsize=10)
            ax2.set_title('Stage III: Performance Probability (Eq.2)', fontsize=11, fontweight='bold')
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1])
            
            ax3 = axes[1, 0]
            ax3.plot(time_steps, simulator.history['theta_t'], '^-', linewidth=2, 
                     markersize=6, color='#2ecc71', label='θ_t (user ability)')
            ax3.plot(time_steps, simulator.history['b_t'], 'v-', linewidth=2, 
                     markersize=6, color='#f39c12', label='b_t (item difficulty)')
            ax3.set_xlabel('Time Step', fontsize=10)
            ax3.set_ylabel('Parameter Value', fontsize=10)
            ax3.set_title('Stage III-IV: Item Difficulty & User Ability (Eq.3-4)', 
                         fontsize=11, fontweight='bold')
            ax3.legend(loc='best', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            ax4 = axes[1, 1]
            ax4.plot(time_steps, simulator.history['delay'], 'D-', linewidth=2, 
                     markersize=6, color='#9b59b6', label='Response delay')
            ax4.set_xlabel('Time Step', fontsize=10)
            ax4.set_ylabel('Delay (seconds)', fontsize=10)
            ax4.set_title('Adaptive Response Delay', fontsize=11, fontweight='bold')
            ax4.legend(loc='best', fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("Mathematical Equations (LaTeX Specification)")
        
        col_eq1, col_eq2 = st.columns(2)
        
        with col_eq1:
            st.markdown("**Eq.(1): Cognitive Load**")
            st.latex(r"L_{cog} = 0.4 \cdot \frac{\Delta t_{resp}}{\bar{t}_{base}} + 0.35 \cdot e_{rate} + 0.25 \cdot \sigma_{att}^2")
            
            st.markdown("**Eq.(2): Performance Probability**")
            st.latex(r"P_t = \frac{1}{1 + \exp(-a \cdot (\theta_t - b_t))}")
        
        with col_eq2:
            st.markdown("**Eq.(3): Item Difficulty Adaptation**")
            st.latex(r"b_{t+1} = \text{clip}_{[0,3]}(b_t + \eta \cdot (P_t - P^*))")
            
            st.markdown("**Eq.(4): User Ability Update**")
            st.latex(r"\theta_{t+1} = 0.7 \cdot \theta_t + 0.3 \cdot \hat{\theta}_t")

else:
    st.info("Adjust parameters in the sidebar and click **Run Simulation** to start")
    
    st.markdown("### System Architecture")
    
    col_arch1, col_arch2 = st.columns(2)
    
    with col_arch1:
        st.markdown("**4-Stage Processing Pipeline:**")
        st.markdown("- **Stage I:** Multimodal Data Sensing")
        st.markdown("- **Stage II:** Context Recognition (Eq.1)")
        st.markdown("- **Stage III:** Response Strategy (Eq.2-3)")
        st.markdown("- **Stage IV:** Feedback Loop (Eq.4)")
    
    with col_arch2:
        st.markdown("**3 Persona Agents:**")
        st.markdown("- **Teacher:** Difficulty adaptation")
        st.markdown("- **Companion:** Emotional support")
        st.markdown("- **Coach:** Motivation & progress")
    
    st.markdown("---")
    st.markdown("*Based on research in cognitive rehabilitation for older adults*")

"""
Multi-Agent LLM Orchestration for Community Health Worker Allocation

True multi-agent architecture addressing the optimization paradox:
- Safety Agent (DeepSeek-V3): Maximize acute risk detection
- Efficiency Agent (Claude Sonnet 4.5): Optimize resource allocation  
- Equity Agent (GPT-5.2): Ensure demographic fairness

Orchestration mechanisms prevent individual optimization from degrading system performance.

Author: Sanjay Basu, MD, PhD
Date: 2026-01-18
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import asyncio
from dataclasses import dataclass
import re

# LLM API clients
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available, using simulation mode")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic not available, using simulation mode")

# Set random seed for reproducibility
np.random.seed(42)

# Paths
RESULTS_DIR = Path("/Users/sanjaybasu/waymark-local/notebooks/causal_orchestration_mhealth/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("MULTI-AGENT LLM ORCHESTRATION SYSTEM")
print("="*80)

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

@dataclass
class MemberProfile:
    """Patient profile for agent decision-making"""
    member_id: str
    age: int
    sex: str
    race_ethnicity: str
    baseline_risk: float  # 30-day acute care probability
    charlson_score: int
    social_needs: List[str]
    prior_ed_visits: int
    current_symptoms: str  # Free-text narrative

@dataclass
class AgentDecision:
    """Individual agent's decision and reasoning"""
    agent_name: str
    action: str  # "contact" or "no_contact"
    confidence: float  # 0-1
    reasoning: str
    priority_score: float  # Agent's internal objective value

# ============================================================================
# AGENT IMPLEMENTATIONS
# ============================================================================

class SafetyAgent:
    """DeepSeek-V3 optimizing for acute risk detection (maximize sensitivity)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.name = "Safety Agent (DeepSeek-V3)"
        self.objective = "Maximize sensitivity for acute care needs"
        self.api_key = api_key
        # DeepSeek via OpenAI-compatible API
        if api_key and OPENAI_AVAILABLE:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
        else:
            self.client = None
    
    def decide(self, member: MemberProfile) -> AgentDecision:
        """Assess acute risk and recommend contact for any concerning signs"""
        
        prompt = f"""You are a safety-focused clinical AI agent. Your ONLY objective is maximizing detection of patients needing urgent medical attention.

Patient Profile:
- Age: {member.age}, Sex: {member.sex}
- Race/Ethnicity: {member.race_ethnicity}
- 30-day acute care risk: {member.baseline_risk*100:.1f}%
- Comorbidities (Charlson): {member.charlson_score}
- Social needs: {', '.join(member.social_needs)}
- Recent ED visits: {member.prior_ed_visits}
- Current status: {member.current_symptoms}

DECISION: Should community health worker contact this patient immediately?
- Consider: ANY signs of acute decompensation, even subtle
- Your metric: Sensitivity (detecting all who might need care)
- You are NOT concerned with cost, equity, or efficiency

Respond ONLY in this format:
DECISION: [contact/no_contact]
CONFIDENCE: [0.0-1.0]
PRIORITY: [0.0-1.0, where 1.0 = highest acute risk]
REASONING: [2-3 sentences explaining clinical reasoning]"""

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=200
                )
                output = response.choices[0].message.content
            except Exception as e:
                print(f"DeepSeek API error: {e}, using heuristic")
                output = self._heuristic_decision(member)
        else:
            output = self._heuristic_decision(member)
        
        # Parse response
        decision = "contact" if "DECISION: contact" in output else "no_contact"
        confidence = float(re.search(r"CONFIDENCE: ([\d.]+)", output).group(1)) if re.search(r"CONFIDENCE: ([\d.]+)", output) else 0.7
        priority = float(re.search(r"PRIORITY: ([\d.]+)", output).group(1)) if re.search(r"PRIORITY: ([\d.]+)", output) else member.baseline_risk
        reasoning_match = re.search(r"REASONING: (.+?)(?:\n|$)", output, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Safety-focused assessment based on clinical risk factors."
        
        return AgentDecision(
            agent_name=self.name,
            action=decision,
            confidence=confidence,
            reasoning=reasoning,
            priority_score=priority
        )
    
    def _heuristic_decision(self, member: MemberProfile) -> str:
        """Fallback heuristic when API unavailable"""
        # Safety agent errs on side of contact
        high_risk = member.baseline_risk > 0.05 or member.charlson_score >= 3 or member.prior_ed_visits > 0
        decision = "contact" if high_risk else "no_contact"
        confidence = 0.8 if high_risk else 0.6
        priority = member.baseline_risk + 0.1 * member.charlson_score
        
        return f"""DECISION: {decision}
CONFIDENCE: {confidence}
PRIORITY: {min(priority, 1.0)}
REASONING: Clinical risk factors suggest {'immediate' if high_risk else 'routine'} assessment warranted. Baseline probability {member.baseline_risk*100:.1f}% with comorbidity burden and utilization history."""


class EfficiencyAgent:
    """Claude Sonnet 4.5 optimizing for cost-effectiveness (maximize benefit/cost)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.name = "Efficiency Agent (Claude Sonnet 4.5)"
        self.objective = "Maximize cost-effectiveness of resource allocation"
        self.api_key = api_key
        if api_key and ANTHROPIC_AVAILABLE:
            self.client = Anthropic(api_key=api_key)
        else:
            self.client = None
    
    def decide(self, member: MemberProfile, cate_estimate: float) -> AgentDecision:
        """Assess cost-effectiveness - only contact if high predicted benefit"""
        
        prompt = f"""You are an efficiency-focused population health AI agent. Your ONLY objective is maximizing value: benefit per dollar spent.

Patient Profile:
- Age: {member.age}, Sex: {member.sex}
- Race/Ethnicity: {member.race_ethnicity}
- 30-day acute care risk: {member.baseline_risk*100:.1f}%
- Comorbidities: {member.charlson_score}
- Social needs: {', '.join(member.social_needs)}
- Predicted intervention effect (CATE): {cate_estimate*100:.1f}% absolute risk reduction

Economics:
- CHW contact cost: ~$150 per engagement
- ED visit cost: ~$1,200
- Expected value = P(prevent ED) × $1,200 - $150

DECISION: Is contacting this patient cost-effective?
- Consider: Expected benefit vs. cost
- Your metric: Return on investment
- You are NOT concerned with individual safety or fairness

Respond ONLY in this format:
DECISION: [contact/no_contact]
CONFIDENCE: [0.0-1.0]
PRIORITY: [0.0-1.0, where 1.0 = highest cost-effectiveness]
REASONING: [2-3 sentences on economic value]"""

        if self.client:
            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4.5-20250514",
                    max_tokens=200,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                output = response.content[0].text
            except Exception as e:
                print(f"Claude API error: {e}, using heuristic")
                output = self._heuristic_decision(member, cate_estimate)
        else:
            output = self._heuristic_decision(member, cate_estimate)
        
        # Parse response
        decision = "contact" if "DECISION: contact" in output else "no_contact"
        confidence = float(re.search(r"CONFIDENCE: ([\d.]+)", output).group(1)) if re.search(r"CONFIDENCE: ([\d.]+)", output) else 0.7
        priority = float(re.search(r"PRIORITY: ([\d.]+)", output).group(1)) if re.search(r"PRIORITY: ([\d.]+)", output) else max(0, -cate_estimate)
        reasoning_match = re.search(r"REASONING: (.+?)(?:\n|$)", output, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Economic analysis based on expected value calculation."
        
        return AgentDecision(
            agent_name=self.name,
            action=decision,
            confidence=confidence,
            reasoning=reasoning,
            priority_score=priority
        )
    
    def _heuristic_decision(self, member: MemberProfile, cate_estimate: float) -> str:
        """Fallback heuristic"""
        # Efficiency agent only contacts if strong predicted benefit
        expected_value = (-cate_estimate) * member.baseline_risk * 1200 - 150  # Benefit - cost
        cost_effective = expected_value > 0 and cate_estimate < -0.01
        
        decision = "contact" if cost_effective else "no_contact"
        confidence = 0.8 if abs(expected_value) > 100 else 0.6
        priority = max(0, expected_value / 500)  # Normalize to 0-1
        
        return f"""DECISION: {decision}
CONFIDENCE: {confidence}
PRIORITY: {min(priority, 1.0)}
REASONING: Expected value ${expected_value:.0f} ({'positive' if expected_value > 0 else 'negative'} ROI). Predicted effect {cate_estimate*100:.1f}% suggests {'worthwhile' if cost_effective else 'limited'} intervention value."""


class EquityAgent:
    """GPT-5.2 optimizing for fairness (minimize demographic disparit

ies)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.name = "Equity Agent (GPT-5.2)"
        self.objective = "Ensure algorithmic fairness across demographics"
        self.api_key = api_key
        if api_key and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
        # Track allocation rates by demographic
        self.allocation_history = {'White': [], 'Black': [], 'Hispanic': [], 'Asian': [], 'Other': []}
    
    def decide(self, member: MemberProfile) -> AgentDecision:
        """Assess whether contact promotes fairness given allocation history"""
        
        # Current allocation rates by race
        current_rates = {
            race: np.mean(decisions) if len(decisions) > 0 else 0.5 
            for race, decisions in self.allocation_history.items()
        }
        
        prompt = f"""You are a fairness-focused health equity AI agent. Your ONLY objective is ensuring equitable treatment across demographic groups.

Patient Profile:
- Age: {member.age}, Sex: {member.sex}
- Race/Ethnicity: {member.race_ethnicity}
- 30-day acute care risk: {member.baseline_risk*100:.1f}%

Current System Allocation Rates:
- White: {current_rates['White']*100:.1f}%
- Black: {current_rates['Black']*100:.1f}%
- Hispanic: {current_rates['Hispanic']*100:.1f}%
- Asian: {current_rates['Asian']*100:.1f}%
- Other: {current_rates['Other']*100:.1f}%

DECISION: Should we contact this patient to promote equity?
- Consider: Whether this decision reduces demographic disparities
- Your metric: Equalized odds, demographic parity
- You are NOT concerned with individual clinical outcomes or costs

Respond ONLY in this format:
DECISION: [contact/no_contact]
CONFIDENCE: [0.0-1.0]
PRIORITY: [0.0-1.0, where 1.0 = most important for equity]
REASONING: [2-3 sentences on fairness impact]"""

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-5.2",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=200
                )
                output = response.choices[0].message.content
            except Exception as e:
                print(f"GPT-5.2 API error: {e}, using heuristic")
                output = self._heuristic_decision(member, current_rates)
        else:
            output = self._heuristic_decision(member, current_rates)
        
        # Parse response
        decision = "contact" if "DECISION: contact" in output else "no_contact"
        confidence = float(re.search(r"CONFIDENCE: ([\d.]+)", output).group(1)) if re.search(r"CONFIDENCE: ([\d.]+)", output) else 0.7
        priority = float(re.search(r"PRIORITY: ([\d.]+)", output).group(1)) if re.search(r"PRIORITY: ([\d.]+)", output) else 0.5
        reasoning_match = re.search(r"REASONING: (.+?)(?:\n|$)", output, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Fairness assessment based on demographic allocation patterns."
        
        # Update history
        self.allocation_history[member.race_ethnicity].append(1 if decision == "contact" else 0)
        
        return AgentDecision(
            agent_name=self.name,
            action=decision,
            confidence=confidence,
            reasoning=reasoning,
            priority_score=priority
        )
    
    def _heuristic_decision(self, member: MemberProfile, current_rates: Dict[str, float]) -> str:
        """Fallback heuristic"""
        # Equity agent tries to equalize allocation rates
        member_group_rate = current_rates[member.race_ethnicity]
        overall_rate = np.mean(list(current_rates.values()))
        
        # Contact if member's group is under-allocated
        under_allocated = member_group_rate < overall_rate - 0.05
        decision = "contact" if under_allocated else "no_contact"
        confidence = 0.7
        priority = abs(member_group_rate - overall_rate)
        
        return f"""DECISION: {decision}
CONFIDENCE: {confidence}
PRIORITY: {min(priority, 1.0)}
REASONING: {member.race_ethnicity} group currently at {member_group_rate*100:.1f}% allocation vs. {overall_rate*100:.1f}% overall. {'Increasing' if under_allocated else 'Maintaining'} allocation promotes demographic parity."""


# ============================================================================
# ORCHESTRATION MECHANISMS
# ============================================================================

class MultiAgentOrchestrator:
    """Resolves conflicts between agents to prevent optimization paradox"""
    
    def __init__(self, strategy: str = "weighted_voting"):
        self.strategy = strategy
        self.decision_history = []
    
    def orchestrate(self, decisions: List[AgentDecision]) -> Tuple[str, Dict]:
        """Combine agent decisions using specified strategy"""
        
        if self.strategy == "majority_vote":
            return self._majority_vote(decisions)
        elif self.strategy == "weighted_voting":
            return self._weighted_voting(decisions)
        elif self.strategy == "priority_weighted":
            return self._priority_weighted(decisions)
        elif self.strategy == "nash_bargaining":
            return self._nash_bargaining(decisions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _majority_vote(self, decisions: List[AgentDecision]) -> Tuple[str, Dict]:
        """Simple majority vote"""
        contact_votes = sum(1 for d in decisions if d.action == "contact")
        final_action = "contact" if contact_votes >= 2 else "no_contact"
        
        metadata = {
            "strategy": "majority_vote",
            "votes": {d.agent_name: d.action for d in decisions},
            "final_action": final_action
        }
        return final_action, metadata
    
    def _weighted_voting(self, decisions: List[AgentDecision]) -> Tuple[str, Dict]:
        """Vote weighted by agent confidence"""
        contact_weight = sum(d.confidence for d in decisions if d.action == "contact")
        no_contact_weight = sum(d.confidence for d in decisions if d.action == "no_contact")
        
        final_action = "contact" if contact_weight > no_contact_weight else "no_contact"
        
        metadata = {
            "strategy": "weighted_voting",
            "contact_weight": contact_weight,
            "no_contact_weight": no_contact_weight,
            "final_action": final_action
        }
        return final_action, metadata
    
    def _priority_weighted(self, decisions: List[AgentDecision]) -> Tuple[str, Dict]:
        """Weight by agent's internal priority score"""
        contact_priority = sum(d.priority_score for d in decisions if d.action == "contact")
        no_contact_priority = sum(d.priority_score for d in decisions if d.action == "no_contact")
        
        final_action = "contact" if contact_priority > no_contact_priority else "no_contact"
        
        metadata = {
            "strategy": "priority_weighted",
            "contact_priority": contact_priority,
            "no_contact_priority": no_contact_priority,
            "final_action": final_action
        }
        return final_action, metadata
    
    def _nash_bargaining(self, decisions: List[AgentDecision]) -> Tuple[str, Dict]:
        """Nash bargaining solution (simplified)"""
        # Each agent's "utility" from final decision
        # If final action matches their recommendation, utility = priority_score
        # Otherwise, utility = 0
        
        # Try contact
        contact_utilities = [
            d.priority_score if d.action == "contact" else 0.01 
            for d in decisions
        ]
        contact_product = np.prod(contact_utilities)  # Nash product
        
        # Try no_contact  
        no_contact_utilities = [
            d.priority_score if d.action == "no_contact" else 0.01
            for d in decisions
        ]
        no_contact_product = np.prod(no_contact_utilities)
        
        final_action = "contact" if contact_product > no_contact_product else "no_contact"
        
        metadata = {
            "strategy": "nash_bargaining",
            "contact_nash_product": contact_product,
            "no_contact_nash_product": no_contact_product,
            "final_action": final_action
        }
        return final_action, metadata


# ============================================================================
# SIMULATION
# ============================================================================

def generate_synthetic_member(member_id: int, target_trial_data: pd.DataFrame) -> MemberProfile:
    """Generate member profile matching target trial distribution"""
    row = target_trial_data.iloc[member_id % len(target_trial_data)]
    
    # Create narrative
    symptoms = []
    if row['charlson'] >= 3:
        symptoms.append("multiple chronic conditions managed")
    if row['baseline_risk'] > 0.06:
        symptoms.append("recent increase in symptoms")
    if 'housing' in str(row.get('housing_insecurity', 0)):
        symptoms.append("housing instability reported")
    
    symptom_text = ", ".join(symptoms) if symptoms else "stable, routine follow-up"
    
    return MemberProfile(
        member_id=f"M{member_id:06d}",
        age=int(row['age']),
        sex='Female' if row.get('female', 0) == 1 else 'Male',
        race_ethnicity=row['race'],
        baseline_risk=row['baseline_risk'],
        charlson_score=int(row['charlson']),
        social_needs=row['social_needs'].split(',') if isinstance(row.get('social_needs'), str) else ['housing', 'food'],
        prior_ed_visits=np.random.poisson(row['baseline_risk'] * 3),
        current_symptoms=symptom_text
    )


def run_multi_agent_simulation(n_decisions: int = 1000, use_apis: bool = False):
    """Run multi-agent system and compare orchestration strategies"""
    
    print(f"\n{'='*80}")
    print(f"RUNNING MULTI-AGENT SIMULATION (N={n_decisions})")
    print(f"API Mode: {'ENABLED' if use_apis else 'SIMULATION (heuristics)'}")
    print(f"{'='*80}\n")
    
    # Load target trial data
    data_path = RESULTS_DIR / "target_trial_data.csv.gz"
    if data_path.exists():
        target_data = pd.read_csv(data_path)
        print(f"✓ Loaded {len(target_data):,} member profiles from target trial\n")
    else:
        print("Warning: Target trial data not found, using synthetic data\n")
        target_data = pd.DataFrame({
            'age': np.random.gamma(3.2, 13.5, 10000),
            'baseline_risk': np.random.beta(4, 80, 10000),
            'charlson': np.random.poisson(2.5, 10000),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 10000, p=[0.38, 0.31, 0.23, 0.04, 0.04]),
            'female': np.random.binomial(1, 0.64, 10000),
            'social_needs': ['housing,food'] * 10000
        })
    
    # Initialize agents
    safety_agent = SafetyAgent(api_key=os.getenv('DEEPSEEK_API_KEY') if use_apis else None)
    efficiency_agent = EfficiencyAgent(api_key=os.getenv('ANTHROPIC_API_KEY') if use_apis else None)
    equity_agent = EquityAgent(api_key=os.getenv('OPENAI_API_KEY') if use_apis else None)
    
    # Test different orchestration strategies
    strategies = ["majority_vote", "weighted_voting", "priority_weighted", "nash_bargaining"]
    results = {strategy: [] for strategy in strategies}
    
    print("Processing decisions...")
    for i in range(min(n_decisions, len(target_data))):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_decisions} decisions processed...")
        
        # Generate member
        member = generate_synthetic_member(i, target_data)
        
        # Simulate CATE estimate (from X-Learner)
        cate_estimate = target_data.iloc[i % len(target_data)].get('true_cate', -0.02)
        
        # Get each agent's decision
        safety_decision = safety_agent.decide(member)
        efficiency_decision = efficiency_agent.decide(member, cate_estimate)
        equity_decision = equity_agent.decide(member)
        
        agent_decisions = [safety_decision, efficiency_decision, equity_decision]
        
        # Test each orchestration strategy
        for strategy in strategies:
            orchestrator = MultiAgentOrchestrator(strategy=strategy)
            final_action, metadata = orchestrator.orchestrate(agent_decisions)
            
            results[strategy].append({
                'member_id': member.member_id,
                'final_action': final_action,
                'safety_action': safety_decision.action,
                'efficiency_action': efficiency_decision.action,
                'equity_action': equity_decision.action,
                'safety_priority': safety_decision.priority_score,
                'efficiency_priority': efficiency_decision.priority_score,
                'equity_priority': equity_decision.priority_score,
                **metadata
            })
    
    print(f"\n✓ Completed {n_decisions} decisions across {len(strategies)} orchestration strategies\n")
    
    # Analyze results
    print("="*80)
    print("ORCHESTRATION STRATEGY COMPARISON")
    print("="*80)
    
    for strategy in strategies:
        df = pd.DataFrame(results[strategy])
        contact_rate = (df['final_action'] == 'contact').mean()
        
        # Agent agreement
        unanimous = ((df['safety_action'] == df['efficiency_action']) & 
                     (df['efficiency_action'] == df['equity_action'])).mean()
        
        # Conflict resolution (when agents disagree)
        conflicts = df[~((df['safety_action'] == df['efficiency_action']) & 
                         (df['efficiency_action'] == df['equity_action']))]
        conflict_rate = len(conflicts) / len(df)
        
        print(f"\n{strategy.upper().replace('_', ' ')}:")
        print(f"  Contact rate: {contact_rate*100:.1f}%")
        print(f"  Agent unanimous: {unanimous*100:.1f}%")
        print(f"  Conflicts resolved: {len(conflicts)} ({conflict_rate*100:.1f}%)")
        
        if len(conflicts) > 0:
            safety_wins = (conflicts['final_action'] == conflicts['safety_action']).mean()
            efficiency_wins = (conflicts['final_action'] == conflicts['efficiency_action']).mean()
            equity_wins = (conflicts['final_action'] == conflicts['equity_action']).mean()
            print(f"  Conflict resolution: Safety {safety_wins*100:.0f}%, Efficiency {efficiency_wins*100:.0f}%, Equity {equity_wins*100:.0f}%")
    
    # Save results
    output_path = RESULTS_DIR / "multi_agent_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'strategies': {
                strategy: pd.DataFrame(results[strategy]).to_dict('records')
                for strategy in strategies
            },
            'summary': {
                strategy: {
                    'contact_rate': (pd.DataFrame(results[strategy])['final_action'] == 'contact').mean(),
                    'n_decisions': len(results[strategy])
                }
                for strategy in strategies
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run simulation
    # Set use_apis=True to use real LLM APIs (requires API keys in environment)
    # Set use_apis=False for fast heuristic simulation
    
    USE_REAL_APIS = False  # Change to True when API keys configured
    
    results = run_multi_agent_simulation(
        n_decisions=1000,
        use_apis=USE_REAL_APIS
    )
    
    print("\n" + "="*80)
    print("MULTI-AGENT ORCHESTRATION COMPLETE")
    print("="*80)
    print("\nKey Finding: Orchestration strategies resolve agent conflicts")
    print("Next step: Embed this in MRT to validate which strategy optimizes outcomes")

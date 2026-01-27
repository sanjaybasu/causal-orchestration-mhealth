"""
Conversational Behavior Analysis Module
Implements LLM-as-judge classification for agent interactions inspired by
Kim et al. 2026 "Reasoning Models Generate Societies of Thought"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json

try:
    from litellm import completion
except ImportError:
    completion = None


class ConversationalBehaviorAnalyzer:
    """
    Analyzes agent interaction transcripts for conversational behaviors
    characteristic of effective reasoning in multi-agent systems.
    """

    def __init__(self, judge_model: str = "gemini/gemini-2.0-flash-exp"):
        """
        Initialize analyzer with LLM-as-judge model.

        Args:
            judge_model: Model to use for classification (default: Gemini 2.0 Flash)
        """
        self.judge_model = judge_model

    def analyze_conversational_behaviors(self, transcript: str) -> Dict[str, bool]:
        """
        Classify transcript for four key conversational behaviors.

        Based on Kim et al. 2026 framework:
        1. Question-Answering: Posing and resolving questions
        2. Perspective Shifts: Exploring alternative viewpoints
        3. Conflicts of Perspectives: Sharp contrasts between viewpoints
        4. Reconciliation: Integration of conflicting views

        Args:
            transcript: Agent interaction transcript (critiques + refinements)

        Returns:
            Dictionary with boolean indicators for each behavior
        """
        prompt = f"""You are analyzing a multi-agent AI conversation for specific reasoning behaviors.

TRANSCRIPT:
{transcript}

TASK: Identify whether each of the following behaviors is present (respond with JSON):

1. QUESTION_ANSWERING: Does one agent pose a clinical question and another agent (or the same agent later) resolve it?
   Example: "But what about the creatinine trend?" ... "The creatinine actually stabilized at 1.2"

2. PERSPECTIVE_SHIFT: Do agents explore alternative clinical viewpoints or approaches?
   Example: "However, from an efficiency standpoint..." or "Looking at this from a social work lens..."

3. CONFLICT_OF_PERSPECTIVES: Do agents sharply contrast or disagree with each other's clinical approaches?
   Example: "This plan is too verbose and will never be read" vs "We cannot omit guideline-recommended steps"

4. RECONCILIATION: Do agents integrate or synthesize conflicting viewpoints into a coherent solution?
   Example: "We can address both concerns by consolidating the diabetes actions into one bullet"

Respond ONLY with valid JSON in this exact format:
{{
  "question_answering": true/false,
  "perspective_shift": true/false,
  "conflict_of_perspectives": true/false,
  "reconciliation": true/false
}}"""

        try:
            response = completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result_text = response.choices[0].message.content
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text.strip())
            return result

        except Exception as e:
            print(f"Error in conversational behavior analysis: {e}")
            return {
                "question_answering": False,
                "perspective_shift": False,
                "conflict_of_perspectives": False,
                "reconciliation": False
            }

    def analyze_socioemotional_roles(self, transcript: str) -> Dict[str, bool]:
        """
        Classify transcript using Bales' Interaction Process Analysis framework.

        Returns indicators for 4 higher-level role categories:
        - Asking (for orientation, opinion, suggestion)
        - Giving (orientation, opinion, suggestion)
        - Negative emotional (disagreement, antagonism, tension)
        - Positive emotional (agreement, solidarity, tension release)

        Args:
            transcript: Agent interaction transcript

        Returns:
            Dictionary with boolean indicators for each role category
        """
        prompt = f"""You are analyzing a multi-agent conversation using Bales' Interaction Process Analysis framework.

TRANSCRIPT:
{transcript}

TASK: Identify whether each role category is present (respond with JSON):

1. ASKING: Does any agent ask for orientation, opinions, or suggestions?
   Example: "What should we do about the rising creatinine?" or "Should we include transportation?"

2. GIVING: Does any agent give orientation, opinions, or suggestions?
   Example: "The patient needs HbA1c monitoring" or "I recommend consolidating the diabetes section"

3. NEGATIVE_EMOTIONAL: Does any agent express disagreement, antagonism, or tension?
   Example: "This approach is unsafe" or "That recommendation contradicts guidelines"

4. POSITIVE_EMOTIONAL: Does any agent express agreement, solidarity, or tension release?
   Example: "Good point" or "That's a reasonable compromise"

Respond ONLY with valid JSON:
{{
  "asking": true/false,
  "giving": true/false,
  "negative_emotional": true/false,
  "positive_emotional": true/false
}}"""

        try:
            response = completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result_text = response.choices[0].message.content
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text.strip())
            return result

        except Exception as e:
            print(f"Error in socioemotional role analysis: {e}")
            return {
                "asking": False,
                "giving": False,
                "negative_emotional": False,
                "positive_emotional": False
            }

    def calculate_role_balance(self, roles: Dict[str, bool]) -> Dict[str, float]:
        """
        Calculate Jaccard index for role balance (both sides of role pair present).

        Args:
            roles: Dictionary of role indicators

        Returns:
            Jaccard indices for asking & giving, positive & negative
        """
        # Jaccard = (both present) / (either present)
        ask_give_jaccard = 0.0
        pos_neg_jaccard = 0.0

        if roles.get('asking') or roles.get('giving'):
            ask_give_jaccard = 1.0 if (roles.get('asking') and roles.get('giving')) else 0.0

        if roles.get('positive_emotional') or roles.get('negative_emotional'):
            pos_neg_jaccard = 1.0 if (roles.get('positive_emotional') and roles.get('negative_emotional')) else 0.0

        return {
            'ask_give_jaccard': ask_give_jaccard,
            'positive_negative_jaccard': pos_neg_jaccard
        }


class PersonalityDiversityAnalyzer:
    """
    Analyzes personality diversity using BFI-10 (Big Five Inventory) framework.
    """

    def __init__(self, judge_model: str = "gemini/gemini-2.0-flash-exp"):
        self.judge_model = judge_model

    def analyze_agent_personality(self, agent_name: str, agent_critiques: List[str]) -> Dict[str, float]:
        """
        Assess agent personality using BFI-10 framework.

        Big Five dimensions (1-5 scale):
        - Extraversion: Talkative, outgoing vs reserved, quiet
        - Agreeableness: Sympathetic, warm vs critical, quarrelsome
        - Conscientiousness: Dependable, organized vs disorganized, careless
        - Neuroticism: Anxious, moody vs calm, emotionally stable
        - Openness: Creative, imaginative vs conventional, uncreative

        Args:
            agent_name: Name of agent (Safety, Efficiency, Equity)
            agent_critiques: List of critique texts from this agent

        Returns:
            Dictionary with scores for each Big Five dimension
        """
        sample_critiques = "\n\n".join(agent_critiques[:5])  # Use first 5 examples

        prompt = f"""You are a personality psychologist assessing an AI agent's personality based on its communication style.

AGENT: {agent_name}

SAMPLE CRITIQUES:
{sample_critiques}

TASK: Rate this agent's personality on the Big Five dimensions (1-5 scale).

Consider:
- Extraversion: How assertive, energetic, and talkative is the communication?
- Agreeableness: How cooperative, sympathetic vs critical and challenging?
- Conscientiousness: How thorough, detail-oriented, and organized?
- Neuroticism: How much worry, tension, or emotional reactivity is expressed?
- Openness: How creative, open to alternatives vs conventional and focused?

Respond ONLY with valid JSON:
{{
  "extraversion": <1-5>,
  "agreeableness": <1-5>,
  "conscientiousness": <1-5>,
  "neuroticism": <1-5>,
  "openness": <1-5>,
  "explanation": "<brief explanation>"
}}"""

        try:
            response = completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result_text = response.choices[0].message.content
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text.strip())
            return {
                'extraversion': float(result.get('extraversion', 3.0)),
                'agreeableness': float(result.get('agreeableness', 3.0)),
                'conscientiousness': float(result.get('conscientiousness', 3.0)),
                'neuroticism': float(result.get('neuroticism', 3.0)),
                'openness': float(result.get('openness', 3.0)),
                'explanation': result.get('explanation', '')
            }

        except Exception as e:
            print(f"Error in personality analysis: {e}")
            return {
                'extraversion': 3.0,
                'agreeableness': 3.0,
                'conscientiousness': 3.0,
                'neuroticism': 3.0,
                'openness': 3.0,
                'explanation': 'Error in analysis'
            }

    def calculate_personality_diversity(self, agent_personalities: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate standard deviation across agents for each personality dimension.

        Args:
            agent_personalities: List of personality dictionaries (one per agent)

        Returns:
            Standard deviations for each Big Five dimension
        """
        dimensions = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']
        diversity = {}

        for dim in dimensions:
            scores = [agent[dim] for agent in agent_personalities]
            diversity[f'{dim}_sd'] = np.std(scores)

        return diversity


class ExpertiseDiversityAnalyzer:
    """
    Analyzes domain expertise diversity using semantic embeddings.
    """

    def __init__(self, judge_model: str = "gemini/gemini-2.0-flash-exp",
                 embedding_model: str = "text-embedding-3-large"):
        self.judge_model = judge_model
        self.embedding_model = embedding_model

    def extract_expertise_description(self, agent_name: str, agent_critiques: List[str]) -> str:
        """
        Extract a concise description of agent's domain expertise.

        Args:
            agent_name: Name of agent
            agent_critiques: Sample critiques from agent

        Returns:
            Brief expertise description string
        """
        sample_critiques = "\n\n".join(agent_critiques[:5])

        prompt = f"""Based on these sample critiques from the {agent_name} agent, write a concise description (1-2 sentences) of this agent's domain expertise and knowledge focus.

SAMPLE CRITIQUES:
{sample_critiques}

Expertise description:"""

        try:
            response = completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error extracting expertise: {e}")
            return f"{agent_name} agent expertise"

    def calculate_expertise_diversity(self, expertise_descriptions: List[str]) -> float:
        """
        Calculate mean pairwise cosine distance between expertise embeddings.

        Args:
            expertise_descriptions: List of expertise description strings

        Returns:
            Mean cosine distance (0-1, higher = more diverse)
        """
        if not completion:
            return 0.0

        try:
            # Get embeddings for each expertise description
            embeddings = []
            for desc in expertise_descriptions:
                response = completion(
                    model=self.embedding_model,
                    input=desc
                )
                embeddings.append(np.array(response.data[0].embedding))

            # Calculate pairwise cosine distances
            distances = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    # Cosine distance = 1 - cosine similarity
                    cos_sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    distances.append(1.0 - cos_sim)

            return np.mean(distances) if distances else 0.0

        except Exception as e:
            print(f"Error calculating expertise diversity: {e}")
            return 0.0


def analyze_full_transcript_sample(transcripts: List[Dict],
                                   sample_size: int = 500) -> pd.DataFrame:
    """
    Run complete conversational analysis on a sample of transcripts.

    Args:
        transcripts: List of dicts with keys 'strategy', 'patient_id', 'transcript',
                     'safety_critique', 'efficiency_critique', 'equity_critique'
        sample_size: Number of transcripts to analyze

    Returns:
        DataFrame with all conversational metrics
    """
    behavior_analyzer = ConversationalBehaviorAnalyzer()
    personality_analyzer = PersonalityDiversityAnalyzer()
    expertise_analyzer = ExpertiseDiversityAnalyzer()

    results = []

    # Sample transcripts
    sampled = np.random.choice(transcripts, min(sample_size, len(transcripts)), replace=False)

    for i, item in enumerate(sampled):
        print(f"Analyzing transcript {i+1}/{len(sampled)}...")

        # Conversational behaviors
        behaviors = behavior_analyzer.analyze_conversational_behaviors(item['transcript'])

        # Socioemotional roles
        roles = behavior_analyzer.analyze_socioemotional_roles(item['transcript'])
        role_balance = behavior_analyzer.calculate_role_balance(roles)

        result = {
            'strategy': item['strategy'],
            'patient_id': item.get('patient_id', i),
            **behaviors,
            **roles,
            **role_balance
        }

        # For Nash system, calculate personality and expertise diversity
        if item['strategy'] == 'nash' and all(k in item for k in ['safety_critique', 'efficiency_critique', 'equity_critique']):
            # Personality diversity
            safety_personality = personality_analyzer.analyze_agent_personality(
                'Safety Agent', [item['safety_critique']]
            )
            efficiency_personality = personality_analyzer.analyze_agent_personality(
                'Efficiency Agent', [item['efficiency_critique']]
            )
            equity_personality = personality_analyzer.analyze_agent_personality(
                'Equity Agent', [item['equity_critique']]
            )

            personality_diversity = personality_analyzer.calculate_personality_diversity([
                safety_personality, efficiency_personality, equity_personality
            ])
            result.update(personality_diversity)

            # Expertise diversity
            safety_expertise = expertise_analyzer.extract_expertise_description(
                'Safety Agent', [item['safety_critique']]
            )
            efficiency_expertise = expertise_analyzer.extract_expertise_description(
                'Efficiency Agent', [item['efficiency_critique']]
            )
            equity_expertise = expertise_analyzer.extract_expertise_description(
                'Equity Agent', [item['equity_critique']]
            )

            expertise_div = expertise_analyzer.calculate_expertise_diversity([
                safety_expertise, efficiency_expertise, equity_expertise
            ])
            result['expertise_diversity'] = expertise_div

        results.append(result)

    return pd.DataFrame(results)

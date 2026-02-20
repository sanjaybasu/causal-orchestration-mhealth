
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "results" / "generative_analysis_results.json"
SUBMISSION_DIR = Path("/Users/sanjaybasu/waymark-local/notebooks/causal_orchestration_mhealth_submission")
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

def generate_radar_chart():
    # Load Results
    results_file = BASE_DIR / "results" / "real_analysis_results.json"
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    categories = ['Safety', 'Efficiency', 'Equity']
    
    # strategies to plot
    strategies = {
        'Safety Agent Only': '#d62728', # Red
        'Efficiency Agent Only': '#1f77b4', # Blue
        'Equity Agent Only': '#2ca02c', # Green
        'Multi-Agent (Nash)': '#9467bd' # Purple
        # Skipping Control for clarity or plot usually handles 3-4 best
    }
    
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories), endpoint=False)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for name, color in strategies.items():
        if name not in data:
            print(f"Warning: {name} not found in results.")
            continue
            
        values = [
            data[name]['Safety Score'],
            data[name]['Efficiency Score'],
            data[name]['Equity Score']
        ]
        # Close the loop
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((label_loc, [label_loc[0]]))
        
        ax.plot(angles, values, label=name, color=color, linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=color)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(label_loc)
    ax.set_xticklabels(categories, fontsize=14, weight='bold')
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.0)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    plt.title('Performance of Generative Strategies by Domain', size=16, weight='bold', y=1.1)
    
    # Save to Submission Folder
    output_path = SUBMISSION_DIR / "Figure2.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    generate_radar_chart()

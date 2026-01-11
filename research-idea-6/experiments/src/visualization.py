"""
Visualization Utilities for Model Diffing

This module provides tools for creating visualizations and dashboards
to explore Cross-Coder latents and Model Diffing results.

Key visualizations:
1. Latent activation heatmaps
2. Chat vs Base contribution plots
3. Differential activation dashboards
4. Token-level latent activation plots
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

# Conditional imports for visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_latent_dashboard(
    analysis_results: Dict,
    output_path: Optional[str] = None,
    title: str = "Model Diffing Dashboard",
) -> Optional[str]:
    """
    Create an interactive dashboard for exploring latent analysis results.

    Args:
        analysis_results: Dictionary from LatentAnalyzer.find_user_model_features()
        output_path: Path to save HTML dashboard (optional)
        title: Dashboard title

    Returns:
        HTML string of the dashboard if plotly available, else None
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not installed. Install with: pip install plotly")
        return None

    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Top User Model Candidates",
            "Latent Activation Distribution",
            "Chat vs Base Contribution",
            "Consistency Across Prompt Pairs",
        ),
        specs=[
            [{"type": "bar"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
    )

    candidates = analysis_results.get("user_model_candidates", [])

    if candidates:
        # 1. Top candidates bar chart
        latent_indices = [c["latent_idx"] for c in candidates[:15]]
        mean_diffs = [c["mean_diff"] for c in candidates[:15]]

        fig.add_trace(
            go.Bar(
                x=latent_indices,
                y=mean_diffs,
                name="Mean Diff (Novice - Expert)",
                marker_color="steelblue",
            ),
            row=1, col=1,
        )

        # 2. Distribution of all latent differences
        all_diffs = analysis_results.get("all_latent_diffs", {})
        all_means = [v["mean"] for v in all_diffs.values()]

        fig.add_trace(
            go.Histogram(
                x=all_means,
                nbinsx=50,
                name="Mean Diff Distribution",
                marker_color="coral",
            ),
            row=1, col=2,
        )

        # 3. Consistency scatter (mean vs std)
        n_pairs = [c["n_pairs"] for c in candidates[:15]]
        std_diffs = [c.get("std_diff", 0) for c in candidates[:15]]

        fig.add_trace(
            go.Scatter(
                x=mean_diffs,
                y=std_diffs,
                mode="markers+text",
                text=[str(idx) for idx in latent_indices],
                textposition="top center",
                marker=dict(
                    size=[n * 5 for n in n_pairs],
                    color=mean_diffs,
                    colorscale="Viridis",
                ),
                name="Latent Consistency",
            ),
            row=2, col=1,
        )

        # 4. Number of pairs per latent
        fig.add_trace(
            go.Bar(
                x=latent_indices,
                y=n_pairs,
                name="# Prompt Pairs",
                marker_color="forestgreen",
            ),
            row=2, col=2,
        )

    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        showlegend=True,
    )

    fig.update_xaxes(title_text="Latent Index", row=1, col=1)
    fig.update_yaxes(title_text="Mean Difference", row=1, col=1)
    fig.update_xaxes(title_text="Mean Difference", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Mean Difference", row=2, col=1)
    fig.update_yaxes(title_text="Std Difference", row=2, col=1)
    fig.update_xaxes(title_text="Latent Index", row=2, col=2)
    fig.update_yaxes(title_text="# Pairs", row=2, col=2)

    if output_path:
        fig.write_html(output_path)
        print(f"Dashboard saved to {output_path}")

    return fig.to_html()


def plot_latent_heatmap(
    latent_activations: np.ndarray,
    tokens: List[str],
    top_k_latents: int = 50,
    output_path: Optional[str] = None,
    title: str = "Latent Activations Across Tokens",
):
    """
    Create a heatmap of latent activations across token positions.

    Args:
        latent_activations: Array of shape [seq_len, n_latents]
        tokens: List of token strings
        top_k_latents: Number of top latents to show
        output_path: Path to save figure
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed. Install with: pip install matplotlib")
        return

    # Find top-k most active latents
    mean_acts = latent_activations.mean(axis=0)
    top_indices = np.argsort(mean_acts)[-top_k_latents:][::-1]

    # Extract subset
    subset = latent_activations[:, top_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    if SEABORN_AVAILABLE:
        sns.heatmap(
            subset.T,
            xticklabels=tokens if len(tokens) <= 50 else False,
            yticklabels=[f"L{i}" for i in top_indices],
            cmap="viridis",
            ax=ax,
        )
    else:
        im = ax.imshow(subset.T, aspect="auto", cmap="viridis")
        plt.colorbar(im, ax=ax)
        if len(tokens) <= 50:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels([f"L{i}" for i in top_indices])

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Latent Index")
    ax.set_title(title)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    plt.show()


def plot_chat_vs_base_contributions(
    latent_info: List[Dict],
    output_path: Optional[str] = None,
    title: str = "Chat vs Base Latent Contributions",
):
    """
    Plot the relative contribution of each latent to Chat vs Base reconstructions.

    Args:
        latent_info: List of latent info dictionaries with chat_ratio, etc.
        output_path: Path to save figure
        title: Plot title
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed")
        return

    indices = [info["index"] for info in latent_info]
    chat_ratios = [info["chat_ratio"] for info in latent_info]
    activations = [info.get("mean_activation", 0) for info in latent_info]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Chat ratio distribution
    colors = ["steelblue" if r > 0.7 else "coral" if r < 0.3 else "gray" for r in chat_ratios]
    axes[0].bar(range(len(indices)), chat_ratios, color=colors)
    axes[0].axhline(y=0.5, color="black", linestyle="--", alpha=0.5)
    axes[0].axhline(y=0.7, color="steelblue", linestyle=":", alpha=0.5, label="Chat-specific threshold")
    axes[0].axhline(y=0.3, color="coral", linestyle=":", alpha=0.5, label="Base-specific threshold")
    axes[0].set_xlabel("Latent Index")
    axes[0].set_ylabel("Chat Ratio")
    axes[0].set_title("Latent Chat/Base Balance")
    axes[0].legend()

    # Plot 2: Activation vs Chat Ratio scatter
    scatter = axes[1].scatter(chat_ratios, activations, c=chat_ratios, cmap="coolwarm", alpha=0.6)
    axes[1].set_xlabel("Chat Ratio")
    axes[1].set_ylabel("Mean Activation")
    axes[1].set_title("Activation Strength vs Chat Specificity")
    plt.colorbar(scatter, ax=axes[1], label="Chat Ratio")

    # Add annotations for top points
    sorted_by_act = sorted(zip(chat_ratios, activations, indices), key=lambda x: x[1], reverse=True)
    for ratio, act, idx in sorted_by_act[:5]:
        axes[1].annotate(f"L{idx}", (ratio, act), fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    plt.show()


def plot_differential_analysis(
    diff_results: Dict,
    output_path: Optional[str] = None,
):
    """
    Plot results of differential analysis between two prompt types.

    Args:
        diff_results: Results from LatentAnalyzer.differential_analysis()
        output_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not installed")
        return

    label_a = diff_results["label_a"]
    label_b = diff_results["label_b"]
    top_a = diff_results[f"top_in_{label_a}"]
    top_b = diff_results[f"top_in_{label_b}"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top in A
    indices_a = [e["latent_idx"] for e in top_a[:15]]
    scores_a = [e["diff_score"] for e in top_a[:15]]
    axes[0].barh(range(len(indices_a)), scores_a, color="steelblue")
    axes[0].set_yticks(range(len(indices_a)))
    axes[0].set_yticklabels([f"Latent {i}" for i in indices_a])
    axes[0].set_xlabel("Differential Score")
    axes[0].set_title(f"Top Latents More Active in '{label_a}'")
    axes[0].invert_yaxis()

    # Top in B
    indices_b = [e["latent_idx"] for e in top_b[:15]]
    scores_b = [e["diff_score"] for e in top_b[:15]]
    axes[1].barh(range(len(indices_b)), scores_b, color="coral")
    axes[1].set_yticks(range(len(indices_b)))
    axes[1].set_yticklabels([f"Latent {i}" for i in indices_b])
    axes[1].set_xlabel("Differential Score")
    axes[1].set_title(f"Top Latents More Active in '{label_b}'")
    axes[1].invert_yaxis()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()


def create_sycophancy_report(
    sycophancy_results: Dict,
    user_model_results: Dict,
    output_path: str,
):
    """
    Create a comprehensive HTML report of sycophancy analysis.

    Args:
        sycophancy_results: Results from SycophancyDetector.identify_sycophancy_features()
        user_model_results: Results from LatentAnalyzer.find_user_model_features()
        output_path: Path to save HTML report
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sycophancy Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; }}
            .card {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; }}
            .highlight {{ background: #e8f4f8; padding: 10px; border-left: 4px solid #3498db; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background: #3498db; color: white; }}
            tr:nth-child(even) {{ background: #f2f2f2; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        </style>
    </head>
    <body>
        <h1>Sycophancy & User Modeling Analysis Report</h1>

        <div class="card">
            <h2>Executive Summary</h2>
            <p>This report identifies Chat-specific latents that may be related to:</p>
            <ul>
                <li><strong>User Modeling</strong>: Features that encode the model's inference about user attributes</li>
                <li><strong>Sycophancy</strong>: Features that trigger validating user beliefs over accuracy</li>
            </ul>
        </div>

        <div class="card">
            <h2>User Modeling Features</h2>
            <p class="highlight">{user_model_results.get('interpretation', 'No interpretation available')}</p>

            <h3>Top Candidate Latents</h3>
            <table>
                <tr>
                    <th>Latent Index</th>
                    <th>Mean Diff (Novice - Expert)</th>
                    <th># Prompt Pairs</th>
                    <th>Interpretation</th>
                </tr>
    """

    for candidate in user_model_results.get("user_model_candidates", [])[:10]:
        html += f"""
                <tr>
                    <td>{candidate['latent_idx']}</td>
                    <td>{candidate['mean_diff']:.4f}</td>
                    <td>{candidate['n_pairs']}</td>
                    <td>{candidate.get('interpretation', 'TBD')}</td>
                </tr>
        """

    html += """
            </table>
        </div>

        <div class="card">
            <h2>Sycophancy-Related Features</h2>
            <p class="highlight">{}</p>

            <h3>Top Candidate Latents</h3>
            <table>
                <tr>
                    <th>Latent Index</th>
                    <th>Mean Diff (Biased - Neutral)</th>
                    <th># Prompts</th>
                    <th>Interpretation</th>
                </tr>
    """.format(sycophancy_results.get('interpretation', 'No interpretation available'))

    for candidate in sycophancy_results.get("sycophancy_candidates", [])[:10]:
        html += f"""
                <tr>
                    <td>{candidate['latent_idx']}</td>
                    <td>{candidate['mean_diff']:.4f}</td>
                    <td>{candidate['n_prompts']}</td>
                    <td>{candidate.get('interpretation', 'TBD')}</td>
                </tr>
        """

    html += """
            </table>
        </div>

        <div class="card">
            <h2>Next Steps</h2>
            <ol>
                <li>Inspect top latents on Neuronpedia or similar tools</li>
                <li>Verify Chat-specificity by checking Base model activations</li>
                <li>Run causal intervention experiments (clamping latents)</li>
                <li>Test steering: Can we reduce sycophancy by ablating these latents?</li>
            </ol>
        </div>

        <div class="card">
            <h2>Methodology</h2>
            <p>
                This analysis uses <strong>Model Diffing</strong> with Cross-Coders to compare
                Gemma 2 2B (Base) vs Gemma 2 2B-IT (Chat) models. The Cross-Coder separates
                features into Shared, Base-specific, and Chat-specific categories.
            </p>
            <p>
                Chat-specific features are hypothesized to include user modeling and sycophancy
                behaviors introduced during RLHF training.
            </p>
        </div>

    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Report saved to {output_path}")


def save_analysis_json(
    results: Dict,
    output_path: str,
):
    """
    Save analysis results to JSON for later use.

    Args:
        results: Analysis results dictionary
        output_path: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj

    json_safe = convert_for_json(results)

    with open(output_path, "w") as f:
        json.dump(json_safe, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    print("Visualization module loaded!")
    print(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    print(f"Seaborn available: {SEABORN_AVAILABLE}")
    print(f"Plotly available: {PLOTLY_AVAILABLE}")

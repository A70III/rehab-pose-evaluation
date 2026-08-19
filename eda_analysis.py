import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_and_analyze_data():
    """Load and perform comprehensive EDA on the UCO Physical Rehabilitation dataset results"""
    
    # Load the batch evaluation summary
    data_path = Path("output_comparison_results/batch_evaluation_summary.csv")
    df = pd.read_csv(data_path)
    
    print("=== UCO Physical Rehabilitation Dataset - Exploratory Data Analysis ===\n")
    
    # Basic dataset information
    print("1. Dataset Overview:")
    print(f"   - Total videos analyzed: {len(df)}")
    print(f"   - Unique exercises: {df['exercise'].nunique()}")
    print(f"   - Exercise types: {', '.join(sorted(df['exercise_name'].unique()))}")
    print(f"   - Camera angles: {', '.join(sorted(df['video'].unique()))}")
    print(f"   - Body regions: {', '.join(df['region'].unique())}")
    print(f"   - Positions: {', '.join(df['position'].unique())}")
    print(f"   - Sides: {', '.join(df['side'].unique())}")
    
    # Exercise distribution
    print("\n2. Exercise Distribution:")
    exercise_counts = df.groupby(['exercise', 'exercise_name']).size().reset_index(name='count')
    for _, row in exercise_counts.iterrows():
        print(f"   - Exercise {row['exercise']:>2}: {row['exercise_name']:<50} ({row['count']} videos)")
    
    # Model performance comparison
    print("\n3. Model Performance Summary:")
    models = ['YOLOv8', 'MoveNet', 'MediaPipe']
    
    for model in models:
        valid_col = f"{model}_valid_pct"
        mae_col = f"{model}_mae"
        
        avg_valid = df[valid_col].mean()
        avg_mae = df[mae_col].mean()
        
        print(f"   - {model:<12}: Avg Valid Detection = {avg_valid:6.1f}%, Avg MAE = {avg_mae:6.2f}°")
    
    return df

def create_visualizations(df):
    """Create comprehensive visualizations for the EDA"""
    
    # Set up the figure layout
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Model Performance Comparison - Detection Rate
    plt.subplot(4, 3, 1)
    models = ['YOLOv8', 'MoveNet', 'MediaPipe']
    detection_rates = [df[f"{model}_valid_pct"].mean() for model in models]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = plt.bar(models, detection_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Average Detection Rate by Model', fontsize=14, fontweight='bold')
    plt.ylabel('Detection Rate (%)')
    plt.ylim(0, 105)
    
    # Add value labels on bars
    for bar, rate in zip(bars, detection_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Model Performance Comparison - MAE
    plt.subplot(4, 3, 2)
    mae_values = [df[f"{model}_mae"].mean() for model in models]
    
    bars = plt.bar(models, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Average Mean Absolute Error by Model', fontsize=14, fontweight='bold')
    plt.ylabel('MAE (degrees)')
    
    # Add value labels on bars
    for bar, mae in zip(bars, mae_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{mae:.2f}°', ha='center', va='bottom', fontweight='bold')
    
    # 3. Exercise Distribution
    plt.subplot(4, 3, 3)
    exercise_counts = df['exercise_name'].value_counts()
    plt.pie(exercise_counts.values, labels=exercise_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=sns.color_palette("Set3", len(exercise_counts)))
    plt.title('Distribution of Exercises in Dataset', fontsize=14, fontweight='bold')
    
    # 4. MAE Distribution by Exercise for each model
    plt.subplot(4, 3, 4)
    mae_by_exercise = []
    exercise_labels = []
    
    for exercise in sorted(df['exercise_name'].unique()):
        exercise_data = df[df['exercise_name'] == exercise]
        mae_by_exercise.append([
            exercise_data['YOLOv8_mae'].mean(),
            exercise_data['MoveNet_mae'].mean(),
            exercise_data['MediaPipe_mae'].mean()
        ])
        exercise_labels.append(exercise.replace(' ', '\n'))
    
    mae_by_exercise = np.array(mae_by_exercise)
    
    x = np.arange(len(exercise_labels))
    width = 0.25
    
    plt.bar(x - width, mae_by_exercise[:, 0], width, label='YOLOv8', color=colors[0], alpha=0.8)
    plt.bar(x, mae_by_exercise[:, 1], width, label='MoveNet', color=colors[1], alpha=0.8)
    plt.bar(x + width, mae_by_exercise[:, 2], width, label='MediaPipe', color=colors[2], alpha=0.8)
    
    plt.title('MAE Comparison by Exercise Type', fontsize=14, fontweight='bold')
    plt.xlabel('Exercise Type')
    plt.ylabel('MAE (degrees)')
    plt.xticks(x, exercise_labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # 5. Detection Rate Distribution
    plt.subplot(4, 3, 5)
    detection_data = [df['YOLOv8_valid_pct'], df['MoveNet_valid_pct'], df['MediaPipe_valid_pct']]
    
    plt.boxplot(detection_data, labels=models, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    plt.title('Detection Rate Distribution by Model', fontsize=14, fontweight='bold')
    plt.ylabel('Detection Rate (%)')
    plt.grid(axis='y', alpha=0.3)
    
    # 6. MAE Distribution
    plt.subplot(4, 3, 6)
    mae_data = [df['YOLOv8_mae'], df['MoveNet_mae'], df['MediaPipe_mae']]
    
    plt.boxplot(mae_data, labels=models, patch_artist=True,
                boxprops=dict(facecolor='lightcoral', alpha=0.7),
                medianprops=dict(color='darkred', linewidth=2))
    plt.title('MAE Distribution by Model', fontsize=14, fontweight='bold')
    plt.ylabel('MAE (degrees)')
    plt.grid(axis='y', alpha=0.3)
    
    # 7. Performance by Body Region
    plt.subplot(4, 3, 7)
    region_performance = df.groupby('region')[['YOLOv8_mae', 'MoveNet_mae', 'MediaPipe_mae']].mean()
    
    x = np.arange(len(region_performance.index))
    width = 0.25
    
    plt.bar(x - width, region_performance['YOLOv8_mae'], width, label='YOLOv8', color=colors[0], alpha=0.8)
    plt.bar(x, region_performance['MoveNet_mae'], width, label='MoveNet', color=colors[1], alpha=0.8)
    plt.bar(x + width, region_performance['MediaPipe_mae'], width, label='MediaPipe', color=colors[2], alpha=0.8)
    
    plt.title('Performance by Body Region', fontsize=14, fontweight='bold')
    plt.xlabel('Body Region')
    plt.ylabel('Average MAE (degrees)')
    plt.xticks(x, region_performance.index)
    plt.legend()
    
    # 8. Performance by Position
    plt.subplot(4, 3, 8)
    position_performance = df.groupby('position')[['YOLOv8_mae', 'MoveNet_mae', 'MediaPipe_mae']].mean()
    
    x = np.arange(len(position_performance.index))
    
    plt.bar(x - width, position_performance['YOLOv8_mae'], width, label='YOLOv8', color=colors[0], alpha=0.8)
    plt.bar(x, position_performance['MoveNet_mae'], width, label='MoveNet', color=colors[1], alpha=0.8)
    plt.bar(x + width, position_performance['MediaPipe_mae'], width, label='MediaPipe', color=colors[2], alpha=0.8)
    
    plt.title('Performance by Exercise Position', fontsize=14, fontweight='bold')
    plt.xlabel('Position')
    plt.ylabel('Average MAE (degrees)')
    plt.xticks(x, position_performance.index)
    plt.legend()
    
    # 9. Performance by Side (Left vs Right)
    plt.subplot(4, 3, 9)
    side_performance = df.groupby('side')[['YOLOv8_mae', 'MoveNet_mae', 'MediaPipe_mae']].mean()
    
    x = np.arange(len(side_performance.index))
    
    plt.bar(x - width, side_performance['YOLOv8_mae'], width, label='YOLOv8', color=colors[0], alpha=0.8)
    plt.bar(x, side_performance['MoveNet_mae'], width, label='MoveNet', color=colors[1], alpha=0.8)
    plt.bar(x + width, side_performance['MediaPipe_mae'], width, label='MediaPipe', color=colors[2], alpha=0.8)
    
    plt.title('Performance by Exercise Side', fontsize=14, fontweight='bold')
    plt.xlabel('Side')
    plt.ylabel('Average MAE (degrees)')
    plt.xticks(x, side_performance.index)
    plt.legend()
    
    # 10. Joint-specific Error Analysis
    plt.subplot(4, 3, 10)
    joint_errors = []
    joint_labels = ['Joint 0', 'Joint 1', 'Joint 2']
    
    for model in models:
        model_joint_errors = []
        for j in range(3):
            col_name = f"{model}_j{j}_err"
            model_joint_errors.append(df[col_name].mean())
        joint_errors.append(model_joint_errors)
    
    joint_errors = np.array(joint_errors)
    
    x = np.arange(len(joint_labels))
    
    for i, model in enumerate(models):
        plt.bar(x + i*width - width, joint_errors[i], width, 
                label=model, color=colors[i], alpha=0.8)
    
    plt.title('Average Joint-Specific Errors', fontsize=14, fontweight='bold')
    plt.xlabel('Joint')
    plt.ylabel('Average Error (degrees)')
    plt.xticks(x, joint_labels)
    plt.legend()
    
    # 11. Correlation Heatmap
    plt.subplot(4, 3, 11)
    correlation_cols = ['YOLOv8_mae', 'MoveNet_mae', 'MediaPipe_mae', 
                       'YOLOv8_valid_pct', 'MoveNet_valid_pct', 'MediaPipe_valid_pct']
    correlation_matrix = df[correlation_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Model Performance Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 12. Best vs Worst Performing Videos
    plt.subplot(4, 3, 12)
    df['avg_mae'] = df[['YOLOv8_mae', 'MoveNet_mae', 'MediaPipe_mae']].mean(axis=1)
    
    # Get best and worst performing videos
    best_5 = df.nsmallest(5, 'avg_mae')[['video', 'exercise_name', 'avg_mae']]
    worst_5 = df.nlargest(5, 'avg_mae')[['video', 'exercise_name', 'avg_mae']]
    
    # Create labels for the plot
    best_labels = [f"{row['video']}\n{row['exercise_name'][:20]}..." for _, row in best_5.iterrows()]
    worst_labels = [f"{row['video']}\n{row['exercise_name'][:20]}..." for _, row in worst_5.iterrows()]
    
    y_pos_best = np.arange(len(best_5))
    y_pos_worst = np.arange(len(worst_5)) + len(best_5) + 1
    
    plt.barh(y_pos_best, best_5['avg_mae'], color='green', alpha=0.7, label='Best 5')
    plt.barh(y_pos_worst, worst_5['avg_mae'], color='red', alpha=0.7, label='Worst 5')
    
    all_labels = best_labels + [''] + worst_labels
    plt.yticks(np.arange(len(all_labels)), all_labels, fontsize=8)
    plt.xlabel('Average MAE (degrees)')
    plt.title('Best vs Worst Performing Videos', fontsize=14, fontweight='bold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('uco_rehabilitation_eda_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_statistics(df):
    """Generate detailed statistical analysis"""
    
    print("\n=== Detailed Statistical Analysis ===\n")
    
    # Model ranking analysis
    print("4. Model Ranking Analysis:")
    models = ['YOLOv8', 'MoveNet', 'MediaPipe']
    
    # Create ranking for each video
    rankings = []
    for idx, row in df.iterrows():
        mae_values = [(row[f"{model}_mae"], model) for model in models]
        mae_values.sort()  # Sort by MAE (lower is better)
        
        video_ranking = {mae_values[i][1]: i+1 for i in range(len(mae_values))}
        video_ranking['video'] = row['video']
        video_ranking['exercise'] = row['exercise_name']
        rankings.append(video_ranking)
    
    ranking_df = pd.DataFrame(rankings)
    
    # Calculate average rankings
    avg_rankings = {}
    for model in models:
        avg_rankings[model] = ranking_df[model].mean()
        
    print("   Average Rankings (1=best, 3=worst):")
    sorted_rankings = sorted(avg_rankings.items(), key=lambda x: x[1])
    for i, (model, avg_rank) in enumerate(sorted_rankings):
        print(f"   {i+1}. {model:<12}: {avg_rank:.2f}")
    
    # Win rate analysis
    print("\n5. Win Rate Analysis (Best MAE per video):")
    win_counts = {model: 0 for model in models}
    
    for idx, row in df.iterrows():
        mae_values = {model: row[f"{model}_mae"] for model in models}
        winner = min(mae_values, key=mae_values.get)
        win_counts[winner] += 1
    
    total_videos = len(df)
    for model in models:
        win_rate = (win_counts[model] / total_videos) * 100
        print(f"   - {model:<12}: {win_counts[model]:>2}/{total_videos} wins ({win_rate:5.1f}%)")
    
    # Statistical significance tests
    from scipy import stats
    
    print("\n6. Statistical Significance Tests (Paired t-tests):")
    mae_data = {model: df[f"{model}_mae"].values for model in models}
    
    comparisons = [
        ('YOLOv8', 'MoveNet'),
        ('YOLOv8', 'MediaPipe'),
        ('MoveNet', 'MediaPipe')
    ]
    
    for model1, model2 in comparisons:
        t_stat, p_value = stats.ttest_rel(mae_data[model1], mae_data[model2])
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"   - {model1} vs {model2}: t={t_stat:6.3f}, p={p_value:.4f} {significance}")
    
    # Performance by exercise characteristics
    print("\n7. Performance by Exercise Characteristics:")
    
    print("\n   By Body Region:")
    region_stats = df.groupby('region')[['YOLOv8_mae', 'MoveNet_mae', 'MediaPipe_mae']].agg(['mean', 'std'])
    for region in region_stats.index:
        print(f"   - {region.upper()} Body:")
        for model in models:
            mean_val = region_stats.loc[region, (f'{model}_mae', 'mean')]
            std_val = region_stats.loc[region, (f'{model}_mae', 'std')]
            print(f"     {model}: {mean_val:6.2f}° ± {std_val:5.2f}°")
    
    print("\n   By Exercise Position:")
    position_stats = df.groupby('position')[['YOLOv8_mae', 'MoveNet_mae', 'MediaPipe_mae']].agg(['mean', 'std'])
    for position in position_stats.index:
        print(f"   - {position}:")
        for model in models:
            mean_val = position_stats.loc[position, (f'{model}_mae', 'mean')]
            std_val = position_stats.loc[position, (f'{model}_mae', 'std')]
            print(f"     {model}: {mean_val:6.2f}° ± {std_val:5.2f}°")

def main():
    """Main function to run the complete EDA"""
    
    # Load and analyze the data
    df = load_and_analyze_data()
    
    # Generate detailed statistics
    generate_detailed_statistics(df)
    
    # Create visualizations
    create_visualizations(df)
    
    print("\n=== EDA Complete ===")
    print("Comprehensive visualization saved as 'uco_rehabilitation_eda_comprehensive.png'")
    print("Check the plots for detailed insights about model performance across different exercises and conditions.")

if __name__ == "__main__":
    main()
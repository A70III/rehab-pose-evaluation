import pandas as pd
import numpy as np

def create_summary_table():
    """Create a comprehensive summary table of model performance"""
    
    # Load the data
    df = pd.read_csv("output_comparison_results/batch_evaluation_summary.csv")
    
    models = ['YOLOv8', 'MoveNet', 'MediaPipe']
    
    # Create summary statistics
    summary_data = []
    
    for model in models:
        valid_col = f"{model}_valid_pct"
        mae_col = f"{model}_mae"
        
        model_data = {
            'Model': model,
            'Detection_Rate_Mean': df[valid_col].mean(),
            'Detection_Rate_Std': df[valid_col].std(),
            'Detection_Rate_Min': df[valid_col].min(),
            'Detection_Rate_Max': df[valid_col].max(),
            'MAE_Mean': df[mae_col].mean(),
            'MAE_Std': df[mae_col].std(),
            'MAE_Min': df[mae_col].min(),
            'MAE_Max': df[mae_col].max(),
            'MAE_Median': df[mae_col].median()
        }
        
        summary_data.append(model_data)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Performance by exercise
    exercise_performance = []
    
    for exercise_name in df['exercise_name'].unique():
        exercise_data = df[df['exercise_name'] == exercise_name]
        
        for model in models:
            mae_col = f"{model}_mae"
            valid_col = f"{model}_valid_pct"
            
            exercise_perf = {
                'Exercise': exercise_name,
                'Model': model,
                'Count': len(exercise_data),
                'Avg_MAE': exercise_data[mae_col].mean(),
                'Std_MAE': exercise_data[mae_col].std(),
                'Avg_Detection_Rate': exercise_data[valid_col].mean()
            }
            
            exercise_performance.append(exercise_perf)
    
    exercise_df = pd.DataFrame(exercise_performance)
    
    # Create pivot table for easy comparison
    mae_pivot = exercise_df.pivot(index='Exercise', columns='Model', values='Avg_MAE')
    detection_pivot = exercise_df.pivot(index='Exercise', columns='Model', values='Avg_Detection_Rate')
    
    print("=== UCO Rehabilitation Dataset - Model Performance Summary ===\n")
    
    print("1. Overall Performance Summary:")
    print(summary_df.round(2).to_string(index=False))
    
    print("\n\n2. Mean Absolute Error by Exercise (degrees):")
    print(mae_pivot.round(2).to_string())
    
    print("\n\n3. Detection Rate by Exercise (%):")
    print(detection_pivot.round(1).to_string())
    
    # Best and worst performing cases
    print("\n\n4. Best Performing Cases (Lowest MAE):")
    df['avg_mae'] = df[['YOLOv8_mae', 'MoveNet_mae', 'MediaPipe_mae']].mean(axis=1)
    best_cases = df.nsmallest(5, 'avg_mae')[['video', 'exercise_name', 'YOLOv8_mae', 'MoveNet_mae', 'MediaPipe_mae', 'avg_mae']]
    print(best_cases.round(2).to_string(index=False))
    
    print("\n\n5. Most Challenging Cases (Highest MAE):")
    worst_cases = df.nlargest(5, 'avg_mae')[['video', 'exercise_name', 'YOLOv8_mae', 'MoveNet_mae', 'MediaPipe_mae', 'avg_mae']]
    print(worst_cases.round(2).to_string(index=False))
    
    # Model rankings
    print("\n\n6. Model Rankings by Video (1=Best, 3=Worst):")
    rankings = []
    for idx, row in df.iterrows():
        mae_values = [(row[f"{model}_mae"], model) for model in models]
        mae_values.sort()
        
        ranking_text = f"{row['video']:<12} {row['exercise_name']:<30}"
        for i, (mae, model) in enumerate(mae_values):
            ranking_text += f" {i+1}.{model}({mae:.1f}°)"
        
        rankings.append(ranking_text)
    
    for ranking in rankings:
        print(ranking)
    
    # Save summary tables to CSV
    summary_df.to_csv('model_performance_summary.csv', index=False)
    mae_pivot.to_csv('mae_by_exercise.csv')
    detection_pivot.to_csv('detection_rate_by_exercise.csv')
    
    print("\n\n=== Summary files saved ===")
    print("- model_performance_summary.csv")
    print("- mae_by_exercise.csv") 
    print("- detection_rate_by_exercise.csv")

if __name__ == "__main__":
    create_summary_table()
import pandas as pd
import numpy as np
import random

def simulate_rtb_data(num_impressions=100000, target_ctr=0.10):
    """
    Simulates a realistic RTB dataset with a target CTR.

    Features:
    - user_id: A unique identifier for the user.
    - ad_creative_quality: A score from 1-10 (influences CTR).
    - time_of_day: Hour of the day (0-23) (influences CTR).
    - is_clicked: The target variable (1 if clicked, 0 if not).
    """
    print(f"Generating {num_impressions} impressions with target CTR {target_ctr*100}%...")

    # 1. Create Base Features
    
    # Simulate a smaller pool of users (e.g., 10% of impressions)
    num_users = int(num_impressions * 0.1)
    user_ids = [f"user_{random.randint(1000, 1000 + num_users)}" for _ in range(num_impressions)]
    
    # Ad quality (1-10)
    ad_creative_quality = np.random.randint(1, 11, size=num_impressions)
    
    # Time of day (0-23)
    # Let's assume prime time (e.g., 18:00 - 22:00) is more common
    time_weights = [0.03] * 18  # Low weight for 00:00-17:00
    time_weights += [0.08, 0.1, 0.1, 0.08, 0.05] # High weight for 18:00-22:00
    time_weights += [0.04] # 23:00
    # Normalize weights to sum to 1 (approximately)
    time_weights = [w / sum(time_weights) for w in time_weights]
    time_of_day = np.random.choice(24, size=num_impressions, p=time_weights)

    # 2. Define Logic for Clicks (This is the core part)
    
    # We will define a "base_click_probability"
    # We want better ads and certain times of day to have a higher chance of a click.
    
    # Start with a very low base probability
    base_prob = np.full(num_impressions, 0.01)
    
    # --- Influence of Ad Quality ---
    # Increase probability for high-quality ads (quality > 7)
    base_prob[ad_creative_quality > 7] += 0.05
    # Slightly increase for medium quality (quality 4-7)
    base_prob[(ad_creative_quality >= 4) & (ad_creative_quality <= 7)] += 0.02
    # Decrease for low quality (quality < 4)
    base_prob[ad_creative_quality < 4] -= 0.005
    
    # --- Influence of Time of Day ---
    # Increase probability during "prime time" (e.g., 18:00 - 22:00)
    prime_time_indices = (time_of_day >= 18) & (time_of_day <= 22)
    base_prob[prime_time_indices] += 0.04

    # --- Adjust to match Target CTR ---
    # The probabilities we've set (e.g., 0.01 + 0.05 + 0.04 = 0.10) are just relative.
    # We need to scale them globally to hit our 10% target.
    
    # Calculate the current average probability
    current_avg_prob = np.mean(base_prob)
    
    # Calculate the scaling factor
    # (Ensure we don't divide by zero, though unlikely)
    if current_avg_prob == 0:
        current_avg_prob = 0.01 
        
    scaling_factor = target_ctr / current_avg_prob
    
    # Apply the scaling factor to all probabilities
    click_probability = base_prob * scaling_factor
    
    # Ensure no probability is > 1.0 (very important)
    click_probability = np.clip(click_probability, 0, 1)

    # 3. Generate Clicks based on Probability
    # Use the probabilities to flip a biased coin for each impression
    is_clicked = (np.random.rand(num_impressions) < click_probability).astype(int)

    # 4. Create DataFrame
    data = {
        'user_id': user_ids,
        'ad_creative_quality': ad_creative_quality,
        'time_of_day': time_of_day,
        'click_probability': click_probability, # Good to keep for checking
        'is_clicked': is_clicked
    }
    df = pd.DataFrame(data)

    return df

# --- Main execution ---
if __name__ == "__main__":
    # Generate the data
    simulated_df = simulate_rtb_data(num_impressions=100000, target_ctr=0.10)
    
    # Print a summary
    print("\n--- Data Simulation Complete ---")
    print(simulated_df.head())
    print("\n--- Feature Analysis ---")
    print(simulated_df.describe())
    
    # Verify the CTR
    actual_ctr = simulated_df['is_clicked'].mean()
    print(f"\nActual CTR in generated data: {actual_ctr * 100:.2f}%")

    # Save to a file (optional, but good practice)
    output_filename = "simulated_rtb_data.csv"
    simulated_df.to_csv(output_filename, index=False)
    print(f"Data saved to {output_filename}")
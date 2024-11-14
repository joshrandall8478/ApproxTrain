import os
import json

# Directory containing the JSON files
stats_dir = "training_stats"

# Loop through each JSON file in the directory
for filename in os.listdir(stats_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(stats_dir, filename)
        with open(file_path, "r") as file:
            data = json.load(file)
        
        # Count epochs by checking the length of "train_loss"
        epoch_count = len(data.get("train_loss", []))
        
        # Get test accuracy
        test_accuracy = data.get("test_accuracy", "N/A")
        
        # Output the results
        print(f"File: {filename}")
        print(f"  Number of epochs: {epoch_count}")
        print(f"  Test accuracy: {test_accuracy}\n")

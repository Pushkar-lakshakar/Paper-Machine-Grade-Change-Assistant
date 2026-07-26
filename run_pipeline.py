import sys
import os

# Add root folder to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.generator import generate_dataset
from models.train import main as train_main

def run():
    print("==================================================")
    # 1. Run generator
    generate_dataset(num_events=200)
    
    print("\n==================================================")
    # 2. Run model training
    train_main()
    
    print("\n==================================================")
    print("Pipeline executed successfully!")
    print("You can now start the dashboard using: streamlit run dashboard/app.py")

if __name__ == "__main__":
    run()

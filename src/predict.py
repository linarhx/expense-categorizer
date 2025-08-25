import pandas as pd  # type: ignore
import joblib  # type: ignore
import os


# Load the trained model
model_path = "../models/model.joblib"

if not os.path.exists(model_path):
    print("Error: Model not found! Please run train.py first.")
    exit(1)

print("Loading trained model...")
model = joblib.load(model_path)
print("Model loaded successfully!")

# Load the training data to see what categories we have
df = pd.read_csv('../data/transactions_sample.csv')
categories = df['category'].unique()
print(f"\nAvailable categories: {list(categories)}")

# Interactive prediction loop
print("\n" + "="*50)
print("💰 EXPENSE CATEGORIZER - PREDICTION MODE")
print("="*50)
print("Enter transaction descriptions to get predictions.")
print("Type 'quit' to exit.\n")

while True:
    # Get user input
    transaction = input("Enter transaction description: ").strip()
    
    if transaction.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not transaction:
        print("Please enter a transaction description.")
        continue
    
    try:
        # Make prediction
        prediction = model.predict([transaction])[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba([transaction])[0]
        
        # Create a nice display of results
        print(f"\n🔍 Transaction: {transaction}")
        print(f"📊 Predicted Category: {prediction}")
        
        # Show all probabilities
        print("\n📈 Category Probabilities:")
        for i, (cat, prob) in enumerate(zip(categories, probabilities)):
            if cat == prediction:
                print(f"  🎯 {cat}: {prob:.2%} ← SELECTED")
            else:
                print(f"     {cat}: {prob:.2%}")
        
        print("-" * 40)
        
    except Exception as e:
        print(f"Error making prediction: {e}")
    
    print()  # Empty line for readability


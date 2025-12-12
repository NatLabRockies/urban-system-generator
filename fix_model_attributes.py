"""
Script to extract the correct model attributes from the pretrained model files
"""

import json
import pickle

# Load the encoding dictionary
with open('resources/pretrained_model/encoding_mapper.json', 'r') as f:
    encoding_dict = json.load(f)

# Load the numerical scaler
with open('resources/pretrained_model/num_scaler.pkl', 'rb') as f:
    num_scaler = pickle.load(f)

# Get all categorical columns from encoding dict
cat_cols = list(encoding_dict.keys())

# Get all numerical columns from scaler
num_cols = list(num_scaler.feature_names_in_)

# Combine all attributes - the order matters!
# Typically categorical columns come first, then numerical
all_attributes = cat_cols + num_cols

print("Total attributes:", len(all_attributes))
print("\nCategorical attributes:", len(cat_cols))
print("Numerical attributes:", len(num_cols))

# Generate the new model_attributes.py content
new_content = '''"""
Model Attributes Schema

This file defines all the attributes that the inference model expects.
These must be in the exact order and format that the model was trained with.

Auto-generated from the pretrained model files.
"""

# Complete list of all model attributes
# This matches the trained model's expected input schema
all_model_attributes = [
'''

# Add categorical attributes with comments
new_content += "    # Categorical attributes\n"
for col in cat_cols:
    new_content += f"    '{col}',\n"

# Add numerical attributes with comments  
new_content += "    \n    # Numerical attributes\n"
for col in num_cols:
    new_content += f"    '{col}',\n"

new_content += "]\n"

# Write the new file
with open('resources/model_attributes_fixed.py', 'w') as f:
    f.write(new_content)

print("\nFixed model attributes saved to resources/model_attributes_fixed.py")
print("\nFirst 5 attributes:", all_attributes[:5])
print("Last 5 attributes:", all_attributes[-5:])

"""
Model Attributes Schema

This file defines all the attributes that the inference model expects.
These must be in the exact order and format that the model was trained with.
"""

# Complete list of all model attributes
# This should match your trained model's expected input schema
all_model_attributes = [
    # Building geometry
    'Geometry Building Type RECS',
    'Geometry Floor Area',
    'Geometry Floor Area Bin',
    'Geometry Stories',
    'Geometry Foundation Type',
    'Geometry Attic Type',
    'Geometry Building Number Units SFA',
    
    # Building age
    'Vintage',
    'Vintage ACS',
    
    # Occupancy
    'Bedrooms',
    
    # HVAC Systems
    'HVAC Cooling Type',
    'HVAC Heating Type',
    'HVAC Heating Efficiency',
    'HVAC Cooling Efficiency',
    'Heating Fuel',
    
    # Insulation
    'Insulation Wall',
    'Insulation Roof',
    'Insulation Foundation',
    
    # Windows
    'Windows',
    'Window Areas',
    
    # Appliances
    'Water Heater',
    'Cooking Range',
    'Refrigerator',
    'Dishwasher',
    'Clothes Washer',
    'Clothes Dryer',
    
    # Lighting
    'Lighting',
    
    # Other
    'Orientation',
    'Neighbors',
    'Occupants',
]

# Note: Replace this list with your actual model's expected attributes
# The order and names must match exactly what the model was trained with
#!/usr/bin/env python
"""
Test script for Urban System Generator
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from usg.inference import USGInference
from resources.model_attributes import all_model_attributes

def test_single_prediction():
    """Test single building prediction"""
    print("\n=== Testing Single Building Prediction ===")
    
    try:
        # Initialize inference engine
        inference = USGInference(
            model_path="resources/pretrained_model/adaptive_model_1.keras",
            cat_scaler_path="resources/pretrained_model/cat_scaler.pkl",
            num_scaler_path="resources/pretrained_model/num_scaler.pkl",
            encoding_dict_path="resources/pretrained_model/encoding_mapper.json",
            all_model_attributes=all_model_attributes,
        )
        print("✓ Model loaded successfully")
        
        # Test with minimal known attributes
        known_attrs = {
            'Geometry Building Type RECS': 'Single-Family Detached',
            'Vintage': '2000s',
            'Geometry Stories': '2',
            'Bedrooms': '3',
        }
        
        print(f"✓ Input attributes: {len(known_attrs)}")
        
        # Predict missing attributes
        completed_attrs = inference.predict_missing_single(known_attrs)
        
        print(f"✓ Output attributes: {len(completed_attrs)}")
        
        # Verify all attributes are present
        assert len(completed_attrs) == len(all_model_attributes), \
            f"Expected {len(all_model_attributes)} attributes, got {len(completed_attrs)}"
        
        # Check that known attributes are preserved
        for key, value in known_attrs.items():
            if key in completed_attrs:
                assert str(completed_attrs[key]) == str(value), \
                    f"Known attribute {key} changed from {value} to {completed_attrs[key]}"
        
        print("✓ All tests passed!")
        
        # Show some predicted attributes
        print("\nSample predicted attributes:")
        sample_attrs = ['HVAC Cooling Type', 'Heating Fuel', 'HVAC Heating Type', 
                       'Insulation Wall', 'Water Heater Fuel']
        for attr in sample_attrs:
            if attr in completed_attrs:
                print(f"  {attr}: {completed_attrs[attr]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False

def test_import():
    """Test that the package can be imported"""
    print("\n=== Testing Package Import ===")
    try:
        import usg
        print(f"✓ Package imported successfully")
        print(f"  Version: {usg.__version__}")
        print(f"  Author: {usg.__author__}")
        
        from usg import USGInference, GeoJSONProcessor, ScaledInputMaskedNN
        print("✓ All main classes imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Urban System Generator - Test Suite")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Import Test", test_import()))
    results.append(("Single Prediction Test", test_single_prediction()))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())

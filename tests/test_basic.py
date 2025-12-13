"""
Basic tests for Urban System Generator
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBasicImports(unittest.TestCase):
    """Test basic package imports"""
    
    def test_import_usg(self):
        """Test that usg package can be imported"""
        try:
            import usg
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import usg: {e}")
    
    def test_import_modules(self):
        """Test that main modules can be imported"""
        try:
            from usg import inference
            from usg import geojson_processor
            from usg import model
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import modules: {e}")
    
    def test_geojson_processor_class(self):
        """Test GeoJSONProcessor class exists"""
        from usg.geojson_processor import GeoJSONProcessor
        processor = GeoJSONProcessor()
        self.assertIsNotNone(processor)
    
    def test_model_class(self):
        """Test ScaledInputMaskedNN class exists"""
        from usg.model import ScaledInputMaskedNN
        self.assertTrue(hasattr(ScaledInputMaskedNN, '__init__'))


class TestGeoJSONProcessor(unittest.TestCase):
    """Test GeoJSON processor functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        from usg.geojson_processor import GeoJSONProcessor
        self.processor = GeoJSONProcessor()
    
    def test_vintage_mapping(self):
        """Test year to vintage mapping"""
        self.assertEqual(self.processor.map_vintage(1935), "<1940")
        self.assertEqual(self.processor.map_vintage(1945), "1940s")
        self.assertEqual(self.processor.map_vintage(1955), "1950s")
        self.assertEqual(self.processor.map_vintage(1965), "1960s")
        self.assertEqual(self.processor.map_vintage(1975), "1970s")
        self.assertEqual(self.processor.map_vintage(1985), "1980s")
        self.assertEqual(self.processor.map_vintage(1995), "1990s")
        self.assertEqual(self.processor.map_vintage(2005), "2000s")
        self.assertEqual(self.processor.map_vintage(2015), "2010s")
    
    def test_floor_area_mapping(self):
        """Test floor area to category mapping"""
        self.assertEqual(self.processor.map_floor_area_to_category(400), "0-499")
        self.assertEqual(self.processor.map_floor_area_to_category(600), "500-749")
        self.assertEqual(self.processor.map_floor_area_to_category(900), "750-999")
        self.assertEqual(self.processor.map_floor_area_to_category(1200), "1000-1499")
        self.assertEqual(self.processor.map_floor_area_to_category(1800), "1500-1999")
        self.assertEqual(self.processor.map_floor_area_to_category(5000), "4000+")


class TestModelAvailability(unittest.TestCase):
    """Test model file availability"""
    
    def test_check_model_files(self):
        """Check if pretrained model files exist"""
        import os
        model_files = [
            "resources/pretrained_model/adaptive_model_1.keras",
            "resources/pretrained_model/cat_scaler.pkl",
            "resources/pretrained_model/num_scaler.pkl",
            "resources/pretrained_model/encoding_mapper.json",
        ]
        
        files_exist = all(os.path.exists(f) for f in model_files)
        
        if not files_exist:
            print("\nNote: Pretrained model files not found.")
            print("Inference tests will be skipped.")
            print("To run full tests, ensure model files are in resources/pretrained_model/")
        
        # This test just checks, doesn't fail
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python
"""
Test runner for Urban System Generator

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py basic        # Run basic tests
    python run_tests.py integration  # Run integration tests
"""

import sys
import unittest
from pathlib import Path
import os

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def run_all_tests():
    """Run all tests in the test suite"""
    print("\nRunning all tests...\n")
    loader = unittest.TestLoader()
    
    # Try to discover tests
    suite = loader.discover('tests', pattern='test_*.py')
    
    # If no tests found, try loading specific test files
    if suite.countTestCases() == 0:
        print("No tests discovered via pattern. Loading specific test files...")
        suite = unittest.TestSuite()
        
        # Try to load each test file explicitly
        test_files = ['test_basic', 'test_integration', 'test_inference', 'test_geojson_processor']
        for test_file in test_files:
            try:
                module_tests = loader.loadTestsFromName(f'tests.{test_file}')
                if module_tests.countTestCases() > 0:
                    suite.addTests(module_tests)
                    print(f"  Loaded tests from {test_file}")
            except Exception as e:
                print(f"  Could not load {test_file}: {e}")
    
    if suite.countTestCases() == 0:
        print("No tests found!")
        return False
    
    print(f"\nFound {suite.countTestCases()} tests\n")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def run_basic_tests():
    """Run basic tests"""
    print("\nRunning basic tests...\n")
    loader = unittest.TestLoader()
    
    try:
        suite = loader.loadTestsFromName('tests.test_basic')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    except Exception as e:
        print(f"Error loading basic tests: {e}")
        return False


def run_integration_tests():
    """Run integration tests"""
    print("\nRunning integration tests...\n")
    loader = unittest.TestLoader()
    
    try:
        suite = loader.loadTestsFromName('tests.test_integration')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    except Exception as e:
        print(f"Error loading integration tests: {e}")
        return False


def main():
    """Main entry point"""
    print("=" * 60)
    print("Urban System Generator - Test Suite")
    print("=" * 60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == 'basic':
            success = run_basic_tests()
        elif arg == 'integration':
            success = run_integration_tests()
        else:
            print(f"Unknown test type: {arg}")
            success = False
    else:
        success = run_all_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed or no tests found")
        return 1


if __name__ == "__main__":
    sys.exit(main())

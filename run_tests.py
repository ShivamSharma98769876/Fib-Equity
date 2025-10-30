#!/usr/bin/env python3
"""
Test runner for the Swing Trade Stock Screener
"""

import unittest
import sys
import os
from pathlib import Path

def run_all_tests():
    """Run all test suites"""
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / "tests"
    suite = loader.discover(str(start_dir), pattern="test_*.py")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

def run_specific_test(test_name):
    """Run a specific test module"""
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Import and run specific test
    test_module = f"tests.{test_name}"
    
    try:
        suite = unittest.TestLoader().loadTestsFromName(test_module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return result.wasSuccessful()
    except ImportError as e:
        print(f"Error importing test module {test_module}: {e}")
        return False

def main():
    """Main test runner"""
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        print(f"Running specific test: {test_name}")
        success = run_specific_test(test_name)
    else:
        # Run all tests
        print("Running all tests...")
        success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

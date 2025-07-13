#!/usr/bin/env python3
"""
Test script to verify AQA package installation
"""

import sys
import os

# Add current directory to Python path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_import():
    """Test basic imports"""
    success_count = 0
    total_tests = 0

    try:
        import aqa
        print(f"✓ AQA version {aqa.__version__} imported successfully")
        success_count += 1
    except ImportError as e:
        print(f"✗ AQA import error: {e}")
    total_tests += 1

    # Test main class imports with individual error handling
    imports_to_test = [
        ('alchemical_calculator', 'aqa.alch_calc'),
        ('APDFT_perturbator', 'aqa.AP_class'),
        ('FcM_like', 'aqa.FcMole'),
        ('calc_finite_difference', 'aqa.finite_difference'),
        ('generate_unique_nuclear_numbers_list', 'aqa.gener_chem_space')
    ]

    for item_name, module_name in imports_to_test:
        try:
            # Try importing from aqa package first
            exec(f"from aqa import {item_name}")
            print(f"✓ {item_name} imported successfully from aqa")
            success_count += 1
        except ImportError:
            try:
                # Fall back to direct module import
                exec(f"from {module_name} import {item_name}")
                print(f"✓ {item_name} imported successfully from {module_name}")
                success_count += 1
            except ImportError as e:
                print(f"✗ {item_name} import error: {e}")
        total_tests += 1

    print(f"\nImport test results: {success_count}/{total_tests} successful")
    return success_count == total_tests

def test_dependencies():
    """Test that required dependencies are available"""
    dependencies = [
        'numpy',
        'scipy',
        'pyscf',
        'basis_set_exchange',
        'matplotlib'
    ]

    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} available")
        except ImportError:
            print(f"✗ {dep} not available")

if __name__ == "__main__":
    print("Testing AQA package installation...")
    print("=" * 40)

    print("\n1. Testing core imports:")
    test_import()

    print("\n2. Testing dependencies:")
    test_dependencies()

    print("\n" + "=" * 40)
    print("Installation test complete!")

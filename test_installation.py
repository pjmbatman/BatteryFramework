"""
Simple test script to verify Battery Foundation installation
"""

import sys
import importlib

def test_imports():
    """Test that all major components can be imported"""
    
    print("🔋 Testing Battery Foundation Model Framework Installation")
    print("=" * 60)
    
    # Test core imports
    modules_to_test = [
        'battery_foundation',
        'battery_foundation.models',
        'battery_foundation.models.lipm',
        'battery_foundation.data',
        'battery_foundation.data.dataset',
        'battery_foundation.training',
        'battery_foundation.training.trainer',
        'battery_foundation.tasks',
        'battery_foundation.tasks.soh',
        'battery_foundation.tasks.soc',
        'battery_foundation.utils',
        'battery_foundation.utils.config',
        'battery_foundation.utils.registry',
        'battery_foundation.evaluation',
        'battery_foundation.evaluation.metrics',
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"⚠️  {module_name}: {e}")
            failed_imports.append(module_name)
    
    print("\n" + "=" * 60)
    
    if failed_imports:
        print(f"❌ {len(failed_imports)} modules failed to import:")
        for module in failed_imports:
            print(f"   - {module}")
        return False
    else:
        print("✅ All modules imported successfully!")
        return True


def test_registries():
    """Test that registries are working"""
    print("\n🔧 Testing Registry System")
    print("-" * 30)
    
    try:
        from battery_foundation.utils.registry import ModelRegistry, DatasetRegistry, TaskRegistry
        
        # Test model registry
        models = ModelRegistry.list_available()
        print(f"✅ Models available: {models}")
        
        # Test dataset registry
        datasets = DatasetRegistry.list_available()
        print(f"✅ Datasets available: {datasets}")
        
        # Test task registry
        tasks = TaskRegistry.list_available()
        print(f"✅ Tasks available: {tasks}")
        
        return True
        
    except Exception as e:
        print(f"❌ Registry system error: {e}")
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\n📋 Testing Configuration System")
    print("-" * 35)
    
    try:
        from battery_foundation.utils.config import Config
        
        # Test creating a basic config
        config = Config()
        print(f"✅ Basic config created: {config.model_name}")
        
        # Test config to dict conversion
        config_dict = config.to_dict()
        print(f"✅ Config serialization: {len(config_dict)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration system error: {e}")
        return False


def test_model_creation():
    """Test basic model creation"""
    print("\n🧠 Testing Model Creation")
    print("-" * 25)
    
    try:
        from battery_foundation.utils.config import Config
        from battery_foundation.models.lipm import LiPMModel
        
        # Create config and model
        config = Config(task="ir_pretrain")
        model = LiPMModel(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ LiPM model created with {total_params:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False


def main():
    """Run all tests"""
    
    tests = [
        ("Import Test", test_imports),
        ("Registry Test", test_registries),  
        ("Config Test", test_config_loading),
        ("Model Test", test_model_creation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("-" * 60)
    print(f"RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Battery Foundation Framework is ready to use!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the installation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
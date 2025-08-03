from typing import Dict, Type, Any, Callable


class Registry:
    """Base registry class for storing and retrieving registered components"""
    
    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}
    
    def register(self, name: str, obj: Any = None):
        """Register an object or use as decorator"""
        if obj is None:
            # Used as decorator
            def decorator(obj):
                self._registry[name] = obj
                return obj
            return decorator
        else:
            # Direct registration
            self._registry[name] = obj
            return obj
    
    def get(self, name: str) -> Any:
        """Get registered object by name"""
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in {self._name} registry. "
                         f"Available: {list(self._registry.keys())}")
        return self._registry[name]
    
    def list_available(self) -> list:
        """List all available registered names"""
        return list(self._registry.keys())
    
    def __contains__(self, name: str) -> bool:
        return name in self._registry


# Global registries
ModelRegistry = Registry("Model")
DatasetRegistry = Registry("Dataset") 
TaskRegistry = Registry("Task")
OptimizerRegistry = Registry("Optimizer")
SchedulerRegistry = Registry("Scheduler")
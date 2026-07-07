__all__ = [
    "MimicGenEnvWrapper",
    "generate_mimicgen_dataset",
    "list_available_tasks",
]


def __getattr__(name):
    if name in __all__:
        from env.wrappers.mimicgen_wrapper import (
            MimicGenEnvWrapper,
            generate_mimicgen_dataset,
            list_available_tasks,
        )

        values = {
            "MimicGenEnvWrapper": MimicGenEnvWrapper,
            "generate_mimicgen_dataset": generate_mimicgen_dataset,
            "list_available_tasks": list_available_tasks,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

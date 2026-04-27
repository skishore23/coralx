"""Service factory and dependency injection for CORAL-X."""

from ..common.config import CoralConfig
from ..common.logging import get_logger
from ..ports.interfaces import Executor, FitnessFn, Plugin
from ..services.genetic_operations import GeneticOperationsService
from ..services.population_manager import PopulationManager
from ..services.progress_tracker import ProgressTracker
from .evolution_orchestrator import EvolutionServices


def create_evolution_services(
    config: CoralConfig,
    fitness_fn: FitnessFn | None = None,
    executor: Executor | None = None,
    run_id: str | None = None,
) -> EvolutionServices:
    """Create evolution services with dependency injection.

    Args:
        config: CORAL-X configuration
        fitness_fn: Fitness function (optional, will create default if None)
        executor: Executor (optional, will create default if None)
        run_id: Run identifier (optional)

    Returns:
        EvolutionServices container with all dependencies
    """
    logger = get_logger(__name__)
    plugin = create_plugin(config)

    # Create core services
    population_manager = PopulationManager(config, config.seed)
    genetic_operations = GeneticOperationsService(config, config.seed)
    progress_tracker = ProgressTracker(config, run_id)

    # Create fitness function if not provided
    if fitness_fn is None:
        fitness_fn = plugin.fitness_fn()

    # Create executor if not provided
    if executor is None:
        executor = create_executor(config)

    logger.info(
        f"Evolution services created for experiment: {config.experiment.name}, executor: {config.infra.executor}, run_id: {run_id}"
    )

    return EvolutionServices(
        population_manager=population_manager,
        genetic_operations=genetic_operations,
        progress_tracker=progress_tracker,
        fitness_fn=fitness_fn,
        executor=executor,
        dataset_provider=plugin.dataset(),
        model_factory=plugin.model_factory(),
        config=config,
    )


def create_plugin(config: CoralConfig) -> Plugin:
    """Create the configured experiment plugin."""
    from plugins.registry import create_plugin as create_registered_plugin

    return create_registered_plugin(config)


def create_fitness_function(
    config: CoralConfig, run_id: str | None = None
) -> FitnessFn:
    """Create fitness function based on configuration.

    Args:
        config: CORAL-X configuration
        run_id: Optional run identifier to pass to plugins

    Returns:
        Fitness function implementation
    """
    logger = get_logger(__name__)
    plugin = create_plugin(config)
    fitness_fn = plugin.fitness_fn()
    logger.info(
        f"Fitness function created: target={config.experiment.target}, type={type(fitness_fn).__name__}"
    )

    return fitness_fn


def create_executor(config: CoralConfig) -> Executor:
    """Create executor based on configuration.

    Args:
        config: CORAL-X configuration

    Returns:
        Executor implementation
    """
    logger = get_logger(__name__)

    executor_type = config.infra.executor

    if executor_type == "local":
        from infra.executors.local import LocalExecutor

        executor = LocalExecutor()
    else:
        raise ValueError(f"Unknown executor type: {executor_type}")

    logger.info(
        f"Executor created: type={executor_type}, class={type(executor).__name__}"
    )

    return executor


class ServiceContainer:
    """Service container for managing application services."""

    def __init__(self, config: CoralConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # Lazy-loaded services
        self._evolution_services: EvolutionServices | None = None

    def evolution_services(
        self,
        fitness_fn: FitnessFn | None = None,
        executor: Executor | None = None,
        run_id: str | None = None,
    ) -> EvolutionServices:
        """Get evolution services (lazy-loaded).

        Args:
            fitness_fn: Override fitness function
            executor: Override executor
            run_id: Run identifier

        Returns:
            EvolutionServices container
        """
        if self._evolution_services is None:
            self._evolution_services = create_evolution_services(
                self.config, fitness_fn, executor, run_id
            )

        return self._evolution_services

    def clear_cache(self) -> None:
        """Clear cached services to force recreation."""
        self._evolution_services = None
        self.logger.info("Service cache cleared")

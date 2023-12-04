class TrainerException(Exception):
    """Base class for exceptions in Trainer package."""

    def __init__(self, message="An error occurred in the Trainer package"):
        self.message = message
        super().__init__(self.message)


class CheckPointLoadException(TrainerException):
    """Base class for exceptions in Trainer package."""

    def __init__(self, message="An error loading checkpoint weights"):
        self.message = message
        super().__init__(self.message)


class UninitializedModelError(TrainerException):
    """Exception raised for errors in the usage of an uninitialized model."""

    def __init__(self, message="Model is not initialized"):
        self.message = message
        super().__init__(self.message)


class UntrainedModelError(TrainerException):
    """Exception raised for errors in the usage of an untrained model."""

    def __init__(self, message="Model is not trained"):
        self.message = message
        super().__init__(self.message)


class KModelValueError(TrainerException):
    """Exception raised for errors in the usage of a k fold model."""

    def __init__(self, message="Invalid value"):
        self.message = message
        super().__init__(self.message)


class UtilsValueException(TrainerException):
    def __init__(self, message="Invalid value"):
        self.message = message
        super().__init__(self.message)


class UtilsIOException(TrainerException):
    def __init__(self, message="Invalid value"):
        self.message = message
        super().__init__(self.message)

from abc import ABC, abstractmethod

class FingerRegionRecognizer(ABC):
    """
    The FingerReGionRecognizer interface declares the operations that all concrete FingerRegionRecognizers
    must implement.
    """
    @abstractmethod
    def normalize_image(self) -> str:
        pass

    @abstractmethod
    def segment_image(self) -> str:
        pass

    @abstractmethod
    def lines_orientation(self) -> str:
        pass

    @abstractmethod
    def calculate_ridge_frequencies(self) -> str:
        pass

    # @abstractmethod
    # def calculate_block_frequencies(self) -> str:
    #     pass

    @abstractmethod
    def apply_gabor_filter(self) -> str:
        pass

    # @abstractmethod
    # def skeletonize(self) -> str:
    #     pass

    # @abstractmethod
    # def detect_minutiae(self) -> str:
    #     pass

    # @abstractmethod
    # def calculate_minutiae_weights(self) -> str:
    #     pass

    # @abstractmethod 
    # def count_lines(self) -> str:
    #     pass

    # @abstractmethod
    # def count_diagonal_lines(self) -> str:
    #     pass

    # @abstractmethod
    # def draw_diagonal(self) -> str:
    #     pass

    # @abstractmethod
    # def get_best_region(self) -> str:
    #     pass

    # @abstractmethod
    # def draw_ridges_count_on_region(self) -> str:
    #     pass

    # @abstractmethod
    # def count_fingerprint_ridges(self) -> str:
    #     pass



"""
Peakfinder8 algorithm for serial crystallography hit finding.

Faithful Python/NumPy implementation of the 8-step Peakfinder8 algorithm
from Cheetah (Barty et al.), used as the state-of-the-art classical baseline.

References:
- Barty et al., J. Appl. Cryst. (2014) 47, 1118-1131
- CrystFEL peakfinder8.c
"""

import numpy as np
from scipy import ndimage
from dataclasses import dataclass


@dataclass
class Peak:
    """A detected Bragg peak."""
    row: float          # intensity-weighted centroid row
    col: float          # intensity-weighted centroid col
    total_intensity: float  # background-subtracted integrated intensity
    snr: float          # peak signal-to-noise ratio
    n_pixels: int       # number of connected pixels


@dataclass
class PeakFinder8Result:
    """Result of running Peakfinder8 on a single frame."""
    peaks: list[Peak]
    n_peaks: int
    is_hit: bool


class PeakFinder8:
    """Peakfinder8 algorithm implementation.

    Args:
        beam_center: (row, col) of the beam center.
        min_snr: SNR threshold for peak pixel detection.
        min_pix_count: Minimum connected pixels per peak.
        max_pix_count: Maximum connected pixels per peak.
        min_adc: Absolute intensity floor.
        min_res: Minimum radius from beam center (pixels).
        max_res: Maximum radius from beam center (pixels).
        n_sigma_clip_iterations: Number of sigma-clipping iterations.
        n_peaks_threshold: Minimum peaks to classify as a hit.
    """

    def __init__(
        self,
        beam_center: tuple[int, int] = (256, 256),
        min_snr: float = 6.0,
        min_pix_count: int = 2,
        max_pix_count: int = 200,
        min_adc: float = 50.0,
        min_res: int = 20,
        max_res: int = 240,
        n_sigma_clip_iterations: int = 5,
        n_peaks_threshold: int = 15,
    ):
        self.beam_center = beam_center
        self.min_snr = min_snr
        self.min_pix_count = min_pix_count
        self.max_pix_count = max_pix_count
        self.min_adc = min_adc
        self.min_res = min_res
        self.max_res = max_res
        self.n_sigma_clip_iterations = n_sigma_clip_iterations
        self.n_peaks_threshold = n_peaks_threshold

        self._radial_map: np.ndarray | None = None
        self._ring_indices: dict | None = None
        self._valid_mask: np.ndarray | None = None

    def _precompute_geometry(self, image_shape: tuple[int, int]) -> None:
        """Step 1: Compute radial distance map and bin pixels into rings."""
        cy, cx = self.beam_center
        y, x = np.ogrid[: image_shape[0], : image_shape[1]]
        self._radial_map = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Integer radius bins
        r_int = self._radial_map.astype(int)

        # Valid pixel mask (within resolution limits)
        self._valid_mask = (
            (self._radial_map >= self.min_res) & (self._radial_map <= self.max_res)
        )

        # Group pixel indices by integer radius
        self._ring_indices = {}
        for r in range(self.min_res, self.max_res + 1):
            mask = (r_int == r) & self._valid_mask
            indices = np.where(mask)
            if len(indices[0]) > 0:
                self._ring_indices[r] = indices

    def _sigma_clip_background(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Steps 2-3: Iterative sigma-clipping to estimate per-ring background.

        Returns:
            mu_map: per-pixel background mean (broadcast from ring values)
            sigma_map: per-pixel background std (broadcast from ring values)
        """
        mu_map = np.zeros_like(image, dtype=np.float64)
        sigma_map = np.ones_like(image, dtype=np.float64)

        for r, indices in self._ring_indices.items():
            values = image[indices].astype(np.float64)

            # Iterative sigma clipping
            mask = np.ones(len(values), dtype=bool)
            for _ in range(self.n_sigma_clip_iterations):
                if mask.sum() < 3:
                    break
                v = values[mask]
                mu = v.mean()
                sigma = v.std()
                if sigma < 1e-10:
                    break
                threshold = mu + self.min_snr * sigma
                mask = mask & (values <= threshold)

            v = values[mask] if mask.sum() >= 3 else values
            mu = v.mean()
            sigma = max(v.std(), 1e-10)

            mu_map[indices] = mu
            sigma_map[indices] = sigma

        return mu_map, sigma_map

    def find_peaks(self, image: np.ndarray) -> PeakFinder8Result:
        """Run the full Peakfinder8 algorithm on a single frame.

        Args:
            image: 2D array (uint16 or float), single diffraction frame.

        Returns:
            PeakFinder8Result with detected peaks and hit classification.
        """
        image = image.astype(np.float64)

        # Step 1: Precompute geometry if needed (cached for same shape)
        if self._radial_map is None or self._radial_map.shape != image.shape:
            self._precompute_geometry(image.shape)

        # Steps 2-3: Background estimation
        mu_map, sigma_map = self._sigma_clip_background(image)

        # Step 3: Threshold map
        threshold_map = np.maximum(
            mu_map + self.min_snr * sigma_map, self.min_adc
        )

        # Step 4: Peak pixel identification
        peak_mask = (image > threshold_map) & self._valid_mask

        # Step 5: Connected-component labeling
        labeled, n_components = ndimage.label(peak_mask)

        # Steps 6-7: Validate and measure peaks
        peaks = []
        for comp_id in range(1, n_components + 1):
            comp_mask = labeled == comp_id
            n_pix = comp_mask.sum()

            # Step 6: Size filtering
            if n_pix < self.min_pix_count or n_pix > self.max_pix_count:
                continue

            # Step 7: Measure peak properties
            rows, cols = np.where(comp_mask)
            intensities = image[rows, cols]
            bg_values = mu_map[rows, cols]
            sigma_values = sigma_map[rows, cols]

            # Background-subtracted intensities
            net_intensities = intensities - bg_values
            total_intensity = net_intensities.sum()

            if total_intensity <= 0:
                continue

            # Intensity-weighted centroid
            weights = np.maximum(net_intensities, 0)
            w_sum = weights.sum()
            if w_sum <= 0:
                continue
            row_com = (rows * weights).sum() / w_sum
            col_com = (cols * weights).sum() / w_sum

            # Peak SNR
            mean_sigma = sigma_values.mean()
            snr = total_intensity / (mean_sigma * np.sqrt(n_pix))

            peaks.append(Peak(
                row=row_com,
                col=col_com,
                total_intensity=total_intensity,
                snr=snr,
                n_pixels=n_pix,
            ))

        # Step 8: Hit classification
        n_peaks = len(peaks)
        is_hit = n_peaks >= self.n_peaks_threshold

        return PeakFinder8Result(peaks=peaks, n_peaks=n_peaks, is_hit=is_hit)

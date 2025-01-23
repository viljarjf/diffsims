# -*- coding: utf-8 -*-
# Copyright 2017-2024 The diffsims developers
#
# This file is part of diffsims.
#
# diffsims is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# diffsims is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with diffsims.  If not, see <http://www.gnu.org/licenses/>.

"""Kinematic Diffraction Simulation Generator."""

from typing import Union, Sequence, Tuple
import numpy as np

from orix.quaternion import Rotation
from orix.crystal_map import Phase

from diffsims.crystallography._diffracting_vector import DiffractingVector
from diffsims.utils.shape_factor_models import (
    linear,
    atanc,
    lorentzian,
    sinc,
    sin2c,
    lorentzian_precession,
    _shape_factor_precession,
)

from diffsims.utils.sim_utils import (
    get_electron_wavelength,
    get_kinematical_intensities,
    is_lattice_hexagonal,
    get_points_in_sphere,
    get_intensities_params,
)

_shape_factor_model_mapping = {
    "linear": linear,
    "atanc": atanc,
    "sinc": sinc,
    "sin2c": sin2c,
    "lorentzian": lorentzian,
}

from diffsims.simulations import Simulation1D, Simulation2D

__all__ = ["SimulationGenerator"]


class SimulationGenerator:
    """
    A class for generating kinematic diffraction simulations.
    """

    def __repr__(self):
        return (
            f"SimulationGenerator(accelerating_voltage={self.accelerating_voltage}, "
            f"scattering_params={self.scattering_params}, "
            f"approximate_precession={self.approximate_precession})"
        )

    def __init__(
        self,
        accelerating_voltage: float = 200,
        scattering_params: str = "lobato",
        precession_angle: float = 0,
        shape_factor_model: str = "lorentzian",
        approximate_precession: bool = True,
        minimum_intensity: float = 1e-20,
        **kwargs,
    ):
        """
        Parameters
        ----------
        accelerating_voltage
            The accelerating voltage of the electrons in keV.
        scattering_params
            The scattering parameters to use. One of 'lobato', 'xtables'
        precession_angle
            The precession angle in degrees. If 0, no precession is applied.
        shape_factor_model
            The shape factor model to use. One of 'linear', 'atanc', 'sinc', 'sin2c', 'lorentzian'
        approximate_precession
            If True, the precession is approximated by a Lorentzian function.
        minimum_intensity
            The minimum intensity of a reflection to be included in the profile.
        kwargs
            Keyword arguments to pass to the shape factor model.

        """
        self.accelerating_voltage = accelerating_voltage
        self.precession_angle = np.abs(precession_angle)
        self.approximate_precession = approximate_precession
        if isinstance(shape_factor_model, str):
            if shape_factor_model in _shape_factor_model_mapping.keys():
                self.shape_factor_model = _shape_factor_model_mapping[
                    shape_factor_model
                ]
            else:
                raise NotImplementedError(
                    f"{shape_factor_model} is not a recognized shape factor "
                    f"model, choose from: {_shape_factor_model_mapping.keys()} "
                    f"or provide your own function."
                )
        else:
            self.shape_factor_model = shape_factor_model
        self.minimum_intensity = minimum_intensity
        self.shape_factor_kwargs = kwargs
        if scattering_params in ["lobato", "xtables", None]:
            self.scattering_params = scattering_params
        else:
            raise NotImplementedError(
                "The scattering parameters `{}` is not implemented. "
                "See documentation for available "
                "implementations.".format(scattering_params)
            )

    @property
    def wavelength(self):
        return get_electron_wavelength(self.accelerating_voltage)

    def calculate_diffraction2d(
        self,
        phase: Union[Phase, Sequence[Phase]],
        rotation: Union[Rotation, Sequence[Rotation]] = Rotation.from_euler(
            (0, 0, 0), degrees=True
        ),
        reciprocal_radius: float = 1.0,
        with_direct_beam: bool = True,
        max_excitation_error: float = 1e-2,
        shape_factor_width: float = None,
        debye_waller_factors: dict = None,
    ):
        """Calculates the diffraction pattern for one or more phases given a list
        of rotations for each phase.

        Parameters
        ----------
        phase:
            The phase(s) for which to derive the diffraction pattern.
        reciprocal_radius
            The maximum radius of the sphere of reciprocal space to
            sample, in reciprocal Angstroms.
        rotation
            The Rotation object(s) to apply to the structure and then
            calculate the diffraction pattern.
        with_direct_beam
            If True, the direct beam is included in the simulated
            diffraction pattern. If False, it is not.
        max_excitation_error
            The cut-off for geometric excitation error in the z-direction
            in units of reciprocal Angstroms. Spots with a larger distance
            from the Ewald sphere are removed from the pattern.
            Related to the extinction distance and roughly equal to 1/thickness.
        shape_factor_width
            Determines the width of the reciprocal rel-rod, for fine-grained
            control. If not set will be set equal to max_excitation_error.
        debye_waller_factors
            Maps element names to their temperature-dependent Debye-Waller factors.

        Returns
        -------
        diffsims.sims.diffraction_simulation.DiffractionSimulation
            The data associated with this structure and diffraction setup.
        """
        if isinstance(phase, Phase):
            phase = [phase]
        if isinstance(rotation, Rotation):
            rotation = [rotation]
        if len(phase) != len(rotation):
            raise ValueError(
                "The number of phases and rotations must be equal. "
                f"Got {len(phase)} phases and {len(rotation)} rotations."
            )

        if debye_waller_factors is None:
            debye_waller_factors = {}
        # Specify variables used in calculation
        wavelength = self.wavelength

        # Rotate using all the rotations in the list
        vectors = []
        for p, rotate in zip(phase, rotation):
            recip = DiffractingVector.from_min_dspacing(
                p,
                min_dspacing=1 / reciprocal_radius,
                include_zero_vector=with_direct_beam,
            )
            recip.sanitise_phase()
            # No need to pre-calculate structure factors if only a few rotations are present
            if rotate.size > 4:
                recip.calculate_structure_factor(
                    scattering_params=self.scattering_params,
                    debye_waller_factors=debye_waller_factors,
                    )
            phase_vectors = []
            for rot in rotate:
                # Calculate the reciprocal lattice vectors that intersect the Ewald sphere.
                intersection, excitation_error = get_intersection_with_ewalds_sphere(
                    recip,
                    rot,
                    wavelength,
                    max_excitation_error,
                    self.precession_angle,
                )
                # Select intersected reflections
                intersected_vectors = recip[intersection].rotate_with_basis(rot)
                excitation_error = excitation_error[intersection]
                r_spot = intersected_vectors.norm

                # Calculate shape factor
                shape_factor = self.get_shape_factor(
                    excitation_error, max_excitation_error, r_spot, shape_factor_width
                )
                # Calculate intensity
                if rotate.size <= 4:
                    v = recip[intersection]
                    v.calculate_structure_factor(
                        scattering_params=self.scattering_params,
                        debye_waller_factors=debye_waller_factors,
                        )
                    f_hkls = v.structure_factor
                else:
                    f_hkls = recip[intersection].structure_factor
                intensities = (f_hkls * f_hkls.conjugate()).real * shape_factor

                # Threshold peaks included in simulation as factor of zero beam intensity.
                peak_mask = intensities > np.max(intensities) * self.minimum_intensity
                intensities = intensities[peak_mask]
                intersected_vectors = intersected_vectors[peak_mask]
                intersected_vectors.intensity = intensities
                phase_vectors.append(intersected_vectors)
            vectors.append(phase_vectors)

        if len(phase) == 1:
            vectors = vectors[0]
            phase = phase[0]
            rotation = rotation[0]
            if rotation.size == 1:
                vectors = vectors[0]

        # Create a simulation object
        sim = Simulation2D(
            phases=phase,
            coordinates=vectors,
            rotations=rotation,
            simulation_generator=self,
            reciprocal_radius=reciprocal_radius,
        )
        return sim

    def calculate_diffraction1d(
        self,
        phase: Phase,
        reciprocal_radius: float = 1.0,
        minimum_intensity: float = 1e-3,
        debye_waller_factors: dict = None,
    ):
        """Calculates the 1-D profile of the diffraction pattern for one phases.

        This is useful for plotting the diffracting reflections for some phases.

        Parameters
        ----------
        phase:
            The phase for which to derive the diffraction pattern.
        reciprocal_radius
            The maximum radius of the sphere of reciprocal space to
            sample, in reciprocal Angstroms.
        minimum_intensity
            The minimum intensity of a reflection to be included in the profile.
        debye_waller_factors
            Maps element names to their temperature-dependent Debye-Waller factors.
        """
        latt = phase.structure.lattice

        # Obtain crystallographic reciprocal lattice points within range
        recip_latt = latt.reciprocal()
        spot_indices, _, spot_distances = get_points_in_sphere(
            recip_latt, reciprocal_radius
        )

        ##spot_indicies is a numpy.array of the hkls allowed in the recip radius
        g_indices, multiplicities, g_hkls = get_intensities_params(
            recip_latt, reciprocal_radius
        )

        i_hkl = get_kinematical_intensities(
            phase.structure,
            g_indices,
            np.asarray(g_hkls),
            prefactor=multiplicities,
            scattering_params=self.scattering_params,
            debye_waller_factors=debye_waller_factors,
        )

        if is_lattice_hexagonal(latt):
            # Use Miller-Bravais indices for hexagonal lattices.
            g_indices = np.array(
                [
                    g_indices[:, 0],
                    g_indices[:, 1],
                    g_indices[:, 0] - g_indices[:, 1],
                    g_indices[:, 2],
                ]
            ).T

        hkls_labels = ["".join([str(int(x)) for x in xs]) for xs in g_indices]

        peaks = []
        for l, i, g in zip(hkls_labels, i_hkl, g_hkls):
            peaks.append((l, [i, g]))

        # Scale intensities so that the max intensity is 100.

        max_intensity = max([v[1][0] for v in peaks])
        reciporical_spacing = []
        intensities = []
        hkls = []
        for p in peaks:
            label, v = p  # (label, [intensity,g])
            if v[0] / max_intensity * 100 > minimum_intensity and (label != "000"):
                reciporical_spacing.append(v[1])
                intensities.append(v[0])
                hkls.append(label)

        intensities = np.asarray(intensities) / max(intensities) * 100

        return Simulation1D(
            phase=phase,
            reciprocal_spacing=reciporical_spacing,
            intensities=intensities,
            hkl=hkls,
            reciprocal_radius=reciprocal_radius,
            wavelength=self.wavelength,
        )

    def get_shape_factor(
        self,
        excitation_error: np.ndarray,
        max_excitation_error: float,
        r_spot: np.ndarray = None,
        shape_factor_width: float = None,
    ) -> np.ndarray:

        if shape_factor_width is None:
            shape_factor_width = max_excitation_error
        # select and evaluate shape factor model
        if self.precession_angle == 0:
            # calculate shape factor
            shape_factor = self.shape_factor_model(
                excitation_error, shape_factor_width, **self.shape_factor_kwargs
            )
        else:
            if self.approximate_precession:
                shape_factor = lorentzian_precession(
                    excitation_error,
                    shape_factor_width,
                    r_spot,
                    np.deg2rad(self.precession_angle),
                )
            else:
                if r_spot is None:
                    raise ValueError(
                        "Must supply `r_spot` parameter when not approximating precession"
                    )
                shape_factor = _shape_factor_precession(
                    excitation_error,
                    r_spot,
                    np.deg2rad(self.precession_angle),
                    self.shape_factor_model,
                    shape_factor_width,
                    **self.shape_factor_kwargs,
                )
        return shape_factor

    def get_intersecting_reflections(
        self,
        recip: DiffractingVector,
        recip_hkl: np.ndarray,
        rot: Rotation,
        wavelength: float,
        max_excitation_error: float,
        shape_factor_width: float = None,
        with_direct_beam: bool = True,
    ):
        """Calculates the reciprocal lattice vectors that intersect the Ewald sphere.

        Parameters
        ----------
        recip
            The reciprocal lattice vectors to rotate.
        rot
            The rotation matrix to apply to the reciprocal lattice vectors.
        wavelength
            The wavelength of the electrons in Angstroms.
        max_excitation_error
            The cut-off for geometric excitation error in the z-direction
            in units of reciprocal Angstroms. Spots with a larger distance
            from the Ewald sphere are removed from the pattern.
            Related to the extinction distance and roungly equal to 1/thickness.
        shape_factor_width
            Determines the width of the reciprocal rel-rod, for fine-grained
            control. If not set will be set equal to max_excitation_error.
        """
        if with_direct_beam:
            # Modify the data directly, as `data` is a property
            recip._data = np.vstack([recip._data, [0, 0, 0]])
            recip_hkl = np.vstack([recip_hkl, [0, 0, 0]])

        intersection, excitation_error = get_intersection_with_ewalds_sphere(
            recip,
            rot,
            wavelength,
            max_excitation_error,
            self.precession_angle,
        )
        intersected_vectors = recip[intersection].rotate_with_basis(rot)
        excitation_error = excitation_error[intersection]
        hkl = recip_hkl[intersection]
        r_spot = intersected_vectors.norm

        shape_factor = self.get_shape_factor(
            excitation_error, max_excitation_error, r_spot, shape_factor_width
        )
        return intersected_vectors, hkl, shape_factor


# TODO consider refactoring into a seperate file
def get_intersection_with_ewalds_sphere(
    recip: DiffractingVector,
    rot: Rotation,
    wavelength: float,
    max_excitation_error: float,
    precession_angle: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the reciprocal lattice vectors that intersect the Ewald sphere.

    Parameters
    ----------
    recip
        The reciprocal lattice vectors to rotate.
    rot
        The rotation to apply to the reciprocal lattice vectors.
    wavelength
        The wavelength of the electrons in Angstroms.
    max_excitation_error
        The cut-off for geometric excitation error in the z-direction
        in units of reciprocal Angstroms. Spots with a larger distance
        from the Ewald sphere are removed from the pattern.
        Related to the extinction distance and roungly equal to 1/thickness.
    precession_angle
        Degrees

    Returns
    -------
    intersection
        Array of bools. True where the vectors intersect
    excitation_error
        Excitation error of all vectors
    """
    # Identify the excitation errors of all points (distance from point to Ewald sphere)

    # Instead of rotating vectors, rotate Ewald's sphere to find intersections.
    # Only rotate the intersecting vectors.
    # Using notation from https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    r = 1 / wavelength
    u = rot.to_matrix().squeeze() @ np.array([0, 0, 1])
    c = r * u
    o = recip.data

    diff = o - c
    dot = np.dot(u, diff.T)
    nabla = dot**2 - np.sum(diff**2, axis=1) + r**2
    # We know the reflections are all going to intersect twice
    # Therefore, no need to look at the sign of nabla
    # We also know only the smaller root is important, i.e. -sqrt(nabla)
    sqrt_nabla = np.sqrt(nabla)
    d = -dot - sqrt_nabla

    excitation_error = d

    # determine the pre-selection reflections
    if precession_angle == 0:
        intersection = np.abs(excitation_error) < max_excitation_error
    else:
        # Using the following script to find equations for upper and lower bounds for precessing Ewald's sphere
        """
        import sympy
        import numpy as np

        a = sympy.Symbol("a") # Precession angle
        r = sympy.Symbol("r") # Ewald's sphere radius
        rho, z = sympy.symbols("rho z") # cylindrical coordinates of reflection

        rot = lambda ang: np.asarray([[sympy.cos(ang), -sympy.sin(ang)],[sympy.sin(ang), sympy.cos(ang)]])

        u = np.asarray([0, 1])
        c = r * u
        cl = rot(a) @ c
        cr = rot(-a) @ c
        o = np.asarray([rho, z])

        def get_d(_c):
            diff = o - _c
            dot = np.dot(u, diff)
            nabla = dot**2 - sum(i**2 for i in diff) + r**2
            sqrt_nabla = nabla**0.5
            return  -dot - sqrt_nabla

        d = get_d(c)
        d_upper = get_d(cl)
        d_lower = get_d(cr)

        print(d.simplify())             # r - z - (r**2 - rho**2)**0.5
        print((d_upper - d).simplify()) # r*cos(a) - r + (r**2 - rho**2)**0.5 - (r**2 - (r*sin(a) + rho)**2)**0.5
        print((d_lower - d).simplify()) # r*cos(a) - r + (r**2 - rho**2)**0.5 - (r**2 - (r*sin(a) - rho)**2)**0.5
        """
        # In the above script, d is the same as before.
        # We need the distance of the reflections from the incident beam, i.e. the cylindrical coordinate rho
        # (using https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Vector_formulation):
        rho = np.linalg.norm(np.dot(o, u)[:, np.newaxis] * u - o, axis=1)
        a = np.deg2rad(precession_angle)
        first_half = r * np.cos(a) - r + (r**2 - rho**2) ** 0.5
        upper = first_half - (r**2 - (r * np.sin(a) + rho) ** 2) ** 0.5
        lower = first_half - (r**2 - (r * np.sin(a) - rho) ** 2) ** 0.5
        intersection = (d < (upper + max_excitation_error)) & (
            d > (lower - max_excitation_error)
        )
    return intersection, excitation_error

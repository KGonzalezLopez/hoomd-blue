// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __PAIR_EVALUATOR_STICKY_SPHERES_H__
#define __PAIR_EVALUATOR_STICKY_SPHERES_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairStickySpheres.h
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif



/*
 * I decided not to change the documentation below, that refers to the LJ potential. It is meant to serve as a general example of a pairwise 
 * interaction, so I just leave it as is. -Geert.
 * */

//! Class for evaluating the LJ pair potential
/*! <b>General Overview</b>

    EvaluatorPairLJ is a low level computation class that computes the LJ pair potential V(r). As the standard
    MD potential, it also serves as a well documented example of how to write additional pair potentials. "Standard"
    pair potentials in hoomd are all handled via the template class PotentialPair. PotentialPair takes a potential
    evaluator as a template argument. In this way, all the complicated data management and other details of computing
    the pair force and potential on every single particle is only written once in the template class and the difference
    V(r) potentials that can be calculated are simply handled with various evaluator classes. Template instantiation is
    equivalent to inlining code, so there is no performance loss.

    In hoomd, a "standard" pair potential is defined as V(rsq, rcutsq, params, di, dj, qi, qj), where rsq is the squared
    distance between the two particles, rcutsq is the cutoff radius at which the potential goes to 0, params is any
    number of per type-pair parameters, di, dj are the diameters of particles i and j, and qi, qj are the charges of
    particles i and j respectively.

    Diameter and charge are not always needed by a given pair evaluator, so it must provide the functions
    needsDiameter() and needsCharge() which return boolean values signifying if they need those quantities or not. A
    false return value notifies PotentialPair that it need not even load those values from memory, boosting performance.

    If needsDiameter() returns true, a setDiameter(Scalar di, Scalar dj) method will be called to set the two diameters.
    Similarly, if needsCharge() returns true, a setCharge(Scalar qi, Scalar qj) method will be called to set the two
    charges.

    All other arguments are common among all pair potentials and passed into the constructor. Coefficients are handled
    in a special way: the pair evaluator class (and PotentialPair) manage only a single parameter variable for each
    type pair. Pair potentials that need more than 1 parameter can specify that their param_type be a compound
    structure and reference that. For coalesced read performance on G200 GPUs, it is highly recommended that param_type
    is one of the following types: Scalar, Scalar2, Scalar4.

    The program flow will proceed like this: When a potential between a pair of particles is to be evaluated, a
    PairEvaluator is instantiated, passing the common parameters to the constructor and calling setDiameter() and/or
    setCharge() if need be. Then, the evalForceAndEnergy() method is called to evaluate the force and energy (more
    on that later). Thus, the evaluator must save all of the values it needs to compute the force and energy in member
    variables.

    evalForceAndEnergy() makes the necessary computations and sets the out parameters with the computed values.
    Specifically after the method completes, \a force_divr must be set to the value
    \f$ -\frac{1}{r}\frac{\partial V}{\partial r}\f$ and \a pair_eng must be set to the value \f$ V(r) \f$ if \a energy_shift is false or
    \f$ V(r) - V(r_{\mathrm{cut}}) \f$ if \a energy_shift is true.

    A pair potential evaluator class is also used on the GPU. So all of its members must be declared with the
    DEVICE keyword before them to mark them __device__ when compiling in nvcc and blank otherwise. If any other code
    needs to diverge between the host and device (i.e., to use a special math function like __powf on the device), it
    can similarly be put inside an ifdef NVCC block.

    <b>LJ specifics</b>

    EvaluatorPairLJ evaluates the function:
    \f[ V_{\mathrm{LJ}}(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
                                            \alpha \left( \frac{\sigma}{r} \right)^{6} \right] \f]
    broken up as follows for efficiency
    \f[ V_{\mathrm{LJ}}(r) = r^{-6} \cdot \left( 4 \varepsilon \sigma^{12} \cdot r^{-6} -
                                            4 \alpha \varepsilon \sigma^{6} \right) \f]
    . Similarly,
    \f[ -\frac{1}{r} \frac{\partial V_{\mathrm{LJ}}}{\partial r} = r^{-2} \cdot r^{-6} \cdot
            \left( 12 \cdot 4 \varepsilon \sigma^{12} \cdot r^{-6} - 6 \cdot 4 \alpha \varepsilon \sigma^{6} \right) \f]

    The LJ potential does not need diameter or charge. Two parameters are specified and stored in a Scalar2. \a lj1 is
    placed in \a params.x and \a lj2 is in \a params.y.

    These are related to the standard lj parameters sigma and epsilon by:
    - \a lj1 = 4.0 * epsilon * pow(sigma,12.0)
    - \a lj2 = alpha * 4.0 * epsilon * pow(sigma,6.0);

*/

//! Parameter type for this potential
struct sticky_spheres_params
    {
    Scalar2 coeffs; //!< Contains 1/sigma^2, r_min2
    Scalar4 smoothing_constants; //!< c0, c2, c4, c6 to smooth potential.
    Scalar2 smoothing_constants_pot; //!< a, b to smooth potential.
};

class EvaluatorPairStickySpheres
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef sticky_spheres_params param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairStickySpheres(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), one_over_sigma2(_params.coeffs.x), r_min2(_params.coeffs.y),
						c0(_params.smoothing_constants.x), c2(_params.smoothing_constants.y), c4(_params.smoothing_constants.z), c6(_params.smoothing_constants.w),
						a(_params.smoothing_constants_pot.x), b(_params.smoothing_constants_pot.y)
            {
            }

        //! LJ doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! LJ doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed
                  in PotentialPair.

            \return True if they are evaluated or false if they are not because we are beyond the cutoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift) {
            if (rsq < rcutsq) {
								Scalar one_over_r2 = 1.0 / rsq;
								Scalar r2_over_sigma2 = rsq * one_over_sigma2;									
								Scalar sigma6_over_r6 = 1.0 / (r2_over_sigma2*r2_over_sigma2*r2_over_sigma2);

								if (rsq < r_min2) {
									force_divr = 24.0*one_over_r2*sigma6_over_r6* ( -1.0 + 2.0*sigma6_over_r6 );
									pair_eng = 4.0*sigma6_over_r6*(sigma6_over_r6 - 1.0);
								} else {
									force_divr = 2.0*one_over_r2*( -r2_over_sigma2*(c2 + r2_over_sigma2*(2.0*c4 + 3.0*c6*r2_over_sigma2)) + sigma6_over_r6*(6.0*a*sigma6_over_r6 - 3.0*b) );
									pair_eng = sigma6_over_r6*( a*sigma6_over_r6 - b ) + c0 + r2_over_sigma2*( c2 + r2_over_sigma2*(c4 + c6*r2_over_sigma2 ) );
								}
								return true;
						}
            else {
                return false;
						}
          }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("stickyspheres");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar one_over_sigma2;
        Scalar r_min2;
        Scalar c0;
        Scalar c2;
        Scalar c4;
        Scalar c6;
        Scalar a;
        Scalar b;
    };


#endif // __PAIR_EVALUATOR_STICKY_SPHERES_H__

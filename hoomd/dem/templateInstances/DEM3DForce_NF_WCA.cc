// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include <hoomd/HOOMDMath.h>

#include "../DEMEvaluator.h"
#include "../NoFriction.h"
#include "../WCAPotential.h"
#include "../SWCAPotential.h"
#include "../DEM2DForceCompute.h"
#include "../DEM2DForceComputeGPU.h"
#include "../DEM3DForceCompute.h"
#include "../DEM3DForceComputeGPU.h"

// Include boost.python to do the exporting
#include <boost/python.hpp>
using namespace boost::python;

void export_NF_WCA_3D()
    {
    typedef WCAPotential<Scalar, Scalar4, NoFriction<Scalar> > WCA;
    typedef DEM3DForceCompute<Scalar, Scalar4, WCA> WCA_DEM_3D;

    class_<WCA_DEM_3D, boost::shared_ptr<WCA_DEM_3D>, bases<ForceCompute>, boost::noncopyable >
        ("WCADEM3D", init< boost::shared_ptr<SystemDefinition>,
        boost::shared_ptr<NeighborList>, Scalar, WCA>())
        .def("setParams", &WCA_DEM_3D::setParams)
        .def("setRcut", &WCA_DEM_3D::setRcut)
        ;

#ifdef ENABLE_CUDA
    typedef DEM3DForceComputeGPU<Scalar, Scalar4, WCA> WCA_DEM_3D_GPU;

    class_<WCA_DEM_3D_GPU, boost::shared_ptr<WCA_DEM_3D_GPU>,
           bases<WCA_DEM_3D>, boost::noncopyable >
        ("WCADEM3DGPU", init< boost::shared_ptr<SystemDefinition>,
        boost::shared_ptr<NeighborList>, Scalar, WCA>())
        .def("setParams", &WCA_DEM_3D_GPU::setParams)
        .def("setRcut", &WCA_DEM_3D_GPU::setRcut)
        .def("setAutotunerParams", &WCA_DEM_3D_GPU::setAutotunerParams)
        ;
#endif
    }
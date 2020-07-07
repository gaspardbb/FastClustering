// #define NDEBUGCERR

#include "utils/vose_alias.hpp"
#include "utils/clustering.hpp"
#include "utils/greenkhorn.hpp"

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::VectorXd;
using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace py = pybind11;

PYBIND11_MODULE(_clustering, m)
{
    using namespace pybind11::literals;
    m.doc() = "Clustering and OT module.";

    // VoseAlias sampler
    py::class_<VoseAlias<float>>(m, "VoseAlias")
        .def(py::init<const Eigen::VectorXf &>())
        .def("sample", &VoseAlias<float>::sampleEig, "n"_a = 1)
        .def("__repr__", [](const VoseAlias<float> &a) {
            std::ostringstream m_str;
            m_str << "<VoseAlias sampler ranging from 0 to " << static_cast<int>(a.getSize()) << ".>";
            return m_str.str();
        });

    // clustering header
    m.def("kmeanspp", &clustering::kmeanspp,
          "points"_a.noconvert(), "k"_a, "eps"_a = 0.);

    m.def("afkmc2", &clustering::afkmc2Eig,
          "points"_a.noconvert(), "k"_a, "m"_a);

    // Khorn stuff
    {
        using namespace greenkhorn;
        py::class_<khorn_return>(m, "KhornReturn")
            .def_readonly("u", &khorn_return::u)
            .def_readonly("v", &khorn_return::v)
            .def_property_readonly("errors", [](const khorn_return &a) {
                return a.errors;
            }) // This can be a costly operation, as it involves copying the array of errors.
               // Define it as a property.
            .def_property_readonly("error_msg", [](const khorn_return &a) {
                return static_cast<int>(a.k_error);
            }) 
            .def_property_readonly("transport_cost", &khorn_return::transportCost)
            .def_property_readonly("cost_matrix_max", &khorn_return::costMatrixMax)
            .def_readonly("cost_matrix", &khorn_return::cost_matrix)
            .def_readonly("transport_plan", &khorn_return::transport_plan)
            .def_readonly("duality_gap", &khorn_return::duality_gap);

        m.def("entropic_wasserstein_green", py::overload_cast<const Ref<const MatrixXdR> &, const Ref<const MatrixXdR> &, double>(entropicWasserstein<KHORN_TYPE::GREEN>),
              "X"_a.noconvert(), "Y"_a.noconvert(), "precision"_a);
        m.def("entropic_wasserstein_sin", py::overload_cast<const Ref<const MatrixXdR> &, const Ref<const MatrixXdR> &, double>(entropicWasserstein<KHORN_TYPE::SIN>),
              "X"_a.noconvert(), "Y"_a.noconvert(), "precision"_a);
        m.def("entropic_wasserstein_sin_warm", &entropicWassersteinAddPoint<KHORN_TYPE::SIN>,
              "C"_a.noconvert(), "X"_a.noconvert(), "Y"_a.noconvert(),
              "x"_a.noconvert(), "y"_a.noconvert(), "precision"_a, 
              "initial_u"_a, "initial_v"_a,
              "Testing purposes. Function is not optimized.");
    }
}

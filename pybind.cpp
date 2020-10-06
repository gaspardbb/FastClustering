// #define DEBUGCERR

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
	{
		using namespace clustering;
		m.def("kmeanspp", &kmeanspp,
			  "points"_a.noconvert(), "k"_a, "eps"_a = 0.);
		m.def("afkmc2", &afkmc2Eig,
			  "points"_a.noconvert(), "k"_a, "m"_a);
		py::class_<OnlineKMeans>(m, "OnlineKMeans")
			.def_property_readonly("centroids", &OnlineKMeans::getCentroids)
			.def_property_readonly("weights", &OnlineKMeans::getWeights)
			.def_property_readonly("lr", &OnlineKMeans::getLr);
	}

	// Khorn stuff
	{
		using namespace greenkhorn;
		py::class_<khorn_return>(m, "KhornReturn")
			.def_readonly("u", &khorn_return::u)
			.def_readonly("v", &khorn_return::v)
			.def_readonly("eta", &khorn_return::eta)
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

		// This is experimental stuff
		m.def("entropic_wasserstein_green", py::overload_cast<const Ref<const MatrixXdR> &, const Ref<const MatrixXdR> &, double, int>(entropicWasserstein<KHORN_TYPE::GREEN>),
			  "X"_a.noconvert(), "Y"_a.noconvert(), "precision"_a, "iter_max"_a = 10000);
		m.def("entropic_wasserstein_sin", py::overload_cast<const Ref<const MatrixXdR> &, const Ref<const MatrixXdR> &, double, int>(entropicWasserstein<KHORN_TYPE::SIN>),
			  "X"_a.noconvert(), "Y"_a.noconvert(), "precision"_a, "iter_max"_a = 10000);
		m.def("entropic_wasserstein_kmeans_green", py::overload_cast<const Ref<const MatrixXdR> &, const Ref<const MatrixXdR> &, double, int>(entropicWassersteinKMeans<KHORN_TYPE::GREEN>),
			  "X"_a.noconvert(), "Y"_a.noconvert(), "precision"_a, "iter_max"_a = 10000);
		m.def("entropic_wasserstein_kmeans_sin", py::overload_cast<const Ref<const MatrixXdR> &, const Ref<const MatrixXdR> &, double, int>(entropicWassersteinKMeans<KHORN_TYPE::SIN>),
			  "X"_a.noconvert(), "Y"_a.noconvert(), "precision"_a, "iter_max"_a = 10000);
		m.def("entropic_wasserstein_sin_warm", &entropicWassersteinAddPoint<KHORN_TYPE::SIN>,
			  "C"_a.noconvert(), "X"_a.noconvert(), "Y"_a.noconvert(),
			  "x"_a.noconvert(), "y"_a.noconvert(), "precision"_a,
			  "initial_u"_a, "initial_v"_a,
			  "Testing purposes. Function is not optimized.");
		m.def("entropic_wasserstein_green_warm", &entropicWassersteinAddPoint<KHORN_TYPE::GREEN>,
			  "C"_a.noconvert(), "X"_a.noconvert(), "Y"_a.noconvert(),
			  "x"_a.noconvert(), "y"_a.noconvert(), "precision"_a,
			  "initial_u"_a, "initial_v"_a,
			  "Testing purposes. Function is not optimized.");

		m.def("greenkhorn", &greenkhorn::greenkhorn,
			  "C"_a.noconvert(), "r"_a.noconvert(), "c"_a.noconvert(),
			  "eps"_a, "eta"_a, "u"_a.noconvert(), "v"_a.noconvert(),
			  "iter_max"_a = 100000, "recover_nan"_a = 1);
		m.def("sinkhorn", &greenkhorn::sinkhorn,
			  "C"_a.noconvert(), "r"_a.noconvert(), "c"_a.noconvert(),
			  "eps"_a, "eta"_a, "u"_a.noconvert(), "v"_a.noconvert(),
			  "iter_max"_a = 100000);

		py::class_<PairedOnlineKMeans>(m, "PairedOnlineKMeans")
			.def(py::init<MatrixXdR &, MatrixXdR &, double>(),
				 "X_init"_a.noconvert(), "Y_init"_a.noconvert(), "lr"_a)
			.def("add_points", &PairedOnlineKMeans::addPoints,
				 "x"_a.noconvert(), "y"_a.noconvert())
			.def("add_points_soft", &PairedOnlineKMeans::addPointsSoft,
				 "x"_a.noconvert(), "y"_a.noconvert())
			.def_readonly("centroids_x", &PairedOnlineKMeans::centroids_x)
			.def_readonly("centroids_y", &PairedOnlineKMeans::centroids_y)
			.def_property_readonly("cost_matrix", &PairedOnlineKMeans::getCost);
	}
}

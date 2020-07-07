#ifndef VOSE_ALIAS
#define VOSE_ALIAS

#include <stack>
#include <cstdint>
#include <iostream>
#include <random>
#include <cassert>
#include <Eigen/Dense>

template <typename T>
class VoseAlias
{
public:
    // Could probably template the index type, to return samples which have the right index.
    // Might be better handled from the user's side, and stick with unsigned int here.
    using index_t = std::uint_fast32_t;

private:
    index_t m_size;
    std::vector<T> m_proba_table{};
    std::vector<index_t> m_alias_table{};
    std::uniform_int_distribution<index_t> m_discrete_distribution;
    std::uniform_real_distribution<T> m_uniform_distribution{0., 1.};
    std::random_device rd_{};
    std::mt19937 m_gen{rd_()};

public:
    /**
     * @brief Construct a new Vose Alias object.
     * 
     * @param probabilities pointer to the array of probabilities. Probabilities must sum to 1. 
     * @param size size of the array of probabilities.
     * 
     * Made a first implementation with arrays. But std::vector are said to be as good. 
     */
    VoseAlias(const T *probabilities, int size) : m_size{static_cast<index_t>(size)},
                                                  m_proba_table(m_size),
                                                  m_alias_table(m_size),
                                                  m_discrete_distribution(0, m_size - 1)
    {
        // Copy of the probability array, so as not to modify it
        // Using reserve instead of the call to the constructor prevent from initializing the array to 0.
        // But this require using push_back() after that. In the end, both are more or less the same, and initializing
        // with basic types is not so costly. If this turns out to be too slow, use C-style array instead.
        std::vector<T> m_probabilities(m_size);

        for (size_t i = 0; i < m_size; i++)
        {
            m_probabilities[i] = probabilities[i];
        }

        // T sum {std::accumulate(m_probabilities.begin(), m_probabilities.end(), 0.)};
        // assert(std::fabs(1. - sum) < 10*std::numeric_limits<T>::epsilon() && "Sum of probabilities must sum to 1.");

        std::stack<index_t> small{};
        std::stack<index_t> large{};

        for (index_t i = 0; i < m_size; i++)
        {
            m_probabilities[i] = probabilities[i] * size;
            m_probabilities[i] < 1. ? small.push(i) : large.push(i);
        }

        index_t l;
        index_t g;
        T proba_g;
        while (!small.empty() && !large.empty())
        {
            l = small.top(), small.pop();
            g = large.top(), large.pop();

            m_proba_table[l] = m_probabilities[l];
            m_alias_table[g] = g;

            proba_g = (m_probabilities[g] + m_probabilities[l]) - 1;
            proba_g < 1 ? small.push(g) : large.push(g);
        }

        // Due to numerical imprecisions, both stacks may not be empty: add ones.
        while (!large.empty())
        {
            g = large.top(), large.pop();
            m_proba_table[g] = 1.;
        }

        while (!small.empty())
        {
            l = small.top(), small.pop();
            m_proba_table[l] = 1.;
        }
    }

    // Eigen interface
    VoseAlias(const Eigen::Matrix<T, Eigen::Dynamic, 1> &probabilities) : VoseAlias(probabilities.data(), probabilities.rows())
    {
    }

    index_t sample()
    {
        index_t i{m_discrete_distribution(m_gen)};
        T alpha{m_uniform_distribution(m_gen)};
        return alpha < m_proba_table[i] ? i : m_alias_table[i];
    }

    std::vector<index_t> sample(const size_t n)
    {
        std::vector<index_t> result(n);
        for (auto &value : result)
        {
            value = sample();
        }

        // Compiler does return by value optimization (RVO)
        return result;
    }

    /**
     * @brief Sample and return an Eigen vector. Useful for interfacing with numpy's arrays. 
     * 
     * @param n Number of samples. 
     * @return Eigen::Matrix<index_t, Eigen::Dynamic, 1> 
     */
    Eigen::Matrix<index_t, Eigen::Dynamic, 1> sampleEig(const size_t n)
    {
        Eigen::Matrix<index_t, Eigen::Dynamic, 1> result{n};
        for (auto i = 0; i < n; i++)
        {
            result(i) = sample();
        }
        return result;
    }

    void sampleInPlace(size_t *out, const size_t n)
    {
        for (auto i = 0; i < n; i++)
        {
            out[i] = sample();
        }
    }

    index_t getSize() const {return m_size;}

    template <typename U>
    friend std::ostream &operator<<(std::ostream &out, const VoseAlias<U> &sampler);
};

/**
 * @brief Counts the value of each id in a vector.
 * 
 * @tparam T Type of the vector
 * @param v Vector containing values ranging from 0 to max_val-1.
 * @param max_val Maximum value of the vector. 
 */
template <typename T>
void mean_count(std::vector<T> &v, size_t max_val)
{
    std::vector<int> values_count(max_val, 0);
    for (auto &&val : v)
    {
        values_count[val] += 1;
    }

    double total_size{static_cast<double>(v.size())};

    for (auto &&val : values_count)
    {
        std::cout << val / total_size << " " << std::endl;
    }
}

#endif /* VOSE_ALIAS */

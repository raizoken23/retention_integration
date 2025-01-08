#pragma once

#include <torch/extension.h>
#include <cute/tensor.hpp>

#include <vector>
#include <tuple>
#include <cassert>
#include <limits>
#include <cmath>
#include <unordered_map>
#include <mutex>
#include <optional>
#include "state.h"


namespace std {
    template<typename... Ts>
    struct hash<std::tuple<Ts...>> {
        size_t operator()(const std::tuple<Ts...>& t) const {
            return std::apply([](const auto&... args) {
                size_t seed = 0;
                (... , (seed ^= std::hash<std::decay_t<decltype(args)>>{}(args) + 0x9e3779b9 + (seed << 6) + (seed >> 2)));
                return seed;
            }, t);
        }
    };
}


template <typename Key, typename Value>
struct Cache {
    static Cache& getInstance() {
        static Cache instance;
        return instance;
    }

    void put(const Key& key, const Value& value) {
        std::lock_guard<std::mutex> lock(mtx);
        cache[key] = value;
    }

    std::optional<Value> get(const Key& key) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
        return std::nullopt;
    }

private:
    Cache() = default;
    ~Cache() = default;

    std::unordered_map<Key, Value> cache;
    std::mutex mtx;
};



struct MultiIndexer
{
    using ResultArray = std::vector<std::vector<int>>;

    int dim;
    int deg;
    int expanded_dim;

    MultiIndexer(int dim, int deg) : dim(dim), deg(deg)
    {
        expanded_dim = safe_combination(dim + deg - 1, deg);
    }

    ResultArray indices()
    {
        ResultArray result = {};
        std::vector<int> current = {};
        generate_next(result, current, 0);
        return result;
    }

    at::Tensor coefficients(at::Tensor &indices)
    {
        using coeff_t = int32_t; // should be enough

        auto coefficients_tensor = at::empty(indices.size(0), indices.options().dtype(torch::kInt32));

        for (int i = 0; i < indices.size(0); ++i)
        {
            auto res = at::unique_dim(indices[i], 0, true, false, true);
            auto unique_indices = std::get<0>(res);
            auto counts = std::get<2>(res);

            coeff_t perm_factorial = safe_factorial(deg);
            assert(perm_factorial > 0 && "Factorial overflow detected");

            coeff_t denominator = 1;
            for (int j = 0; j < counts.numel(); ++j)
            {
                int count_value = counts[j].item<int>();
                int count_factorial = safe_factorial(count_value);
                assert(count_value > 0 && "Count value is zero");
                assert(count_factorial != -1 && "Factorial overflow detected");

                assert(std::numeric_limits<coeff_t>::max() / denominator > count_factorial && "Overflow in denominator multiplication");
                denominator *= count_factorial;
            }

            assert(denominator != 0 && "Denominator is zero, division by zero error");
            coeff_t coefficient = perm_factorial / denominator;
            assert(coefficient > 0 && "Coefficient is zero");
            coefficients_tensor[i] = coefficient;
        }

        return coefficients_tensor;
    }

    // return the multi-index and corresponding coefficients
    // TODO (sean): remove this padding
    std::tuple<at::Tensor, at::Tensor> to_torch()
    {
        auto key = std::make_tuple(dim, deg);
        auto cached_result = Cache<decltype(key), std::tuple<at::Tensor, at::Tensor>>::getInstance().get(key);
        if (cached_result) {
            return cached_result.value();
        }

        auto indices = this->indices();
        auto options = torch::TensorOptions().dtype(torch::kInt32);
        std::vector<int> flat_indices(indices.size() * deg);

        for (size_t i = 0; i < indices.size(); ++i)
        {
            std::copy(indices[i].begin(), indices[i].end(), flat_indices.begin() + i * deg);
        }

        auto indices_tensor = torch::from_blob(flat_indices.data(), {static_cast<index_t>(indices.size()), deg}, options).clone();

        auto coefficients_tensor = this->coefficients(indices_tensor);

        // move tensors to CUDA
        indices_tensor = indices_tensor.to(torch::kCUDA).contiguous();
        coefficients_tensor = coefficients_tensor.to(torch::kCUDA).contiguous();

        Cache<decltype(key), std::tuple<at::Tensor, at::Tensor>>::getInstance().put(key, std::make_tuple(indices_tensor, coefficients_tensor));
        return std::make_tuple(indices_tensor, coefficients_tensor);
    }


    /**
     * @brief Calculate the coefficient of a multi-index.
     * 
     * @param indices The multi-index.
     * @return The coefficient of the multi-index.
     */
    int coeffcient(const std::vector<int> &indices)
    {
        int coeff = 1, i = 1;
        auto counts = count(indices);
        for (auto c : counts) {
            if (c != 0) {
                for (int j = 1; j <= c; ++j) {
                    coeff *= i++;
                    coeff /= j;
                }
            }
        }
        return coeff;
    }

    /**
     * @brief Count the number of appearances of each index in the multi-index.
     * 
     * @param indices The multi-index.
     * @return The number of appearances of each index.
     */
    std::vector<int> count(const std::vector<int> &indices)
    {
        std::vector<int> counts(dim, 0);
        for (int i = 0; i < deg; ++i) {
            counts[indices[i]]++;
        }
        return counts;
    }
    

    /**
     * @brief Generate multi-index starts and chunks of coefficients.
     * 
     * This function generates the starting multi-index and chunks of coefficients for a 
     * given BlockD.
     * 
     * @return A tuple containing:
     *         - at::Tensor: A 2D tensor of starting multi-indices, shape (num_block_D, deg), dtype int32.
     *         - at::Tensor: A 2D tensor of coefficient chunks, shape (num_block_D, BlockD), dtype int32.
     *         Both tensors are moved to CUDA and made contiguous.
     * 
     * @note The number of chunks is determined by dividing the expanded dimension by BlockD
     *       and rounding up to the nearest integer.
     */
    std::tuple<at::Tensor, at::Tensor> make_starting_indices_and_coeffs(const int BlockD)
    {
        auto key = std::make_tuple(dim, deg, BlockD);
        auto cached_result = Cache<decltype(key), std::tuple<at::Tensor, at::Tensor>>::getInstance().get(key);
        if (cached_result) {
            return cached_result.value();
        }

        auto numBlockD = (expanded_dim + BlockD - 1) / BlockD;
        auto options = torch::TensorOptions().dtype(torch::kInt32);
        auto starting_indices = at::empty({numBlockD, deg}, options);
        auto coefficient_chunks = at::empty({numBlockD, BlockD}, options);

        std::vector<int> indices(deg, 0);
        std::vector<int> coeffs(BlockD, 0);
        int counter = 0;

        while (!(indices[0] == (dim - 1))) {
            for (int i = deg - 1; i >= 0; --i) {
                if (indices[i] < dim - 1) {
                    indices[i]++;
                    for (int j = i + 1; j < deg; ++j) {
                        indices[j] = indices[i];
                    }
                    coeffs[counter % BlockD] = coeffcient(indices);
                    if (counter % BlockD == 0) {
                        int block_index = counter / BlockD;
                        // Convert std::vector<int> to a tensor
                        auto indices_tensor = torch::from_blob(indices.data(), {deg}, options).clone();
                        starting_indices.index_put_({block_index}, indices_tensor);
                    }
                    if ((counter + 1) % BlockD == 0) {
                        auto c_tensor = torch::from_blob(coeffs.data(), {BlockD}, options).clone();
                        coefficient_chunks.index_put_({counter / BlockD}, c_tensor);
                        // reset coeffs
                        std::fill(coeffs.begin(), coeffs.end(), 0);
                    }
                    ++counter;
                    break;
                }
            }
        }

        // put the last chunk of coefficients in tensor
        auto c_tensor = torch::from_blob(coeffs.data(), {BlockD}, options).clone();
        coefficient_chunks.index_put_({numBlockD - 1}, c_tensor);

        // move tensors to CUDA
        starting_indices = starting_indices.to(torch::kCUDA).contiguous();
        coefficient_chunks = coefficient_chunks.to(torch::kCUDA).contiguous();

        Cache<decltype(key), std::tuple<at::Tensor, at::Tensor>>::getInstance().put(key, std::make_tuple(starting_indices, coefficient_chunks));
        return std::make_tuple(starting_indices, coefficient_chunks);
    }

private:
    void generate_next(ResultArray &result, std::vector<int> &indices, int start)
    {
        if (indices.size() == static_cast<size_t>(deg))
        {
            result.push_back(indices);
            return;
        }
        for (int d = start; d < dim; d++)
        {
            std::vector<int> new_indices = indices;
            new_indices.push_back(d);
            generate_next(result, new_indices, d);
        }
    }
};
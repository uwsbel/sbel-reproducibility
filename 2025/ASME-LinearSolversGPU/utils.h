#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <tuple>
#include <algorithm>
#include <cmath>

// Add CUDA compatibility
#ifdef __CUDACC__
#define UTILS_HOST_DEVICE __host__ __device__
#define UTILS_HOST __host__
#define UTILS_DEVICE __device__
#else
#define UTILS_HOST_DEVICE
#define UTILS_HOST
#define UTILS_DEVICE
#endif

/**
 * Reads a matrix in COO format from a file and converts it to CSR format.
 *
 * @param filename Path to the input file containing matrix data in COO format
 * @param values Output vector to store the non-zero values
 * @param rowIndex Output vector to store the row pointers
 * @param columns Output vector to store the column indices
 * @param n Output parameter to store the matrix dimension
 */
template <typename T>
void readMatrixCSR(const std::string &filename,
                   std::vector<T> &values,
                   std::vector<int> &rowIndex,
                   std::vector<int> &columns,
                   int &n)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    // Read all triplets first to determine matrix size
    std::vector<std::tuple<int, int, double>> triplets;
    int row, col;
    double value;
    int max_row = 0, max_col = 0;

    // Read all entries
    while (file >> row >> col >> value)
    {
        // Convert from 1-based to 0-based indexing if needed
        row--;
        col--;

        // Keep track of matrix dimensions
        max_row = std::max(max_row, row);
        max_col = std::max(max_col, col);

        triplets.emplace_back(row, col, value);
    }

    // Matrix dimensions are max indices + 1 (since we converted to 0-based)
    n = max_row + 1;

    // Check if matrix is square
    if (max_row != max_col)
    {
        std::cerr << "Error: Matrix is not square. Rows: " << max_row + 1 << ", Cols: " << max_col + 1 << std::endl;
        exit(1);
    }

    // Sort triplets by row, then by column for CSR format
    std::sort(triplets.begin(), triplets.end());

    // Initialize CSR arrays
    values.resize(triplets.size());
    columns.resize(triplets.size());
    rowIndex.resize(n + 1, 0);

    // Fill in the CSR arrays
    int current_row = -1;
    for (size_t i = 0; i < triplets.size(); i++)
    {
        int row = std::get<0>(triplets[i]);
        int col = std::get<1>(triplets[i]);
        double val = std::get<2>(triplets[i]);

        // Update row index array
        while (current_row < row)
        {
            current_row++;
            rowIndex[current_row] = i;
        }

        // Store column index and value
        columns[i] = col;
        values[i] = static_cast<T>(val);
    }

    // Set the last element of rowIndex
    rowIndex[n] = triplets.size();

    file.close();
}

/**
 * Reads a vector from a file.
 *
 * @param filename Path to the input file containing vector data
 * @return Vector of values
 */
template <typename T>
std::vector<T> readVector(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::vector<T> values;
    double value;

    // Read all values from the file
    while (file >> value)
    {
        values.push_back(static_cast<T>(value));
    }

    // Check if we read anything
    if (values.empty())
    {
        std::cerr << "Warning: No data read from " << filename << std::endl;
    }

    file.close();
    return values;
}

/**
 * Reads the known solution by combining and transforming Dv and Dl files.
 *
 * @param dvFilename Path to the Dv file
 * @param dlFilename Path to the Dl file
 * @return Combined solution vector
 */
template <typename T>
std::vector<T> readKnownSolution(const std::string &dvFilename, const std::string &dlFilename)
{
    std::vector<T> dvPart = readVector<T>(dvFilename);
    std::vector<T> dlPart = readVector<T>(dlFilename);

    // Negate dlPart before combining
    for (auto &val : dlPart)
    {
        val = -val;
    }

    // Create combined vector
    std::vector<T> solution;
    solution.reserve(dvPart.size() + dlPart.size());
    solution.insert(solution.end(), dvPart.begin(), dvPart.end());
    solution.insert(solution.end(), dlPart.begin(), dlPart.end());

    return solution;
}

/**
 * Writes a vector to a file.
 *
 * @param vector Vector of values to write
 * @param filename Path to the output file
 */
template <typename T>
void writeVectorToFile(const std::vector<T> &vector, const std::string &filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        exit(1);
    }

    // Set precision for output
    file.precision(16);
    file << std::scientific;

    // Write each element on a new line
    for (size_t i = 0; i < vector.size(); i++)
    {
        file << vector[i] << std::endl;
    }

    file.close();
    std::cout << "Solution written to " << filename << std::endl;
}

/**
 * Calculates the relative error between two vectors.
 * Host-only version that can use std::vector and print error messages.
 *
 * @param computed Computed solution vector
 * @param reference Reference solution vector
 * @return Relative error
 */
template <typename T>
UTILS_HOST
    T
    calculateRelativeError(const std::vector<T> &computed, const std::vector<T> &reference)
{
    if (computed.size() != reference.size())
    {
        std::cerr << "Error: Vector sizes don't match for error calculation" << std::endl;
        return static_cast<T>(-1.0);
    }

    T norm_diff = static_cast<T>(0.0);
    T norm_ref = static_cast<T>(0.0);

    for (size_t i = 0; i < computed.size(); i++)
    {
        T diff = computed[i] - reference[i];
        norm_diff += diff * diff;
        norm_ref += reference[i] * reference[i];
    }

    return std::sqrt(norm_diff) / std::sqrt(norm_ref);
}

/**
 * Calculates the relative error between two arrays.
 * Device-compatible version with raw pointers and size parameters.
 *
 * @param computed Computed solution array
 * @param reference Reference solution array
 * @param size Size of both arrays
 * @return Relative error
 */
template <typename T>
UTILS_HOST_DEVICE
    T
    calculateRelativeErrorRaw(const T *computed, const T *reference, int size)
{
    T norm_diff = static_cast<T>(0.0);
    T norm_ref = static_cast<T>(0.0);

    for (int i = 0; i < size; i++)
    {
        T diff = computed[i] - reference[i];
        norm_diff += diff * diff;
        norm_ref += reference[i] * reference[i];
    }

    return std::sqrt(norm_diff) / std::sqrt(norm_ref);
}

/**
 * Calculates the residual-based backward error for a linear system Ax = b.
 * The backward error is defined as ||r||_2 / (||A||_F * ||x||_2 + ||b||_2)
 * where r = b - Ax is the residual, and ||.||_F is the Frobenius norm.
 *
 * @param values Array of non-zero values in CSR format
 * @param rowIndex Array of row pointers in CSR format
 * @param columns Array of column indices in CSR format 
 * @param x Solution vector
 * @param b Right-hand side vector
 * @return Backward error
 */
template <typename T>
UTILS_HOST
T calculateBackwardError(
    const std::vector<T> &values,
    const std::vector<int> &rowIndex,
    const std::vector<int> &columns,
    const std::vector<T> &x,
    const std::vector<T> &b)
{
    int n = b.size();
    if (rowIndex.size() != n + 1 || x.size() != n)
    {
        std::cerr << "Error: Dimensions don't match for backward error calculation" << std::endl;
        return static_cast<T>(-1.0);
    }

    // Calculate residual r = b - Ax
    std::vector<T> r = b;
    for (int i = 0; i < n; i++)
    {
        for (int j = rowIndex[i]; j < rowIndex[i + 1]; j++)
        {
            int col = columns[j];
            r[i] -= values[j] * x[col];
        }
    }

    // Calculate ||r||_2
    T r_norm = static_cast<T>(0.0);
    for (int i = 0; i < n; i++)
    {
        r_norm += r[i] * r[i];
    }
    r_norm = std::sqrt(r_norm);

    // Calculate ||A||_F (Frobenius norm)
    T A_norm = static_cast<T>(0.0);
    for (size_t i = 0; i < values.size(); i++)
    {
        A_norm += values[i] * values[i];
    }
    A_norm = std::sqrt(A_norm);

    // Calculate ||x||_2
    T x_norm = static_cast<T>(0.0);
    for (int i = 0; i < n; i++)
    {
        x_norm += x[i] * x[i];
    }
    x_norm = std::sqrt(x_norm);

    // Calculate ||b||_2
    T b_norm = static_cast<T>(0.0);
    for (int i = 0; i < n; i++)
    {
        b_norm += b[i] * b[i];
    }
    b_norm = std::sqrt(b_norm);

    // Calculate backward error: ||r||_2 / (||A||_F * ||x||_2 + ||b||_2)
    T denominator = A_norm * x_norm + b_norm;
    return r_norm / denominator;
}

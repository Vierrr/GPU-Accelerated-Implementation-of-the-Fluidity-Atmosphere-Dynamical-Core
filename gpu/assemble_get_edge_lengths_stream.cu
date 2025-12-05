#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cmath>

using namespace std;
#define BLOCK_SIZE 128
extern cudaStream_t stream_get_edge;

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

template <typename T, int size> class myArray {
    private:
        T ptr[size];

    public:
        __device__  myArray(){};
        __device__  myArray(T arr[]){
            for (int i = 0; i < size; ++i)
                ptr[i] = arr[i];
        };
        __device__  myArray(std::initializer_list<T> list){
            int i =0;
            for (const auto& x : list){
                ptr[i++]=x;
            }
        };
        __device__  const T& operator[](int i) const {
            return ptr[i];
        };
        __device__  T& operator[](int i) {
            return ptr[i];
        };
        __device__  myArray & operator=(const myArray &a)
        {
            for(int i=0; i< size;++i){
            this->ptr[i]=a[i];
            }
            return *this;
        }

};
template <typename T>
class SortEigenstuff
{
public:
    __device__ void operator()(int32_t sortType, bool isRotation,
        myArray<T, 3>& eval, myArray<myArray<T, 3>, 3>& evec)
    {
        if (sortType != 0)
        {
            // Sort the eigenvalues to eval[0] <= eval[1] <= eval[2].
            myArray<size_t, 3> index{};
            if (eval[0] < eval[1])
            {
                if (eval[2] < eval[0])
                {
                    // even permutation
                    index[0] = 2;
                    index[1] = 0;
                    index[2] = 1;
                }
                else if (eval[2] < eval[1])
                {
                    // odd permutation
                    index[0] = 0;
                    index[1] = 2;
                    index[2] = 1;
                    isRotation = !isRotation;
                }
                else
                {
                    // even permutation
                    index[0] = 0;
                    index[1] = 1;
                    index[2] = 2;
                }
            }
            else
            {
                if (eval[2] < eval[1])
                {
                    // odd permutation
                    index[0] = 2;
                    index[1] = 1;
                    index[2] = 0;
                    isRotation = !isRotation;
                }
                else if (eval[2] < eval[0])
                {
                    // even permutation
                    index[0] = 1;
                    index[1] = 2;
                    index[2] = 0;
                }
                else
                {
                    // odd permutation
                    index[0] = 1;
                    index[1] = 0;
                    index[2] = 2;
                    isRotation = !isRotation;
                }
            }

            if (sortType == -1)
            {
                // The request is for eval[0] >= eval[1] >= eval[2]. This
                // requires an odd permutation, (i0,i1,i2) -> (i2,i1,i0).
                //swap(index[0], index[2]);
                T tmp=index[0];
                index[0]=index[2];
                index[2]=tmp;
                isRotation = !isRotation;
            }

            myArray<T, 3> unorderedEVal = eval;
            myArray<myArray<T, 3>, 3> unorderedEVec = evec;
            for (size_t j = 0; j < 3; ++j)
            {
                size_t i = index[j];
                eval[j] = unorderedEVal[i];
                evec[j] = unorderedEVec[i];
            }
        }

        // Ensure the ordered eigenvectors form a right-handed basis.
        if (!isRotation)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                evec[2][j] = -evec[2][j];
            }
        }
    }
};
    template <typename T>
    class SymmetricEigensolver3x3
    {
    public:
        // The input matrix must be symmetric, so only the unique elements
        // must be specified: a00, a01, a02, a11, a12, and a22.
        //
        // If 'aggressive' is 'true', the iterations occur until a
        // superdiagonal entry is exactly zero.  If 'aggressive' is 'false',
        // the iterations occur until a superdiagonal entry is effectively
        // zero compared to the/ sum of magnitudes of its diagonal neighbors.
        // Generally, the nonaggressive convergence is acceptable.
        //
        // The order of the eigenvalues is specified by sortType:
        // -1 (decreasing), 0 (no sorting) or +1 (increasing).  When sorted,
        // the eigenvectors are ordered accordingly, and
        // {evec[0], evec[1], evec[2]} is guaranteed to/ be a right-handed
        // orthonormal set.  The return value is the number of iterations
        // used by the algorithm.

        __device__ int32_t operator()(T const& a00, T const& a01, T const& a02, T const& a11,
            T const& a12, T const& a22, bool aggressive, int32_t sortType,
            myArray<T, 3>& eval, myArray<myArray<T, 3>, 3>& evec) const
        {
            // Compute the Householder reflection H0 and B = H0*A*H0, where
            // b02 = 0. H0 = {{c,s,0},{s,-c,0},{0,0,1}} with each inner
            // triple a row of H0.
            T const zero = static_cast<T>(0);
            T const one = static_cast<T>(1);
            T const half = static_cast<T>(0.5);
            bool isRotation = false;
            T c = zero, s = zero;
            GetCosSin(a12, -a02, c, s);
            T term0 = c * a00 + s * a01;
            T term1 = c * a01 + s * a11;
            T term2 = s * a00 - c * a01;
            T term3 = s * a01 - c * a11;
            T b00 = c * term0 + s * term1;
            T b01 = s * term0 - c * term1;
            //T b02 = c * a02 + s * a12;  // 0
            T b11 = s * term2 - c * term3;
            T b12 = s * a02 - c * a12;
            T b22 = a22;

            // Maintain Q as the product of the reflections. Initially,
            // Q = H0. Updates by Givens reflections G are Q <- Q * G. The
            // columns of the final Q are the estimates for the eigenvectors.
            myArray<myArray<T, 3>, 3> Q =
            { {
                { c, s, zero },
                { s, -c, zero },
                { zero, zero, one }
            } };

            // The smallest subnormal number is 2^{-alpha}. The value alpha is
            // 149 for 'float' and 1074 for 'double'.
            int32_t constexpr alpha = std::numeric_limits<T>::digits - std::numeric_limits<T>::min_exponent;
            int32_t i = 0, imax = 0, power = 0;
            T c2 = zero, s2 = zero;

            if (std::fabs(b12) <= std::fabs(b01))
            {
                // It is known that |currentB12| < 2^{-i/2} * |initialB12|.
                // Compute imax so that 0 is the closest floating-point number
                // to 2^{-imax/2} * |initialB12|.
                (void)std::frexp(b12, &power);
                imax = 2 * (power + alpha + 1);

                for (i = 0; i < imax; ++i)
                {
                    // Compute the Givens reflection
                    // G = {{c,0,-s},{s,0,c},{0,1,0}} where each inner triple
                    // is a row of G.
                    GetCosSin(half * (b00 - b11), b01, c2, s2);
                    s = std::sqrt(half * (one - c2));
                    c = half * s2 / s;

                    // Update Q <- Q * G.
                    for (size_t r = 0; r < 3; ++r)
                    {
                        term0 = c * Q[r][0] + s * Q[r][1];
                        term1 = Q[r][2];
                        term2 = c * Q[r][1] - s * Q[r][0];
                        Q[r][0] = term0;
                        Q[r][1] = term1;
                        Q[r][2] = term2;
                    }
                    isRotation = !isRotation;

                    // Update B <- Q^T * B * Q, ensuring that b02 is zero and
                    // |b12| has strictly decreased.
                    term0 = c * b00 + s * b01;
                    term1 = c * b01 + s * b11;
                    term2 = s * b00 - c * b01;
                    term3 = s * b01 - c * b11;
                    //b02 = s * c * (b11 - b00) + (c * c - s * s) * b01; // 0
                    b00 = c * term0 + s * term1;
                    b01 = s * b12;
                    b11 = b22;
                    b12 = c * b12;
                    b22 = s * term2 - c * term3;

                    if (Converged(aggressive, b00, b11, b01))
                    {
                        // Compute the Householder reflection
                        // H1 = {{c,s,0},{s,-c,0},{0,0,1}} where each inner
                        // triple is a row of H1.
                        GetCosSin(half * (b00 - b11), b01, c2, s2);
                        s = std::sqrt(half * (one - c2));
                        c = half * s2 / s;

                        // Update Q <- Q * H1.
                        for (size_t r = 0; r < 3; ++r)
                        {
                            term0 = c * Q[r][0] + s * Q[r][1];
                            term1 = s * Q[r][0] - c * Q[r][1];
                            Q[r][0] = term0;
                            Q[r][1] = term1;
                        }
                        isRotation = !isRotation;

                        // Compute the diagonal estimate D = Q^T * B * Q.
                        term0 = c * b00 + s * b01;
                        term1 = c * b01 + s * b11;
                        term2 = s * b00 - c * b01;
                        term3 = s * b01 - c * b11;
                        b00 = c * term0 + s * term1;
                        b11 = s * term2 - c * term3;
                        break;
                    }
                }
            }
            else
            {
                // It is known that |currentB01| < 2^{-i/2} * |initialB01|.
                // Compute imax so that 0 is the closest floating-point number
                // to 2^{-imax/2} * |initialB01|.
                (void)std::frexp(b01, &power);
                imax = 2 * (power + alpha + 1);

                for (i = 0; i < imax; ++i)
                {
                    // Compute the Givens reflection
                    // G = {{0,1,0},{c,0,-s},{s,0,c}} where each inner triple
                    // is a row of G.
                    GetCosSin(half * (b11 - b22), b12, c2, s2);
                    s = std::sqrt(half * (one - c2));
                    c = half * s2 / s;

                    // Update Q <- Q * G.
                    for (size_t r = 0; r < 3; ++r)
                    {
                        term0 = c * Q[r][1] + s * Q[r][2];
                        term1 = Q[r][0];
                        term2 = c * Q[r][2] - s * Q[r][1];
                        Q[r][0] = term0;
                        Q[r][1] = term1;
                        Q[r][2] = term2;
                    }
                    isRotation = !isRotation;

                    // Update B <- Q^T * B * Q, ensuring that b02 is zero and
                    // |b01| has strictly decreased.
                    term0 = c * b11 + s * b12;
                    term1 = c * b12 + s * b22;
                    term2 = s * b11 - c * b12;
                    term3 = s * b12 - c * b22;
                    //b02 = s * c * (b22 - b11) + (c * c - s * s) * b12;  // 0
                    b22 = s * term2 - c * term3;
                    b12 = -s * b01;
                    b11 = b00;
                    b01 = c * b01;
                    b00 = c * term0 + s * term1;

                    if (Converged(aggressive, b11, b22, b12))
                    {
                        // Compute the Householder reflection
                        // H1 = {{1,0,0},{0,c,s},{0,s,-c}} where each inner
                        // triple is a row of H1.
                        GetCosSin(half * (b11 - b22), b12, c2, s2);
                        s = std::sqrt(half * (one - c2));
                        c = half * s2 / s;

                        // Update Q <- Q * H1.
                        for (size_t r = 0; r < 3; ++r)
                        {
                            term0 = c * Q[r][1] + s * Q[r][2];
                            term1 = s * Q[r][1] - c * Q[r][2];
                            Q[r][1] = term0;
                            Q[r][2] = term1;
                        }
                        isRotation = !isRotation;

                        // Compute the diagonal estimate D = Q^T * B * Q.
                        term0 = c * b11 + s * b12;
                        term1 = c * b12 + s * b22;
                        term2 = s * b11 - c * b12;
                        term3 = s * b12 - c * b22;
                        b11 = c * term0 + s * term1;
                        b22 = s * term2 - c * term3;
                        break;
                    }
                }
            }

            eval = { b00, b11, b22 };
            for (size_t row = 0; row < 3; ++row)
            {
                for (size_t col = 0; col < 3; ++col)
                {
                    evec[row][col] = Q[col][row];
                }
            }

            SortEigenstuff<T>()(sortType, isRotation, eval, evec);
            return i;
        }

    private:
        // Normalize (u,v) to (c,s) with c <= 0 when (u,v) is not (0,0).
        // If (u,v) = (0,0), the function returns (c,s) = (-1,0). When used
        // to generate a Householder reflection, it does not matter whether
        // (c,s) or (-c,-s) is returned. When generating a Givens reflection,
        // c = cos(2*theta) and s = sin(2*theta). Having a negative cosine
        // for the double-angle term ensures that the single-angle terms
        // c = cos(theta) and s = sin(theta) satisfy |c| < 1/sqrt(2) < |s|.
        static __device__ void GetCosSin(T const& u, T const& v, T& c, T& s)
        {
            T const zero = static_cast<T>(0);
            T length = std::sqrt(u * u + v * v);
            if (length > zero)
            {
                c = u / length;
                s = v / length;
                if (c > zero)
                {
                    c = -c;
                    s = -s;
                }
            }
            else
            {
                c = static_cast<T>(-1);
                s = zero;
            }
        }

        static __device__ bool Converged(bool aggressive, T const& diagonal0,
            T const& diagonal1, T const& superdiagonal)
        {
            if (aggressive)
            {
                // Test whether the superdiagonal term is zero.
                return superdiagonal == static_cast<T>(0);
            }
            else
            {
                // Test whether the superdiagonal term is effectively zero
                // compared to its diagonal neighbors.
                T sum = std::fabs(diagonal0) + std::fabs(diagonal1);
                return sum + std::fabs(superdiagonal) == sum;
            }
        }
    };

template <typename T>
class NISymmetricEigensolver3x3
{
public:
    // The input matrix must be symmetric, so only the unique elements
    // must be specified: a00, a01, a02, a11, a12, and a22.  The
    // eigenvalues are sorted in ascending order: eval0 <= eval1 <= eval2.

    __device__ void operator()(T a00, T a01, T a02, T a11, T a12, T a22,
        int32_t sortType, myArray<T, 3>& eval, myArray<myArray<T, 3>, 3>& evec) const
    {
        // Precondition the matrix by factoring out the maximum absolute
        // value of the components.  This guards against floating-point
        // overflow when computing the eigenvalues.
        T max0 = fmax(fabs(a00), fabs(a01));
        T max1 = fmax(fabs(a02), fabs(a11));
        T max2 = fmax(fabs(a12), fabs(a22));
        T maxAbsElement = fmax(fmax(max0, max1), max2);
        if (maxAbsElement == (T)0)
        {
            // A is the zero matrix.
            eval[0] = (T)0;
            eval[1] = (T)0;
            eval[2] = (T)0;
            evec[0] = { (T)1, (T)0, (T)0 };
            evec[1] = { (T)0, (T)1, (T)0 };
            evec[2] = { (T)0, (T)0, (T)1 };
            return;
        }

        T invMaxAbsElement = (T)1 / maxAbsElement;
        a00 *= invMaxAbsElement;
        a01 *= invMaxAbsElement;
        a02 *= invMaxAbsElement;
        a11 *= invMaxAbsElement;
        a12 *= invMaxAbsElement;
        a22 *= invMaxAbsElement;

        T norm = a01 * a01 + a02 * a02 + a12 * a12;
        if (norm > (T)0)
        {
            // Compute the eigenvalues of A.

            // In the PDF mentioned previously, B = (A - q*I)/p, where
            // q = tr(A)/3 with tr(A) the trace of A (sum of the diagonal
            // entries of A) and where p = sqrt(tr((A - q*I)^2)/6).
            T q = (a00 + a11 + a22) / (T)3;

            // The matrix A - q*I is represented by the following, where
            // b00, b11 and b22 are computed after these comments,
            //   +-           -+
            //   | b00 a01 a02 |
            //   | a01 b11 a12 |
            //   | a02 a12 b22 |
            //   +-           -+
            T b00 = a00 - q;
            T b11 = a11 - q;
            T b22 = a22 - q;

            // The is the variable p mentioned in the PDF.
            T p = sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * (T)2) / (T)6);

            // We need det(B) = det((A - q*I)/p) = det(A - q*I)/p^3.  The
            // value det(A - q*I) is computed using a cofactor expansion
            // by the first row of A - q*I.  The cofactors are c00, c01
            // and c02 and the determinant is b00*c00 - a01*c01 + a02*c02.
            // The det(B) is then computed finally by the division
            // with p^3.
            T c00 = b11 * b22 - a12 * a12;
            T c01 = a01 * b22 - a12 * a02;
            T c02 = a01 * a12 - b11 * a02;
            T det = (b00 * c00 - a01 * c01 + a02 * c02) / (p * p * p);

            // The halfDet value is cos(3*theta) mentioned in the PDF. The
            // acos(z) function requires |z| <= 1, but will fail silently
            // and return NaN if the input is larger than 1 in magnitude.
            // To avoid this problem due to rounding errors, the halfDet
            // value is clamped to [-1,1].
            T halfDet = det * (T)0.5;
            halfDet = fmin(fmax(halfDet, (T)-1), (T)1);

            // The eigenvalues of B are ordered as
            // beta0 <= beta1 <= beta2.  The number of digits in
            // twoThirdsPi is chosen so that, whether float or double,
            // the floating-point number is the closest to theoretical
            // 2*pi/3.
            T angle = acos(halfDet) / (T)3;
            T const twoThirdsPi = (T)2.09439510239319549;
            T beta2 = cos(angle) * (T)2;
            T beta0 = cos(angle + twoThirdsPi) * (T)2;
            T beta1 = -(beta0 + beta2);

            // The eigenvalues of A are ordered as
            // alpha0 <= alpha1 <= alpha2.
            eval[0] = q + p * beta0;
            eval[1] = q + p * beta1;
            eval[2] = q + p * beta2;

            // Compute the eigenvectors so that the set
            // {evec[0], evec[1], evec[2]} is right handed and
            // orthonormal.
            if (halfDet >= (T)0)
            {
                ComputeEigenvector0(a00, a01, a02, a11, a12, a22, eval[2], evec[2]);
                ComputeEigenvector1(a00, a01, a02, a11, a12, a22, evec[2], eval[1], evec[1]);
                evec[0] = Cross(evec[1], evec[2]);
            }
            else
            {
                ComputeEigenvector0(a00, a01, a02, a11, a12, a22, eval[0], evec[0]);
                ComputeEigenvector1(a00, a01, a02, a11, a12, a22, evec[0], eval[1], evec[1]);
                evec[2] = Cross(evec[0], evec[1]);
            }
        }
        else
        {
            // The matrix is diagonal.
            eval[0] = a00;
            eval[1] = a11;
            eval[2] = a22;
            evec[0] = { (T)1, (T)0, (T)0 };
            evec[1] = { (T)0, (T)1, (T)0 };
            evec[2] = { (T)0, (T)0, (T)1 };
        }

        // The preconditioning scaled the matrix A, which scales the
        // eigenvalues.  Revert the scaling.
        eval[0] *= maxAbsElement;
        eval[1] *= maxAbsElement;
        eval[2] *= maxAbsElement;

        SortEigenstuff<T>()(sortType, true, eval, evec);
    }

private:
    static __device__ myArray<T, 3> Multiply(T s, myArray<T, 3> const& U)
    {
        myArray<T, 3> product = { s * U[0], s * U[1], s * U[2] };
        return product;
    }

    static __device__ myArray<T, 3> Subtract(myArray<T, 3> const& U, myArray<T, 3> const& V)
    {
        myArray<T, 3> difference = { U[0] - V[0], U[1] - V[1], U[2] - V[2] };
        return difference;
    }

    static __device__ myArray<T, 3> Divide(myArray<T, 3> const& U, T s)
    {
        T invS = (T)1 / s;
        myArray<T, 3> division = { U[0] * invS, U[1] * invS, U[2] * invS };
        return division;
    }

    static __device__ T Dot(myArray<T, 3> const& U, myArray<T, 3> const& V)
    {
        T dot = U[0] * V[0] + U[1] * V[1] + U[2] * V[2];
        return dot;
    }

    static __device__ myArray<T, 3> Cross(myArray<T, 3> const& U, myArray<T, 3> const& V)
    {
        myArray<T, 3> cross =
        {
            U[1] * V[2] - U[2] * V[1],
            U[2] * V[0] - U[0] * V[2],
            U[0] * V[1] - U[1] * V[0]
        };
        return cross;
    }

    __device__ void ComputeOrthogonalComplement(myArray<T, 3> const& W,
        myArray<T, 3>& U, myArray<T, 3>& V) const
    {
        // Robustly compute a right-handed orthonormal set { U, V, W }.
        // The vector W is guaranteed to be unit-length, in which case
        // there is no need to worry about a division by zero when
        // computing invLength.
        T invLength;
        if (fabs(W[0]) > fabs(W[1]))
        {
            // The component of maximum absolute value is either W[0]
            // or W[2].
            invLength = (T)1 / sqrt(W[0] * W[0] + W[2] * W[2]);
            U = { -W[2] * invLength, (T)0, +W[0] * invLength };
        }
        else
        {
            // The component of maximum absolute value is either W[1]
            // or W[2].
            invLength = (T)1 / sqrt(W[1] * W[1] + W[2] * W[2]);
            U = { (T)0, +W[2] * invLength, -W[1] * invLength };
        }
        V = Cross(W, U);
    }

    __device__ void ComputeEigenvector0(T a00, T a01, T a02, T a11, T a12, T a22,
        T eval0, myArray<T, 3>& evec0) const
    {
        // Compute a unit-length eigenvector for eigenvalue[i0].  The
        // matrix is rank 2, so two of the rows are linearly independent.
        // For a robust computation of the eigenvector, select the two
        // rows whose cross product has largest length of all pairs of
        // rows.
        myArray<T, 3> row0 = { a00 - eval0, a01, a02 };
        myArray<T, 3> row1 = { a01, a11 - eval0, a12 };
        myArray<T, 3> row2 = { a02, a12, a22 - eval0 };
        myArray<T, 3>  r0xr1 = Cross(row0, row1);
        myArray<T, 3>  r0xr2 = Cross(row0, row2);
        myArray<T, 3>  r1xr2 = Cross(row1, row2);
        T d0 = Dot(r0xr1, r0xr1);
        T d1 = Dot(r0xr2, r0xr2);
        T d2 = Dot(r1xr2, r1xr2);

        T dmax = d0;
        int32_t imax = 0;
        if (d1 > dmax)
        {
            dmax = d1;
            imax = 1;
        }
        if (d2 > dmax)
        {
            imax = 2;
        }

        if (imax == 0)
        {
            evec0 = Divide(r0xr1, sqrt(d0));
        }
        else if (imax == 1)
        {
            evec0 = Divide(r0xr2, sqrt(d1));
        }
        else
        {
            evec0 = Divide(r1xr2, sqrt(d2));
        }
    }

    __device__ void ComputeEigenvector1(T a00, T a01, T a02, T a11, T a12, T a22,
        myArray<T, 3> const& evec0, T eval1, myArray<T, 3>& evec1) const
    {
        // Robustly compute a right-handed orthonormal set
        // { U, V, evec0 }.
        myArray<T, 3> U, V;
        ComputeOrthogonalComplement(evec0, U, V);

        // Let e be eval1 and let E be a corresponding eigenvector which
        // is a solution to the linear system (A - e*I)*E = 0.  The matrix
        // (A - e*I) is 3x3, not invertible (so infinitely many
        // solutions), and has rank 2 when eval1 and eval are different.
        // It has rank 1 when eval1 and eval2 are equal.  Numerically, it
        // is difficult to compute robustly the rank of a matrix.  Instead,
        // the 3x3 linear system is reduced to a 2x2 system as follows.
        // Define the 3x2 matrix J = [U V] whose columns are the U and V
        // computed previously.  Define the 2x1 vector X = J*E.  The 2x2
        // system is 0 = M * X = (J^T * (A - e*I) * J) * X where J^T is
        // the transpose of J and M = J^T * (A - e*I) * J is a 2x2 matrix.
        // The system may be written as
        //     +-                        -++-  -+       +-  -+
        //     | U^T*A*U - e  U^T*A*V     || x0 | = e * | x0 |
        //     | V^T*A*U      V^T*A*V - e || x1 |       | x1 |
        //     +-                        -++   -+       +-  -+
        // where X has row entries x0 and x1.

        myArray<T, 3> AU =
        {
            a00 * U[0] + a01 * U[1] + a02 * U[2],
            a01 * U[0] + a11 * U[1] + a12 * U[2],
            a02 * U[0] + a12 * U[1] + a22 * U[2]
        };

        myArray<T, 3> AV =
        {
            a00 * V[0] + a01 * V[1] + a02 * V[2],
            a01 * V[0] + a11 * V[1] + a12 * V[2],
            a02 * V[0] + a12 * V[1] + a22 * V[2]
        };

        T m00 = U[0] * AU[0] + U[1] * AU[1] + U[2] * AU[2] - eval1;
        T m01 = U[0] * AV[0] + U[1] * AV[1] + U[2] * AV[2];
        T m11 = V[0] * AV[0] + V[1] * AV[1] + V[2] * AV[2] - eval1;

        // For robustness, choose the largest-length row of M to compute
        // the eigenvector.  The 2-tuple of coefficients of U and V in the
        // assignments to eigenvector[1] lies on a circle, and U and V are
        // unit length and perpendicular, so eigenvector[1] is unit length
        // (within numerical tolerance).
        T absM00 = fabs(m00);
        T absM01 = fabs(m01);
        T absM11 = fabs(m11);
        T maxAbsComp;
        if (absM00 >= absM11)
        {
            maxAbsComp = fmax(absM00, absM01);
            if (maxAbsComp > (T)0)
            {
                if (absM00 >= absM01)
                {
                    m01 /= m00;
                    m00 = (T)1 / sqrt((T)1 + m01 * m01);
                    m01 *= m00;
                }
                else
                {
                    m00 /= m01;
                    m01 = (T)1 / sqrt((T)1 + m00 * m00);
                    m00 *= m01;
                }
                evec1 = Subtract(Multiply(m01, U), Multiply(m00, V));
            }
            else
            {
                evec1 = U;
            }
        }
        else
        {
            maxAbsComp = fmax(absM11, absM01);
            if (maxAbsComp > (T)0)
            {
                if (absM11 >= absM01)
                {
                    m01 /= m11;
                    m11 = (T)1 / sqrt((T)1 + m01 * m01);
                    m01 *= m11;
                }
                else
                {
                    m11 /= m01;
                    m01 = (T)1 / sqrt((T)1 + m11 * m11);
                    m11 *= m01;
                }
                evec1 = Subtract(Multiply(m11, U), Multiply(m01, V));
            }
            else
            {
                evec1 = U;
            }
        }
    }
};
// __device__ void dspev(double * __restrict__ ap, double* __restrict__ out_val, double *__restrict__ out_vec){
//     myArray<double, 3> eigen_value={0.0};
//     myArray<myArray<double, 3>, 3> eigen_vec={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
//     NISymmetricEigensolver3x3 <double>dspev;
//     dspev(*ap,*(ap+1),*(ap+3),*(ap+2),*(ap+4),*(ap+5),1,eigen_value,eigen_vec);
//     int i,j;
//     for (i=0;i<3;i++){
//         out_val[i]=eigen_value[i];
//         for (j=0;j<3;j++)out_vec[i*3+j]=eigen_vec[i][j];
//     }
//     return;
// }

extern "C" __device__ void get_edge_matmul(double* a, double* b, double*C, int m,int n,int k){
    //mkn循环法
    //a b都是列排序
    for (int mi = 0; mi < m; mi ++) {
        for (int ki = 0; ki < k; ki ++) {
            for (int ni = 0; ni < n; ni ++) {
                C[ni* m + mi] +=  a[ki * m + mi] * b[k * ni + ki];
            }
        }
    }
}


//这个可能不适合内置
 __device__ int get_idx(int i,int j,int dim){
    //但要想清楚！！！，fortran里这边i、j都是取值1、2、3
    int k,l;
    k = min(i, j);
    l = max(i, j);

    if (k == 0)return l;
    if (dim == 3)return l + k + 1;  //原本这个值最大可以到l=3 k=3 l+k=6，在C里最大只有4.。。。所以相加后加1.。
    //因为dim =3 不存在第三种情况return l + k ;
/*
    if (k == 1)return 1;
    if (dim == 3)return l + k ;  //原本这个值最大可以到l=3 k=3 l+k=6，在C里最大只有4.。。。所以相加后加1.。
    return l + k -1;
    */
    return 0;
}

//适合内置
 __device__ inline int coeff(int k,int l){
    
    if(k==l)return 1;
    return 2;
}

extern "C"  __device__ void solve(double mat[42], int m, int n){
    //a按列存储
    //要把a|b拼成增广矩阵
    //直接拼成mat传进来，因为是按列存 要接在尾巴还挺容易
	//下面代码将增广矩阵化为上三角矩阵，并判断增广矩阵秩是否为 n
    int k;
    double tmp=0.0;
    double max;
    int max_k;
	for (int i = 0; i < m; ++i)
	{
		//寻找第 i 列绝对值最大的元素
        max = 1e-20;
        max_k = -1;
        tmp=0.0;
		for (k = i; k < m; ++k)
		{
			if (fabs(mat[i*m+k]) > max){
				//不同线程会有分歧
                max = fabs(mat[i*m+k]);
                max_k = k;
                //！！！！！！！！！！！！！！！！！！！
            }
		}

		if (max_k>i)
        {    //说明第 i 列有不为0的元素，且abs最大的元素不在i行，进行交换
			//交换第 i 行和第 max_k 行所有元素。增广矩阵每行有n+1个元素
			for (int j = i; j <= n; ++j)//从每行第 i 个元素交换即可，因为前面的元素都为0
			{//使用中间变量交换元素
				tmp = mat[i+j*m]; mat[i+j*m] = mat[max_k+j*m]; mat[max_k+j*m] = tmp;
			}
        }

        if(max_k>-1){
            //交换完进行消元
			double c;//倍数
			for (int j = i + 1; j < m; j++)
			{   //利用mat[i][i]消去i行下方的行(j行) 第i列的值，同时更新j行其他列的值
				c = -mat[j+i*m] / mat[i+i*m];
				for (k = i+1; k <= n; k++)
				{//第i列位于i行下方的值可以默认为0，可不操作.最后的第n+1列就也得修改，所以停止条件为k=n+1
					mat[j+k*m] += c * mat[i+k*m];//第 i 行 a 倍加到第 j 行
				}
			}
		}
		else //没有找到则说明系数矩阵秩不为 n ，说明方程组中有效方程的个数小于 n
		{
			//cout << "系数矩阵奇异，线性方程组无解或有无数解" << endl;
			return;
		}
	}

	for (int i = m-1; i >= 0; --i)
	{
		//解放在b原地
		for (int j = m-1; j > i; --j)
		{
			mat[n*m+i] -= mat[n*m+j] * mat[i+j*m];  //mat[n*m+j] 即x_j是已经求解出的部分
		}
		mat[n*m+i] = mat[n*m+i]/mat[i+i*m];
	}
    return;
}



extern "C" __global__ void get_edge_lengths(double* field_val,int* ndglno,int ddim, int dngi, int dloc, int ele_count, double* h_bar){
                //double* metric,double* eval, double* evec){
#ifndef DOUBLEP
    #define DOUBLEP
#endif
    //ele_tensor = simplex_tensor(coordinates, ele);
    //-----------begin 实现simplex_tensor-------------------------------------------------
    //double ele_tensor[9]={0.0};
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=ele_count)return;
    int dim = ddim;
    int loc = dloc;

    int index[4]={0};//循环展开，128bits可以有什么优化读取嘛
    index[0]= ndglno[idx*loc]-1;
    index[1]= ndglno[idx*loc+1]-1;
    index[2]= ndglno[idx*loc+2]-1;
    index[3]= ndglno[idx*loc+3]-1;
    int i,j,k,l,n,p;

    double pos_ele[12]={0.0};//fortran dim(3,4) 一列的3个元素连续存
    for(i=0;i<loc;++i){
        //为每个顶点 取三个连续的坐标
        pos_ele[i*dim] = field_val[index[i]*dim];
        pos_ele[i*dim+1] = field_val[index[i]*dim+1];
        pos_ele[i*dim+2] = field_val[index[i]*dim+2];
    }

    double A[6*7] = {0.0};  //6条边 A实际是A|b的增广矩阵
    double* x = &A[6*6];    //x指向b部分数据
    for(i=0;i<6;++i){
        x[i] = 1.0;
    }
    double diff[3] = {0.0};
    n=0;
    for(i=0;i<loc;++i){
        for(j=i+1;j<loc;++j){
            diff[0] = pos_ele[j*dim]-pos_ele[i*dim];
            diff[1] = pos_ele[j*dim+1]-pos_ele[i*dim+1];
            diff[2] = pos_ele[j*dim+2]-pos_ele[i*dim+2];
            //计算每2点之间各坐标的距离
            //if(blockIdx.x==1&&threadIdx.x==100)printf("diff:%f,%f,%f \n",diff[0],diff[1],diff[2]);
            for(k=0;k<dim;++k){
                for(l=0;l<dim;++l){
                    //考虑是否展开
                    p = get_idx(k,l,dim);
                    A[p*6+n]=diff[k]*diff[l]*coeff(k,l);
                    //p为列号：不太清楚，是dim间的关系（？）； n为行:不同的边
                }
            }
            ++n;

        }
    }

    solve(A,6,6); //A就用一次 占用这么大，考虑下放global或是多多复用
    double m[9]={0.0};
    
    for(j=0;j<dim;++j){
        for(i=0;i<dim;++i){
            m[j*dim+i]=x[get_idx(i,j,dim)];
        }

    }
    //---------------------end 实现simplex_tensor(coordinates, ele)-------------------------


    for(j=0;j<dim;++j){
        for(i=0;i<=j;++i){
            //x[i+1+ j*(j+1)/2-1]= m[j*dim+i]; 
            //先按fortran的坐标算 即i、j为1 based，再对最终算出的坐标结果减1
            //i理解为行，j理解为列
            x[i + j*(j+1)/2 ]= m[j*dim+i];
            
        }
    }

#ifdef DOUBLEP
//调device
    //dspev(x, diff, m);
    myArray<double, 3> eigen_value={0.0};
    myArray<myArray<double, 3>, 3> eigen_vec={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
    SymmetricEigensolver3x3 <double>DSPEV;
    DSPEV(x[0],x[1],x[3],x[2],x[4],x[5],false, 1,eigen_value,eigen_vec);
    for (i=0;i<3;i++){
        diff[i]=eigen_value[i];
        for (j=0;j<3;j++){
            m[i*3+j]=eigen_vec[i][j];
        }
    }
#else
    
#endif
//------------------------end 实现eigendecomposition_symmetric(m,evecs,evals)------------------------------------------

//------------------------begin 实现edge_length_from_eigenvalue_vector(evals)---------------------
//evals复用的diff
    
    for(i=0;i<3;++i){
        if(diff[i]<1e-20){
            printf("block %d   thread %d gg %16.lf\n",blockIdx.x,threadIdx.x,diff[i]);
        }else{
            diff[i] = 1.0/sqrt(abs(diff[i]));
        }
    }
//-----------------------end 实现edge_length_from_eigenvalue_vector(evals)---------------------
//李老师说这样取特征值并进行的一系列计算是为了得到一个矩阵的平方（开方是另一个函数）
//----------------begin eigenrecomposition(edge, evecs, edge_length_from_eigenvalue_vector(evals))
//eigenrecomposition(M, V, A) A复用的diff V复用的m
//double edge[3*3];//复用A

    for(i=0;i<3;++i){
        A[i*3]=diff[i]*m[i*3];
        A[i*3+1]=diff[i]*m[i*3+1];
        A[i*3+2]=diff[i]*m[i*3+2];
    }
    //需要转置m,用pos_ele装 transpose(m)，然后再用m装矩阵乘的结果

    for(i=0;i<3;++i){
        for(j=0;j<3;++j){
            pos_ele[j*3+i] = m[i*3+j];
            m[i*3+j]=0.0;   //为矩阵乘清空结果
        }
    }
    get_edge_matmul(A,pos_ele,m,3,3,3);
//----------------end eigenrecomposition----------------------------


    //m就是最后要复制到ngi维度上的h_bar结果
    for(j=0;j<dngi;++j){
        for(i=0;i<9;++i){
                h_bar[idx*11*9+j*9+i]=m[i];
        }        
    }
    return;

}

extern double *x_d_field_val;
extern int *x_d_ndglno;
extern double *kmk_d_shape;
extern double *kmk_d_detwei;
extern double *kmk_d_hbar;

extern "C" void get_edge_lengths_gpu_(int* dim, int* loc, int* ngi, int* ele_count, int* vertex_num,double* field_val, int* ndglno){

    double* d_hbar=kmk_d_hbar;
    double* d_field_val=x_d_field_val;
    int* d_ndglno = x_d_ndglno;
    cudaError_t error;
    //在transform先传了
    int block_size = 128;

    int num_block=((*ele_count) + block_size - 1) / block_size;
    dim3 grid(num_block);
    dim3 block(block_size); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    get_edge_lengths<<<grid,block>>>(d_field_val,d_ndglno,*dim,*ngi,*loc,*ele_count,d_hbar);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    error = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel get edge lengths took %f ms\n", milliseconds);
}

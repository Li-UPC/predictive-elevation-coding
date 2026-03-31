# Lossless heightmap compressor

Lossless compressor for 2D elevation grids (heightmaps). It combines a **Lagrange predictor** with **adaptive arithmetic coding** to achieve high compression ratios on integer matrices.

---

## How it works

### 1. Elevation data prediction

Let `M` be a grid of points `(i, j)` with `i = 0, ..., N` and `j = 0, ..., M`.

For every point `M[i][j]` with `i >= 2` and `j >= 2`, its value is predicted as the nearest integer to a linear combination of 12 neighboring points:

```
μ1·z1 + μ2·z2 + μ3·z3 + μ4·z4 + μ5·z5 + α1·b1 + α2·b2 + ... + α7·b7
```

The local neighborhood around the point `x` to predict is:

```
+----+----+----+----+----+
| b3 | b4 | b5 | b6 | b7 |
+----+----+----+----+----+
| b2 | z2 | z3 | z4 | z5 |
+----+----+----+----+----+
| b1 | z1 |  x |    |    |
+----+----+----+----+----+
```

Points `b1`, `b2`, `z4`, and `z5` are taken as zero if they fall outside the grid.

#### Computing the coefficients

The 12 coefficients are computed using the **Lagrange multiplier method**, minimizing the sum of squared prediction errors over the entire grid:

```
f(μ1,...,μ5, α1,...,α7) = Σ ( M[i][j] - μ1·z1 - ... - μ5·z5 - α1·b1 - ... - α7·b7 )²
```

subject to the constraint that the **mean prediction error is zero**:

```
Σ ( M[i][j] - μ1·z1 - ... - μ5·z5 - α1·b1 - ... - α7·b7 ) = 0
```

Taking derivatives with respect to the coefficients and the Lagrange multiplier yields a **13×13 linear system**, solved in Python using `scipy.linalg.solve`.

---

### 2. Arithmetic coding

The prediction residuals are encoded using **adaptive arithmetic coding**.

#### Why arithmetic coding?

Unlike Huffman coding, which assigns an integer number of bits to each symbol, arithmetic coding assigns **fractional bits**, approaching the theoretical entropy limit `H = -Σ p·log₂(p)`. This matters because prediction residuals are heavily concentrated near zero: a dominant zero residual ideally needs only ~0.1 bits, but Huffman would be forced to assign 1 bit — a 10x overhead for the most frequent symbol alone.

#### Why three frequency tables?

A single table covering the full residual range would contain many symbols with near-zero frequency, slowing down adaptation and reducing compression efficiency. Instead, three separate tables are used, each learning its own distribution faster:

1. **freqs** — values in `[-70, 70]` (the vast majority of residuals)
2. **freqs2** — values in `(-200, -70)` and `(70, 200)`
3. **freqs3** — digits `0–9` for values with absolute value > 200

Each table contains an extra **escape symbol** to signal a table switch to the decoder, which always tracks the last table used.

---

### 3. Structure of the compressed file

1. IEEE 754 encoding of the 12 predictor coefficients (48 bytes).
2. Number of rows, number of columns, and `M[0][0]`.
3. First two rows and columns encoded as differences.
4. Prediction residuals for the remaining elements.

---

## Implementation overview

Compression is a two-step process:

1. **Python** (`compress.py`) loads the input matrix and solves the 13×13 linear system to obtain the 12 Lagrange predictor coefficients.
2. **C++** (`compressor`) receives the coefficients, computes the prediction residuals, and encodes them using adaptive arithmetic coding.

Decompression is handled entirely by the C++ executable, which reads the coefficients from the file header and reconstructs the original matrix.

---

## Dependencies

- g++ with C++11 support
- Python 3 with: `numpy`, `pandas`, `scipy`

```
pip install numpy pandas scipy
```

## Setup

First, compile the C++ executable from the `src/` directory (required before running `compress.py`):

```
cd src
g++ -std=c++11 -Os -s \
    compressor.cpp ArithmeticCoder.cpp BitIoStream.cpp FrequencyTable.cpp \
    -o compressor
```

## Usage

The input file must be a space-delimited matrix of integers (one row per line).
All commands must be run from the `src/` directory.

**Compress:**
```
python3 compress.py c infile outfile
```

**Decompress:**
```
python3 compress.py d infile outfile
```

**Example:**
```
cd src
python3 compress.py c ../tests/file11111.txt output.cdi
python3 compress.py d output.cdi recovered.txt
```

## Test results

All test files compress losslessly (decompressed output is bit-for-bit identical to the original).

| File | Original | Compressed | Ratio | Bits/value |
|------|----------|------------|-------|------------|
| file11111 | 169 MB | 19.5 MB | 8.27x | 5.44 bpp |
| file21212 | 172 MB | 17.8 MB | 9.67x | 4.96 bpp |
| file22121 | 172 MB | 16.4 MB | 10.50x | 4.57 bpp |
| file22222 | 172 MB | 16.7 MB | 10.29x | 4.66 bpp |

---

## Attribution

The arithmetic coding implementation (`ArithmeticCoder.cpp/hpp`, `BitIoStream.cpp/hpp`, `FrequencyTable.cpp/hpp`) is taken from
[Reference arithmetic coding](https://github.com/nayuki/Reference-arithmetic-coding) by [Project Nayuki](https://www.nayuki.io/page/reference-arithmetic-coding),
published under the MIT License.

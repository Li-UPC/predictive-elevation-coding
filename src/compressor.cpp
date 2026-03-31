#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "ArithmeticCoder.hpp"
#include "BitIoStream.hpp"
#include "FrequencyTable.hpp"
#include <sstream>
#include <vector>
#include <limits>

using namespace std;

using std::uint32_t;

const int SMALL_THRESHOLD = 70, LARGE_THRESHOLD = 200;
int paddingCols = 2;

/****** DECOMPRESSION ******/

// Reverses the digits of n (e.g. 1230 -> 0321 = 321).
// Used to undo the digit-reversal applied during encoding of large residuals.

int reverseDigits(int n) {
    int result = 0;

    while (n > 0) {
        int digit = n % 10;           // extract last digit
        result = result * 10 + digit;
        n /= 10;                      // remove last digit
    }
	return result;

}

// Decodes a large residual (absolute value > LARGE_THRESHOLD) encoded digit by digit.
// Digits were stored in reverse order terminated by symbol 10.
int decodeLarge(SimpleFrequencyTable &freqs3, ArithmeticDecoder &dec){
	uint32_t symbol = dec.read(freqs3);
	freqs3.increment(symbol);
	int number = 0;
	while (int(symbol) != 10){
		number = number*10 + int(symbol);
		symbol = dec.read(freqs3);
		freqs3.increment(symbol);
	}
	return number;
}

// Decodes a single residual value using a three-tier frequency table scheme.
// - freqs  handles small values in [-SMALL_THRESHOLD, SMALL_THRESHOLD]
// - freqs2 handles medium values in (-LARGE_THRESHOLD, -SMALL_THRESHOLD) and (SMALL_THRESHOLD, LARGE_THRESHOLD)
// - freqs3 handles large values with absolute value > LARGE_THRESHOLD (digit by digit)
// tier tracks which frequency table is active: 0 = small (freqs), 1 = medium/large (freqs2/freqs3).
int decodeSimple(SimpleFrequencyTable &freqs,SimpleFrequencyTable &freqs2,SimpleFrequencyTable &freqs3, ArithmeticDecoder &dec, int& tier){
	if (tier == 0){
		uint32_t symbol = dec.read(freqs);
		freqs.increment(symbol);

		if (int(symbol) == (2*SMALL_THRESHOLD+1)){ // escape code: switch to freqs2
			tier = 1;
			return decodeSimple(freqs, freqs2, freqs3, dec, tier);
		}
		return int(symbol) - SMALL_THRESHOLD;
	}
	else if (tier == 1){
		uint32_t symbol = dec.read(freqs2);
		freqs2.increment(symbol);

		if (int(symbol) ==(2*(LARGE_THRESHOLD-SMALL_THRESHOLD) +1)){ // large negative
			symbol = decodeLarge(freqs3, dec);
			return -int(symbol);

		}
		else if (int(symbol) == (2*(LARGE_THRESHOLD - SMALL_THRESHOLD)+ 2)){ // large positive
			symbol = decodeLarge(freqs3, dec);
			return int(symbol);
		}
		else if (int(symbol) == (2*(LARGE_THRESHOLD-SMALL_THRESHOLD)+3)){ // escape code: switch back to freqs
			tier = 0;
			return decodeSimple(freqs, freqs2, freqs3, dec, tier);
		}
		int res = int(symbol) - (LARGE_THRESHOLD - SMALL_THRESHOLD);
		if (res < 0) res -= SMALL_THRESHOLD;
		else res += SMALL_THRESHOLD;
		return res;
	}
	return 0;
}


/****** COMPRESSION ******/

// Reads a space-delimited matrix of integers from a text file.
// Appends paddingCols zeros to each row (used as padding for the Lagrange predictor).
std::vector<std::vector<int>>
loadtxt(const std::string& ruta, char delimiter = ' ')
{
    std::ifstream file(ruta);
    if (!file)
        throw std::runtime_error("Could not open file: " + ruta);

    std::vector<std::vector<int>> data;
    std::string line;
    std::size_t ncols = 0;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        std::stringstream ss(line);
        std::vector<int> row;
        int value;

        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == delimiter) ss.ignore();
        }
		for (int i = 0; i < paddingCols; ++i) row.push_back(0);

		// Check all rows have the same number of columns
        if (ncols == 0)
            ncols = row.size();
        else if (row.size() != ncols)
            throw std::runtime_error("Rows have different number of columns");

        data.push_back(std::move(row));
    }
    return data;
}


// Applies the 12-coefficient Lagrange predictor to the input matrix and returns
// the prediction residuals (actual - predicted) as a (n-2) x (m-2) matrix.
// The first two rows and columns are handled separately as boundary conditions.
std::vector<std::vector<int>> linearPrediction(const std::vector<std::vector<int>>& nums, const vector<float>& sol) {
    size_t n = nums.size();
    size_t m = nums.empty() ? 0 : (nums[0].size() - paddingCols );
    if (n < 2 || m < 2) {
        return {};
    }

    std::vector<std::vector<int>> dif(n-2, std::vector<int>(m-2));

    for (size_t i = 2; i < n; ++i) {
        for (size_t j = 2; j < m; ++j) {
            dif[i-2][j-2] = int(sol[0]*nums[i][j-1] + sol[1]*nums[i-1][j-1] + sol[2]* nums[i-1][j] + sol[3]*nums[i-1][j+1] + sol[4]*nums[i-1][j+2] + sol[5]*nums[i][j-2] + sol[6]*nums[i-1][j-2] + sol[7]*nums[i-2][j-2] + sol[8]*nums[i-2][j-1] + sol[9]*nums[i-2][j] + sol[10]*nums[i-2][j+1] + sol[11]*nums[i-2][j+2] + 0.5) - nums[i][j];
        }
    }

    return dif;
}


// Encodes a single residual value using a three-tier frequency table scheme.
// - freqs  handles small values in [-SMALL_THRESHOLD, SMALL_THRESHOLD]
// - freqs2 handles medium values in (-LARGE_THRESHOLD, -SMALL_THRESHOLD) and (SMALL_THRESHOLD, LARGE_THRESHOLD)
// - freqs3 handles large values with absolute value > LARGE_THRESHOLD (digit by digit)
// tier tracks which frequency table is active: 0 = small (freqs), 1 = medium/large (freqs2/freqs3).
void encode(ArithmeticEncoder& enc, SimpleFrequencyTable& freqs, SimpleFrequencyTable& freqs2, SimpleFrequencyTable& freqs3, int& tier, int symbol){
	if (abs(symbol) <= SMALL_THRESHOLD){
		if (tier == 0){
			symbol += SMALL_THRESHOLD;
			enc.write(freqs, static_cast<uint32_t>(symbol));
			freqs.increment(static_cast<uint32_t>(symbol));
		}
		else {
			tier = 0;
			enc.write(freqs2, static_cast<uint32_t>(2*(LARGE_THRESHOLD - SMALL_THRESHOLD)+3));
			freqs2.increment(static_cast<uint32_t>(2*(LARGE_THRESHOLD - SMALL_THRESHOLD)+3));
			encode(enc, freqs, freqs2, freqs3, tier, symbol);
		}
	}
	else if (SMALL_THRESHOLD < abs(symbol) and abs(symbol) <= LARGE_THRESHOLD){
		if (tier == 1){
			if (symbol < 0) symbol += SMALL_THRESHOLD;
			else symbol -= SMALL_THRESHOLD;
			symbol += (LARGE_THRESHOLD-SMALL_THRESHOLD);
			enc.write(freqs2, static_cast<uint32_t>(symbol));
			freqs2.increment(static_cast<uint32_t>(symbol));
		}
		else{
			tier = 1;
			enc.write(freqs, static_cast<uint32_t>(2*SMALL_THRESHOLD+1));
			freqs.increment(static_cast<uint32_t>(2*SMALL_THRESHOLD+1));
			encode(enc, freqs, freqs2, freqs3, tier, symbol);
		}
	}
	else{
		if (tier == 1){
			if (symbol<0) {
				symbol = -symbol;
				enc.write(freqs2, static_cast<uint32_t>(2*(LARGE_THRESHOLD - SMALL_THRESHOLD)+1));
				freqs2.increment(static_cast<uint32_t>(2*(LARGE_THRESHOLD - SMALL_THRESHOLD)+1));
			}
			else{
				enc.write(freqs2, static_cast<uint32_t>(2*(LARGE_THRESHOLD - SMALL_THRESHOLD)+2));
				freqs2.increment(static_cast<uint32_t>(2*(LARGE_THRESHOLD - SMALL_THRESHOLD)+ 2));
			}

			int trailingZeros = 0;
			int copy = symbol;
			while (copy%10 == 0){
				++trailingZeros;
				copy /= 10;
			}

			symbol = reverseDigits(symbol);

			while (symbol != 0){
				int aux = symbol%10;
				enc.write(freqs3, static_cast<uint32_t>(aux));
				freqs3.increment(static_cast<uint32_t>(aux));
				symbol /= 10;
			}
			while (trailingZeros) {
				int aux = 0;
				enc.write(freqs3, static_cast<uint32_t>(aux));
				freqs3.increment(static_cast<uint32_t>(aux));
				--trailingZeros;
			}

			enc.write(freqs3, static_cast<uint32_t>(10));
			freqs3.increment(static_cast<uint32_t>(10));
		}
		else {
			tier = 1;
			enc.write(freqs, static_cast<uint32_t>(2*SMALL_THRESHOLD+1));
			freqs.increment(static_cast<uint32_t>(2*SMALL_THRESHOLD+1));
			encode(enc, freqs, freqs2, freqs3, tier, symbol);
		}
	}
}


// Compresses inputFile into outputFile using Lagrange prediction + arithmetic coding.
// The 12 predictor coefficients (sol) are stored in the first 48 bytes of the output
// so that decompression does not require them to be passed again.
int compress(const char* inputFile, const char* outputFile, const vector<float>& sol){

	auto matrix = loadtxt(inputFile);
	auto dif = linearPrediction(matrix, sol);
	int n = matrix.size();
	int m = matrix[0].size()-paddingCols;


	int rows = dif.size();
	int cols = dif[0].size();


	std::ofstream out(outputFile, std::ios::binary);
	for (int i = 0; i < 12; ++i){
		out.write(reinterpret_cast<const char*>(&sol[i]), sizeof(float));
	}
	BitOutputStream bout(out);
	try {
		int tier = 1;

		SimpleFrequencyTable freqs(FlatFrequencyTable(2*SMALL_THRESHOLD + 2));
		SimpleFrequencyTable freqs2(FlatFrequencyTable(2*(LARGE_THRESHOLD-SMALL_THRESHOLD) + 5));
		SimpleFrequencyTable freqs3(FlatFrequencyTable(11));
		ArithmeticEncoder enc(32, bout);

		// encode matrix dimensions and boundary rows/columns
		encode(enc, freqs, freqs2, freqs3, tier, n);
		encode(enc, freqs, freqs2, freqs3, tier, m);
		encode(enc, freqs, freqs2, freqs3, tier, matrix[0][0]);

		for(int j = 1; j < m; ++j) encode(enc, freqs, freqs2, freqs3, tier, matrix[0][j]-matrix[0][j-1]);
		for(int i = 1; i < n; ++i) encode(enc, freqs, freqs2, freqs3, tier, matrix[i][0]-matrix[i-1][0]);

		for(int j = 1; j < m; ++j) encode(enc, freqs, freqs2, freqs3, tier, matrix[1][j]-matrix[1][j-1]);
		for(int i = 2; i < n; ++i) encode(enc, freqs, freqs2, freqs3, tier, matrix[i][1]-matrix[i-1][1]);

		// encode prediction residuals
		for (int i = 0; i < rows; ++i){
			for (int j = 0; j < cols; ++j){
				int symbol = dif[i][j];
				encode(enc, freqs, freqs2, freqs3, tier, symbol);
			}
		}
		
		enc.finish();  // Flush remaining code bits
		bout.finish();
		return EXIT_SUCCESS;
		
	} catch (const char *msg) {
		std::cerr << msg << std::endl;
		return EXIT_FAILURE;
	}
}


// Decompresses inputFile into outputFile.
// Reads the 12 predictor coefficients from the file header, then reconstructs
// the original matrix by decoding residuals and reversing the Lagrange prediction.
int decompress(const char* inputFile, const char* outputFile){
	std::ifstream in(inputFile, std::ios::binary);
	vector<float> sol(12);
	for (int i = 0; i<12; ++i) {
		in.read(reinterpret_cast<char*>(&sol[i]), sizeof(float));
	}

	std::ofstream fout(outputFile);

	BitInputStream bin(in);
	try {
		int tier = 1;
		SimpleFrequencyTable freqs(FlatFrequencyTable(2*SMALL_THRESHOLD + 2));
		SimpleFrequencyTable freqs2(FlatFrequencyTable(2*(LARGE_THRESHOLD-SMALL_THRESHOLD) + 5));
		SimpleFrequencyTable freqs3(FlatFrequencyTable(11));
		ArithmeticDecoder dec(32, bin);

		int rows = decodeSimple(freqs, freqs2, freqs3, dec, tier);
		int cols = decodeSimple(freqs, freqs2, freqs3, dec, tier);
		std::vector<std::vector<int>> v(rows, std::vector<int>(cols+paddingCols));
		v[0][0] = decodeSimple(freqs, freqs2, freqs3, dec, tier);

		for (int j = 1; j < cols; ++j) {
			v[0][j] = v[0][j-1] + decodeSimple(freqs, freqs2, freqs3, dec, tier);
		}
		for (int i = 1; i < rows; ++i) v[i][0] = v[i-1][0] + decodeSimple(freqs, freqs2, freqs3, dec, tier);

		for (int j = 1; j < cols; ++j) {
			v[1][j] = v[1][j-1] + decodeSimple(freqs, freqs2, freqs3, dec, tier);
		}
		for (int i = 2; i < rows; ++i) {
			v[i][1] = v[i-1][1] + decodeSimple(freqs, freqs2, freqs3, dec, tier);
		}

		for (int i = 2; i < rows; ++i){
			for (int j = 2; j < cols; ++j){
				v[i][j] = decodeSimple(freqs, freqs2, freqs3, dec, tier);
				v[i][j]= int(sol[0]*v[i][j-1] + sol[1]*v[i-1][j-1] + sol[2]*v[i-1][j] + sol[3]*v[i-1][j+1] + sol[4]*v[i-1][j+2] + sol[5]*v[i][j-2] + sol[6]*v[i-1][j-2] + sol[7]*v[i-2][j-2] + sol[8]*v[i-2][j-1] + sol[9]*v[i-2][j] + sol[10]*v[i-2][j+1] + sol[11]*v[i-2][j+2] + 0.5) - v[i][j];

			}
		}

		for (const auto& row : v) {
			for (size_t j = 0; j < row.size() - paddingCols; ++j) {
				fout << row[j];
				if (j + 1 < row.size() - paddingCols)
					fout << ' ';
			}
			fout << '\n';
		}

    	fout.close();

		return EXIT_SUCCESS;
		
	} catch (const char *msg) {
		std::cerr << msg << std::endl;
		return EXIT_FAILURE;
	}
}



int main(int argc, char *argv[]) {
	char t = argv[1][0];
	const char *inputFile  = argv[2];
	const char *outputFile = argv[3];

	if (t == 'c'){
		vector<float> sol(12);
		for (int i= 0; i < 12; ++i){
			sol[i] = std::atof(argv[4 + i]);
		}
		compress(inputFile, outputFile,sol);
	}
	else if(t == 'd')
		decompress(inputFile, outputFile);
}



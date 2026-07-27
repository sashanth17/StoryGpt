#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <regex>

using namespace std;

// Struct to represent a pair of integer tokens
struct Pair {
  int left;
  int right;
  bool operator==(const Pair &o) const {
    return left == o.left && right == o.right;
  }
};

struct PairHash {
  size_t operator()(const Pair &p) const {
    return ((uint64_t)p.left << 32) | (uint32_t)p.right;
  }
};

class Tokenizer {
public:
    unordered_map<string, int> token_to_id;
    vector<string> id_to_token;
    unordered_map<Pair, int, PairHash> merges; // Maps a pair of token IDs to its merge rank

    Tokenizer(const char* bin_path, const char* merges_path) {
        // 1. Load Vocab from Binary File (No JSON parsing!)
        ifstream bin_file(bin_path, ios::binary);
        if (bin_file.is_open()) {
            uint32_t total_tokens;
            bin_file.read(reinterpret_cast<char*>(&total_tokens), sizeof(total_tokens));
            for (uint32_t i = 0; i < total_tokens; i++) {
                uint32_t len;
                bin_file.read(reinterpret_cast<char*>(&len), sizeof(len));
                string token(len, '\0');
                bin_file.read(&token[0], len);
                
                id_to_token.push_back(token);
                token_to_id[token] = i;
            }
        } else {
            cerr << "Warning: Could not open " << bin_path << endl;
        }

        // 2. Load Merges from bin file (No string parsing!)
        ifstream merges_file(merges_path, ios::binary);
        if (merges_file.is_open()) {
            uint32_t total_merges;
            merges_file.read(reinterpret_cast<char*>(&total_merges), sizeof(total_merges));
            for (uint32_t i = 0; i < total_merges; i++) {
                uint32_t left, right;
                merges_file.read(reinterpret_cast<char*>(&left), sizeof(left));
                merges_file.read(reinterpret_cast<char*>(&right), sizeof(right));
                Pair p = {(int)left, (int)right};
                merges[p] = i; // rank is exactly the insertion index
            }
        } else {
            cerr << "Warning: Could not open " << merges_path << endl;
        }
        cout << "Loaded vocab size: " << token_to_id.size() << endl;
        cout << "Loaded merges size: " << merges.size() << endl;
    }

    // Standard BPE algorithm for inference (applies rank-ordered merges)
    vector<int> bpe(const string& word) {
        vector<int> tokens;
        for (char c : word) {
            tokens.push_back(static_cast<uint8_t>(c)); // initial byte tokens (0-255)
        }

        while (tokens.size() >= 2) {
            int best_rank = 1e9;
            int best_idx = -1;
            Pair best_pair = {-1, -1};

            for (size_t i = 0; i < tokens.size() - 1; i++) {
                Pair p = {tokens[i], tokens[i+1]};
                if (merges.find(p) != merges.end()) {
                    if (merges[p] < best_rank) {
                        best_rank = merges[p];
                        best_idx = i;
                        best_pair = p;
                    }
                }
            }

            if (best_idx == -1) break; // no more possible merges

            // Apply merge
            string merged_str = id_to_token[best_pair.left] + id_to_token[best_pair.right];
            int merged_id = token_to_id[merged_str]; // string to ID

            vector<int> new_tokens;
            for (size_t i = 0; i < tokens.size(); i++) {
                if (i == best_idx) {
                    new_tokens.push_back(merged_id);
                    i++; // skip the right part of the merged pair
                } else {
                    new_tokens.push_back(tokens[i]);
                }
            }
            tokens = new_tokens;
        }
        return tokens;
    }

    vector<int> encode(const string& text) {
        // Pre-tokenize text precisely as we did during training
        regex gpt2_regex(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+(?!\S)|\s+)");
        sregex_iterator it(text.begin(), text.end(), gpt2_regex);
        sregex_iterator end;

        vector<int> all_tokens;
        while (it != end) {
            vector<int> word_tokens = bpe(it->str());
            all_tokens.insert(all_tokens.end(), word_tokens.begin(), word_tokens.end());
            ++it;
        }
        return all_tokens;
    }
};

// ----------------------------------------------------
// C API for Python ctypes
// ----------------------------------------------------
extern "C" {
    void* create_tokenizer(const char* bin_path, const char* merges_path) {
        return new Tokenizer(bin_path, merges_path);
    }

    int* encode(void* tokenizer, const char* text, int* out_length) {
        Tokenizer* tok = static_cast<Tokenizer*>(tokenizer);
        vector<int> tokens = tok->encode(string(text));
        
        *out_length = tokens.size();
        int* result = new int[tokens.size()];
        copy(tokens.begin(), tokens.end(), result);
        return result;
    }

    void free_tokens(int* tokens) {
        delete[] tokens;
    }

    void free_tokenizer(void* tokenizer) {
        delete static_cast<Tokenizer*>(tokenizer);
    }
}

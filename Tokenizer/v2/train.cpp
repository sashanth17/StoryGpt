#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <regex>
#include <algorithm>

using namespace std;

// 1. Issue 3 Fixed: Linked-list nodes for O(1) updates
struct TokenNode {
  int id;
  int prev;
  int next;
};

struct Word {
  vector<TokenNode> nodes;
  int head;
  int count;
};

// Struct to represent a pair of integer tokens
struct Pair {
  int left;
  int right;
  bool operator==(const Pair &o) const {
    return left == o.left && right == o.right;
  }
};

// 2. Issue 1 Fixed: Better Hash Function
struct PairHash {
  size_t operator()(const Pair &p) const {
    return ((uint64_t)p.left << 32) | (uint32_t)p.right;
  }
};

struct PairFreq {
  Pair p;
  uint64_t freq;
  bool operator<(const PairFreq &o) const {
    return freq < o.freq; // Max-heap based on frequency
  }
};

string escape_json(const string &s) {
  string res;
  for (char c : s) {
    if (c == '"') res += "\\\"";
    else if (c == '\\') res += "\\\\";
    else if (c == '\b') res += "\\b";
    else if (c == '\f') res += "\\f";
    else if (c == '\n') res += "\\n";
    else if (c == '\r') res += "\\r";
    else if (c == '\t') res += "\\t";
    else res += c;
  }
  return res;
}

int main() {
  string dataset = "../validation.csv";
  int vocab_size = 4096;

  cout << "Enter desired vocabulary size (default " << vocab_size << "): ";
  string input;
  getline(cin, input);
  if (!input.empty()) {
    vocab_size = stoi(input);
  }

  // ----------------------------
  // 1. Initialize Byte-level Vocab (Issue 5 Fixed)
  // ----------------------------
  unordered_map<string, int> token_to_id;
  vector<string> id_to_token;
  for (int i = 0; i < 256; i++) {
    string s(1, static_cast<char>(static_cast<uint8_t>(i)));
    token_to_id[s] = i;
    id_to_token.push_back(s);
  }

  auto get_or_add_token = [&](const string &s) {
    if (token_to_id.find(s) == token_to_id.end()) {
      token_to_id[s] = id_to_token.size();
      id_to_token.push_back(s);
    }
    return token_to_id[s];
  };

  // ----------------------------
  // 2. Load dataset with streaming and Regex splitting (Issues 4 & 6 Fixed)
  // ----------------------------
  ifstream file(dataset, ios::binary);
  if (!file.is_open()) {
    cerr << "Failed to open " << dataset << endl;
    return 1;
  }

  unordered_map<string, int> word_counts;
  string line;
  
  // Approximate GPT-2 Regex pattern (without full unicode categories due to C++ limits)
  regex gpt2_regex(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+(?!\S)|\s+)");

  cout << "Reading and pre-tokenizing dataset..." << endl;
  while (getline(file, line)) {
    line += "\n"; // getline strips newline, so we add it back
    sregex_iterator it(line.begin(), line.end(), gpt2_regex);
    sregex_iterator end;
    while (it != end) {
      word_counts[it->str()]++;
      ++it;
    }
  }
  file.close();

  // ----------------------------
  // 3. Create initial corpus (Doubly-Linked List)
  // ----------------------------
  vector<Word> corpus;
  for (const auto &kv : word_counts) {
    Word w;
    w.count = kv.second;
    w.head = 0;
    for (size_t i = 0; i < kv.first.size(); i++) {
      uint8_t b = static_cast<uint8_t>(kv.first[i]);
      TokenNode node;
      node.id = b;
      node.prev = (i == 0) ? -1 : i - 1;
      node.next = (i == kv.first.size() - 1) ? -1 : i + 1;
      w.nodes.push_back(node);
    }
    corpus.push_back(w);
  }

  // ----------------------------
  // 4. Initial Counting and Indexing
  // ----------------------------
  unordered_map<Pair, uint64_t, PairHash> pair_counts;
  
  // Issue 2 Fixed: vector instead of unordered_set
  unordered_map<Pair, vector<int>, PairHash> pair_to_words;

  auto add_pair_to_word = [&](const Pair& p, int word_idx) {
      if (pair_to_words[p].empty() || pair_to_words[p].back() != word_idx) {
          pair_to_words[p].push_back(word_idx);
      }
  };

  for (int word_idx = 0; word_idx < corpus.size(); word_idx++) {
    const auto &w = corpus[word_idx];
    int curr = w.head;
    while (curr != -1) {
      int nxt = w.nodes[curr].next;
      if (nxt != -1) {
        Pair p = {w.nodes[curr].id, w.nodes[nxt].id};
        pair_counts[p] += w.count;
        add_pair_to_word(p, word_idx);
      }
      curr = nxt;
    }
  }

  priority_queue<PairFreq> heap;
  for (const auto &kv : pair_counts) {
    heap.push({kv.first, kv.second});
  }

  vector<pair<string, string>> merges;
  vector<Pair> merges_ids;

  // ----------------------------
  // 5. High Performance BPE Loop
  // ----------------------------
  cout << "Starting BPE merges..." << endl;
  int merges_done = 0;
  
  while (id_to_token.size() < vocab_size) {
    // Find best pair using Lazy Deletion Heap
    Pair best_pair = {-1, -1};
    uint64_t max_freq = 0;

    while (!heap.empty()) {
      PairFreq top = heap.top();
      heap.pop();

      if (pair_counts.find(top.p) != pair_counts.end() && pair_counts[top.p] == top.freq) {
        best_pair = top.p;
        max_freq = top.freq;
        break;
      }
    }

    if (max_freq == 0) break; 

    // Create new token
    string str_left = id_to_token[best_pair.left];
    string str_right = id_to_token[best_pair.right];
    string new_token_str = str_left + str_right;

    merges.push_back({str_left, str_right});
    merges_ids.push_back(best_pair);
    int new_token_id = get_or_add_token(new_token_str);

    cout << "merge " << merges_done + 1 << ": ('" << str_left << "', '" << str_right 
         << "') -> " << new_token_str << " (Freq: " << max_freq << ")" << endl;

    // Fetch and unique affected words
    vector<int> affected_words = pair_to_words[best_pair];
    pair_to_words.erase(best_pair); 
    pair_counts.erase(best_pair); // this pair is fully merged out
    
    sort(affected_words.begin(), affected_words.end());
    affected_words.erase(unique(affected_words.begin(), affected_words.end()), affected_words.end());

    for (int word_idx : affected_words) {
      auto &w = corpus[word_idx];

      int curr = w.head;
      while (curr != -1) {
        int nxt = w.nodes[curr].next;
        if (nxt == -1) break;

        if (w.nodes[curr].id == best_pair.left && w.nodes[nxt].id == best_pair.right) {
          int prev = w.nodes[curr].prev;
          int nxt_nxt = w.nodes[nxt].next;

          // 1. Decrement old adjacent pairs
          if (prev != -1) {
            Pair p_left = {w.nodes[prev].id, w.nodes[curr].id};
            pair_counts[p_left] -= w.count;
          }
          if (nxt_nxt != -1) {
            Pair p_right = {w.nodes[nxt].id, w.nodes[nxt_nxt].id};
            pair_counts[p_right] -= w.count;
          }

          // 2. Apply merge (in-place linked list update)
          w.nodes[curr].id = new_token_id;
          w.nodes[curr].next = nxt_nxt;
          if (nxt_nxt != -1) w.nodes[nxt_nxt].prev = curr;
          w.nodes[nxt].id = -1; // Mark as deleted

          // 3. Add new adjacent pairs and push to heap
          if (prev != -1) {
            Pair p_new_left = {w.nodes[prev].id, w.nodes[curr].id};
            pair_counts[p_new_left] += w.count;
            add_pair_to_word(p_new_left, word_idx);
            heap.push({p_new_left, pair_counts[p_new_left]});
          }
          if (nxt_nxt != -1) {
            Pair p_new_right = {w.nodes[curr].id, w.nodes[nxt_nxt].id};
            pair_counts[p_new_right] += w.count;
            add_pair_to_word(p_new_right, word_idx);
            heap.push({p_new_right, pair_counts[p_new_right]});
          }

          // Skip over the newly merged token
          nxt = nxt_nxt;
        }
        curr = nxt;
      }
    }

    merges_done++;
    
    // Issue 8 Fixed: Periodic Heap Rebuild
    if (merges_done % 1000 == 0) {
      priority_queue<PairFreq> new_heap;
      for (const auto &kv : pair_counts) {
        if (kv.second > 0) {
          new_heap.push({kv.first, kv.second});
        }
      }
      heap = std::move(new_heap);
    }
  }

  // ----------------------------
  // 6. Save results (Issue 7 Fixed: JSON and Binary)
  // ----------------------------
  cout << "Saving vocab.json..." << endl;
  ofstream vocab_file("vocab.json");
  vocab_file << "{\n";
  for (size_t i = 0; i < id_to_token.size(); i++) {
    vocab_file << "  \"" << escape_json(id_to_token[i]) << "\": " << i;
    if (i < id_to_token.size() - 1) vocab_file << ",";
    vocab_file << "\n";
  }
  vocab_file << "}\n";

  cout << "Saving merges.txt..." << endl;
  ofstream merges_file("merges.txt");
  for (const auto &m : merges) {
    merges_file << m.first << " " << m.second << "\n";
  }
  
  cout << "Saving merges.bin (Fast Binary format)..." << endl;
  ofstream merges_bin("merges.bin", ios::binary);
  uint32_t total_merges = merges_ids.size();
  merges_bin.write(reinterpret_cast<const char*>(&total_merges), sizeof(total_merges));
  for (const auto &m : merges_ids) {
      uint32_t left = m.left;
      uint32_t right = m.right;
      merges_bin.write(reinterpret_cast<const char*>(&left), sizeof(left));
      merges_bin.write(reinterpret_cast<const char*>(&right), sizeof(right));
  }
  
  cout << "Saving id_to_token.bin (Fast Binary format)..." << endl;
  ofstream bin_file("id_to_token.bin", ios::binary);
  uint32_t total_tokens = id_to_token.size();
  bin_file.write(reinterpret_cast<const char*>(&total_tokens), sizeof(total_tokens));
  for (const string& token : id_to_token) {
      uint32_t len = token.size();
      bin_file.write(reinterpret_cast<const char*>(&len), sizeof(len));
      bin_file.write(token.data(), len);
  }

  cout << "Training finished\n";
  cout << "Vocabulary size: " << id_to_token.size() << endl;
  return 0;
}

#ifndef parser_h
#define parser_h

// #include <map>
#include <string>

using namespace std;

vector<pair<pair<int, int>, double>> parse_qubo(const string& input);
string read_file(const string& filename);

#endif
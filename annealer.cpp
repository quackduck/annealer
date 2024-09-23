#include <iostream>
#include <ostream>
#include <vector>
#include <random>
#include <thread>
#include <algorithm>
#include <functional>
#include <map>

#include "parser.hh"

using namespace std;

using solution_t = vector<bool>;
using qubo_t = map<pair<int, int>, double>;
using scheduler_t = function<double(double T_0, double T, int iter, int max_iter)>;

ostream& operator << (ostream& os, const solution_t& x) {
    for (auto xi : x) os << xi << ' ';
    return os;
}

ostream& operator << (ostream& os, const qubo_t& Q) {
    for (const auto& entry : Q) {
        os << '[' << entry.first.first << ' ' << entry.first.second << "] : " << entry.second << endl;
    }
    return os;
}

struct settings {
    int max_iter;
    double T_0;
    scheduler_t temp_scheduler;
    unsigned seed;
};

struct result {
    solution_t solution;
    double energy;
};

double evaluate(const solution_t& x, const qubo_t& Q) {
    double value = 0.0;
    for (const auto& entry : Q) {
        // value += entry.second * x[entry.first.first] * x[entry.first.second];
        if (x[entry.first.first] && x[entry.first.second]) value += entry.second;
    }
    return value;
}

double evaluateDiff(const solution_t& x, const qubo_t& Q, int flip_idx) {
    double diff = 0.0;
    for (const auto& entry : Q) {
        if (entry.first.first == flip_idx || entry.first.second == flip_idx) {
            if (x[entry.first.first] && x[entry.first.second]) {
                // both on, flipping either will turn it off.
                diff -= entry.second;
            } else if (x[entry.first.first] || x[entry.first.second]) {
                // one on. flip it on if the one we're flipping is off.
                if (entry.first.first == flip_idx && !x[entry.first.first]) {
                    diff += entry.second;
                } else if (entry.first.second == flip_idx && !x[entry.first.second]) {
                    diff += entry.second;
                }
            } else {
                // both off. flip it on only if both are on.
                if (entry.first.first == flip_idx && entry.first.second == flip_idx) {
                    diff += entry.second;
                }
            }
        }
    }
    return diff;

    // for (int i = 0; i < x.size(); i++) {
    //     // double coupling = Q[{flip_idx, i}] + Q[{i, flip_idx}];
    //     // Q is const, so we can't use the above line.
    //     double coupling = 0;
    //     auto it = Q.find({flip_idx, i});
    //     if (it != Q.end()) {
    //         coupling += it->second;
    //     }
    //     if (i != flip_idx) {
    //         it = Q.find({i, flip_idx});
    //         if (it != Q.end()) {
    //             coupling += it->second;
    //         }
    //     }

    //     if (coupling != 0) {
    //         bool i_on = x[i];
    //         bool flip_on = x[flip_idx];
    //         if (i_on && flip_on) {
    //             // currently on. flipping either will turn it off.
    //             diff -= coupling;
    //         } else if (i_on || flip_on) {
    //             // one of them is on. 
    //             // both will become on if the one we're flipping is off.
    //             if (!flip_on) {
    //                 diff += coupling;
    //             }
    //         } else {
    //             // both off. flip it on only if both will become on.
    //             if (i == flip_idx) {
    //                 diff += coupling;
    //             }
    //         }
    //     }
    // }
    // return diff;
}

int qubo_size(const qubo_t& Q) {
    int n = 0;
    for (const auto& entry : Q) {
        n = max(n, entry.first.first);
    }
    return n + 1; // 0 indexed
}

result sim_anneal(const qubo_t& Q, const settings s) { // intentionally get copy of settings
    int n = qubo_size(Q);

    mt19937 gen(s.seed);
    uniform_real_distribution<> dis(0.0, 1.0);
    uniform_int_distribution<> flip_dis(0, n - 1);

    solution_t x(n, 0);
    for (auto xi : x) {
        xi = dis(gen) < 0.5 ? 0 : 1;
    }
    double f_x = evaluate(x, Q);

    solution_t best_x = x;
    double best_f_x = f_x;

    double T = s.T_0;
    for (int iter = 0; iter < s.max_iter; iter++) {
        solution_t x_prime = x;
        int flip_idx = flip_dis(gen);

        double f_x_prime = evaluateDiff(x, Q, flip_idx) + f_x;

        x_prime[flip_idx] = !x_prime[flip_idx];

        // double f_x_prime = evaluate(x_prime, Q);

        if (f_x_prime < f_x || dis(gen) < exp((f_x - f_x_prime) / T)) {
            x = x_prime;
            f_x = f_x_prime;
        }

        if (f_x < best_f_x) {
            best_x = x;
            best_f_x = f_x;
        }

        T = s.temp_scheduler(s.T_0, T, iter, s.max_iter);
    }

    return {best_x, best_f_x};
}

// returns sorted results of length num_threads * samples_per_thread
vector<result> multithreaded_sim_anneal(const qubo_t& Q, const settings s, int num_threads, int samples_per_thread = 1) {
    vector<thread> threads;
    vector<result> results(num_threads * samples_per_thread);
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&results, i, &Q, s, samples_per_thread](){
            for (int j = 0; j < samples_per_thread; j++) {
                settings s_copy = s;
                s_copy.seed += i * samples_per_thread + j;
                results[i * samples_per_thread + j] = sim_anneal(Q, s_copy);
            }
        });
    }

    for (auto& t : threads) t.join();

    sort(results.begin(), results.end(), [](const result& a, const result& b) {
        return a.energy < b.energy;
    });

    return results;
}

double linear_scheduler(double T_0, double T, int iter, int max_iter) {
    return T_0 - (T_0 / max_iter) * iter;
}

// double geometric_scheduler(double T_0, double T, int iter, int max_iter) {
//     return T * 0.99;
// }

scheduler_t make_geometric_scheduler(double alpha) {
    return [alpha](double T_0, double T, int iter, int max_iter) {
        return T * alpha;
    };
}


void trial(solution_t x, const qubo_t& Q) {
    cout << "For solution: " << x << endl;
    cout << "Energy: " << evaluate(x, Q) << endl << endl;
}

bool operator < (const result& a, const result& b) {
    return a.energy > b.energy; // reverse order
}

void present_results(const vector<result>& results) {
    map<result, int> counts;
    for (const auto& r : results) {
        counts[r]++;
    }

    for (const auto& entry : counts) {
        cout << "Energy: " << entry.first.energy << " (" << entry.second << "x)" << '\n';
        cout << "Solution: " << entry.first.solution << '\n';
    }
}

qubo_t sparsen(vector<vector<double>> dense) {
    qubo_t sparse;

    for (int i = 0; i < dense.size(); i++) {
        for (int j = 0; j < dense[i].size(); j++) {
            if (dense[i][j] != 0) { // only add non-zero entries
                sparse[{i, j}] = dense[i][j];
            }
        }
    }

    return sparse;
}

/* todo:
maybe flip all bits in sequence instead of random ones?
optimize evaluate by getting diff caused by flipping one bit
*/

int main() {
    // qubo_t Q = sparsen({
    //     { -2,  1,  1,  0,  0 },
    //     {  1, -2,  0,  1,  0 },
    //     {  1,  0, -3,  1,  1 },
    //     {  0,  1,  1, -3,  1 },
    //     {  0,  0,  1,  1, -2 }
    // });

    qubo_t Q = parse_qubo(read_file("qubo.txt"));

    // qubo_t Q = sparsen({
    //     {-17, 10, 10, 10, 0, 20},
    //     {10, -18, 10, 10, 10, 20},
    //     {10, 10, -29, 10, 20, 20},
    //     {10, 10, 10, -19, 10, 10},
    //     {0, 10, 20, 10, -17, 10},
    //     {20, 20, 20, 10, 10, -28}
    // });

    // qubo_t Q = unsparse(sparse); // just for now.

    // cout << Q << endl;

    // trial({1, 0, 0, 0, 1, 0}, Q);

    // unsigned seed = time(0); // 42
    random_device rd;
    unsigned seed = rd();

    settings s = {.max_iter = 40000, .T_0 = 100.0, .temp_scheduler = make_geometric_scheduler(0.999), .seed = seed};

    // settings s = {.max_iter = 5, .T_0 = 100.0, .temp_scheduler = linear_scheduler, .seed = 2};

    vector<result> results = multithreaded_sim_anneal(Q, s, 4, 4);
    result best = results[0];

    cout << "Best energy: " << best.energy << '\n';
    cout << "Solution: " << best.solution << '\n';
    cout << '\n';

    present_results(results);

    return 0;
}
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
// using scheduler_t = double (*)(double T_0, double T, int iter, int max_iter);
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
        x_prime[flip_idx] = !x_prime[flip_idx];

        double f_x_prime = evaluate(x_prime, Q);

        if (f_x_prime < f_x || dis(gen) < exp((f_x - f_x_prime) / T)) {
            x = x_prime;
            f_x = f_x_prime;
        }

        if (f_x < best_f_x) {
            best_x = x;
            best_f_x = f_x;
            // cout << "New best energy: " << best_f_x << endl;
            // cout << "New best solution: ";
            // for (auto xi : best_x) {
            //     cout << xi << " ";
            // }
        }

        T = s.temp_scheduler(s.T_0, T, iter, s.max_iter);
    }

    // cout << "Best energy: " << best_f_x << endl;

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

    settings s = {.max_iter = 10000, .T_0 = 100.0, .temp_scheduler = make_geometric_scheduler(0.99), .seed = seed};

    vector<result> results = multithreaded_sim_anneal(Q, s, 8, 4);
    result best = results[0];

    cout << "Best energy: " << best.energy << '\n';
    cout << "Solution: " << best.solution << '\n';
    cout << '\n';

    present_results(results);

    return 0;
}